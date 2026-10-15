"""Dense float64 oracles for the fixed-inducing PINN VaLLA adaptation."""
from dataclasses import replace

import pytest
import torch

from laplace_hypopinn.config import LaplaceConfig, TrainingConfig
from laplace_hypopinn.curvature.jacobians import iter_jacobians
from laplace_hypopinn.curvature.residuals import make_residual_plan, residual_values
from laplace_hypopinn.model import flatten_parameters
from laplace_hypopinn.types import RandomStreams, TrainingBatch


def polynomial(parameters, coordinates):
    z, x = coordinates.unbind(-1)
    return parameters['offset'] + parameters['coefficients'][0]*z**2 + parameters['coefficients'][1]*x


def fixture():
    parameters = {'offset': torch.tensor(-.7, dtype=torch.float64, requires_grad=True),
                  'coefficients': torch.tensor([.4, -.2], dtype=torch.float64, requires_grad=True)}
    coordinates = torch.tensor([[.2, .3], [.5, .7], [.8, .4]], dtype=torch.float64)
    batch = TrainingBatch(coordinates, torch.tensor([2., 2.2, 2.4], dtype=torch.float64),
                          coordinates[:2], torch.tensor([.1, .3], dtype=torch.float64))
    plan = make_residual_plan(batch, TrainingConfig(pde_weight=.7, data_weight=1.3,
                                                   positivity_weight=.2))
    config = LaplaceConfig(method='valla', inducing_count=100, variational_steps=35,
                           prior_precision=1.4, scale=.8, chunk_size=2)
    return parameters, plan, config


def cache_fixture(config=None):
    from laplace_hypopinn.curvature.valla import build_valla_cache
    parameters, plan, default = fixture()
    config = default if config is None else config
    cache = build_valla_cache(parameters, plan, config, RandomStreams(7), forward=polynomial)
    return parameters, plan, config, cache


def test_complete_weighted_cache_and_full_inducing_exact_covariance_kl():
    from laplace_hypopinn.curvature.valla import valla_terms
    parameters, plan, config, cache = cache_fixture()
    jacobian = torch.cat(list(iter_jacobians(parameters, plan, 2, forward=polynomial)))*config.scale**.5
    mean = torch.cat([residual_values(parameters, b, forward=polynomial)*b.alpha.sqrt()
                      for b in plan])*config.scale**.5
    torch.testing.assert_close(cache.inducing_jacobian, jacobian)
    torch.testing.assert_close(cache.residual_mean, mean)
    torch.testing.assert_close(cache.prior_variance, jacobian.square().sum(1)/config.prior_precision)
    torch.testing.assert_close(cache.kernel_xz, jacobian@jacobian.T/config.prior_precision)
    factor = torch.eye(jacobian.shape[0], dtype=torch.float64)
    variance, kl = valla_terms(factor, cache, config.prior_precision)
    identity = torch.eye(jacobian.shape[1], dtype=torch.float64)
    precision = config.prior_precision*identity+jacobian.T@jacobian
    covariance = torch.linalg.solve(precision, identity)
    torch.testing.assert_close(variance, (jacobian@covariance*jacobian).sum(1))
    expected_kl = .5*(config.prior_precision*covariance.trace()-identity.shape[0]
                       -torch.linalg.slogdet(config.prior_precision*covariance).logabsdet)
    torch.testing.assert_close(kl, expected_kl)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    assert not cache.inducing_jacobian.requires_grad
    assert not cache.kernel_xz.requires_grad


@pytest.mark.parametrize('alpha', [0, 1])
def test_author_data_terms_kl_and_variational_gradients(alpha):
    from laplace_hypopinn.curvature.valla import valla_objective, valla_terms
    _, _, config, cache = cache_fixture()
    factor = (torch.eye(cache.inducing_jacobian.shape[0], dtype=torch.float64)*.7).requires_grad_()
    variance, kl = valla_terms(factor, cache, config.prior_precision)
    mean = cache.residual_mean
    data = .5*(mean.square()+variance) if alpha == 0 else .5*torch.log1p(variance)+.5*mean.square()/(1+variance)
    objective = valla_objective(factor, cache, config.prior_precision, alpha)
    torch.testing.assert_close(objective, data.sum()+kl)
    assert torch.autograd.gradcheck(lambda value: valla_objective(value, cache, config.prior_precision, alpha),
                                   (factor,), eps=1e-6, atol=1e-5)
    objective.backward()
    assert torch.isfinite(factor.grad).all()
    assert factor.grad.tril().abs().sum() > 0
    assert factor.grad.triu(1).count_nonzero() == 0


def test_nondiagonal_factor_function_covariance_and_kl_dense_oracle():
    from laplace_hypopinn.curvature.valla import valla_terms
    _, _, config, cache = cache_fixture()
    count = cache.inducing_jacobian.shape[0]
    factor = torch.linspace(-.2, .8, count*count, dtype=torch.float64).reshape(count, count).tril()
    variance, kl = valla_terms(factor, cache, config.prior_precision)
    columns = cache.inducing_jacobian.T@factor
    identity = torch.eye(columns.shape[0], dtype=torch.float64)
    covariance = torch.linalg.solve(config.prior_precision*identity+columns@columns.T, identity)
    jacobian = cache.inducing_jacobian
    torch.testing.assert_close(variance, (jacobian@covariance*jacobian).sum(1))
    expected = .5*(config.prior_precision*covariance.trace()-identity.shape[0]
                  -torch.linalg.slogdet(config.prior_precision*covariance).logabsdet)
    torch.testing.assert_close(kl, expected)


def test_selection_cache_frozen_and_chunk_independent():
    from laplace_hypopinn.curvature.valla import build_valla_cache
    parameters, plan, config = fixture()
    config = replace(config, inducing_count=3)
    cache = build_valla_cache(parameters, plan, config, RandomStreams(33), forward=polynomial)
    other = build_valla_cache(parameters, plan, replace(config, chunk_size=1), RandomStreams(33), forward=polynomial)
    assert cache.inducing_identities == other.inducing_identities
    assert len(set(cache.inducing_identities)) == 3
    assert {identity[1] for identity in cache.inducing_identities} <= {'data','pde','positivity'}
    torch.testing.assert_close(cache.kernel_xz, other.kernel_xz)
    snapshot = cache.inducing_jacobian.clone()
    with torch.no_grad():
        parameters['offset'].add_(10)
        plan[0].coordinates.add_(10)
        plan[0].alpha.add_(10)
    torch.testing.assert_close(snapshot, cache.inducing_jacobian)


@pytest.mark.parametrize('alpha', [0, 1])
def test_fit_only_variational_factor_and_reduces_objective(alpha):
    from laplace_hypopinn.curvature.valla import fit_valla
    from laplace_hypopinn.curvature.dense import precision_matrix
    parameters, plan, config = fixture()
    before, _ = flatten_parameters(parameters)
    before = before.detach().clone()
    posterior = fit_valla(parameters, plan, replace(config, inducing_count=3, alpha=alpha),
                          RandomStreams(7), forward=polynomial)
    after, _ = flatten_parameters(parameters)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(before, posterior.mean)
    assert all(value.grad is None for value in parameters.values())
    assert posterior.prediction_kind == 'linearized'
    assert posterior.forward is polynomial
    history = posterior.metadata['objective_history']
    assert len(history) == config.variational_steps+1
    assert history[-1] < history[0]
    assert torch.linalg.eigvalsh(precision_matrix(posterior)).min() >= config.prior_precision-1e-12
    assert torch.isfinite(posterior.low_rank).all()


def test_linearized_jvp_and_shared_draw_are_coherent_across_grid_chunks():
    from laplace_hypopinn.curvature.valla import fit_valla, linearized_predict
    from laplace_hypopinn.curvature.sampling import sample_vectors
    parameters, plan, config = fixture()
    posterior = fit_valla(parameters, plan, replace(config, variational_steps=1), RandomStreams(5), forward=polynomial)
    delta = sample_vectors(posterior, 1, RandomStreams(14))[0]-posterior.mean
    coordinates = torch.linspace(0, 1, 26, dtype=torch.float64).reshape(13, 2)
    expected = polynomial(parameters, coordinates)+delta[0]+delta[1]*coordinates[:,0]**2+delta[2]*coordinates[:,1]
    full = linearized_predict(parameters, delta, coordinates, forward=polynomial)
    torch.testing.assert_close(full, expected)
    for chunk in (1, 3, 8):
        chunked = torch.cat([linearized_predict(parameters, delta, x, forward=polynomial)
                             for x in coordinates.split(chunk)])
        torch.testing.assert_close(full, chunked)
        assert full.argmin() == chunked.argmin()


@pytest.mark.parametrize('prior', [0, -1, float('nan')])
def test_nonpositive_or_nonfinite_prior_rejected(prior):
    from laplace_hypopinn.curvature.valla import valla_objective
    _, _, _, cache = cache_fixture()
    with pytest.raises(ValueError, match='prior'):
        valla_objective(torch.eye(8, dtype=torch.float64), cache, prior, 0)


def test_unsupported_alpha_and_changed_prior_rejected():
    from laplace_hypopinn.curvature.valla import valla_objective
    _, _, config, cache = cache_fixture()
    factor = torch.eye(cache.inducing_jacobian.shape[0], dtype=torch.float64)
    with pytest.raises(ValueError, match='alpha'):
        valla_objective(factor, cache, config.prior_precision, .5)
    with pytest.raises(ValueError, match='fixed'):
        valla_objective(factor, cache, config.prior_precision+1, 0)


def test_shared_low_rank_sampler_covariance_matches_valla_precision():
    from laplace_hypopinn.curvature.valla import fit_valla
    from laplace_hypopinn.curvature.sampling import sample_vectors
    from laplace_hypopinn.curvature.dense import precision_matrix
    parameters, plan, config = fixture()
    posterior = fit_valla(parameters, plan, replace(config, variational_steps=2), RandomStreams(6), forward=polynomial)
    samples = sample_vectors(posterior, 25000, RandomStreams(21))-posterior.mean
    precision = precision_matrix(posterior)
    expected = torch.linalg.solve(precision, torch.eye(precision.shape[0], dtype=torch.float64))
    torch.testing.assert_close(torch.cov(samples.T), expected, rtol=.06, atol=.004)


def test_jvp_prediction_preserves_nonlinear_map_and_uses_linearized_perturbation():
    from laplace_hypopinn.curvature.valla import linearized_predict
    parameters = {'amplitude': torch.tensor(.7, dtype=torch.float64)}
    def nonlinear(values, coordinates):
        return (values['amplitude'].square()*coordinates[:, :1]).sin()
    coordinates = torch.tensor([[.2, .1], [.5, .4]], dtype=torch.float64)
    delta = torch.tensor([.3], dtype=torch.float64)
    expected = nonlinear(parameters, coordinates)+torch.cos(.7**2*coordinates[:, :1])*2*.7*coordinates[:, :1]*.3
    actual = linearized_predict(parameters, delta, coordinates, forward=nonlinear)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, nonlinear({'amplitude': parameters['amplitude']+delta[0]}, coordinates))
    torch.testing.assert_close(linearized_predict(parameters, delta*0, coordinates, forward=nonlinear),
                               nonlinear(parameters, coordinates))


def test_cache_dense_memory_guard_and_empty_plan():
    from laplace_hypopinn.curvature.valla import build_valla_cache
    parameters, plan, config = fixture()
    with pytest.raises(MemoryError):
        build_valla_cache(parameters, plan, replace(config, max_dense_bytes=1), RandomStreams(4), forward=polynomial)
    with pytest.raises(ValueError, match='residual'):
        build_valla_cache(parameters, (), config, RandomStreams(4), forward=polynomial)


def test_fit_reports_actual_roundoff_variance_correction():
    from laplace_hypopinn.curvature.valla import fit_valla
    parameters, plan, config = fixture()
    posterior = fit_valla(parameters, plan, replace(config, variational_steps=1), RandomStreams(6), forward=polynomial)
    assert posterior.metadata['variance_clipped_count'] == 0
    assert posterior.metadata['variance_clipped_magnitude'] == 0
    assert posterior.metadata['variance_roundoff_tolerance'] > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('jacobian_magnitude', [1e4, 1e10])
@pytest.mark.parametrize('factor_value', [.5, .9, 1.])
def test_strongly_informed_variance_and_objective_gradient(dtype, jacobian_magnitude, factor_value):
    from laplace_hypopinn.curvature.residuals import ResidualBatch
    from laplace_hypopinn.curvature.valla import build_valla_cache, valla_objective, valla_terms
    parameters = {'weight': torch.tensor(0., dtype=dtype)}
    def scalar_model(values, coordinates):
        return values['weight']*coordinates[:, 0]
    batch = ResidualBatch('data', torch.tensor([[jacobian_magnitude, 0.]], dtype=dtype),
                          torch.zeros(1, dtype=dtype), torch.ones(1, dtype=dtype))
    config = LaplaceConfig(method='valla', inducing_count=1)
    cache = build_valla_cache(parameters, (batch,), config, RandomStreams(5), forward=scalar_model)
    factor = torch.tensor([[factor_value]], dtype=dtype, requires_grad=True)
    variance, _ = valla_terms(factor, cache, 1.)
    gradient = torch.autograd.grad(valla_objective(factor, cache, 1., 0), factor)[0]
    # Independent scalar closed form; avoid a subtraction of two large kernel terms.
    j2 = jacobian_magnitude**2
    precision = 1+j2*factor_value**2
    expected_variance = j2/precision
    expected_gradient = -j2*j2*factor_value/precision**2+j2*factor_value/precision-j2*factor_value/precision**2
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(variance, torch.tensor([expected_variance], dtype=dtype),
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(gradient, torch.tensor([[expected_gradient]], dtype=dtype),
                               atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('parameter_count', [1, 2, 4])
def test_strong_rank_deficient_inducing_features_preserve_prior_directions(dtype, parameter_count):
    from laplace_hypopinn.curvature.residuals import ResidualBatch
    from laplace_hypopinn.curvature.valla import build_valla_cache, valla_terms
    parameters = {'weights': torch.zeros(parameter_count, dtype=dtype)}
    def rank_one_model(values, coordinates):
        return values['weights'].sum()*coordinates[:, 0]
    magnitude = 1e10
    batch = ResidualBatch('data', torch.tensor([[magnitude, 0.], [magnitude, 0.]], dtype=dtype),
                          torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype))
    cache = build_valla_cache(parameters, (batch,), LaplaceConfig(method='valla', inducing_count=2),
                              RandomStreams(5), forward=rank_one_model)
    factor = (torch.eye(2, dtype=dtype)*.5).requires_grad_()
    variance, kl = valla_terms(factor, cache, 1.)
    j2 = parameter_count*magnitude**2
    informed_precision = 1+.5*j2
    torch.testing.assert_close(variance, torch.full((2,), j2/informed_precision, dtype=dtype),
                               atol=2e-5, rtol=2e-5)
    expected_kl = .5*(torch.log(torch.tensor(informed_precision, dtype=torch.float64))-1+1/informed_precision)
    torch.testing.assert_close(kl, expected_kl.to(dtype), atol=2e-5, rtol=2e-5)
    gradient = torch.autograd.grad(.5*variance.sum()+kl, factor)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_large_informed_projection_preserves_uninformed_query_variance(dtype):
    from laplace_hypopinn.curvature.residuals import ResidualBatch
    from laplace_hypopinn.curvature.valla import build_valla_cache, valla_terms
    parameters = {'weights': torch.zeros(3, dtype=dtype)}
    def mixed_features(values, coordinates):
        return (values['weights'][0]+values['weights'][1])*coordinates[:, 0]+values['weights'][2]*coordinates[:, 1]
    magnitude = 1e10
    batch = ResidualBatch('data', torch.tensor([[magnitude, 0.], [magnitude, 2.]], dtype=dtype),
                          torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype))
    cache = build_valla_cache(parameters, (batch,), LaplaceConfig(method='valla', inducing_count=1),
                              RandomStreams(1), forward=mixed_features)
    assert cache.inducing_identities == ((0, 'data', 0),)
    variance, _ = valla_terms(torch.full((1, 1), .5, dtype=dtype), cache, 1.)
    informed = 2*magnitude**2/(1+.5*magnitude**2)
    torch.testing.assert_close(variance, torch.tensor([informed, informed+4], dtype=dtype),
                               atol=2e-5, rtol=2e-5)


def test_cache_guard_accounts_for_augmented_precision_square_root():
    from laplace_hypopinn.curvature.residuals import ResidualBatch
    from laplace_hypopinn.curvature.valla import build_valla_cache
    parameters = {'weights': torch.zeros(2, dtype=torch.float32)}
    def linear(values, coordinates):
        return coordinates@values['weights']
    batch = ResidualBatch('data', torch.eye(2, dtype=torch.float32),
                          torch.zeros(2), torch.ones(2))
    # 2x2 caches fit in 32 work-dtype bytes; the 4x2 augmented QR does not.
    config = LaplaceConfig(method='valla', inducing_count=2, chunk_size=2, max_dense_bytes=32)
    with pytest.raises(MemoryError):
        build_valla_cache(parameters, (batch,), config, RandomStreams(5), forward=linear)
