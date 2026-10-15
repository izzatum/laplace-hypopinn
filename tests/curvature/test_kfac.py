import torch

from laplace_hypopinn.config import LaplaceConfig
from laplace_hypopinn.curvature.residuals import ResidualBatch
from laplace_hypopinn.curvature.jacobians import iter_jacobians
from laplace_hypopinn.curvature.jets import residual_layer_terms
from laplace_hypopinn.curvature.kfac import accumulate_factors, fit_kfac
from laplace_hypopinn.model import initialize_model
from laplace_hypopinn.types import RandomStreams


def fixture(widths=(2, 3, 1)):
    params = initialize_model(widths, streams=RandomStreams(81), dtype=torch.float64)
    coords = torch.tensor([[.2, .4], [-.3, .8], [.9, -.2]], dtype=torch.float64)
    batches = tuple(ResidualBatch(family, coords[:count], torch.full((count,), 2., dtype=torch.float64), torch.full((count,), .7/count, dtype=torch.float64)) for family, count in [('pde', 3), ('data', 2), ('positivity', 3)])
    return params, batches


def test_factors_match_explicit_exposure_formula_and_are_psd():
    params, plan = fixture()
    factors = accumulate_factors(params, plan, chunk_size=2)
    terms = [residual_layer_terms(params, batch) for batch in plan]
    exposure = sum(batch.alpha.sum() * (3 if batch.family == 'pde' else 1) for batch in plan)
    for index, factor in enumerate(factors):
        a = sum(torch.einsum('n,nsi,nsj->ij', batch.alpha, term.layers[index].activations, term.layers[index].activations) for batch, term in zip(plan, terms)) / exposure
        g = sum(torch.einsum('n,nsi,nsj->ij', batch.alpha, term.layers[index].adjoints, term.layers[index].adjoints) for batch, term in zip(plan, terms))
        torch.testing.assert_close(factor.A, a)
        torch.testing.assert_close(factor.G, g)
        assert torch.linalg.eigvalsh(a).min() > -1e-13
        assert torch.linalg.eigvalsh(g).min() > -1e-13
        torch.testing.assert_close(factor.exposure, exposure)


def test_duplicate_with_normalized_weights_and_chunking_invariant():
    params, plan = fixture()
    doubled = tuple(ResidualBatch(b.family, b.coordinates.repeat(2, 1), b.targets.repeat(2), b.alpha.repeat(2)/2) for b in plan)
    reference = accumulate_factors(params, plan, chunk_size=1)
    actual = accumulate_factors(params, doubled, chunk_size=4)
    for left, right in zip(reference, actual):
        torch.testing.assert_close(left.A, right.A)
        torch.testing.assert_close(left.G, right.G)


def test_affine_receiver_exactness_and_additive_prior_eigen_covariance():
    params, plan = fixture((2, 1))
    plan = (plan[1],)
    config = LaplaceConfig(method='kfac', prior_precision=.23, scale=1.7)
    posterior = fit_kfac(params, plan, config)
    rows = torch.cat(tuple(iter_jacobians(params, plan, 2)))
    dense_precision = config.scale * rows.T @ rows + config.prior_precision * torch.eye(rows.shape[1], dtype=torch.float64)
    assert posterior.representation == 'kronecker'
    qg, qa, precision = posterior.eigenbasis[0]
    q = torch.kron(qg.contiguous(), qa.contiguous())
    covariance = (q / precision.reshape(-1)[None, :]) @ q.T
    torch.testing.assert_close(covariance, torch.linalg.inv(dense_precision), atol=1e-12, rtol=1e-12)
    assert posterior.blocks == ()


def test_product_eigenvalues_match_explicit_kron_row_major():
    params, plan = fixture()
    config = LaplaceConfig(method='kfac', prior_precision=.8, scale=.4)
    factors = accumulate_factors(params, plan, chunk_size=2)
    posterior = fit_kfac(params, plan, config)
    for factor, (qg, qa, precision) in zip(factors, posterior.eigenbasis):
        q = torch.kron(qg.contiguous(), qa.contiguous())
        actual = (q * precision.flatten()[None, :]) @ q.T
        expected = config.scale*torch.kron(factor.G.contiguous(), factor.A.contiguous()) + config.prior_precision*torch.eye(actual.shape[0], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_sampler_uses_canonical_matrix_basis_without_dense_kron_storage():
    from laplace_hypopinn.curvature.sampling import sample_vectors

    params, plan = fixture((2, 2, 1))
    posterior = fit_kfac(params, plan, LaplaceConfig(method='kfac'))
    actual = sample_vectors(posterior, 3, generator=torch.Generator().manual_seed(93))
    generator = torch.Generator().manual_seed(93)
    expected = []
    for _ in range(3):
        pieces = []
        for qg, qa, precision in posterior.eigenbasis:
            noise = torch.randn(precision.shape, dtype=torch.float64, generator=generator)
            pieces.append((qg @ (noise / precision.sqrt()) @ qa.T).flatten())
        expected.append(torch.cat(pieces) + posterior.mean)
    torch.testing.assert_close(actual, torch.stack(expected))
    assert posterior.blocks == () and posterior.factor is None and posterior.low_rank is None


def test_receiver_only_factors_reduce_to_standard_single_channel_formula():
    params, plan = fixture()
    batch = plan[1]
    terms = residual_layer_terms(params, batch)
    factors = accumulate_factors(params, (batch,))
    for layer, factor in zip(terms.layers, factors):
        a = layer.activations[:, 0]
        b = layer.adjoints[:, 0]
        torch.testing.assert_close(factor.A, (a.T * batch.alpha) @ a / batch.alpha.sum())
        torch.testing.assert_close(factor.G, (b.T * batch.alpha) @ b)
