import pytest
import torch
from laplace_hypopinn.model import flatten_parameters, initialize_model
from laplace_hypopinn.types import RandomStreams


def test_covariance_root_and_seeded_empirical_samples():
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import covariance_root, sample_vectors
    mean, layout = flatten_parameters({'w': torch.zeros(2, dtype=torch.float64)})
    precision = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
    posterior = Posterior(mean, layout, 'dense', factor=torch.linalg.cholesky(precision))
    root = covariance_root(posterior)
    covariance = torch.linalg.solve(precision, torch.eye(2, dtype=precision.dtype))
    torch.testing.assert_close(root@root.T, covariance)
    samples = sample_vectors(posterior, 30000, RandomStreams(91))
    torch.testing.assert_close(torch.cov(samples.T), covariance, rtol=.04, atol=.008)
    assert torch.isfinite(samples).all()
    torch.testing.assert_close(posterior.mean, mean)


def test_legacy_layerwise_noise_order_and_nonmutation():
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import draw_parameters
    parameters = initialize_model((2, 2, 1), dtype=torch.float64)
    mean, layout = flatten_parameters(parameters)
    posterior = Posterior(mean.detach().clone(), layout, 'diagonal', diagonal=torch.ones_like(mean))
    actual = draw_parameters(posterior, 2, RandomStreams(41))
    rng = RandomStreams(41).torch
    for sample in range(2):
        for block in layout.blocks:
            noise = torch.randn(block.shape, dtype=mean.dtype, generator=rng)
            weight, bias = block.parameter_names
            torch.testing.assert_close(actual[weight][sample], parameters[weight]+noise[:, :-1])
            torch.testing.assert_close(actual[bias][sample], parameters[bias]+noise[:, -1])
    torch.testing.assert_close(flatten_parameters(parameters)[0], mean)


@pytest.mark.parametrize('rank', [0, 1, 2])
def test_low_rank_precision_sampling_and_root(rank):
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import covariance_root, sample_vectors
    mean, layout = flatten_parameters({'w': torch.zeros(3, dtype=torch.float64)})
    diagonal = torch.tensor([1., 2., 3.], dtype=mean.dtype)
    columns = torch.tensor([[.2, .3], [.4, -.5], [.6, .1]], dtype=mean.dtype)[:, :rank]
    posterior = Posterior(mean, layout, 'low_rank', diagonal=diagonal, low_rank=columns)
    covariance = torch.linalg.solve(diagonal.diag()+columns@columns.T, torch.eye(3, dtype=mean.dtype))
    root = covariance_root(posterior)
    torch.testing.assert_close(root@root.T, covariance)
    values = sample_vectors(posterior, 40000, RandomStreams(21))
    torch.testing.assert_close(torch.cov(values.T), covariance, rtol=.05, atol=.012)


def test_invalid_posterior_precision_rejected():
    from laplace_hypopinn.curvature.base import Posterior
    mean, layout = flatten_parameters({'w': torch.zeros(2)})
    with pytest.raises(ValueError):
        Posterior(mean, layout, 'diagonal', diagonal=torch.tensor([1., -1.]))


@pytest.mark.parametrize('representation', ['dense', 'low_rank', 'diagonal', 'block_diagonal', 'kronecker'])
def test_sampling_chunk_sizes_do_not_change_random_draws(representation):
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import sample_vectors
    mean, layout = flatten_parameters(initialize_model((2, 2, 1), dtype=torch.float64))
    kwargs = {}
    if representation == 'dense':
        kwargs['factor'] = torch.eye(mean.numel(), dtype=mean.dtype)
    elif representation == 'low_rank':
        kwargs['diagonal'] = torch.ones_like(mean)
        kwargs['low_rank'] = torch.ones((mean.numel(), 2), dtype=mean.dtype)/10
    elif representation == 'diagonal':
        kwargs['diagonal'] = torch.ones_like(mean)
    elif representation == 'block_diagonal':
        kwargs['blocks'] = tuple(torch.eye(b.numel, dtype=mean.dtype) for b in layout.blocks)
    else:
        kwargs['eigenbasis'] = tuple((torch.eye(b.shape[0], dtype=mean.dtype),
                                      torch.eye(b.shape[1], dtype=mean.dtype),
                                      torch.ones(b.shape, dtype=mean.dtype)) for b in layout.blocks)
    posterior = Posterior(mean, layout, representation, **kwargs)
    whole = sample_vectors(posterior, 7, RandomStreams(101))
    streams = RandomStreams(101)
    chunked = torch.cat([sample_vectors(posterior, n, streams) for n in [2, 1, 4]])
    torch.testing.assert_close(whole, chunked, rtol=0, atol=0)


def test_kronecker_rotation_and_block_factor_roots():
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.dense import precision_matrix
    from laplace_hypopinn.curvature.sampling import covariance_root, sample_vectors
    params = initialize_model((2, 1), dtype=torch.float64)
    mean, layout = flatten_parameters(params)
    qa = torch.tensor([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]], dtype=mean.dtype)
    eigenvalues = torch.tensor([[2., 3., 4.]], dtype=mean.dtype)
    posterior = Posterior(mean, layout, 'kronecker', eigenbasis=((torch.ones((1, 1), dtype=mean.dtype), qa, eigenvalues),))
    precision = qa@eigenvalues.reshape(-1).diag()@qa.T
    torch.testing.assert_close(precision_matrix(posterior), precision)
    root = covariance_root(posterior)
    torch.testing.assert_close(root@root.T, torch.linalg.solve(precision, torch.eye(3, dtype=mean.dtype)))
    streams = RandomStreams(7)
    expected_noise = torch.randn((1, 3), dtype=mean.dtype, generator=streams.torch)
    expected = mean+(expected_noise/eigenvalues.sqrt())@qa.T
    torch.testing.assert_close(sample_vectors(posterior, 1, RandomStreams(7)), expected)
    block = Posterior(mean, layout, 'block_diagonal', blocks=(torch.linalg.cholesky(precision),))
    torch.testing.assert_close(covariance_root(block)@covariance_root(block).T, root@root.T)
