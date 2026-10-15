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


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('rank', [0, 2])
def test_low_rank_factorization_is_built_once_and_reused(monkeypatch, dtype, rank):
    from laplace_hypopinn.curvature import base, sampling
    calls = []
    original_qr, original_svd = torch.geqrf, torch.linalg.svd

    def counted_qr(matrix):
        calls.append(('qr', matrix.shape))
        return original_qr(matrix)

    def counted_svd(matrix, **kwargs):
        calls.append(('svd', matrix.shape))
        return original_svd(matrix, **kwargs)

    # Count only cached factorizations; random draws and transforms stay real.
    monkeypatch.setattr(torch, 'geqrf', counted_qr)
    monkeypatch.setattr(torch.linalg, 'svd', counted_svd)
    mean, layout = flatten_parameters({'w': torch.zeros(3, dtype=dtype)})
    diagonal = torch.tensor([1., 2., 3.], dtype=dtype)
    columns = torch.tensor([[.2, .3], [.4, -.5], [.6, .1]], dtype=dtype)[:, :rank]
    original_diagonal, original_columns = diagonal.clone(), columns.clone()
    global_rng = torch.random.get_rng_state().clone()
    posterior = base.Posterior(mean, layout, 'low_rank', diagonal=diagonal, low_rank=columns)
    expected_calls = [('qr', (3, rank)), ('svd', (rank, rank))] if rank else []
    assert calls == expected_calls
    if rank:
        assert all(value.dtype == dtype for value in (posterior._low_rank_qr,
                   posterior._low_rank_tau, posterior._low_rank_rotation, posterior._low_rank_scales))
    first = sampling.sample_vectors(posterior, 2, RandomStreams(10))
    second = sampling.sample_vectors(posterior, 1, RandomStreams(11))
    assert calls == expected_calls
    assert first.dtype == second.dtype == dtype
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    torch.testing.assert_close(diagonal, original_diagonal)
    torch.testing.assert_close(columns, original_columns)


def test_low_rank_cache_is_isolated_from_caller_tensor_mutation():
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import sample_vectors
    mean, layout = flatten_parameters({'w': torch.zeros(2, dtype=torch.float64)})
    diagonal = torch.tensor([1., 2.], dtype=mean.dtype)
    columns = torch.tensor([[.2], [.3]], dtype=mean.dtype)
    posterior = Posterior(mean, layout, 'low_rank', diagonal=diagonal, low_rank=columns)
    expected = sample_vectors(posterior, 2, RandomStreams(10))
    diagonal.fill_(5.)
    columns.fill_(7.)
    actual = sample_vectors(posterior, 2, RandomStreams(10))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonfinite_low_rank_whitening_is_rejected_at_construction():
    from laplace_hypopinn.curvature.base import Posterior
    mean, layout = flatten_parameters({'w': torch.zeros(2, dtype=torch.float32)})
    with pytest.raises(ValueError, match='finite'):
        Posterior(mean, layout, 'low_rank', diagonal=torch.full_like(mean, 1e-30),
                  low_rank=torch.full((2, 1), 1e30, dtype=mean.dtype))


@pytest.mark.parametrize('dtype,magnitude', [(torch.float32, 1e4), (torch.float32, 1e5),
                                            (torch.float64, 1e8), (torch.float64, 1e10)])
def test_strong_low_rank_precision_preserves_small_variance(dtype, magnitude):
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import covariance_root, sample_vectors
    mean, layout = flatten_parameters({'w': torch.zeros(1, dtype=dtype)})
    posterior = Posterior(mean, layout, 'low_rank', diagonal=torch.ones_like(mean),
                          low_rank=torch.full((1, 1), magnitude, dtype=dtype))
    # Scalar float64 arithmetic is independent of the sampler's factorization.
    variance = 1/(1+float(magnitude)**2)
    root = covariance_root(posterior)
    assert root.dtype == dtype
    torch.testing.assert_close(root.square(), torch.full_like(root, variance), rtol=3e-6, atol=0)
    samples = sample_vectors(posterior, 5000, RandomStreams(984))
    assert samples.dtype == dtype
    torch.testing.assert_close(samples.var(), torch.tensor(variance, dtype=dtype), rtol=.06, atol=0)
    assert int((samples == 0).sum()) < 5


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rotated_low_rank_root_matches_independent_eigen_oracle(dtype):
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import covariance_root
    mean, layout = flatten_parameters({'w': torch.zeros(3, dtype=dtype)})
    basis = torch.tensor([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]], dtype=dtype)
    diagonal = torch.tensor([.5, 2., 3.], dtype=dtype)
    columns = diagonal.sqrt()[:, None]*basis[:, :2]*torch.tensor([1e4, 3.], dtype=dtype)
    posterior = Posterior(mean, layout, 'low_rank', diagonal=diagonal, low_rank=columns)
    root = covariance_root(posterior)
    # Check covariance in known whitened eigendirections, including the small variance.
    projected = basis.T@(diagonal.sqrt()[:, None]*root)
    actual = projected.double()@projected.double().T
    expected = torch.diag(torch.tensor([1/(1+1e8), .1, 1.], dtype=torch.float64))
    torch.testing.assert_close(actual.diag(), expected.diag(), rtol=4e-4, atol=0)
    torch.testing.assert_close(actual-actual.diag().diag(), torch.zeros_like(actual), rtol=0, atol=2e-7 if dtype == torch.float32 else 1e-14)


def test_low_rank_cache_and_sampling_do_not_materialize_full_basis(monkeypatch):
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import sample_vectors
    size, rank = 257, 2
    mean, layout = flatten_parameters({'w': torch.zeros(size, dtype=torch.float64)})
    columns = torch.arange(size*rank, dtype=mean.dtype).reshape(size, rank)/100
    original_eye, original_svd = torch.eye, torch.linalg.svd
    def bounded_eye(n, *args, **kwargs):
        assert n <= rank
        return original_eye(n, *args, **kwargs)
    def bounded_svd(matrix, *args, **kwargs):
        assert matrix.shape[0] <= rank and matrix.shape[1] <= rank
        return original_svd(matrix, *args, **kwargs)
    monkeypatch.setattr(torch, 'eye', bounded_eye)
    monkeypatch.setattr(torch.linalg, 'svd', bounded_svd)
    posterior = Posterior(mean, layout, 'low_rank', diagonal=torch.ones_like(mean), low_rank=columns)
    assert sample_vectors(posterior, 2, RandomStreams(3)).shape == (2, size)


def test_low_rank_with_more_columns_than_parameters():
    from laplace_hypopinn.curvature.base import Posterior
    from laplace_hypopinn.curvature.sampling import covariance_root
    mean, layout = flatten_parameters({'w': torch.zeros(2, dtype=torch.float64)})
    diagonal = torch.tensor([1., 3.], dtype=mean.dtype)
    columns = torch.tensor([[1., 2., 3., 4.], [2., 0., 1., 3.]], dtype=mean.dtype)
    posterior = Posterior(mean, layout, 'low_rank', diagonal=diagonal, low_rank=columns)
    root = covariance_root(posterior)
    expected = torch.linalg.solve(diagonal.diag()+columns@columns.T, torch.eye(2, dtype=mean.dtype))
    torch.testing.assert_close(root@root.T, expected)
