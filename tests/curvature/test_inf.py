"""Small float64 dense oracles for rank-limited EFB plus diagonal correction."""

from types import SimpleNamespace

import pytest
import torch

from laplace_hypopinn.config import LaplaceConfig
from laplace_hypopinn.curvature.efb import EFBStats
from laplace_hypopinn.curvature.sampling import covariance_root, sample_vectors
from laplace_hypopinn.model import flatten_parameters, initialize_model
from laplace_hypopinn.types import RandomStreams


def raw_fixture():
    qg = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    qa, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., -1., 1.], [1., 0., -1.]], dtype=torch.float64))
    basis = (SimpleNamespace(QG=qg, QA=qa),)
    values = torch.tensor([[9., 1., 2.], [3., 8., 0.]], dtype=torch.float64)
    diagonal = torch.full((2, 3), 10., dtype=torch.float64)
    return basis, EFBStats((values,), (diagonal,))


def test_exact_paired_rank_budget_and_canonical_kronecker_order():
    from laplace_hypopinn.curvature.inf import approximate_inf_blocks

    basis, stats = raw_fixture()
    block, = approximate_inf_blocks(basis, stats, rank=2)
    assert block.effective_rank == 2
    assert block.selected_pairs.tolist() == [[0, 0], [1, 1]]
    q = torch.kron(basis[0].QG, basis[0].QA.contiguous())
    torch.testing.assert_close(block.vectors, q[:, [0, 4]])
    torch.testing.assert_close(block.eigenvalues, torch.tensor([9., 8.], dtype=torch.float64))
    expected = stats.diagonals[0].flatten() - (q[:, [0, 4]].square() * torch.tensor([9., 8.])).sum(1)
    torch.testing.assert_close(block.diagonal_remainder, expected)


@pytest.mark.parametrize('rank, expected', [(0, 0), (2, 2), (20, 6)])
def test_repeated_and_zero_eigenvalues_respect_actual_budget(rank, expected):
    from laplace_hypopinn.curvature.inf import approximate_inf_blocks

    basis, stats = raw_fixture()
    stats = EFBStats((torch.tensor([[3., 3., 0.], [0., 0., 0.]], dtype=torch.float64),), stats.diagonals)
    block, = approximate_inf_blocks(basis, stats, rank)
    assert block.effective_rank == expected
    assert block.vectors.shape == (6, expected)
    if rank == 2:
        assert block.selected_pairs.tolist() == [[0, 0], [0, 1]]


def test_negative_remainder_is_recorded_and_raw_statistics_are_unchanged():
    from laplace_hypopinn.curvature.inf import approximate_inf_blocks

    basis, stats = raw_fixture()
    stats = EFBStats(stats.lambdas, (torch.zeros_like(stats.diagonals[0]),))
    values_before = stats.lambdas[0].clone()
    block, = approximate_inf_blocks(basis, stats, rank=2)
    q = torch.kron(basis[0].QG, basis[0].QA.contiguous())[:, [0, 4]]
    negative = (q.square() * torch.tensor([9., 8.])).sum(1)
    assert block.clipped_count == int((negative > 0).sum())
    assert block.clipped_mass == pytest.approx(float(negative.sum()))
    assert torch.equal(block.diagonal_remainder, torch.zeros(6, dtype=torch.float64))
    torch.testing.assert_close(stats.lambdas[0], values_before, rtol=0, atol=0)
    assert torch.count_nonzero(stats.diagonals[0]) == 0


def test_precision_covariance_and_perturb_solve_with_singular_gram():
    from laplace_hypopinn.curvature.inf import fit_inf

    basis, stats = raw_fixture()
    # A named arbitrary matrix is a valid canonical block, no preset shape required.
    parameters = {'custom.matrix': torch.zeros((2, 3), dtype=torch.float64)}
    stats = EFBStats((torch.tensor([[4., 0., 0.], [0., 0., 0.]], dtype=torch.float64),), stats.diagonals)
    config = LaplaceConfig(method='inf', rank=6, scale=.7, prior_precision=.3)
    posterior = fit_inf(parameters, basis, stats, config)
    assert posterior.low_rank.shape == (6, 6)
    assert torch.linalg.matrix_rank(posterior.low_rank) == 1
    precision = torch.diag(posterior.diagonal) + posterior.low_rank @ posterior.low_rank.T
    covariance = torch.linalg.solve(precision, torch.eye(6, dtype=torch.float64))
    root = covariance_root(posterior)
    torch.testing.assert_close(root @ root.T, covariance)
    # Build the entire deterministic perturb-and-solve root, including zero columns.
    noise_root = torch.cat((torch.diag(posterior.diagonal.sqrt()), posterior.low_rank), 1)
    sampling_root = torch.linalg.solve(precision, noise_root)
    torch.testing.assert_close(sampling_root @ sampling_root.T, covariance)
    draws = sample_vectors(posterior, 14000, generator=torch.Generator().manual_seed(18))
    torch.testing.assert_close(torch.cov(draws.T), covariance, atol=.006, rtol=.08)
    generator = torch.Generator().manual_seed(9)
    expected = []
    for _ in range(3):
        h = posterior.diagonal.sqrt() * torch.randn(6, dtype=torch.float64, generator=generator)
        h += posterior.low_rank @ torch.randn(6, dtype=torch.float64, generator=generator)
        expected.append(torch.linalg.solve(precision, h))
    actual = sample_vectors(posterior, 3, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(actual, torch.stack(expected))


def test_zero_rank_is_exact_diagonal_limit_with_additive_prior():
    from laplace_hypopinn.curvature.inf import fit_inf

    basis, stats = raw_fixture()
    parameters = {'arbitrary': torch.zeros((2, 3), dtype=torch.float64)}
    config = LaplaceConfig(method='inf', rank=0, scale=2.5, prior_precision=.4)
    posterior = fit_inf(parameters, basis, stats, config)
    assert posterior.low_rank.shape == (6, 0)
    torch.testing.assert_close(posterior.diagonal, 2.5 * stats.diagonals[0].flatten() + .4)
    assert posterior.metadata['effective_rank'] == [0]
    assert posterior.metadata['clipped_count'] == 0


@pytest.mark.parametrize('bad', ['negative', 'nonfinite', 'shape', 'dtype', 'basis', 'count', 'rank'])
def test_material_invalid_inputs_are_rejected(bad):
    from laplace_hypopinn.curvature.inf import approximate_inf_blocks

    basis, stats = raw_fixture()
    rank = 2
    if bad == 'negative':
        stats = EFBStats((-stats.lambdas[0],), stats.diagonals)
    elif bad == 'nonfinite':
        values = stats.diagonals[0].clone()
        values[0, 0] = float('nan')
        stats = EFBStats(stats.lambdas, (values,))
    elif bad == 'shape':
        stats = EFBStats((stats.lambdas[0].flatten(),), stats.diagonals)
    elif bad == 'dtype':
        stats = EFBStats((stats.lambdas[0].float(),), stats.diagonals)
    elif bad == 'basis':
        basis = (SimpleNamespace(QG=basis[0].QG * 2, QA=basis[0].QA),)
    elif bad == 'count':
        basis = ()
    else:
        rank = -1
    with pytest.raises(ValueError):
        approximate_inf_blocks(basis, stats, rank)


def test_short_nonlinear_pinn_multiple_arbitrary_widths():
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues
    from laplace_hypopinn.curvature.inf import fit_inf
    from laplace_hypopinn.curvature.kfac import accumulate_factors
    from laplace_hypopinn.curvature.residuals import ResidualBatch

    for widths in ((2, 2, 1), (2, 3, 2, 1)):
        parameters = initialize_model(widths, streams=RandomStreams(12), dtype=torch.float64)
        points = torch.tensor([[.2, .8], [-.4, .3]], dtype=torch.float64)
        plan = tuple(ResidualBatch(family, points, torch.ones(2, dtype=torch.float64), torch.full((2,), .2, dtype=torch.float64)) for family in ('data', 'pde', 'positivity'))
        basis = accumulate_factors(parameters, plan, chunk_size=1)
        stats = accumulate_eigenvalues(parameters, plan, basis, chunk_size=1)
        posterior = fit_inf(parameters, basis, stats, LaplaceConfig(method='inf', rank=2))
        mean, layout = flatten_parameters(parameters)
        assert posterior.low_rank.shape == (mean.numel(), 2 * len(layout.blocks))
        assert posterior.metadata['effective_rank'] == [2] * len(layout.blocks)
        samples = sample_vectors(posterior, 2, generator=torch.Generator().manual_seed(3))
        assert samples.shape == (2, mean.numel()) and torch.isfinite(samples).all()
        for block, count in zip(layout.blocks, posterior.metadata['effective_rank'], strict=True):
            assert count <= block.numel
