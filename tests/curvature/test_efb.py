"""Dense float64 oracles for the fixed-basis EKFAC-style correction."""

from dataclasses import replace

import pytest
import torch

from laplace_hypopinn.config import LaplaceConfig
from laplace_hypopinn.curvature.jacobians import iter_jacobians
from laplace_hypopinn.curvature.kfac import accumulate_factors
from laplace_hypopinn.curvature.residuals import ResidualBatch
from laplace_hypopinn.curvature.sampling import covariance_root, sample_vectors
from laplace_hypopinn.model import flatten_parameters, initialize_model
from laplace_hypopinn.types import RandomStreams


def fixture(widths=(2, 3, 2, 1)):
    parameters = initialize_model(widths, streams=RandomStreams(81), dtype=torch.float64)
    coordinates = torch.tensor([[.2, .4], [-.3, .8], [.9, -.2]], dtype=torch.float64)
    plan = tuple(
        ResidualBatch(
            family, coordinates, torch.full((3,), 2., dtype=torch.float64),
            torch.tensor([.11, .47, .23], dtype=torch.float64),
        )
        for family in ('data', 'pde', 'positivity')
    )
    return parameters, plan


def test_eigenvalues_and_raw_diagonal_match_full_mixed_derivative_ggn():
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues

    parameters, plan = fixture()
    basis = accumulate_factors(parameters, plan, chunk_size=2)
    stats = accumulate_eigenvalues(parameters, plan, basis, chunk_size=1)
    rows = torch.cat(tuple(iter_jacobians(parameters, plan, 2)))
    _, layout = flatten_parameters(parameters)
    for factor, block, values, diagonal in zip(
        basis, layout.blocks, stats.lambdas, stats.diagonals, strict=True,
    ):
        q = torch.kron(factor.QG.contiguous(), factor.QA.contiguous())
        layer_rows = rows[:, block.slice]
        dense = layer_rows.T @ layer_rows
        expected = torch.diag(q.T @ dense @ q).reshape(block.shape)
        torch.testing.assert_close(values, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(diagonal, dense.diag().reshape(block.shape))


def test_all_batches_accumulate_in_the_same_frozen_basis(monkeypatch):
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues, fit_efb

    parameters, plan = fixture()
    basis = accumulate_factors(parameters, plan)
    snapshots = tuple((factor.QG.clone(), factor.QA.clone()) for factor in basis)
    separately = tuple(accumulate_eigenvalues(parameters, batch, basis) for batch in plan)

    def no_eigensolve(*args, **kwargs):
        raise AssertionError('the second pass must not recompute an eigenbasis')

    monkeypatch.setattr(torch.linalg, 'eigh', no_eigensolve)
    config = LaplaceConfig(method='efb', scale=2.3, prior_precision=.31, chunk_size=2)
    posterior = fit_efb(parameters, plan, basis, config)
    for index, (qg, qa, precision) in enumerate(posterior.eigenbasis):
        expected = sum(stats.lambdas[index] for stats in separately)
        torch.testing.assert_close(precision, config.scale * expected + config.prior_precision)
        torch.testing.assert_close(qg, snapshots[index][0], rtol=0, atol=0)
        torch.testing.assert_close(qa, snapshots[index][1], rtol=0, atol=0)
    assert posterior.metadata['method'] == 'efb'
    assert 'ekfac' in posterior.metadata['approximation'].lower()


def test_normalized_duplicates_and_chunk_boundaries_preserve_statistics():
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues

    parameters, plan = fixture()
    basis = accumulate_factors(parameters, plan)
    doubled = tuple(
        ResidualBatch(batch.family, batch.coordinates.repeat(2, 1),
                      batch.targets.repeat(2), batch.alpha.repeat(2) / 2)
        for batch in plan
    )
    reference = accumulate_eigenvalues(parameters, plan, basis, chunk_size=1)
    actual = accumulate_eigenvalues(parameters, doubled, basis, chunk_size=4)
    for left, right in zip(reference.lambdas, actual.lambdas, strict=True):
        torch.testing.assert_close(left, right)
    for left, right in zip(reference.diagonals, actual.diagonals, strict=True):
        torch.testing.assert_close(left, right)


def test_covariance_root_and_seeded_draws_match_dense_efb_precision():
    from laplace_hypopinn.curvature.efb import fit_efb

    parameters, plan = fixture((2, 2, 1))
    basis = accumulate_factors(parameters, plan)
    posterior = fit_efb(parameters, plan, basis, LaplaceConfig(method='efb', scale=.7))
    dense_blocks = []
    for qg, qa, precision in posterior.eigenbasis:
        q = torch.kron(qg.contiguous(), qa.contiguous())
        dense_blocks.append((q * precision.flatten()[None, :]) @ q.T)
    dense_precision = torch.block_diag(*dense_blocks)
    root = covariance_root(posterior)
    torch.testing.assert_close(root @ root.T, torch.linalg.inv(dense_precision))

    samples = sample_vectors(posterior, 4, generator=torch.Generator().manual_seed(19))
    generator = torch.Generator().manual_seed(19)
    expected = []
    for _ in range(4):
        pieces = []
        for qg, qa, precision in posterior.eigenbasis:
            q = torch.kron(qg.contiguous(), qa.contiguous())
            noise = torch.randn(precision.shape, dtype=torch.float64, generator=generator)
            pieces.append(q @ (noise.flatten() / precision.flatten().sqrt()))
        expected.append(posterior.mean + torch.cat(pieces))
    torch.testing.assert_close(samples, torch.stack(expected))
    draws = sample_vectors(posterior, 12000, generator=torch.Generator().manual_seed(93))
    empirical = torch.cov((draws - posterior.mean).T)
    torch.testing.assert_close(empirical, torch.linalg.inv(dense_precision), atol=.04, rtol=.08)


@pytest.mark.parametrize('bad_basis', ['count', 'shape', 'orthogonality', 'nonfinite'])
def test_invalid_basis_is_rejected_before_replay(bad_basis):
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues

    parameters, plan = fixture()
    basis = accumulate_factors(parameters, plan)
    if bad_basis == 'count':
        basis = basis[:-1]
    else:
        qa = basis[0].QA.clone()
        if bad_basis == 'shape':
            qa = qa[:1]
        elif bad_basis == 'orthogonality':
            qa *= 2
        else:
            qa[0, 0] = float('nan')
        basis = (replace(basis[0], QA=qa), *basis[1:])
    with pytest.raises(ValueError, match='basis'):
        accumulate_eigenvalues(parameters, plan, basis)


def test_empty_residual_plan_is_rejected():
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues

    parameters, plan = fixture()
    basis = accumulate_factors(parameters, plan)
    with pytest.raises(ValueError, match='nonempty'):
        accumulate_eigenvalues(parameters, (), basis)
