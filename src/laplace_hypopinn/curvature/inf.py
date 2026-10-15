"""Rank-limited fixed EFB eigenpairs with an explicit PSD diagonal correction.

Each layer selects paired Kronecker eigenvectors in row-major ``G ⊗ A`` order.
No Cartesian closure or dense parameter covariance is constructed. Clipping a
negative diagonal remainder is a substantive approximation and is reported.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..model import predict
from .base import DEFAULT_MAX_DENSE_BYTES, Posterior, guard_dense, require_scalar
from .efb import _freeze_basis


@dataclass(frozen=True)
class INFBlock:
    """Unscaled low-rank data curvature and nonnegative parameter diagonal."""

    vectors: Tensor
    eigenvalues: Tensor
    selected_pairs: Tensor
    diagonal_remainder: Tensor
    requested_rank: int
    effective_rank: int
    clipped_count: int
    clipped_mass: float
    input_corrections: dict


def _nonnegative(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    magnitude = value.abs().max() if value.numel() else value.new_zeros(())
    tolerance = 100 * torch.finfo(value.dtype).eps * max(value.numel(), 1) * magnitude
    if (value < -tolerance).any():
        raise ValueError(f'{name} must be nonnegative; materially negative statistic')
    corrections = {
        'tolerance': float(tolerance), 'count': int((value < 0).sum()),
        'magnitude': float(-value.clamp_max(0).sum()),
    }
    return value.detach().clamp_min(0), corrections


def approximate_inf_blocks(kfac_basis, efb_stats, rank, *, max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
    """Select at most ``rank`` eigenpairs per matrix-shaped EFB block.

    Ties follow canonical row-major order. Selected zero eigenvalues retain
    their slots, so effective rank denotes selected columns, including any
    zero columns after scaling. Input statistics and bases are never mutated.
    ``max_dense_bytes`` bounds each selected-vector matrix before allocation.
    """
    if type(rank) is not int or rank < 0:
        raise ValueError('rank must be a nonnegative integer')
    factors = tuple(kfac_basis)
    lambdas, diagonals = tuple(efb_stats.lambdas), tuple(efb_stats.diagonals)
    if not factors or len(factors) != len(lambdas) or len(factors) != len(diagonals):
        raise ValueError('basis, eigenvalue and diagonal block counts must agree and be nonempty')
    results = []
    reference = factors[0].QG
    if reference.dtype not in (torch.float32, torch.float64):
        raise ValueError('INF basis must use float32 or float64')
    for factor, values, diagonal in zip(factors, lambdas, diagonals, strict=True):
        qg, qa = factor.QG, factor.QA
        for basis in (qg, qa):
            if (basis.ndim != 2 or basis.shape[0] != basis.shape[1] or basis.shape[0] == 0
                    or basis.dtype != reference.dtype or basis.device != reference.device
                    or not torch.isfinite(basis).all()):
                raise ValueError('invalid INF basis shape, dtype/device or values')
            identity = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device)
            if not torch.allclose(basis.T @ basis, identity, rtol=1e-5, atol=1e-6):
                raise ValueError('INF basis must be orthonormal')
        shape = (qg.shape[0], qa.shape[0])
        for statistic in (values, diagonal):
            if (statistic.shape != shape or statistic.dtype != reference.dtype
                    or statistic.device != reference.device):
                raise ValueError('INF statistics must match basis shape and dtype/device')
        values, value_correction = _nonnegative(values, 'EFB eigenvalues')
        diagonal, diagonal_correction = _nonnegative(diagonal, 'GGN diagonal')
        effective = min(rank, values.numel())
        guard_dense(values.numel() * effective, values.dtype, max_dense_bytes)
        indices = torch.argsort(values.flatten(), descending=True, stable=True)[:effective]
        rows, columns = indices // shape[1], indices % shape[1]
        pairs = torch.stack((rows, columns), dim=1)
        # Build only the selected qg[:,j] ⊗ qa[:,k] vectors, never full QG ⊗ QA.
        vectors = (qg[:, rows, None] * qa[:, columns].T[None, :, :]).permute(0, 2, 1).reshape(values.numel(), effective)
        vectors = vectors.detach()
        selected = values.flatten()[indices]
        remainder = diagonal.flatten() - (vectors.square() * selected).sum(1)
        results.append(INFBlock(
            vectors, selected, pairs, remainder.clamp_min(0), rank, effective,
            int((remainder < 0).sum()), float(-remainder.clamp_max(0).sum()),
            {'eigenvalues': value_correction, 'diagonal': diagonal_correction},
        ))
    return tuple(results)


def fit_inf(parameters, kfac_basis, efb_stats, config, *, forward=predict) -> Posterior:
    """Fit ``D + B B.T`` from a fixed basis and unregularized EFB statistics.

    ``D = scale * max(diag(F)-diag(U Lambda U.T), 0) + prior_precision`` and
    ``B = U sqrt(scale * Lambda)``. The shared low-rank posterior sampler uses
    exact perturb-and-solve draws, including singular low-rank Gram matrices.
    Named custom matrix blocks are supported with supplied compatible bases.
    """
    require_scalar(config.prior_precision, 'prior_precision', positive=True)
    require_scalar(config.scale, 'scale', positive=True)
    factors = tuple(kfac_basis)
    _, mean, layout = _freeze_basis(parameters, factors)
    approximations = approximate_inf_blocks(
        factors, efb_stats, config.rank, max_dense_bytes=config.max_dense_bytes,
    )
    total_rank = sum(block.effective_rank for block in approximations)
    guard_dense(layout.numel * total_rank, mean.dtype, config.max_dense_bytes)
    columns = mean.new_zeros((layout.numel, total_rank))
    diagonal = torch.empty_like(mean)
    start = 0
    for block, approximation in zip(layout.blocks, approximations, strict=True):
        stop = start + approximation.effective_rank
        columns[block.slice, start:stop] = approximation.vectors * (config.scale * approximation.eigenvalues).sqrt()
        diagonal[block.slice] = config.scale * approximation.diagonal_remainder + config.prior_precision
        start = stop
    return Posterior(
        mean=mean.detach().clone(), layout=layout, representation='low_rank',
        diagonal=diagonal, low_rank=columns, forward=forward,
        metadata={
            'method': 'inf', 'target': 'weighted_residual_ggn_approximation',
            'approximation': 'rank_limited_efb_plus_nonnegative_diagonal_correction',
            'prior_precision': config.prior_precision, 'scale': config.scale,
            'regularization': 'additive_isotropic_precision; prior only if matching MAP training',
            'requested_rank': config.rank,
            'effective_rank': [block.effective_rank for block in approximations],
            'total_effective_rank': total_rank,
            'selected_eigenpairs': [block.selected_pairs.tolist() for block in approximations],
            'clipped_count': sum(block.clipped_count for block in approximations),
            'clipped_mass': sum(block.clipped_mass for block in approximations),
            'block_clipped_count': [block.clipped_count for block in approximations],
            'block_clipped_mass': [block.clipped_mass for block in approximations],
            'input_statistic_corrections': [block.input_corrections for block in approximations],
        },
    )
