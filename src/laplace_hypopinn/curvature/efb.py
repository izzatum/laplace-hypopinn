"""EKFAC-style diagonal correction in a fixed Kronecker eigenbasis.

EFB uses the exact weighted residual Jacobians in the KFAC eigenvectors. This
is the eigenvalue correction idea of EKFAC, adapted to the residual GGN of a
PINN: coordinate-jet channel cross terms are included before squaring. It
discards off-diagonal curvature in that basis and remains an approximation.
The caller must replay the same fixed residual plan used to fit KFAC.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..model import flatten_parameters, predict
from .base import Posterior
from .kfac import iter_layer_terms


@dataclass(frozen=True)
class EFBStats:
    """Unscaled, unregularized per-layer GGN statistics for EFB and INF.

    ``lambdas`` are diagonal entries in the fixed eigenbasis; ``diagonals``
    are diagonal entries in parameter coordinates. Both use matrix-shaped
    row-major ``[W|b]`` blocks and include every weighted residual row.
    """

    lambdas: tuple[Tensor, ...]
    diagonals: tuple[Tensor, ...]


def _freeze_basis(parameters, kfac_basis):
    mean, layout = flatten_parameters(parameters)
    factors = tuple(kfac_basis)
    if len(factors) != len(layout.blocks):
        raise ValueError('KFAC basis blocks must match parameter layout')
    basis = []
    for factor, block in zip(factors, layout.blocks, strict=True):
        if len(block.shape) != 2:
            raise ValueError('KFAC basis requires matrix parameter blocks')
        rows, columns = block.shape
        pair = []
        for value, size in ((factor.QG, rows), (factor.QA, columns)):
            if (
                value.shape != (size, size) or not torch.isfinite(value).all()
                or value.dtype != mean.dtype or value.device != mean.device
            ):
                raise ValueError('KFAC basis shape, dtype/device or values are invalid')
            identity = torch.eye(size, dtype=value.dtype, device=value.device)
            if not torch.allclose(value.T @ value, identity, rtol=1e-5, atol=1e-6):
                raise ValueError('KFAC basis must be orthonormal')
            pair.append(value.detach().clone())
        basis.append(tuple(pair))
    return tuple(basis), mean, layout


def _accumulate(parameters, residual_plan, basis, layout, chunk_size, forward, jet_adapter):
    lambdas = tuple(qg.new_zeros(block.shape) for (qg, _), block in zip(basis, layout.blocks))
    diagonals = tuple(torch.zeros_like(values) for values in lambdas)
    has_rows = False
    for batch, terms in iter_layer_terms(
        parameters, residual_plan, chunk_size, forward=forward, jet_adapter=jet_adapter,
    ):
        if len(terms.layers) != len(layout.blocks):
            raise ValueError('jet adapter layers must match canonical parameter blocks')
        has_rows = True
        for layer, block, (qg, qa), values, diagonal in zip(
            terms.layers, layout.blocks, basis, lambdas, diagonals, strict=True,
        ):
            matrices = layer.matrices
            if matrices.shape != (len(batch.alpha), *block.shape):
                raise ValueError('jet adapter matrix shapes must match canonical parameter blocks')
            if not torch.isfinite(matrices).all():
                raise ValueError('nonfinite residual layer Jacobian')
            rotated = qg.T @ matrices @ qa
            values.add_(torch.einsum('n,nij->ij', batch.alpha, rotated.square()))
            diagonal.add_(torch.einsum('n,nij->ij', batch.alpha, matrices.square()))
    if not has_rows:
        raise ValueError('EFB requires a nonempty residual plan')
    if any(not torch.isfinite(value).all() for value in (*lambdas, *diagonals)):
        raise ValueError('nonfinite accumulated EFB statistics')
    return EFBStats(lambdas, diagonals)


def accumulate_eigenvalues(
    parameters, residual_plan, kfac_basis, *, chunk_size=64, forward=predict, jet_adapter=None,
):
    """Replay exact layer Jacobians once in supplied, frozen KFACFactor bases.

    Accumulate ``sum_i alpha_i * (QG.T @ M_i @ QA)**2`` and
    ``sum_i alpha_i * M_i**2`` without recomputing eigenvectors or multiplying
    Jacobians by residual values. No scale or prior is applied here.
    """
    basis, _, layout = _freeze_basis(parameters, kfac_basis)
    return _accumulate(
        parameters, residual_plan, basis, layout, chunk_size, forward, jet_adapter,
    )


def fit_efb(
    parameters, residual_plan, kfac_basis, config, *, forward=predict, jet_adapter=None,
) -> Posterior:
    """Fit ``scale * Lambda + prior_precision`` in a fixed Kronecker basis.

    Arbitrary-width/depth MLPs use exact coordinate jets; custom forwards must
    provide a compatible explicit jet adapter, as in the KFAC first pass.
    Additive regularization is a prior only when matching MAP training.
    """
    basis, mean, layout = _freeze_basis(parameters, kfac_basis)
    stats = _accumulate(
        parameters, residual_plan, basis, layout, config.chunk_size, forward, jet_adapter,
    )
    eigenbasis = tuple(
        (qg, qa, config.scale * values + config.prior_precision)
        for (qg, qa), values in zip(basis, stats.lambdas, strict=True)
    )
    return Posterior(
        mean=mean.detach().clone(), layout=layout, representation='kronecker',
        eigenbasis=eigenbasis, forward=forward,
        metadata={
            'method': 'efb', 'target': 'weighted_residual_ggn_approximation',
            'approximation': 'ekfac_style_exact_jacobian_diagonal_in_fixed_kfac_basis',
            'prior_precision': config.prior_precision, 'scale': config.scale,
            'regularization': 'additive_isotropic_precision; prior only if matching MAP training',
            'basis': 'fixed_kronecker_eigenvectors_from_kfac',
            'includes_coordinate_channel_cross_terms': True,
        },
    )
