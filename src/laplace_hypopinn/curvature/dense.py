"""Streamed weighted-residual GGN reference estimators."""

import torch

from ..model import flatten_parameters, predict
from .base import (DEFAULT_MAX_DENSE_BYTES, Posterior, cholesky_with_jitter,
                   guard_dense, require_scalar)


def fit_ggn(parameters, residual_plan, *, structure='diagonal', prior_precision=1.,
            scale=1., max_dense_bytes=DEFAULT_MAX_DENSE_BYTES, chunk_size=64,
            forward=predict):
    from .jacobians import iter_jacobians

    require_scalar(prior_precision, 'prior_precision', positive=True)
    require_scalar(scale, 'scale')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    aliases = {'full': 'dense', 'block': 'block_diagonal'}
    structure = aliases.get(structure, structure)
    if structure not in {'dense', 'diagonal', 'block_diagonal'}:
        raise ValueError('structure must be diagonal, block_diagonal or full')
    mean, layout = flatten_parameters(parameters)
    mean = mean.detach().clone()
    if structure == 'dense':
        guard_dense(layout.numel**2, mean.dtype, max_dense_bytes)
        values = [torch.zeros((layout.numel, layout.numel), dtype=mean.dtype, device=mean.device)]
    elif structure == 'block_diagonal':
        guard_dense(sum(block.numel**2 for block in layout.blocks), mean.dtype, max_dense_bytes)
        values = [torch.zeros((b.numel, b.numel), dtype=mean.dtype, device=mean.device)
                  for b in layout.blocks]
    else:
        values = [torch.zeros_like(mean)]
    for rows in iter_jacobians(parameters, residual_plan, chunk_size, forward=forward):
        rows = rows.detach()
        if not torch.isfinite(rows).all():
            raise ValueError('residual Jacobian contains nonfinite values')
        if structure == 'diagonal':
            values[0].add_(rows.square().sum(0), alpha=float(scale))
        elif structure == 'dense':
            values[0].add_(rows.T@rows, alpha=float(scale))
        else:
            for value, block in zip(values, layout.blocks, strict=True):
                local = rows[:, block.slice]
                value.add_(local.T@local, alpha=float(scale))
    metadata = {'estimator': 'weighted_residual_ggn', 'scale': float(scale),
                'prior_precision': float(prior_precision)}
    if structure == 'diagonal':
        return Posterior(mean, layout, structure, diagonal=values[0]+prior_precision,
                         metadata={**metadata, 'jitter': 0.}, forward=forward)
    factors, jitters = [], []
    for value in values:
        value.diagonal().add_(prior_precision)
        factor, jitter = cholesky_with_jitter(value)
        factors.append(factor)
        jitters.append(jitter)
    metadata['jitter'] = tuple(jitters)
    kwargs = {'factor': factors[0]} if structure == 'dense' else {'blocks': tuple(factors)}
    return Posterior(mean, layout, structure, metadata=metadata, forward=forward, **kwargs)


def precision_matrix(posterior, *, max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
    """Materialize precision for diagnostics only, after checking its byte cost."""
    guard_dense(posterior.layout.numel**2, posterior.mean.dtype, max_dense_bytes)
    kind = posterior.representation
    if kind == 'diagonal':
        return torch.diag(posterior.diagonal)
    if kind == 'dense':
        return posterior.factor@posterior.factor.T
    if kind == 'block_diagonal':
        return torch.block_diag(*(factor@factor.T for factor in posterior.blocks))
    if kind == 'low_rank':
        return torch.diag(posterior.diagonal)+posterior.low_rank@posterior.low_rank.T
    if kind == 'kronecker':
        values = []
        for qg, qa, eigenvalues in posterior.eigenbasis:
            basis = torch.kron(qg.contiguous(), qa.contiguous())
            values.append((basis*eigenvalues.reshape(-1))@basis.T)
        return torch.block_diag(*values)
    raise ValueError(f'unsupported posterior representation {kind!r}')
