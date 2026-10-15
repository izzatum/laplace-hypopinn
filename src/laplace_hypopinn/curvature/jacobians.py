"""Chunked derivative-aware residual rows in the canonical parameter layout."""

import torch

from ..model import flatten_parameters, predict, unflatten_parameters
from .residuals import ResidualBatch, scalar_residual


def iter_jacobians(parameters, plan, chunk_size, *, forward=predict):
    """Yield sqrt(alpha)*dr/dtheta, retaining fixed row weights across passes.

    Canonical flattening pairs MLP weights and biases row by row. Generic
    parameter mappings preserve insertion order and arbitrary tensor shapes.
    Only the current row chunk's Jacobian is materialized.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    vector, layout = flatten_parameters(parameters)
    if isinstance(plan, ResidualBatch):
        plan = (plan,)
    for batch in plan:

        def row(value, coordinate, target):
            restored = unflatten_parameters(value, layout)
            return scalar_residual(restored, coordinate, target, batch.family, forward)

        derivative = torch.func.jacrev(row, argnums=0)
        for start in range(0, batch.coordinates.shape[0], chunk_size):
            stop = start + chunk_size
            rows = torch.func.vmap(derivative, in_dims=(None, 0, 0))(
                vector,
                batch.coordinates[start:stop],
                batch.targets.reshape(-1)[start:stop],
            )
            yield rows * batch.alpha[start:stop].sqrt()[:, None]


def jacobian_blocks(parameters, batch, chunk_size, *, forward=predict):
    """Single-family alias of :func:`iter_jacobians`."""
    yield from iter_jacobians(parameters, batch, chunk_size, forward=forward)
