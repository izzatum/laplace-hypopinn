"""Curvature foundations: weighted scalar residuals and parameter Jacobians."""

from .jacobians import iter_jacobians, jacobian_blocks
from .residuals import ResidualBatch, ResidualPlan, make_residual_plan, residual_values

__all__ = [
    "ResidualBatch",
    "ResidualPlan",
    "make_residual_plan",
    "residual_values",
    "iter_jacobians",
    "jacobian_blocks",
]
