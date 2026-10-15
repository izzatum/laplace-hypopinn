"""Replayable scalar residuals and globally normalized likelihood weights.

For a family with term weight w and N observations, alpha=2*w/N gives
``mean(w*r**2) = .5*sum(alpha*r**2)``. The parameter prior is separate.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..config import TrainingConfig
from ..model import predict
from ..types import Forward, Parameters, TrainingBatch

ResidualFamily = Literal["data", "pde", "positivity"]


@dataclass(frozen=True)
class ResidualBatch:
    """One fixed family: targets are times for data, velocities for PDE.

    Positivity targets are unused. All arrays have a common leading row axis;
    alpha is the precision of each scalar residual, including normalization.
    """

    family: ResidualFamily
    coordinates: Tensor
    targets: Tensor
    alpha: Tensor

    def __post_init__(self):
        if self.family not in ("data", "pde", "positivity"):
            raise ValueError("unknown residual family")
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("residual coordinates must have shape (n,2)")
        n = self.coordinates.shape[0]
        if self.targets.shape not in ((n,), (n, 1)) or self.alpha.shape != (n,):
            raise ValueError("targets and alpha must match the number of residual rows")
        if not torch.isfinite(self.alpha).all() or (self.alpha < 0).any():
            raise ValueError("residual alpha must be finite and nonnegative")


ResidualPlan = tuple[ResidualBatch, ...]


def make_residual_plan(batch: TrainingBatch, config: TrainingConfig) -> ResidualPlan:
    """Snapshot the training selection once; iteration never samples new rows."""
    plan = []
    for family, coordinates, targets, weight in (
        ("data", batch.receiver_coordinates, batch.receiver_times, config.data_weight),
        ("pde", batch.coordinates, batch.velocity, config.pde_weight),
        (
            "positivity",
            batch.coordinates,
            torch.zeros_like(batch.velocity),
            config.positivity_weight,
        ),
    ):
        if weight == 0:
            continue
        count = coordinates.shape[0]
        plan.append(
            ResidualBatch(
                family,
                coordinates.detach().clone(),
                targets.detach().clone().reshape(-1),
                coordinates.new_full((count,), 2 * weight / count),
            )
        )
    return tuple(plan)


def scalar_prediction(parameters: Parameters, coordinate: Tensor, forward: Forward) -> Tensor:
    """Evaluate a pointwise scalar model, accepting (n,) or (n,1) outputs."""
    output = forward(parameters, coordinate.unsqueeze(0))
    if output.shape not in ((1,), (1, 1)):
        raise ValueError("forward must return one scalar per point, shape (n,) or (n,1)")
    return output.reshape(())


def scalar_residual(parameters, coordinate, target, family, forward):
    if family == "pde":
        gradient = torch.func.grad(scalar_prediction, argnums=1)(parameters, coordinate, forward)
        return gradient.square().sum() - target.square().reciprocal()
    prediction = scalar_prediction(parameters, coordinate, forward)
    if family == "data":
        return prediction - target
    return 2 * torch.relu(-prediction)


def residual_values(
    parameters: Parameters,
    batch: ResidualBatch,
    *,
    forward: Forward = predict,
) -> Tensor:
    """Unweighted residuals; PDE rows retain mixed coordinate/parameter derivatives."""
    return torch.func.vmap(
        lambda coordinate, target: scalar_residual(
            parameters, coordinate, target, batch.family, forward
        )
    )(batch.coordinates, batch.targets.reshape(-1))
