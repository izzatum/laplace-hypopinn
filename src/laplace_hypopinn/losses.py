"""Pure notebook-compatible Eikonal, receiver, positivity, and prior losses."""

import torch

from .config import TrainingConfig
from .curvature.residuals import ResidualBatch, residual_values
from .model import predict
from .types import Forward, LossTerms, Parameters, TrainingBatch


def loss_terms(
    parameters: Parameters,
    batch: TrainingBatch,
    config: TrainingConfig,
    *,
    forward: Forward = predict,
) -> LossTerms:
    """Return raw mean loss components and their weighted total.

    Positivity uses 2*relu(-T), equivalent to the notebook's
    (1-sign(T))*abs(T), including a zero loss gradient at T=0. The Gaussian
    parameter prior contributes .5*prior_precision*sum(theta**2) once.
    """
    zero = next(iter(parameters.values())).new_zeros(())
    components = {}
    for family, coordinates, targets in (
        ("pde", batch.coordinates, batch.velocity),
        ("data", batch.receiver_coordinates, batch.receiver_times),
        ("positivity", batch.coordinates, torch.zeros_like(batch.velocity)),
    ):
        residual_batch = ResidualBatch(
            family,
            coordinates,
            targets,
            coordinates.new_ones(coordinates.shape[0]),
        )
        components[family] = (
            residual_values(parameters, residual_batch, forward=forward).square().mean()
        )
    prior = (
        0.5
        * config.prior_precision
        * sum((value.square().sum() for value in parameters.values()), zero)
    )
    total = (
        config.pde_weight * components["pde"]
        + config.data_weight * components["data"]
        + config.positivity_weight * components["positivity"]
        + prior
    )
    return LossTerms(total=total, prior=prior, **components)
