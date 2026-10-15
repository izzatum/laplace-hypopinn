"""Pure notebook-compatible Eikonal, receiver, positivity, and prior losses."""

import torch
from torch.nn import functional as F

from .config import TrainingConfig
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
    # Reuse the same collocation graph for PDE and positivity. Computing two
    # mathematically equal forwards changes float32 gradient accumulation order
    # and the historical Adam trajectory. Corrected GGN estimators intentionally
    # retain the independent scalar residual graph in curvature.residuals.
    coordinates = batch.coordinates.detach().clone().requires_grad_()

    def values(points):
        output = forward(parameters, points)
        if output.shape not in ((points.shape[0],), (points.shape[0], 1)):
            raise ValueError('forward must return one scalar per point, shape (n,) or (n,1)')
        return output.reshape(-1, 1)

    times = values(coordinates)
    receiver_times = values(batch.receiver_coordinates)
    gradient = (torch.autograd.grad(
        times, coordinates, grad_outputs=torch.ones_like(times),
        create_graph=True, allow_unused=True,
    )[0] if times.requires_grad else None)
    if gradient is None:
        gradient = torch.zeros_like(coordinates)
    components = {
        'pde': F.mse_loss(gradient[:, 1:2].square() + gradient[:, 0:1].square(),
                          1.0 / batch.velocity.reshape(-1, 1).square()),
        'data': F.mse_loss(receiver_times, batch.receiver_times.reshape(-1, 1)),
        'positivity': F.mse_loss((1-torch.sign(times))*times.abs(), torch.zeros_like(times)),
    }
    zero = next(iter(parameters.values())).new_zeros(())
    prior = (
        0.5
        * config.prior_precision
        * sum((value.square().sum() for value in parameters.values()), zero)
    )
    total = (
        config.pde_weight * components["pde"]
        + config.data_weight * components["data"]
        + config.positivity_weight * components["positivity"]
    )
    if config.prior_precision:
        total = total + prior
    return LossTerms(total=total, prior=prior, **components)
