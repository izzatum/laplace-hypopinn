"""Historical batch-gradient-square statistic, explicitly distinct from a GGN."""

import numpy as np
import torch

from ..config import TrainingConfig
from ..model import flatten_parameters, predict
from ..types import TrainingBatch
from .base import Posterior, require_scalar


def fit_legacy_diagonal(parameters, problem, config, streams, *, forward=predict,
                        training_config=None):
    from ..losses import loss_terms

    require_scalar(config.add, 'add', positive=True)
    require_scalar(config.multiply, 'multiply')
    if config.batches < 1 or config.batch_size < 1:
        raise ValueError('batches and batch_size must be positive')
    training_config = TrainingConfig() if training_config is None else training_config
    local = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    mean, layout = flatten_parameters(local)
    statistic = torch.zeros_like(mean)
    coordinates = problem.grid.coordinates
    velocity = problem.grid.velocity.reshape(-1)
    for _ in range(config.batches):
        indices = streams.numpy.choice(np.arange(coordinates.shape[0]), config.batch_size, replace=True)
        indices = torch.as_tensor(indices, device=coordinates.device)
        batch = TrainingBatch(coordinates[indices], velocity[indices],
                              problem.receiver_coordinates, problem.receiver_times)
        total = loss_terms(local, batch, training_config, forward=forward).total
        if not torch.isfinite(total):
            raise ValueError('nonfinite legacy curvature loss')
        gradients = torch.autograd.grad(total, tuple(local.values()), allow_unused=True)
        gradient_map = {name: torch.zeros_like(value) if grad is None else grad
                        for (name, value), grad in zip(local.items(), gradients, strict=True)}
        gradient, _ = flatten_parameters(gradient_map)
        if not torch.isfinite(gradient).all():
            raise ValueError('nonfinite legacy batch gradient')
        statistic.add_(gradient.detach().square(), alpha=config.batch_size)
    precision = config.add+config.multiply*statistic
    return Posterior(mean.detach().clone(), layout, 'diagonal', diagonal=precision,
                     metadata={'estimator': 'legacy_batch_gradient_square',
                               'legacy_statistic': statistic, 'add': config.add,
                               'multiply': config.multiply, 'batches': config.batches,
                               'batch_size': config.batch_size, 'jitter': 0.}, forward=forward)
