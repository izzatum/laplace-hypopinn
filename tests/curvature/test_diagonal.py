import numpy as np
import torch

from laplace_hypopinn.config import LaplaceConfig, TrainingConfig
from laplace_hypopinn.model import flatten_parameters, initialize_model, predict
from laplace_hypopinn.types import Grid, Problem, RandomStreams, TrainingBatch


def test_legacy_accumulation_matches_archived_diagonal_batch_gradient():
    from laplace_hypopinn.curvature.diagonal import fit_legacy_diagonal
    from laplace_hypopinn.losses import loss_terms
    parameters = initialize_model((2, 2, 1), dtype=torch.float64)
    axes = torch.tensor([0., 1.], dtype=torch.float64)
    grid = Grid(axes, axes, torch.full((2, 2), 2., dtype=torch.float64))
    problem = Problem(grid, grid.coordinates[:2], torch.tensor([.1, .2], dtype=torch.float64))
    config = LaplaceConfig(batches=3, batch_size=3, add=2., multiply=.3)
    training = TrainingConfig(pde_weight=.7, data_weight=.2, positivity_weight=.4)
    reference = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Mish(), torch.nn.Linear(2, 1)).double()
    with torch.no_grad():
        for layer, i in [(reference[0], 0), (reference[2], 1)]:
            layer.weight.copy_(parameters[f'layers.{i}.weight'])
            layer.bias.copy_(parameters[f'layers.{i}.bias'])
    # Import the unchanged archive as an independent historical oracle.
    import sys
    from pathlib import Path
    archive = Path(__file__).parents[2]/'archive'/'legacy'
    sys.path.insert(0, str(archive))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from curvature.curvatures import Diagonal
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.path.remove(str(archive))
    legacy = Diagonal(reference)
    rng = np.random.RandomState(18)
    for _ in range(config.batches):
        indices = rng.choice(np.arange(4), config.batch_size, replace=True)
        batch = TrainingBatch(grid.coordinates[indices], grid.velocity.reshape(-1)[indices],
                              problem.receiver_coordinates, problem.receiver_times)
        local = {'layers.0.weight': reference[0].weight, 'layers.0.bias': reference[0].bias,
                 'layers.1.weight': reference[2].weight, 'layers.1.bias': reference[2].bias}
        reference.zero_grad()
        loss_terms(local, batch, training, forward=predict).total.backward()
        legacy.update(config.batch_size)
    statistic = torch.cat([value.reshape(-1) for value in legacy.state.values()])
    before = {key: value.clone() for key, value in parameters.items()}
    posterior = fit_legacy_diagonal(parameters, problem, config, RandomStreams(18), training_config=training)
    torch.testing.assert_close(posterior.diagonal, config.add+config.multiply*statistic)
    torch.testing.assert_close(posterior.metadata['legacy_statistic'], statistic)
    assert posterior.metadata['estimator'] == 'legacy_batch_gradient_square'
    for key in parameters:
        torch.testing.assert_close(parameters[key], before[key])
    torch.testing.assert_close(posterior.mean, flatten_parameters(parameters)[0])
