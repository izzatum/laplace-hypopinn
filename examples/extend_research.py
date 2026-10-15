"""Executable short example of observed-data inversion with a custom model.

Run: python examples/extend_research.py --output /tmp/custom-hypopinn
The source and reference field are deliberately absent. Any deterministic,
pointwise scalar differentiable forward may replace this quadratic model.
"""
import argparse

import torch

from laplace_hypopinn.config import DataConfig, ExperimentConfig, LaplaceConfig, TrainingConfig
from laplace_hypopinn.experiment import run_experiment
from laplace_hypopinn.types import Grid, Problem


def forward(parameters, coordinates):
    return ((coordinates-parameters['center']).square()*parameters['scale'].square()).sum(-1) + parameters['offset']


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    z, x = torch.linspace(0., 2., 11), torch.linspace(0., 3., 13)
    grid = Grid(z, x, torch.full((11, 13), 2.))
    receivers = torch.stack((torch.zeros(5), torch.linspace(0., 3., 5)), -1)
    times = torch.tensor([.9, .65, .5, .65, .9])
    problem = Problem(grid, receivers, times)
    parameters = {'center': torch.tensor([.8, 1.2]), 'scale': torch.tensor([.5, .5]),
                  'offset': torch.tensor(.05)}
    config = ExperimentConfig(name='custom-observed', data=DataConfig(collocation_count=8),
                              training=TrainingConfig(steps=2, log_every=1),
                              laplace=LaplaceConfig(method='full', samples=3, chunk_size=4))
    result = run_experiment(config, problem=problem, parameters=parameters,
                            forward=forward, output=args.output)
    print('MAP source (z,x) km:', result.map_location.tolist())
    return result


if __name__ == '__main__':
    main()
