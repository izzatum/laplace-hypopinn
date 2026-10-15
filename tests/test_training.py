"""Notebook optimizer parity and caller-isolated training of custom models."""

import numpy as np
import pytest
import torch

from laplace_hypopinn.config import TrainingConfig
from laplace_hypopinn.model import initialize_model, predict
from laplace_hypopinn.training import train
from laplace_hypopinn.types import RandomStreams, TrainingBatch


def custom_forward(parameters, coordinates):
    weights = parameters['kernel'].reshape(-1)
    return weights[0]*coordinates[:, 0].square() + weights[1]*coordinates[:, 1] + parameters['offset']


def example():
    parameters = {'kernel': torch.tensor([[[.4, -.3]]], dtype=torch.float64, requires_grad=True),
                  'offset': torch.tensor(-.2, dtype=torch.float64, requires_grad=True)}
    coordinates = torch.tensor([[0., 0.], [.5, 1.], [1., .2]], dtype=torch.float64, requires_grad=True)
    batch = TrainingBatch(coordinates, torch.tensor([2., 3., 2.5], dtype=torch.float64),
                          torch.tensor([[.1, .2], [.8, .6]], dtype=torch.float64),
                          torch.tensor([[.2], [.5]], dtype=torch.float64))
    return parameters, batch


@pytest.mark.parametrize('prior_precision', [0., .7])
def test_one_step_matches_notebook_adam_cosine_and_keeps_caller_unchanged(prior_precision):
    parameters, batch = example()
    snapshots = {name: value.detach().clone() for name, value in parameters.items()}
    reference = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    config = TrainingConfig(steps=7, prior_precision=prior_precision)
    optimizer = torch.optim.Adam(list(reference.values()), lr=.005, betas=(.5, .9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
    coordinates = batch.coordinates.detach().clone().requires_grad_()
    times = custom_forward(reference, coordinates)
    gradient = torch.autograd.grad(times.sum(), coordinates, create_graph=True)[0]
    loss = ((gradient.square().sum(-1) - batch.velocity.pow(-2)).square().mean()
            + (custom_forward(reference, batch.receiver_coordinates) - batch.receiver_times.reshape(-1)).square().mean()
            + ((1-torch.sign(times))*times.abs()).square().mean()
            + .5*prior_precision*sum(value.square().sum() for value in reference.values()))
    loss.backward()
    optimizer.step()
    scheduler.step()
    result = train(parameters, batch, config, RandomStreams(), forward=custom_forward, stop_after=1)
    assert result.state.step == 1
    assert result.state.scheduler_state['last_epoch'] == 1
    assert result.state.optimizer_state['param_groups'][0]['lr'] == optimizer.param_groups[0]['lr']
    assert result.history[0]['total'] == pytest.approx(loss.item())
    for name, value in result.parameters.items():
        torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        torch.testing.assert_close(parameters[name], snapshots[name], rtol=0, atol=0)
        assert parameters[name].grad is None
    assert batch.coordinates.grad is None


def test_runs_ignore_global_rng_and_keep_global_streams_unchanged():
    parameters, batch = example()
    config = TrainingConfig(steps=4, log_every=2)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first = train(parameters, batch, config, RandomStreams(31), forward=custom_forward)
    second = train(parameters, batch, config, RandomStreams(92), forward=custom_forward)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, np.random.get_state()))
    assert first.history == second.history
    for name in parameters:
        assert torch.equal(first.parameters[name], second.parameters[name])
    assert [record['step'] for record in first.history] == [1, 2, 4]


def test_default_mish_network_matches_notebook_updates():
    _, batch = example()
    parameters = initialize_model(streams=RandomStreams(4321), dtype=torch.float64)
    reference = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    config = TrainingConfig(steps=3001)
    optimizer = torch.optim.Adam(list(reference.values()), lr=.005, betas=(.5, .9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3001)
    for _ in range(3):
        optimizer.zero_grad()
        coordinates = batch.coordinates.detach().clone().requires_grad_()
        times = predict(reference, coordinates)
        gradient = torch.autograd.grad(times, coordinates, grad_outputs=torch.ones_like(times), create_graph=True)[0]
        loss = ((gradient.square().sum(-1) - batch.velocity.pow(-2)).square().mean()
                + (predict(reference, batch.receiver_coordinates).reshape(-1) - batch.receiver_times.reshape(-1)).square().mean()
                + ((1-torch.sign(times))*times.abs()).square().mean())
        loss.backward()
        optimizer.step()
        scheduler.step()
    result = train(parameters, batch, config, RandomStreams(), stop_after=3)
    for name in parameters:
        torch.testing.assert_close(result.parameters[name], reference[name], rtol=1e-11, atol=1e-13)


def test_nonfinite_error_identifies_update_step():
    parameters, batch = example()
    def bad_forward(parameters, coordinates):
        return custom_forward(parameters, coordinates) * float('nan')
    with pytest.raises(FloatingPointError, match='step 1'):
        train(parameters, batch, TrainingConfig(steps=2), RandomStreams(), forward=bad_forward)


def test_invalid_stop_boundary_rejected():
    parameters, batch = example()
    with pytest.raises(ValueError, match='stop_after'):
        train(parameters, batch, TrainingConfig(steps=2), RandomStreams(), forward=custom_forward, stop_after=3)
