"""Safe checkpoint persistence and bit-exact continuation across local streams."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from laplace_hypopinn.artifacts import load_checkpoint, save_checkpoint
from laplace_hypopinn.config import TrainingConfig
from laplace_hypopinn.training import train
from laplace_hypopinn.types import RandomStreams
from test_training import custom_forward, example


def test_safe_roundtrip_and_exact_resume(tmp_path):
    parameters, batch = example()
    config = TrainingConfig(steps=9, log_every=3)
    streams = RandomStreams(31)
    streams.numpy.normal(size=5)
    torch.rand(5, generator=streams.torch)
    full = train(parameters, batch, config, streams, forward=custom_forward)
    partial = train(parameters, batch, config, streams, forward=custom_forward, stop_after=4)
    path = tmp_path/'checkpoints'/'partial.pt'
    save_checkpoint(path, partial.state)
    payload = torch.load(path, weights_only=True)
    assert payload['schema_version'] == 1
    state = load_checkpoint(path, config)
    fresh_streams = RandomStreams(900)
    resumed = train(parameters, batch, config, fresh_streams, forward=custom_forward, resume=state)
    assert resumed.history == full.history
    assert resumed.state.step == config.steps
    assert torch.equal(fresh_streams.torch.get_state(), streams.torch.get_state())
    assert np.array_equal(fresh_streams.numpy.get_state()[1], streams.numpy.get_state()[1])
    for name in parameters:
        assert torch.equal(full.parameters[name], resumed.parameters[name])
    assert full.state.scheduler_state == resumed.state.scheduler_state
    for index, value in full.state.optimizer_state['state'].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, resumed.state.optimizer_state['state'][index][name])


def test_reject_overwrite_and_incompatible_config_or_layout(tmp_path):
    parameters, batch = example()
    config = TrainingConfig(steps=3)
    state = train(parameters, batch, config, RandomStreams(), forward=custom_forward, stop_after=1).state
    path = tmp_path/'state.pt'
    save_checkpoint(path, state)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, state)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='config'):
        load_checkpoint(path, replace(config, learning_rate=.01))
    changed = {'kernel': torch.zeros((2, 1, 1), dtype=torch.float64), 'offset': parameters['offset']}
    with pytest.raises(ValueError, match='layout'):
        train(changed, batch, config, RandomStreams(), forward=custom_forward, resume=state)
    with pytest.raises(ValueError, match='layout'):
        train(dict(reversed(list(parameters.items()))), batch, config, RandomStreams(), forward=custom_forward, resume=state)


def test_reject_corrupted_schema_and_layout(tmp_path):
    parameters, batch = example()
    config = TrainingConfig(steps=3)
    state = train(parameters, batch, config, RandomStreams(), forward=custom_forward, stop_after=1).state
    path = tmp_path/'state.pt'
    save_checkpoint(path, state)
    payload = torch.load(path, weights_only=True)
    payload['schema_version'] = 999
    torch.save(payload, tmp_path/'bad-schema.pt')
    with pytest.raises(ValueError, match='schema'):
        load_checkpoint(tmp_path/'bad-schema.pt', config)
    payload['schema_version'] = 1
    payload['parameters']['kernel'] = torch.ones(3, dtype=torch.float64)
    torch.save(payload, tmp_path/'bad-layout.pt')
    with pytest.raises(ValueError, match='layout'):
        load_checkpoint(tmp_path/'bad-layout.pt', config)


def test_reject_optimizer_config_and_tensor_shape_corruption(tmp_path):
    parameters, batch = example()
    config = TrainingConfig(steps=3)
    state = train(parameters, batch, config, RandomStreams(), forward=custom_forward, stop_after=1).state
    path = tmp_path/'state.pt'
    save_checkpoint(path, state)
    payload = torch.load(path, weights_only=True)
    payload['optimizer_state']['param_groups'][0]['betas'] = (.9, .999)
    torch.save(payload, tmp_path/'bad-optimizer.pt')
    with pytest.raises(ValueError, match='optimizer config'):
        load_checkpoint(tmp_path/'bad-optimizer.pt', config)
    payload['optimizer_state']['param_groups'][0]['betas'] = config.betas
    payload['optimizer_state']['state'][0]['exp_avg'] = torch.zeros(3, dtype=torch.float64)
    torch.save(payload, tmp_path/'bad-optimizer-layout.pt')
    with pytest.raises(ValueError, match='optimizer layout'):
        load_checkpoint(tmp_path/'bad-optimizer-layout.pt', config)
