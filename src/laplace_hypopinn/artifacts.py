"""Versioned tensor-only checkpoints and atomic, overwrite-safe publication."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .config import ExperimentConfig, TrainingConfig
from .training import SCHEMA_VERSION, TrainingState, validate_state


def save_checkpoint(path, state: TrainingState) -> None:
    """Publish a complete checkpoint atomically, refusing any existing path.

    The payload consists only of tensors and ordinary Python containers/numbers,
    and is compatible with ``torch.load(weights_only=True)``. NumPy's uint32
    random state is serialized as an int64 tensor, avoiding unsafe pickle globals.
    """
    validate_state(state)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = state.numpy_rng_state
    payload = {
        'schema_version': state.schema_version, 'config': asdict(state.config),
        'layout': state.layout,
        'parameters': {name: value.detach().clone() for name, value in state.parameters.items()},
        'optimizer_state': state.optimizer_state, 'scheduler_state': state.scheduler_state,
        'step': state.step, 'history': list(state.history),
        'numpy_rng_state': (numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                            int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])),
        'torch_rng_state': state.torch_rng_state,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.', delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        # Same-filesystem hard-link publication is atomic and cannot replace an
        # existing destination, including one created after the initial check.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, expected_config: TrainingConfig | ExperimentConfig) -> TrainingState:
    """Load safely and require exact training settings and tensor layout metadata.

    ExperimentConfig inputs validate their training field; callers own validation
    of experiment data, forward implementation and model configuration separately.
    """
    config = expected_config.training if isinstance(expected_config, ExperimentConfig) else expected_config
    if not isinstance(config, TrainingConfig):
        raise TypeError('expected_config must be TrainingConfig or ExperimentConfig')
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported training checkpoint schema')
    if payload.get('config') != asdict(config):
        raise ValueError('checkpoint config does not match expected config')
    try:
        layout = tuple((name, tuple(shape), dtype, device) for name, shape, dtype, device in payload['layout'])
        metadata = {name: (shape, dtype, device) for name, shape, dtype, device in layout}
        parameters = {}
        for name, value in payload['parameters'].items():
            shape, dtype, device = metadata[name]
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError('checkpoint parameter layout does not match stored tensors')
            parameters[name] = value.to(device).detach().clone().requires_grad_()
        numpy_state = payload['numpy_rng_state']
        state = TrainingState(
            parameters=parameters, optimizer_state=payload['optimizer_state'],
            scheduler_state=payload['scheduler_state'], step=payload['step'],
            numpy_rng_state=(numpy_state[0], numpy_state[1].numpy().astype(np.uint32),
                             numpy_state[2], numpy_state[3], numpy_state[4]),
            torch_rng_state=payload['torch_rng_state'], config=config,
            layout=layout, history=tuple(payload['history']),
        )
        validate_state(state)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid checkpoint structure or parameter layout') from exc
    return state
