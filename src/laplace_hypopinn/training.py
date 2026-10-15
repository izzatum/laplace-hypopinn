"""Caller-isolated Adam training with explicit, resumable local random streams."""

from dataclasses import asdict, dataclass
import copy
from collections.abc import Mapping

import torch

from .config import TrainingConfig
from .losses import loss_terms
from .model import predict
from .types import Forward, Parameters, RandomStreams, TrainingBatch

SCHEMA_VERSION = 1


def parameter_layout(parameters):
    """A named tensor layout, including insertion order, dtype, and device."""
    if not parameters:
        raise ValueError('parameter layout must be nonempty')
    layout = []
    for name, value in parameters.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('parameter layout requires named tensors')
        if not value.is_floating_point() or value.numel() == 0:
            raise ValueError('parameter layout requires nonempty floating tensors')
        layout.append((name, tuple(value.shape), str(value.dtype), str(value.device)))
    return tuple(layout)


@dataclass(frozen=True)
class TrainingState:
    """Snapshot after ``step`` completed updates; all optimizer tensors are owned.

    ``config.steps`` always denotes the full cosine horizon. History contains
    pre-update losses at update 1, logging intervals, and the scheduled last step.
    Layout is independent of architecture and accepts arbitrary named tensors.
    """

    parameters: Parameters
    optimizer_state: dict
    scheduler_state: dict
    step: int
    numpy_rng_state: tuple
    torch_rng_state: torch.Tensor
    config: TrainingConfig
    layout: tuple
    history: tuple[dict, ...] = ()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class TrainingResult:
    state: TrainingState

    @property
    def parameters(self):
        return self.state.parameters

    @property
    def history(self):
        return self.state.history


def _copy_batch(batch):
    # Coordinate leaves owned by the caller must never accumulate derivatives.
    return TrainingBatch(**{
        name: getattr(batch, name).detach().clone()
        for name in ('coordinates', 'velocity', 'receiver_coordinates', 'receiver_times')
    })


def validate_state(state):
    """Validate the snapshot contract before loading optimizer tensors."""
    if state.schema_version != SCHEMA_VERSION:
        raise ValueError('unsupported training checkpoint schema')
    if not isinstance(state.config, TrainingConfig):
        raise ValueError('checkpoint config must be TrainingConfig')
    if type(state.step) is not int or not 0 <= state.step <= state.config.steps:
        raise ValueError('checkpoint step is outside configured schedule')
    if state.layout != parameter_layout(state.parameters):
        raise ValueError('checkpoint parameter layout does not match tensors')
    if not all(torch.isfinite(value).all() for value in state.parameters.values()):
        raise ValueError('checkpoint contains nonfinite parameters')
    groups = state.optimizer_state.get('param_groups', [])
    if len(groups) != 1 or len(groups[0].get('params', [])) != len(state.parameters):
        raise ValueError('checkpoint optimizer layout is incompatible')
    group = groups[0]
    if (tuple(group.get('betas', ())) != state.config.betas
            or group.get('eps') != 1e-8 or group.get('weight_decay') != 0
            or group.get('amsgrad') or group.get('maximize')):
        raise ValueError('checkpoint optimizer config is incompatible')
    indices = group['params']
    if indices != list(range(len(state.parameters))):
        raise ValueError('checkpoint optimizer layout is incompatible')
    optimizer_values = state.optimizer_state.get('state', {})
    if not set(optimizer_values).issubset(indices):
        raise ValueError('checkpoint optimizer layout is incompatible')
    for index, parameter in enumerate(state.parameters.values()):
        entry = optimizer_values.get(index)
        if entry is None:  # A parameter unused by the forward may never receive a gradient.
            continue
        for name in ('exp_avg', 'exp_avg_sq'):
            value = entry.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise ValueError('checkpoint optimizer layout or moment is incompatible')
        count = entry.get('step')
        if (not isinstance(count, torch.Tensor) or count.numel() != 1
                or not torch.isfinite(count).all() or not 0 <= count.item() <= state.step):
            raise ValueError('checkpoint optimizer step is incompatible')
    if state.scheduler_state.get('last_epoch') != state.step:
        raise ValueError('checkpoint scheduler step is incompatible')
    if state.scheduler_state.get('T_max') != state.config.steps:
        raise ValueError('checkpoint scheduler config is incompatible')
    if (state.scheduler_state.get('eta_min') != state.config.eta_min
            or state.scheduler_state.get('base_lrs') != [state.config.learning_rate]
            or state.scheduler_state.get('_last_lr') != [group.get('lr')]):
        raise ValueError('checkpoint scheduler config is incompatible')


def train(
    parameters: Mapping[str, torch.Tensor], batch: TrainingBatch,
    config: TrainingConfig, streams: RandomStreams, *, forward: Forward = predict,
    resume: TrainingState | None = None, stop_after: int | None = None,
) -> TrainingResult:
    """Train a private parameter copy; ``stop_after`` is an absolute step bound.

    A partial run retains the full configured cosine schedule. Resuming restores
    the provided local streams, optimizer and scheduler; global RNGs are untouched.
    Custom forwards must be deterministic, differentiable pointwise scalar models.
    """
    layout = parameter_layout(parameters)
    end = config.steps if stop_after is None else stop_after
    start = 0 if resume is None else resume.step
    if type(end) is not int or not start <= end <= config.steps:
        raise ValueError('stop_after must lie between resumed step and config.steps')
    if resume is not None:
        validate_state(resume)
        if asdict(config) != asdict(resume.config):
            raise ValueError('resume config does not match training config')
        if layout != resume.layout:
            raise ValueError('resume parameter layout does not match initial parameters')
    source = parameters if resume is None else resume.parameters
    owned = {name: value.detach().clone().requires_grad_() for name, value in source.items()}
    if not all(torch.isfinite(value).all() for value in owned.values()):
        raise ValueError('initial parameters must be finite')
    optimizer = torch.optim.Adam(list(owned.values()), lr=config.learning_rate, betas=config.betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.eta_min,
    )
    history = [] if resume is None else copy.deepcopy(list(resume.history))
    if resume is not None:
        # Optimizer.load_state_dict can retain references to tensors in its input.
        optimizer.load_state_dict(copy.deepcopy(resume.optimizer_state))
        scheduler.load_state_dict(copy.deepcopy(resume.scheduler_state))
        streams.numpy.set_state(copy.deepcopy(resume.numpy_rng_state))
        streams.torch.set_state(resume.torch_rng_state.cpu().clone())
    local_batch = _copy_batch(batch)
    for step in range(start + 1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(owned, local_batch, config, forward=forward)
        metrics = torch.stack([getattr(terms, name).detach() for name in ('total', 'pde', 'data', 'positivity', 'prior')])
        if not torch.isfinite(metrics).all():
            raise FloatingPointError(f'nonfinite training loss at step {step}')
        terms.total.backward()
        if any(value.grad is not None and not torch.isfinite(value.grad).all() for value in owned.values()):
            raise FloatingPointError(f'nonfinite training gradient at step {step}')
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        if any(not torch.isfinite(value).all() for value in owned.values()):
            raise FloatingPointError(f'nonfinite training parameter at step {step}')
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            values = metrics.cpu().tolist()
            history.append(dict(step=step, learning_rate=learning_rate,
                                **dict(zip(('total', 'pde', 'data', 'positivity', 'prior'), values, strict=True))))
    state = TrainingState(
        parameters={name: value.detach().clone().requires_grad_() for name, value in owned.items()},
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        scheduler_state=copy.deepcopy(scheduler.state_dict()), step=end,
        numpy_rng_state=copy.deepcopy(streams.numpy.get_state()),
        torch_rng_state=streams.torch.get_state().clone(), config=config, layout=layout,
        history=tuple(history),
    )
    return TrainingResult(state)
