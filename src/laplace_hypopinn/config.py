"""Immutable scientific settings; the historical MLP is only a default preset."""

from dataclasses import dataclass, field
import math

import torch

DEFAULT_WIDTHS = (2, 16, 16, 32, 32, 16, 16, 1)
INITIALIZATIONS = ("kaiming_normal", "xavier_normal", "kaiming_uniform", "xavier_uniform")
ACTIVATIONS = ("mish", "tanh", "silu")


def validate_widths(widths):
    if (
        len(widths) < 2
        or any(type(w) is not int or w <= 0 for w in widths)
        or widths[0] != 2
        or widths[-1] != 1
    ):
        raise ValueError("widths must be positive integers starting with 2 and ending with 1")
    return tuple(widths)


def resolve_device(device="cpu"):
    try:
        selected = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device {device!r}") from exc
    if selected.type == "cpu":
        if selected.index is not None:
            raise ValueError("CPU device must be 'cpu'")
        return selected
    if selected.type != "cuda":
        raise ValueError("device must be cpu or an explicitly available CUDA device")
    if not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    index = 0 if selected.index is None else selected.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return torch.device("cuda", index)


def resolve_dtype(dtype="float32"):
    if dtype in ("float32", torch.float32):
        return torch.float32
    if dtype in ("float64", torch.float64):
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _positive(name, value, *, zero=False):
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _count(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ModelConfig:
    widths: tuple[int, ...] = DEFAULT_WIDTHS
    activation: str = "mish"
    initialization: str = "kaiming_normal"

    def __post_init__(self):
        object.__setattr__(self, "widths", validate_widths(self.widths))
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f"initialization must be one of {INITIALIZATIONS}")


@dataclass(frozen=True)
class DataConfig:
    kind: str = "gradient"
    velocity_path: str | None = None
    observed_path: str | None = None
    source: tuple[float, float] | None = (1.0, 1.5)
    collocation_count: int = 2501
    z_bounds: tuple[float, float] = (0.0, 2.0)
    x_bounds: tuple[float, float] = (0.0, 3.0)
    grid_shape: tuple[int, int] = (101, 151)
    surface_velocity: float = 2.0
    velocity_gradient: float = 0.5
    receiver_count: int = 11
    receiver_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.kind not in ("gradient", "constant", "otway", "observed"):
            raise ValueError("unknown data kind")
        _count("collocation_count", self.collocation_count)
        _count("receiver_count", self.receiver_count)
        for bounds in (self.z_bounds, self.x_bounds):
            if (
                len(bounds) != 2
                or not all(math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise ValueError("grid bounds must be finite and increasing")
        if len(self.grid_shape) != 2:
            raise ValueError("grid_shape must have two entries")
        for n in self.grid_shape:
            _count("grid_shape", n)
            if n < 2:
                raise ValueError("grid axes require at least two samples")
        _positive("surface_velocity", self.surface_velocity)
        if not math.isfinite(self.velocity_gradient):
            raise ValueError("velocity_gradient must be finite")
        if self.source is not None and (
            len(self.source) != 2 or not all(math.isfinite(v) for v in self.source)
        ):
            raise ValueError("source must contain finite (z,x) coordinates")
        for name in ("source", "z_bounds", "x_bounds", "grid_shape", "receiver_indices"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 3001
    learning_rate: float = 0.005
    betas: tuple[float, float] = (0.5, 0.9)
    eta_min: float = 0.0
    log_every: int = 100
    pde_weight: float = 1.0
    data_weight: float = 1.0
    positivity_weight: float = 1.0
    prior_precision: float = 0.0

    def __post_init__(self):
        _count("steps", self.steps)
        _count("log_every", self.log_every)
        _positive("learning_rate", self.learning_rate)
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("Adam betas must lie in [0,1)")
        object.__setattr__(self, "betas", tuple(self.betas))
        for name in (
            "eta_min",
            "pde_weight",
            "data_weight",
            "positivity_weight",
            "prior_precision",
        ):
            _positive(name, getattr(self, name), zero=True)


@dataclass(frozen=True)
class LaplaceConfig:
    method: str = "legacy_diagonal"
    batches: int = 1000
    batch_size: int = 2501
    samples: int = 1000
    add: float = 1000.0
    multiply: float = 0.001
    prior_precision: float = 1.0
    scale: float = 1.0
    chunk_size: int = 64
    max_dense_bytes: int = 256 * 1024 * 1024
    rank: int = 32
    inducing_count: int = 32
    variational_steps: int = 200
    variational_lr: float = 0.01
    alpha: int = 0

    def __post_init__(self):
        if self.method not in (
            "legacy_diagonal",
            "diagonal",
            "block_diagonal",
            "block",
            "full",
            "kfac",
            "efb",
            "inf",
            "valla",
        ):
            raise ValueError("unknown Laplace method")
        for name in (
            "batches",
            "batch_size",
            "samples",
            "chunk_size",
            "max_dense_bytes",
            "inducing_count",
            "variational_steps",
        ):
            _count(name, getattr(self, name))
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        _positive("add", self.add)
        _positive("multiply", self.multiply, zero=True)
        _positive("prior_precision", self.prior_precision)
        _positive("scale", self.scale)
        _positive("variational_lr", self.variational_lr)
        if self.alpha not in (0, 1):
            raise ValueError("alpha must be 0 or 1")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "ex2"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    seed: int = 4321
    dtype: str = "float32"
    device: str = "cpu"
    prediction_chunk_size: int = 8192
    posterior_batch_size: int = 1
    save_posterior_fields: bool = False

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        resolve_dtype(self.dtype)
        resolve_device(self.device)
        _count("prediction_chunk_size", self.prediction_chunk_size)
        _count("posterior_batch_size", self.posterior_batch_size)
