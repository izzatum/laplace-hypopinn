"""Tensor data contracts. Frozen records protect structure, not tensor contents."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

Parameters = dict[str, Tensor]
ParametersBatch = dict[str, Tensor]
Forward = Callable[[Mapping[str, Tensor], Tensor], Tensor]


class RandomStreams:
    """Explicit local generators; consuming their state is intentional."""

    def __init__(self, seed=4321, device="cpu", *, numpy=None, torch_generator=None):
        from .config import resolve_device

        selected = resolve_device(device)
        self.numpy = np.random.RandomState(seed) if numpy is None else numpy
        self.torch = (
            torch.Generator(device=selected).manual_seed(seed)
            if torch_generator is None
            else torch_generator
        )


Streams = RandomStreams


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    kind: str = "tensor"

    @property
    def slice(self):
        return slice(self.start, self.stop)

    @property
    def numel(self):
        return self.stop - self.start


@dataclass(frozen=True)
class ParameterLayout:
    blocks: tuple[ParameterBlock, ...]
    names: tuple[str, ...]
    numel: int

    @property
    def size(self):
        return self.numel


@dataclass(frozen=True)
class Grid:
    z: Tensor
    x: Tensor
    velocity: Tensor

    def __post_init__(self):
        for name in ("z", "x"):
            axis = getattr(self, name)
            if (
                axis.ndim != 1
                or axis.numel() < 2
                or not torch.isfinite(axis).all()
                or not (axis[1:] > axis[:-1]).all()
            ):
                raise ValueError(
                    f"{name} axis must be finite, one-dimensional and strictly increasing"
                )
        if self.velocity.shape != (self.z.numel(), self.x.numel()):
            raise ValueError("velocity shape must be (len(z),len(x))")
        if not torch.isfinite(self.velocity).all() or not (self.velocity > 0).all():
            raise ValueError("velocity must be finite and positive")

    @property
    def coordinates(self):
        z, x = torch.meshgrid(self.z, self.x, indexing="ij")
        return torch.stack((z.reshape(-1), x.reshape(-1)), dim=-1)

    @property
    def shape(self):
        return self.velocity.shape


def _validate_receivers(coordinates, times):
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] == 0:
        raise ValueError("receiver_coordinates must have nonempty shape (n,2)")
    if times.shape not in ((coordinates.shape[0],), (coordinates.shape[0], 1)):
        raise ValueError("receiver_times must have shape (n,) or (n,1)")
    if not torch.isfinite(coordinates).all() or not torch.isfinite(times).all():
        raise ValueError("receiver coordinates and times must be finite")


@dataclass(frozen=True)
class Problem:
    grid: Grid
    receiver_coordinates: Tensor
    receiver_times: Tensor
    reference_times: Tensor | None = None
    true_source: Tensor | None = None

    def __post_init__(self):
        _validate_receivers(self.receiver_coordinates, self.receiver_times)
        coordinates = self.receiver_coordinates
        if not (
            (coordinates[:, 0] >= self.grid.z[0])
            & (coordinates[:, 0] <= self.grid.z[-1])
            & (coordinates[:, 1] >= self.grid.x[0])
            & (coordinates[:, 1] <= self.grid.x[-1])
        ).all():
            raise ValueError("receiver coordinates must be in the grid domain")
        if self.reference_times is not None and (
            self.reference_times.shape != self.grid.shape
            or not torch.isfinite(self.reference_times).all()
        ):
            raise ValueError("reference_times must be finite and match the grid shape")
        if self.true_source is not None and (
            self.true_source.shape != (2,) or not torch.isfinite(self.true_source).all()
        ):
            raise ValueError("true_source must have finite shape (2,)")


@dataclass(frozen=True)
class TrainingBatch:
    coordinates: Tensor
    velocity: Tensor
    receiver_coordinates: Tensor
    receiver_times: Tensor

    def __post_init__(self):
        if (
            self.coordinates.ndim != 2
            or self.coordinates.shape[1] != 2
            or self.coordinates.shape[0] == 0
        ):
            raise ValueError("coordinates must have nonempty shape (n,2)")
        n = self.coordinates.shape[0]
        if self.velocity.shape not in ((n,), (n, 1)):
            raise ValueError("velocity must have shape (n,) or (n,1)")
        if (
            not torch.isfinite(self.coordinates).all()
            or not torch.isfinite(self.velocity).all()
            or not (self.velocity > 0).all()
        ):
            raise ValueError("training coordinates must be finite and velocity finite and positive")
        _validate_receivers(self.receiver_coordinates, self.receiver_times)


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    pde: Tensor
    data: Tensor
    positivity: Tensor
    prior: Tensor
