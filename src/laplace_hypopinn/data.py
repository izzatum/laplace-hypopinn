"""Validated velocity grids and travel times in (z,x) kilometres and seconds.

Analytic inputs support constant velocity and a vertical linear gradient. Otway
CSV inputs follow the archived orientation (CSV columns are depth), and require
an explicit path. Only Otway reference generation imports the fast-marching
extra. Observed inputs need neither a known source nor reference travel times.
"""
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig, ExperimentConfig, resolve_device, resolve_dtype
from .types import Grid, Problem, RandomStreams, TrainingBatch


def _axis(bounds, count, *, dtype, device, linear=False):
    if linear:
        return torch.linspace(*bounds, count, dtype=dtype, device=device)
    step = (bounds[1] - bounds[0]) / (count - 1)
    # arange preserves the archived float32 grid (linspace differs by one ulp).
    axis = torch.arange(bounds[0], bounds[1] + .5 * step, step,
                        dtype=dtype, device=device)
    if axis.numel() != count:
        raise ValueError("grid bounds and shape cannot form the requested arange axis")
    return axis


def _source(config, grid):
    if config.source is None:
        raise ValueError("synthetic reference generation requires a source")
    source = torch.tensor(config.source, dtype=grid.z.dtype, device=grid.z.device)
    if not ((grid.z[0] <= source[0]) & (source[0] <= grid.z[-1]) &
            (grid.x[0] <= source[1]) & (source[1] <= grid.x[-1])):
        raise ValueError("source must lie within the grid domain")
    return source


def _receiver_indices(config, grid):
    if config.receiver_indices is not None:
        values = config.receiver_indices
        if (not values or any(type(i) is not int or i < 0 or i >= grid.velocity.numel()
                              for i in values)):
            raise ValueError("receiver indices must be nonempty integers within the flattened grid")
        return torch.tensor(values, device=grid.z.device, dtype=torch.long)
    if config.receiver_count > grid.x.numel():
        raise ValueError("receiver_count cannot exceed the number of surface grid points")
    return torch.linspace(0, grid.x.numel() - 1, config.receiver_count,
                          device=grid.z.device).round().long()


def prepare_problem(config: ExperimentConfig | DataConfig, *, dtype=None, device=None) -> Problem:
    """Prepare a synthetic or observed problem with explicit precision and device.

    ExperimentConfig supplies dtype/device defaults; DataConfig alone defaults
    to float32/CPU. Keyword overrides take precedence. Source/reference fields in
    an observed NPZ are optional and never synthesized from config.source.
    Otway uses CSV transpose, configured bounds, linspace axes, and the archived
    tensor floor-division source cell on those axes.
    """
    if isinstance(config, ExperimentConfig):
        dtype = config.dtype if dtype is None else dtype
        device = config.device if device is None else device
        data = config.data
    elif isinstance(config, DataConfig):
        data = config
    else:
        raise TypeError("config must be an ExperimentConfig or DataConfig")
    dtype = resolve_dtype("float32" if dtype is None else dtype)
    device = resolve_device("cpu" if device is None else device)
    if data.kind == "observed":
        if data.observed_path is None:
            raise ValueError("observed_path is required for observed data")
        return load_observed_problem(data.observed_path, dtype=dtype, device=device)

    if data.kind == "otway":
        if data.velocity_path is None:
            raise ValueError("velocity_path is required for Otway; supply the CSV path explicitly")
        raw = np.loadtxt(Path(data.velocity_path), delimiter=",").T.copy()
        if raw.ndim != 2:
            raise ValueError("velocity CSV must be two-dimensional")
        shape = raw.shape
    else:
        shape = data.grid_shape
    z = _axis(data.z_bounds, shape[0], dtype=dtype, device=device, linear=data.kind == "otway")
    x = _axis(data.x_bounds, shape[1], dtype=dtype, device=device, linear=data.kind == "otway")
    if data.kind == "otway":
        velocity = torch.as_tensor(raw, dtype=dtype, device=device)
    else:
        gradient = data.velocity_gradient if data.kind == "gradient" else 0.
        # Keep archived source-centred arithmetic for bitwise regression parity.
        if data.source is None:
            raise ValueError("synthetic reference generation requires a source")
        source_z = torch.tensor(data.source[0], dtype=dtype, device=device)
        surface = torch.tensor(data.surface_velocity, dtype=dtype, device=device)
        source_velocity = surface + gradient * (source_z - z[0])
        velocity = (source_velocity + gradient * (z[:, None] - source_z)).expand(-1, x.numel()).clone()
    grid = Grid(z, x, velocity)
    source = _source(data, grid)
    if data.kind == "otway":
        try:
            import skfmm
        except ImportError as exc:
            raise ImportError("Otway reference generation requires the fast-marching extra: "
                              "pip install 'laplace-hypopinn[fast-marching]'") from exc
        # FMM is CPU/NumPy based; floor in the requested torch precision first.
        dz, dx = z[1] - z[0], x[1] - x[0]
        iz = int((source[0] - z[0]) // dz)
        ix = int((source[1] - x[0]) // dx)
        phi = np.ones_like(raw)
        phi[iz, ix] = -1.
        reference = torch.as_tensor(skfmm.travel_time(phi, raw, dx=(float(dz), float(dx))),
                                    dtype=dtype, device=device)
    else:
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        distance_squared = (xx - source[1])**2 + (zz - source[0])**2
        if gradient == 0:
            reference = torch.sqrt(distance_squared) / data.surface_velocity
        else:
            reference = torch.acosh(1. + .5 * (1. / velocity) * (1. / source_velocity) *
                                     gradient**2 * distance_squared) / abs(gradient)
    indices = _receiver_indices(data, grid)
    return Problem(grid, grid.coordinates[indices], reference.flatten()[indices], reference, source)


def load_observed_problem(path, *, dtype=torch.float32, device="cpu") -> Problem:
    """Read an NPZ without pickle; required arrays are z,x,velocity and receivers.

    receiver_coordinates use (z,x), receiver_times are seconds; optional arrays
    are true_source (2,) and reference_times (nz,nx). All inputs are validated by
    the shared Grid and Problem contracts. Paths are used explicitly; use
    presets.load_config to resolve config-relative paths before calling here.
    """
    dtype, device = resolve_dtype(dtype), resolve_device(device)
    required = ("z", "x", "velocity", "receiver_coordinates", "receiver_times")
    with np.load(Path(path), allow_pickle=False) as arrays:
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"observed NPZ is missing required arrays: {', '.join(missing)}")
        tensors = {name: torch.tensor(arrays[name], dtype=dtype, device=device)
                   for name in (*required, "true_source", "reference_times") if name in arrays}
    grid = Grid(tensors["z"], tensors["x"], tensors["velocity"])
    return Problem(grid, tensors["receiver_coordinates"], tensors["receiver_times"],
                   tensors.get("reference_times"), tensors.get("true_source"))


def sample_training_batch(problem: Problem, count: int, streams: RandomStreams, *,
                          replace: bool = False) -> TrainingBatch:
    """Sample flattened grid points using only the caller's local NumPy RNG.

    Default training selection is without replacement; legacy curvature batches
    must request replace=True. Returned coordinates do not require gradients;
    differentiating consumers make their own coordinate leaves.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    size = problem.grid.velocity.numel()
    if not replace and count > size:
        raise ValueError("cannot sample more grid points than available without replacement")
    indices = streams.numpy.choice(np.arange(size), count, replace=replace)
    indices = torch.as_tensor(indices, dtype=torch.long, device=problem.grid.z.device)
    return TrainingBatch(problem.grid.coordinates[indices], problem.grid.velocity.flatten()[indices],
                         problem.receiver_coordinates, problem.receiver_times)
