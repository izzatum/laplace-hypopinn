"""Chunked fields and coherent posterior draws in physical (z,x) coordinates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import predict, unflatten_parameters


@dataclass(frozen=True)
class PosteriorSummary:
    locations: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    field_mean: torch.Tensor | None = None
    field_std: torch.Tensor | None = None
    field_store: str | None = None


def _chunks(grid, chunk_size):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    nx = grid.x.numel()
    for start in range(0, grid.velocity.numel(), chunk_size):
        index = torch.arange(start, min(start+chunk_size, grid.velocity.numel()), device=grid.z.device)
        yield start, torch.stack((grid.z[index//nx], grid.x[index % nx]), dim=-1)


def _values(value, size):
    if value.shape not in ((size,), (size, 1)):
        raise ValueError('forward must return one scalar per point with shape (n,) or (n,1)')
    value = value.detach().reshape(-1)
    if not torch.isfinite(value).all():
        raise ValueError('prediction contains nonfinite values')
    return value


def predict_grid(parameters, grid, chunk_size=8192, *, forward=predict):
    """Evaluate a pointwise scalar forward without mutating its parameters."""
    output = grid.velocity.new_empty(grid.velocity.numel())
    with torch.no_grad():
        for start, coordinates in _chunks(grid, chunk_size):
            value = _values(forward(parameters, coordinates), len(coordinates))
            output[start:start+len(value)] = value
    return output.reshape(grid.shape)


def grid_location(field, grid):
    """First row-major minimum, using actual grid axes including their origins."""
    if field.shape != grid.shape or not torch.isfinite(field).all():
        raise ValueError('field must be finite and match the grid shape')
    index = int(field.reshape(-1).argmin())
    return torch.stack((grid.z[index//grid.x.numel()], grid.x[index % grid.x.numel()]))


def sample_locations(posterior, grid, config, generator, *, field_store=None, field_moments=True):
    """Stream posterior fields, retaining locations and population field moments.

    One complete parameter vector is drawn per sample. For linearized posteriors,
    its difference from the MAP is reused over every grid chunk. Complete field
    storage requires an explicit new disk path and never allocates S*N in RAM.
    """
    from .curvature.sampling import sample_vectors

    if posterior.prediction_kind not in ('travel_time', 'linearized'):
        raise ValueError(f'unsupported posterior prediction kind {posterior.prediction_kind!r}')
    samples = config.laplace.samples
    chunk_size = config.prediction_chunk_size
    batch_size = config.posterior_batch_size
    shape = tuple(grid.shape)
    size = grid.velocity.numel()
    location_samples = posterior.mean.new_empty((samples, 2))
    mean = posterior.mean.new_zeros(size) if field_moments else None
    m2 = torch.zeros_like(mean) if field_moments else None
    store = None
    path = None
    if config.save_posterior_fields or field_store is not None:
        if field_store is None:
            raise ValueError('save_posterior_fields requires an explicit field_store path or run output')
        path = Path(field_store)
        # Reserve before opening the mmap: no clobbering an existing array.
        with path.open('xb'):
            pass
        dtype = np.float64 if posterior.mean.dtype == torch.float64 else np.float32
        store = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(samples, *shape))
    center = unflatten_parameters(posterior.mean, posterior.layout)
    try:
        for begin in range(0, samples, batch_size):
            vectors = sample_vectors(posterior, min(batch_size, samples-begin), generator=generator)
            for local_index, vector in enumerate(vectors):
                sample_index = begin+local_index
                parameters = unflatten_parameters(vector, posterior.layout)
                minimum = float('inf')
                minimum_index = 0
                for start, coordinates in _chunks(grid, chunk_size):
                    if posterior.prediction_kind == 'linearized':
                        from .curvature.valla import linearized_predict
                        values = linearized_predict(center, vector-posterior.mean, coordinates,
                                                    forward=posterior.forward)
                    else:
                        with torch.no_grad():
                            values = posterior.forward(parameters, coordinates)
                    values = _values(values, len(coordinates))
                    stop = start+len(values)
                    local_minimum, index = values.min(0)
                    if float(local_minimum) < minimum:
                        minimum, minimum_index = float(local_minimum), start+int(index)
                    if field_moments:
                        delta = values-mean[start:stop]
                        mean[start:stop].add_(delta/(sample_index+1))
                        m2[start:stop].add_(delta*(values-mean[start:stop]))
                    if store is not None:
                        store[sample_index].reshape(-1)[start:stop] = values.cpu().numpy()
                location_samples[sample_index] = torch.stack((grid.z[minimum_index//grid.x.numel()],
                                                               grid.x[minimum_index % grid.x.numel()]))
        if store is not None:
            store.flush()
    finally:
        del store
    return PosteriorSummary(location_samples, location_samples.mean(0),
                            location_samples.std(0, correction=0),
                            mean.reshape(shape) if field_moments else None,
                            (m2/samples).clamp_min(0).sqrt().reshape(shape) if field_moments else None,
                            str(path) if path is not None else None)
