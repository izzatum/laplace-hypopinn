from dataclasses import replace

import numpy as np
import torch

from laplace_hypopinn.config import ExperimentConfig, LaplaceConfig
from laplace_hypopinn.curvature.base import Posterior
from laplace_hypopinn.model import flatten_parameters
from laplace_hypopinn.prediction import predict_grid, sample_locations
from laplace_hypopinn.types import Grid


def bowl(parameters, coordinates):
    return ((coordinates - parameters['center']) ** 2).sum(-1)


def fixture():
    grid = Grid(torch.tensor([4., 5., 6.]), torch.tensor([8., 10., 12.]), torch.ones(3, 3))
    p = {'center': torch.tensor([4., 8.])}
    mean, layout = flatten_parameters(p)
    posterior = Posterior(mean, layout, 'diagonal', diagonal=torch.full_like(mean, 5.), forward=bowl)
    return p, grid, posterior


def test_grid_coordinates_ties_and_no_mutation():
    p, grid, _ = fixture()
    original = p['center'].clone()
    expected = bowl(p, grid.coordinates).reshape(grid.shape)
    torch.testing.assert_close(predict_grid(p, grid, 2, forward=bowl), expected)
    torch.testing.assert_close(p['center'], original)
    def flat(p, x):
        return x[:, 0] * 0 + p['center'].sum() * 0
    posterior = fixture()[2]
    posterior = replace(posterior, forward=flat)
    config = replace(ExperimentConfig(), laplace=LaplaceConfig(samples=3), prediction_chunk_size=2)
    summary = sample_locations(posterior, grid, config, torch.Generator().manual_seed(4))
    torch.testing.assert_close(summary.locations, torch.tensor([[4., 8.]]).repeat(3, 1))
    torch.testing.assert_close(summary.std, torch.zeros(2))


def test_chunks_batches_and_disk_fields(tmp_path):
    _, grid, posterior = fixture()
    config = replace(ExperimentConfig(), laplace=LaplaceConfig(samples=7), prediction_chunk_size=2)
    a = sample_locations(posterior, grid, config, torch.Generator().manual_seed(42))
    other = replace(config, posterior_batch_size=3, prediction_chunk_size=5, save_posterior_fields=True)
    b = sample_locations(posterior, grid, other, torch.Generator().manual_seed(42), field_store=tmp_path/'fields.npy')
    torch.testing.assert_close(a.locations, b.locations)
    torch.testing.assert_close(a.field_mean, b.field_mean)
    torch.testing.assert_close(a.std, a.locations.std(dim=0, correction=0))
    fields = np.load(tmp_path/'fields.npy', mmap_mode='r')
    np.testing.assert_allclose(fields.mean(0), b.field_mean.numpy(), atol=1e-5)
    np.testing.assert_allclose(fields.std(0), b.field_std.numpy(), atol=1e-5)


def test_linearized_samples_use_same_delta_over_all_chunks():
    p, grid, posterior = fixture()
    posterior = replace(posterior, prediction_kind='linearized')
    config = replace(ExperimentConfig(), laplace=LaplaceConfig(samples=1), prediction_chunk_size=2)
    seed = 34
    summary = sample_locations(posterior, grid, config, torch.Generator().manual_seed(seed))
    from laplace_hypopinn.curvature.sampling import sample_vectors
    from laplace_hypopinn.curvature.valla import linearized_predict
    draw = sample_vectors(posterior, 1, generator=torch.Generator().manual_seed(seed))[0]
    expected = linearized_predict(p, draw-posterior.mean, grid.coordinates, forward=bowl).reshape(grid.shape)
    torch.testing.assert_close(summary.field_mean, expected)
    nonlinear = predict_grid({'center': draw}, grid, 4, forward=bowl)
    assert not torch.allclose(expected, nonlinear)


def test_field_storage_requires_new_explicit_path(tmp_path):
    import pytest
    _, grid, posterior = fixture()
    config = replace(ExperimentConfig(), laplace=LaplaceConfig(samples=1), save_posterior_fields=True)
    with pytest.raises(ValueError, match='field_store'):
        sample_locations(posterior, grid, config, torch.Generator().manual_seed(1))
    path = tmp_path/'existing.npy'
    path.write_bytes(b'untouched')
    with pytest.raises(FileExistsError):
        sample_locations(posterior, grid, config, torch.Generator().manual_seed(1), field_store=path)
    assert path.read_bytes() == b'untouched'
