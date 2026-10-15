"""Scientific input contracts and archived data preparation regressions."""
import builtins
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from laplace_hypopinn.config import DataConfig, ExperimentConfig
from laplace_hypopinn.data import (
    load_observed_problem, prepare_problem, sample_training_batch,
)
from laplace_hypopinn.presets import load_config, load_preset
from laplace_hypopinn.types import RandomStreams

ROOT = Path(__file__).resolve().parents[1]


def test_constant_analytic_distance_with_nonzero_origin():
    config = DataConfig(kind="constant", source=(2., 4.), z_bounds=(1., 3.),
                        x_bounds=(3., 5.), grid_shape=(5, 7), receiver_count=4,
                        surface_velocity=3.)
    problem = prepare_problem(config, dtype=torch.float64)
    expected = torch.linalg.vector_norm(problem.grid.coordinates - problem.true_source, dim=1) / 3
    torch.testing.assert_close(problem.reference_times.flatten(), expected, rtol=1e-14, atol=1e-14)
    assert problem.grid.z.dtype == torch.float64
    assert problem.receiver_times.shape == (4,)


def test_gradient_grid_matches_notebook_arange_and_receivers_exactly():
    problem = prepare_problem(ExperimentConfig())
    z = torch.arange(0., 2.02, .02)
    x = torch.arange(0., 3.02, .02)
    assert problem.grid.shape == (101, 151)
    assert torch.equal(problem.grid.z, z)
    assert torch.equal(problem.grid.x, x)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    velocity = 2.5 + .5 * (zz - 1.)
    expected = torch.acosh(1. + .5 * (1. / velocity) * (1. / 2.5) * .25 *
                           ((xx - 1.5)**2 + (zz - 1.)**2)) / .5
    assert torch.equal(problem.grid.velocity, velocity)
    assert torch.equal(problem.reference_times, expected)
    indices = torch.arange(0, 151, 15)
    assert torch.equal(problem.receiver_coordinates, problem.grid.coordinates[indices])
    assert torch.equal(problem.receiver_times, expected.flatten()[indices])


@pytest.mark.parametrize("gradient", [.7, -.3, 0.])
def test_gradient_vertical_analytic_integral(gradient):
    config = DataConfig(source=(.5, 0.), velocity_gradient=gradient,
                        z_bounds=(0., 1.), x_bounds=(0., 1.), grid_shape=(21, 11))
    problem = prepare_problem(config, dtype="float64")
    z = problem.grid.z
    if gradient:
        expected = (torch.log((2. + gradient * z) / (2. + gradient * .5)) / gradient).abs()
    else:
        expected = (z - .5).abs() / 2.
    torch.testing.assert_close(problem.reference_times[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(problem.reference_times).all()


def test_otway_bytes_transpose_bounds_receivers_and_float32_source_floor():
    skfmm = pytest.importorskip("skfmm")
    path = ROOT / "otway2.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "25ebfe9aa0dd1a78f79acad489f9db64e93765bac8f5dea03c7e87921343ccdf"
    raw = np.loadtxt(path, delimiter=",").T.copy()
    config = load_config(ROOT / "configs/ex3.toml")
    problem = prepare_problem(config)
    assert problem.grid.shape == (246, 305)
    assert torch.equal(problem.grid.velocity, torch.tensor(raw, dtype=torch.float32))
    z, x = torch.linspace(0., 1.28, 246), torch.linspace(0., 1.704, 305)
    assert torch.equal(problem.grid.z, z)
    assert torch.equal(problem.grid.x, x)
    source_index = tuple(int(s // d) for s, d in zip(torch.tensor([.7, 1.]), (z[1], x[1])))
    assert source_index == (133, 178)
    phi = np.ones_like(raw)
    phi[source_index] = -1.
    reference = skfmm.travel_time(phi, raw, dx=(z[1], x[1]))
    assert torch.equal(problem.reference_times, torch.tensor(reference, dtype=torch.float32))
    assert torch.equal(problem.receiver_coordinates, problem.grid.coordinates[torch.arange(0, 305, 60)])


def test_fast_marching_optional_dependency_error(monkeypatch):
    original = builtins.__import__
    def without_skfmm(name, *args, **kwargs):
        if name == "skfmm":
            raise ModuleNotFoundError("No module named 'skfmm'", name="skfmm")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_skfmm)
    with pytest.raises(ImportError, match="fast-marching"):
        prepare_problem(load_config(ROOT / "configs/ex3.toml"))
    # Analytic and observed inputs never require this extra.
    assert prepare_problem(DataConfig()).reference_times is not None


def observed_arrays():
    return dict(z=np.array([0., 1.]), x=np.array([0., 1., 2.]),
                velocity=np.full((2, 3), 2.), receiver_coordinates=np.array([[0., 0.], [0., 2.]]),
                receiver_times=np.array([.3, .4]))


def test_observed_input_needs_no_true_source_or_reference(tmp_path):
    path = tmp_path / "observed.npz"
    np.savez(path, **observed_arrays())
    problem = load_observed_problem(path, dtype="float64")
    assert problem.true_source is None
    assert problem.reference_times is None
    assert problem.grid.velocity.dtype == torch.float64
    config = DataConfig(kind="observed", observed_path=str(path), source=None)
    assert prepare_problem(config).true_source is None


def test_observed_optional_truth_is_preserved(tmp_path):
    path = tmp_path / "truth.npz"
    arrays = observed_arrays() | dict(true_source=np.array([.5, 1.]), reference_times=np.ones((2, 3)))
    np.savez(path, **arrays)
    problem = load_observed_problem(path)
    assert torch.equal(problem.true_source, torch.tensor([.5, 1.]))
    assert torch.equal(problem.reference_times, torch.ones(2, 3))


@pytest.mark.parametrize("field,value,message", [
    ("velocity", np.zeros((2, 3)), "positive"),
    ("velocity", np.ones((3, 2)), "shape"),
    ("velocity", np.full((2, 3), np.nan), "finite"),
    ("z", np.array([1., 0.]), "increasing"),
    ("x", np.array([0., 0., 2.]), "increasing"),
    ("receiver_times", np.ones(3), "shape"),
    ("receiver_times", np.array([np.inf, .2]), "finite"),
    ("receiver_coordinates", np.array([[0., 3.], [0., 0.]]), "domain"),
])
def test_reject_invalid_observed_arrays(tmp_path, field, value, message):
    path = tmp_path / "bad.npz"
    arrays = observed_arrays()
    arrays[field] = value
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_observed_problem(path)


def test_observed_missing_required_key(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, z=np.array([0., 1.]))
    with pytest.raises(ValueError, match="missing.*(x|velocity)"):
        load_observed_problem(path)


@pytest.mark.parametrize("kwargs,message", [
    (dict(source=None), "source"),
    (dict(source=(5., 1.)), "source.*domain"),
    (dict(velocity_gradient=-3.), "positive"),
    (dict(receiver_indices=(0, 100000)), "receiver.*indices"),
    (dict(receiver_indices=(1.5, 2)), "receiver.*indices"),
])
def test_reject_invalid_synthetic_problem(kwargs, message):
    with pytest.raises(ValueError, match=message):
        prepare_problem(DataConfig(**kwargs))


def test_sampling_without_replacement_matches_local_numpy_and_preserves_global_state():
    problem = prepare_problem(DataConfig())
    expected = np.random.RandomState(4321).choice(np.arange(problem.grid.z.numel() * problem.grid.x.numel()), 2501, replace=False)
    state = np.random.get_state()
    batch = sample_training_batch(problem, 2501, RandomStreams())
    assert torch.equal(batch.coordinates, problem.grid.coordinates[expected])
    assert torch.equal(batch.velocity, problem.grid.velocity.flatten()[expected])
    assert not batch.coordinates.requires_grad
    assert torch.equal(batch.receiver_times, problem.receiver_times)
    assert np.array_equal(np.random.get_state()[1], state[1])
    with pytest.raises(ValueError, match="without replacement"):
        sample_training_batch(problem, problem.grid.velocity.numel() + 1, RandomStreams())


def test_sampling_with_replacement_is_explicit():
    problem = prepare_problem(DataConfig(grid_shape=(2, 2), receiver_count=2))
    batch = sample_training_batch(problem, 10, RandomStreams(), replace=True)
    assert batch.coordinates.shape == (10, 2)
    with pytest.raises(ValueError, match="positive integer"):
        sample_training_batch(problem, 0, RandomStreams())


def test_config_paths_resolved_against_config_file(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    directory.mkdir()
    np.savez(tmp_path / "observed.npz", **observed_arrays())
    path = directory / "observed.toml"
    path.write_text('name="observed"\ndtype="float64"\n[data]\nkind="observed"\nobserved_path="../observed.npz"\n')
    monkeypatch.chdir("/tmp")
    config = load_config(path)
    assert config.data.observed_path == str(tmp_path / "observed.npz")
    assert prepare_problem(config).true_source is None
    assert prepare_problem(config).grid.z.dtype == torch.float64


@pytest.mark.parametrize("name,source,points,add", [
    ("ex1", (1., 1.5), 2501, 1000.), ("ex2", (1., 1.5), 2501, 1000.),
    ("ex3", (.7, 1.), 5001, 5000.), ("ex3_2", (.75, 1.1), 5001, 5000.),
    ("ex3_3", (.65, .9), 5001, 5000.),
])
def test_presets_validate_and_match_checked_in_config(name, source, points, add):
    config = load_config(ROOT / "configs" / (name + ".toml"))
    builtin = load_preset(name, velocity_path=ROOT / "otway2.csv" if name.startswith("ex3") else None)
    assert config == builtin
    assert config.data.source == source
    assert config.data.collocation_count == points
    assert config.laplace.batch_size == points
    assert config.laplace.add == add
    assert config.seed == 4321
    assert config.training.steps == 3001


def test_unknown_config_keys_and_preset_names_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[data]\nkind="gradient"\nunknown_option=1\n')
    with pytest.raises((TypeError, ValueError), match="unknown_option"):
        load_config(path)
    with pytest.raises(ValueError, match="preset"):
        load_preset("fake")


def test_import_has_no_fast_marching_side_effect():
    code = 'import sys; import laplace_hypopinn.data; assert "skfmm" not in sys.modules'
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
