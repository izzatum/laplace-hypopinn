"""Plots use persisted research artifacts, including unknown-source observations."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def artifact_run(tmp_path, *, truth=True):
    directory = tmp_path / "run"
    directory.mkdir()
    z, x = np.array([0.0, 0.3, 1.8]), np.array([2.0, 2.2, 3.5, 6.0])
    arrays = dict(
        z=z, x=x, velocity=np.full((3, 4), 2.5),
        receiver_coordinates=np.array([[0.0, 2.2], [0.3, 6.0]]),
        receiver_times=np.array([0.4, 0.7]), map_field=np.arange(12).reshape(3, 4) / 10,
        map_location=np.array([0.3, 3.5]),
        posterior_locations=np.array([[0.0, 2.2], [1.8, 6.0], [0.3, 3.5]]),
        field_mean=np.ones((3, 4)), field_std=np.full((3, 4), 0.2),
    )
    if truth:
        arrays.update(true_source=np.array([1.8, 2.2]), reference_times=np.ones((3, 4)))
    np.savez_compressed(directory / "arrays.npz", **arrays)
    (directory / "metrics.json").write_text(json.dumps({"history": [
        {"step": 1, "total": 3.0, "pde": 2.0, "data": 1.0},
        {"step": 3, "total": 1.5, "pde": 1.0, "data": 0.5},
    ]}))
    return directory, arrays


def test_headless_plots_preserve_artifact_coordinates_and_units(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from laplace_hypopinn.plotting import plot_run

    run, arrays = artifact_run(tmp_path)
    snapshots = {}
    original = Figure.savefig

    def inspect_figure(figure, path, *args, **kwargs):
        axis = figure.axes[0]
        snapshots[Path(path).stem] = dict(
            xlabel=axis.get_xlabel(), ylabel=axis.get_ylabel(),
            lines={line.get_label(): (line.get_xdata().copy(), line.get_ydata().copy())
                   for line in axis.lines},
            collections={collection.get_label(): np.asarray(collection.get_offsets()).copy()
                         for collection in axis.collections},
            mesh=axis.collections[0].get_coordinates().copy()
                 if axis.collections and hasattr(axis.collections[0], "get_coordinates") else None,
            values=np.asarray(axis.collections[0].get_array()).copy()
                   if axis.collections and hasattr(axis.collections[0], "get_coordinates") else None,
            colorbar_label=figure.axes[1].get_ylabel() if len(figure.axes) > 1 else None,
            marker_clipping={collection.get_label(): collection.get_clip_on()
                             for collection in axis.collections
                             if collection.get_label() in ("Receivers", "MAP source",
                                                           "True source", "Posterior samples")},
        )
        return original(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    before = plt.get_fignums()
    outputs = plot_run(run)
    assert {path.stem for path in outputs} == {
        "velocity", "map_field", "source_locations", "training_history", "reference_times",
        "field_mean", "field_std",
    }
    assert all(path.parent == run / "plots" and path.stat().st_size > 1000 for path in outputs)
    assert plt.get_fignums() == before
    field = snapshots["map_field"]
    assert (field["xlabel"], field["ylabel"]) == ("x (km)", "z (km)")
    assert field["colorbar_label"] == "Travel time (s)"
    np.testing.assert_array_equal(field["values"], arrays["map_field"])
    assert snapshots["velocity"]["colorbar_label"] == "Velocity (km/s)"
    mesh = field["mesh"]
    # Mesh cell edges follow the nonuniform physical coordinates, rather than pixel indices.
    np.testing.assert_allclose((mesh[0, :-1, 0] + mesh[0, 1:, 0]) / 2,
                               [2.0, 2.475, 3.8, 6.0])
    assert not any(snapshots["source_locations"]["marker_clipping"].values())
    locations = snapshots["source_locations"]["collections"]
    for label, key in (("Receivers", "receiver_coordinates"),
                       ("Posterior samples", "posterior_locations")):
        np.testing.assert_allclose(locations[label], arrays[key][:, ::-1])
    np.testing.assert_allclose(locations["MAP source"], arrays["map_location"][None, ::-1])
    np.testing.assert_allclose(locations["True source"], arrays["true_source"][None, ::-1])
    np.testing.assert_array_equal(snapshots["training_history"]["lines"]["total"][0], [1, 3])


def test_observed_artifacts_need_no_ground_truth_and_reject_overwrite(tmp_path):
    from laplace_hypopinn.plotting import plot_run

    run, _ = artifact_run(tmp_path, truth=False)
    output = tmp_path / "figures"
    paths = plot_run(run, output)
    assert "reference_times" not in {path.stem for path in paths}
    saved = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        plot_run(run, output)
    assert all(path.read_bytes() == contents for path, contents in saved.items())


def test_plotting_import_does_not_import_optional_matplotlib():
    result = subprocess.run([sys.executable, "-c", (
        "import sys; import laplace_hypopinn.plotting; "
        "assert not any(name.startswith('matplotlib') for name in sys.modules)"
    )], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
