"""Headless figures from saved run arrays; Matplotlib is an optional dependency."""

import json
from pathlib import Path

import numpy as np


def plot_run(run_directory, output_directory=None) -> list[Path]:
    """Plot persisted physical fields, source locations and training history.

    Coordinates in ``arrays.npz`` are (z,x) in km; times are seconds and velocity
    is km/s. Truth and posterior fields are optional. Existing figure paths are
    rejected before any figure is written. Importing this module needs no plot
    extra or graphical display.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires the optional extra: pip install 'laplace-hypopinn[plot]'") from exc

    directory = Path(run_directory)
    output = directory / "plots" if output_directory is None else Path(output_directory)
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    history = metrics.get("history", [])
    z, x = arrays["z"], arrays["x"]
    fields = {
        "velocity": ("Velocity model", "Velocity (km/s)"),
        "map_field": ("MAP travel time", "Travel time (s)"),
        "reference_times": ("Reference travel time", "Travel time (s)"),
        "field_mean": ("Posterior mean travel time", "Travel time (s)"),
        "field_std": ("Posterior travel-time standard deviation", "Standard deviation (s)"),
    }
    selected = [key for key in fields if key in arrays]
    for key in selected:
        if arrays[key].shape != (len(z), len(x)):
            raise ValueError(f"{key} must have shape (len(z), len(x))")
    names = selected + ["source_locations"] + (["training_history"] if history else [])
    paths = [output / f"{name}.png" for name in names]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    output.mkdir(parents=True, exist_ok=True)

    def source_markers(axis, posterior=False):
        markers = [("receiver_coordinates", "Receivers", "v", "tab:orange", 35),
                   ("map_location", "MAP source", "*", "red", 120),
                   ("true_source", "True source", "X", "white", 70)]
        if posterior and "posterior_locations" in arrays:
            coordinates = arrays["posterior_locations"]
            axis.scatter(coordinates[:, 1], coordinates[:, 0], s=16, alpha=0.35,
                         label="Posterior samples", color="tab:blue", clip_on=False)
        for key, label, marker, color, size in markers:
            if key in arrays:
                coordinates = np.atleast_2d(arrays[key])
                axis.scatter(coordinates[:, 1], coordinates[:, 0], label=label,
                             marker=marker, color=color, edgecolors="black", s=size, clip_on=False)
        axis.set(xlabel="x (km)", ylabel="z (km)", xlim=(x[0], x[-1]), ylim=(z[-1], z[0]))
        axis.set_aspect("equal")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best", fontsize="small")

    for name, path in zip(names, paths, strict=True):
        figure = Figure(figsize=(7, 5), layout="constrained")
        FigureCanvasAgg(figure)
        try:
            axis = figure.subplots()
            if name in fields:
                title, units = fields[name]
                image = axis.pcolormesh(x, z, arrays[name], shading="nearest", cmap="viridis")
                figure.colorbar(image, ax=axis, label=units)
                source_markers(axis)
                axis.set_title(title)
            elif name == "source_locations":
                source_markers(axis, posterior=True)
                axis.set_title("Source locations")
            else:
                steps = [row["step"] for row in history]
                for key in ("total", "pde", "data", "positivity", "prior"):
                    if all(key in row for row in history):
                        axis.plot(steps, [row[key] for row in history], label=key)
                axis.set(xlabel="Adam update", ylabel="Loss", title="Training history")
                axis.legend()
                axis.grid(alpha=0.2)
            figure.savefig(path, dpi=150)
        finally:
            plt.close(figure)
            figure.clear()
    return paths
