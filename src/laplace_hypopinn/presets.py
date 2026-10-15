"""Historical examples and strict TOML loading, without working-directory state."""
from dataclasses import replace
from pathlib import Path
import tomllib

from .config import DataConfig, ExperimentConfig, LaplaceConfig, ModelConfig, TrainingConfig


def load_config(path) -> ExperimentConfig:
    """Load TOML and resolve data paths against its directory, rejecting unknown keys."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        fields = tomllib.load(stream)
    constructors = {"model": ModelConfig, "data": DataConfig, "training": TrainingConfig,
                    "laplace": LaplaceConfig}
    for name, constructor in constructors.items():
        if name in fields:
            section = fields[name]
            if name == "data":
                for key in ("velocity_path", "observed_path"):
                    if key in section:
                        candidate = Path(section[key]).expanduser()
                        if not candidate.is_absolute():
                            candidate = path.parent / candidate
                        section[key] = str(candidate.resolve())
            fields[name] = constructor(**section)
    return ExperimentConfig(**fields)


def load_preset(name: str, *, velocity_path=None) -> ExperimentConfig:
    """Return an archived example's settings, without implicitly finding data.

    Ex1's four sequential initializations are orchestrated by the experiment
    runner; its ModelConfig records the first initialization. Otway presets
    require an explicit velocity_path before data preparation and work from an
    installed wheel without a source checkout. CSV bounds are verified values,
    avoiding NumPy arange endpoint drift in the historical notebooks.
    """
    sources = {"ex1": (1., 1.5), "ex2": (1., 1.5), "ex3": (.7, 1.),
               "ex3_2": (.75, 1.1), "ex3_3": (.65, .9)}
    if name not in sources:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(sources)}")
    data = DataConfig(source=sources[name])
    laplace = LaplaceConfig()
    if name.startswith("ex3"):
        data = replace(data, kind="otway", source=sources[name], collocation_count=5001,
                       velocity_path=None if velocity_path is None else str(Path(velocity_path).expanduser().resolve()),
                       grid_shape=(246, 305), z_bounds=(0., 1.28), x_bounds=(0., 1.704),
                       receiver_count=6, receiver_indices=(0, 60, 120, 180, 240, 300))
        laplace = replace(laplace, batch_size=5001, add=5000.)
    return ExperimentConfig(name=name, data=data, laplace=laplace)
