"""Functional HypoPINN numerical core; optional tools are imported on demand."""

from .config import DataConfig, ExperimentConfig, LaplaceConfig, ModelConfig, TrainingConfig
from .model import (
    flatten_parameters,
    initialize_model,
    make_mlp_forward,
    module_adapter,
    predict,
    unflatten_parameters,
)
from .types import (
    Forward,
    Grid,
    LossTerms,
    ParameterLayout,
    Parameters,
    Problem,
    RandomStreams,
    Streams,
    TrainingBatch,
)

__version__ = "0.1.0"
__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "LaplaceConfig",
    "ModelConfig",
    "TrainingConfig",
    "flatten_parameters",
    "initialize_model",
    "make_mlp_forward",
    "module_adapter",
    "predict",
    "unflatten_parameters",
    "Forward",
    "Grid",
    "LossTerms",
    "ParameterLayout",
    "Parameters",
    "Problem",
    "RandomStreams",
    "Streams",
    "TrainingBatch",
]
