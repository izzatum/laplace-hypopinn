from dataclasses import FrozenInstanceError
import pytest
import torch

from laplace_hypopinn.config import (
    DEFAULT_WIDTHS,
    DataConfig,
    ExperimentConfig,
    LaplaceConfig,
    ModelConfig,
    TrainingConfig,
    resolve_device,
    resolve_dtype,
)
from laplace_hypopinn.types import RandomStreams


def test_historical_defaults_and_immutability():
    config = ExperimentConfig()
    assert config.model.widths == DEFAULT_WIDTHS
    assert config.training.steps == 3001
    assert config.training.learning_rate == 0.005
    assert config.training.betas == (0.5, 0.9)
    assert config.data.collocation_count == 2501
    assert config.laplace.method == "legacy_diagonal"
    assert config.laplace.add == 1000
    assert config.laplace.multiply == 0.001
    assert config.seed == 4321
    assert resolve_dtype(config.dtype) == torch.float32
    assert resolve_device(config.device) == torch.device("cpu")
    with pytest.raises(FrozenInstanceError):
        config.seed = 1


@pytest.mark.parametrize(
    "widths", [(), (2,), (3, 1), (2, 3, 2), (2, 0, 1), (2, 2.5, 1), (2, True, 1)]
)
def test_invalid_widths_fail(widths):
    with pytest.raises(ValueError, match="width"):
        ModelConfig(widths=widths)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (TrainingConfig, {"steps": 0}),
        (DataConfig, {"collocation_count": 0}),
        (LaplaceConfig, {"add": -1}),
        (LaplaceConfig, {"multiply": float("nan")}),
        (ModelConfig, {"activation": "bad"}),
        (ExperimentConfig, {"dtype": "float16"}),
    ],
)
def test_invalid_configuration_fails_early(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_invalid_devices_and_rng_independence():
    with pytest.raises(ValueError):
        resolve_device("mps")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda:0")
    state = torch.random.get_rng_state().clone()
    first, second = RandomStreams(4321), RandomStreams(4321)
    assert torch.equal(torch.rand(5, generator=first.torch), torch.rand(5, generator=second.torch))
    assert (first.numpy.rand(4) == second.numpy.rand(4)).all()
    assert torch.equal(state, torch.random.get_rng_state())


def test_corrected_precision_and_extension_configuration():
    assert LaplaceConfig(method="inf", rank=0).rank == 0
    assert LaplaceConfig(method="block_diagonal").method == "block_diagonal"
    assert LaplaceConfig(method="valla").variational_steps == 200
    assert LaplaceConfig(method="valla").variational_lr == 0.01
    assert LaplaceConfig(method="valla", alpha=1).alpha == 1
    for kwargs in (
        {"method": "diagonal", "prior_precision": 0},
        {"method": "diagonal", "scale": 0},
        {"add": 0},
        {"rank": -1},
        {"variational_lr": 0},
        {"variational_steps": 0},
        {"alpha": 0.5},
        {"method": "low_rank"},
    ):
        with pytest.raises(ValueError):
            LaplaceConfig(**kwargs)
