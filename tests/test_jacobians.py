import pytest
import torch

from laplace_hypopinn.config import TrainingConfig
from laplace_hypopinn.model import (
    flatten_parameters,
    initialize_model,
    predict,
    unflatten_parameters,
)
from test_losses import example, polynomial


def test_mixed_coordinate_parameter_jacobians_match_finite_differences():
    from laplace_hypopinn.curvature.residuals import make_residual_plan, residual_values
    from laplace_hypopinn.curvature.jacobians import iter_jacobians

    parameters, batch = example()
    plan = make_residual_plan(batch, TrainingConfig())
    vector, layout = flatten_parameters(parameters)

    def residuals(vector):
        restored = unflatten_parameters(vector, layout)
        return torch.cat(
            [
                entry.alpha.sqrt() * residual_values(restored, entry, forward=polynomial)
                for entry in plan
            ]
        )

    step = 1e-6
    eye = torch.eye(vector.numel(), dtype=vector.dtype)
    finite_difference = torch.stack(
        [
            (residuals(vector + step * direction) - residuals(vector - step * direction))
            / (2 * step)
            for direction in eye
        ],
        -1,
    )
    rows = torch.cat(list(iter_jacobians(parameters, plan, 2, forward=polynomial)))
    torch.testing.assert_close(rows, finite_difference, rtol=1e-7, atol=1e-8)


def test_chunk_sizes_and_duplicate_half_weights_preserve_ggn():
    from laplace_hypopinn.curvature.residuals import ResidualBatch, make_residual_plan
    from laplace_hypopinn.curvature.jacobians import iter_jacobians

    parameters, batch = example()
    plan = make_residual_plan(batch, TrainingConfig())

    def ggn(plan, chunk_size):
        rows = torch.cat(list(iter_jacobians(parameters, plan, chunk_size, forward=polynomial)))
        return rows.T @ rows

    expected = ggn(plan, 100)
    for chunk_size in (1, 2, 3, 5):
        torch.testing.assert_close(ggn(plan, chunk_size), expected)
    doubled = tuple(
        ResidualBatch(
            entry.family,
            entry.coordinates.repeat(2, 1),
            entry.targets.repeat(2),
            entry.alpha.repeat(2) / 2,
        )
        for entry in plan
    )
    torch.testing.assert_close(ggn(doubled, 2), expected)
    torch.testing.assert_close(ggn(plan, 2), expected)


def test_mlp_jacobian_uses_canonical_weight_bias_layout():
    from laplace_hypopinn.curvature.residuals import ResidualBatch
    from laplace_hypopinn.curvature.jacobians import jacobian_blocks

    parameters = initialize_model((2, 2, 1), dtype=torch.float64)
    coords = torch.tensor([[0.3, 0.2], [0.1, 0.9]], dtype=torch.float64)
    batch = ResidualBatch(
        "data",
        coords,
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    vector, layout = flatten_parameters(parameters)
    expected = torch.func.jacrev(
        lambda value: predict(unflatten_parameters(value, layout), coords).reshape(-1)
    )(vector)
    actual = torch.cat(list(jacobian_blocks(parameters, batch, 1)))
    torch.testing.assert_close(actual, expected * batch.alpha.sqrt()[:, None])


def test_invalid_chunk_size_rejected():
    from laplace_hypopinn.curvature.jacobians import iter_jacobians

    parameters, _ = example()
    with pytest.raises(ValueError, match="positive"):
        list(iter_jacobians(parameters, (), 0, forward=polynomial))


def test_custom_forward_with_scalar_and_rank_three_parameters():
    from laplace_hypopinn.curvature.residuals import make_residual_plan, residual_values
    from laplace_hypopinn.curvature.jacobians import iter_jacobians

    _, batch = example()
    parameters = {
        "kernel": torch.arange(4, dtype=torch.float64).reshape(1, 2, 2) / 10,
        "constant": torch.tensor(-0.7, dtype=torch.float64),
    }

    def custom(parameters, coordinates):
        return (
            torch.einsum("ni,ij,nj->n", coordinates, parameters["kernel"][0], coordinates)
            + parameters["constant"]
        )

    plan = make_residual_plan(batch, TrainingConfig())
    vector, layout = flatten_parameters(parameters)
    expected = torch.func.jacrev(
        lambda value: torch.cat(
            [
                entry.alpha.sqrt()
                * residual_values(unflatten_parameters(value, layout), entry, forward=custom)
                for entry in plan
            ]
        )
    )(vector)
    rows = torch.cat(list(iter_jacobians(parameters, plan, 2, forward=custom)))
    torch.testing.assert_close(rows, expected)
    assert rows.shape == (8, 5)
