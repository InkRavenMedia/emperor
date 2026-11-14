import unittest
from dataclasses import dataclass, replace

import torch
from torch import nn

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    LayerNormPositionOptions,
    LayerState,
    NormalizationOptions,
    RecurrentLayerConfig,
)
from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.nn import Module


@dataclass
class BufferedScaleConfig(ConfigBase):
    input_dim: int | None = optional_field("Input width.")
    output_dim: int | None = optional_field("Output width.")

    def _registry_owner(self) -> type:
        return BufferedScale


class BufferedScale(Module):
    def __init__(self, cfg, overrides=None):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.register_buffer("scale", torch.tensor(2.0))
        self.factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, state):
        return replace(state, hidden=state.hidden * self.scale * self.factor)


class TestRecurrentRuntimeStateGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.starting_rng_state = torch.get_rng_state()
        self.guard = RecurrentRuntimeStateGuard()
        self.module = nn.Module()
        self.module.register_buffer("progress", torch.tensor(2.0))
        self.module.register_buffer("optional", None)
        self.progress_buffer = self.module.progress

    def tearDown(self) -> None:
        torch.set_rng_state(self.starting_rng_state)

    def test_provisional_branch_always_restores_buffers_and_rng(self) -> None:
        torch.manual_seed(17)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(17)

        with self.guard.isolate_provisional_branch(self.module):
            self.module.progress.add_(5.0)
            self.module.register_buffer(
                "progress",
                torch.tensor(99.0),
                persistent=False,
            )
            self.module.register_buffer(
                "temporary",
                torch.tensor(1.0),
                persistent=False,
            )
            self.module.register_buffer("optional", torch.tensor(4.0))
            torch.rand(())

        self.assertIs(self.module.progress, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(2.0))
        self.assertIsNone(self.module.optional)
        self.assertFalse(hasattr(self.module, "temporary"))
        self.assertEqual(set(self.module.state_dict()), {"progress"})
        torch.testing.assert_close(torch.rand(()), expected_next_random_value)

    def test_failed_handoff_restores_buffers_and_rng(self) -> None:
        torch.manual_seed(23)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(23)

        with self.assertRaisesRegex(RuntimeError, "handoff failed"):
            with self.guard.rollback_handoff_on_failure(self.module):
                self.module.progress.add_(4.0)
                self.module.register_buffer("progress", torch.tensor(77.0))
                torch.rand(())
                raise RuntimeError("handoff failed")

        self.assertIs(self.module.progress, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(2.0))
        torch.testing.assert_close(torch.rand(()), expected_next_random_value)

    def test_successful_handoff_commits_buffers_and_rng(self) -> None:
        torch.manual_seed(31)
        expected_values = torch.rand(2)
        torch.manual_seed(31)

        with self.guard.rollback_handoff_on_failure(self.module):
            self.module.progress.add_(3.0)
            actual_in_scope_random_value = torch.rand(())

        self.assertIs(self.module.progress, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(5.0))
        torch.testing.assert_close(actual_in_scope_random_value, expected_values[0])
        torch.testing.assert_close(torch.rand(()), expected_values[1])
        self.assertEqual(vars(self.guard), {})

    def test_restoration_preserves_backward_through_unchanged_buffer_aliases(self):
        self.module.register_buffer("alias", self.progress_buffer)
        hidden = torch.ones(2, requires_grad=True)
        before_branch = hidden * self.module.progress

        with self.guard.isolate_provisional_branch(self.module):
            inside_branch = hidden * self.module.alias
            # Rebinding a name must not mutate the original saved tensor.
            self.module.register_buffer("progress", torch.tensor(99.0))

        (before_branch + inside_branch).sum().backward()

        torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 4.0))
        self.assertIs(self.module.progress, self.progress_buffer)
        self.assertIs(self.module.alias, self.progress_buffer)

    def test_restoration_keeps_modified_buffer_aliases_and_values(self):
        self.module.register_buffer("alias", self.progress_buffer)
        with self.guard.isolate_provisional_branch(self.module):
            self.module.alias.add_(3)
        self.assertIs(self.module.progress, self.progress_buffer)
        self.assertIs(self.module.alias, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(2.0))

    def test_restoration_supports_buffers_created_in_inference_mode(self):
        with torch.inference_mode():
            module = nn.Module()
            module.register_buffer("progress", torch.tensor(2.0))
            original_buffer = module.progress
            with self.guard.isolate_provisional_branch(module):
                module.progress.add_(3)
            self.assertIs(module.progress, original_buffer)
            torch.testing.assert_close(module.progress, torch.tensor(2.0))

    def test_restoration_preserves_native_batch_norm_running_statistics(self):
        module = nn.BatchNorm1d(2)
        original_buffers = dict(module.named_buffers())
        initial_values = {
            name: buffer.clone() for name, buffer in original_buffers.items()
        }
        with self.guard.isolate_provisional_branch(module):
            module(torch.tensor([[2.0, 4.0], [4.0, 8.0]]))
            self.assertFalse(
                torch.equal(module.running_mean, initial_values["running_mean"])
            )
        for name, buffer in module.named_buffers():
            self.assertIs(buffer, original_buffers[name])
            torch.testing.assert_close(buffer, initial_values[name])

    def test_unchanged_nan_buffer_does_not_invalidate_saved_tensors(self):
        self.module.register_buffer("nan_scale", torch.tensor(float("nan")))
        hidden = torch.ones((), requires_grad=True)
        with self.guard.isolate_provisional_branch(self.module):
            output = hidden * self.module.nan_scale
        output.backward()
        self.assertTrue(torch.isnan(hidden.grad))

    def test_restoration_preserves_backward_through_conjugated_buffer_views(self):
        scale = torch.tensor([1 + 2j, 3 + 4j]).conj()
        self.module.register_buffer("complex_scale", scale)
        hidden = torch.ones_like(scale, requires_grad=True)
        with self.guard.isolate_provisional_branch(self.module):
            output = hidden * self.module.complex_scale
        output.real.sum().backward()
        torch.testing.assert_close(hidden.grad, scale.conj())
        self.assertIs(self.module.complex_scale, scale)

    def test_buffered_block_outputs_and_gradients_match_depth_equations(self):
        for mode in ("stable", "gradient_suffix", "full_gradient"):
            with self.subTest(mode=mode):
                model = RecurrentLayerConfig(
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    recurrent_normalization=NormalizationOptions.LAYER_NORM,
                    input_dim=2,
                    output_dim=2,
                    max_steps=3,
                    initial_iterations=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    block_config=BufferedScaleConfig(),
                    smooth_iteration_growth_flag=mode != "stable",
                    gradient_transition_count=(None if mode == "full_gradient" else 2),
                    no_gradient_transition_count=(
                        0 if mode == "full_gradient" else None
                    ),
                ).build()
                model.recurrent_iteration_schedule.forward_call_progress.fill_(4)
                scale_buffer = model.block_model.scale
                checkpoint_keys = tuple(model.state_dict())
                hidden = torch.ones(1, 2, requires_grad=True)
                actual = model(LayerState(hidden)).hidden

                expected_input = hidden.detach().clone().requires_grad_()
                expected_factor = torch.tensor(1.0, requires_grad=True)
                source = expected_input * (2 * expected_factor) ** 2
                if mode == "full_gradient":
                    target = expected_input * (2 * expected_factor) ** 3
                else:
                    target_prefix = (expected_input * 2 * expected_factor).detach()
                    target = target_prefix * (2 * expected_factor) ** 2
                expected = target if mode == "stable" else (source + target) * 0.5

                actual_gradients = torch.autograd.grad(
                    actual.sum(), (hidden, model.block_model.factor), allow_unused=True
                )
                expected_gradients = torch.autograd.grad(
                    expected.sum(), (expected_input, expected_factor), allow_unused=True
                )
                torch.testing.assert_close(actual, expected)
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    if expected_gradient is None:
                        self.assertIsNone(actual_gradient)
                    else:
                        torch.testing.assert_close(actual_gradient, expected_gradient)
                self.assertIs(model.block_model.scale, scale_buffer)
                torch.testing.assert_close(scale_buffer, torch.tensor(2.0))
                self.assertEqual(tuple(model.state_dict()), checkpoint_keys)


if __name__ == "__main__":
    unittest.main()
