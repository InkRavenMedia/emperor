import copy
import unittest
from contextlib import nullcontext

import torch

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters._decay import DecayPolicy
from emperor.nn import (
    ForwardSchedule,
    freeze_forward_schedules,
    grouped_forward_calls,
    once_per_recurrent_run,
)


def decay_policy(schedule=WeightDecayScheduleOptions.EXPONENTIAL, warmup=0):
    return DecayPolicy(
        AdditiveDynamicBiasConfig(
            decay_schedule=schedule,
            decay_rate=0.1,
            decay_warmup_batches=warmup,
        )
    )


class ForwardScheduleTests(unittest.TestCase):
    def test_bound_schedule_follows_copied_owner_and_replaced_buffers(self):
        from emperor.nn import ForwardScheduleCounterOptions

        policy = decay_policy()
        counter = ForwardScheduleCounterOptions.DECAY_STEP
        policy.schedule.set_value(counter, 2)
        restored = copy.deepcopy(policy).to(dtype=torch.float64)
        restored.schedule.set_value(counter, 5)

        self.assertEqual(policy.schedule.value(counter).item(), 2)
        self.assertEqual(restored.schedule.value(counter).item(), 5)
        self.assertEqual(restored.schedule.value(counter).dtype, torch.float64)
        self.assertEqual(set(restored.state_dict()), {"warmup_step", "decay_step"})
        replacement = torch.tensor([7.0], dtype=torch.float64)
        restored.decay_step = replacement
        restored.schedule.set_value(counter, 8)
        torch.testing.assert_close(
            replacement, torch.tensor([8.0], dtype=torch.float64)
        )
        self.assertEqual(policy.schedule.value(counter).item(), 2)

    def test_counter_access_requires_enum_members_in_every_context(self):
        policy = decay_policy()
        accesses = (
            (policy.schedule.value, ()),
            (policy.schedule.set_value, (1,)),
            (policy.schedule.run_value, ()),
            (policy.schedule.set_run_value, (1,)),
        )
        for context in (
            nullcontext,
            once_per_recurrent_run,
            freeze_forward_schedules,
        ):
            with context():
                for counter in ("decay_step", WeightDecayScheduleOptions.EXPONENTIAL):
                    for access, arguments in accesses:
                        with self.subTest(
                            context=context.__name__,
                            counter=counter,
                            access=access.__name__,
                        ):
                            with self.assertRaisesRegex(
                                TypeError, "ForwardScheduleCounterOptions"
                            ):
                                access(counter, *arguments)
        self.assertEqual(policy.decay_step.item(), 0)

    def test_counter_options_preserve_buffer_names_and_grouped_run_updates(self):
        from emperor.nn import ForwardScheduleCounterOptions

        module = torch.nn.Module()
        module.schedule = ForwardSchedule(module)
        counters = (
            (ForwardScheduleCounterOptions.WARMUP_STEP, "warmup_step"),
            (ForwardScheduleCounterOptions.DECAY_STEP, "decay_step"),
            (
                ForwardScheduleCounterOptions.FORWARD_CALL_PROGRESS,
                "forward_call_progress",
            ),
        )
        for counter, buffer_name in counters:
            module.register_buffer(buffer_name, torch.zeros(1))
            module.schedule.set_value(counter, 2)

        with grouped_forward_calls() as group:
            for _ in range(2):
                with group(), once_per_recurrent_run():
                    for counter, _ in counters:
                        self.assertEqual(module.schedule.run_value(counter).item(), 2)
                        module.schedule.set_run_value(counter, 3)
                        self.assertEqual(module.schedule.run_value(counter).item(), 2)

        self.assertEqual(set(module.state_dict()), {name for _, name in counters})
        for counter, buffer_name in counters:
            self.assertEqual(module.schedule.value(counter).item(), 3)
            self.assertEqual(module.state_dict()[buffer_name].item(), 3)

    def test_groups_replay_native_calls_and_commit_only_one_progression(self):
        for option in (
            WeightDecayScheduleOptions.EXPONENTIAL,
            WeightDecayScheduleOptions.LINEAR,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
        ):
            with self.subTest(schedule=option):
                schedule = decay_policy(option, warmup=1)
                reference = decay_policy(option, warmup=1)
                parameter = torch.ones(2, requires_grad=True)
                expected = [reference(parameter) for _ in range(3)]
                outputs = []
                with grouped_forward_calls() as group:
                    for _ in range(4):
                        with group():
                            actual = [schedule(parameter) for _ in range(3)]
                            for value, target in zip(actual, expected, strict=False):
                                torch.testing.assert_close(value, target)
                            outputs.extend(actual)
                self.assertEqual(schedule.warmup_step.item(), 1)
                self.assertEqual(schedule.decay_step.item(), 2)
                self.assertLess(outputs[-1][0].item(), 1)
                sum(value.sum() for value in outputs).backward()
                self.assertTrue(torch.isfinite(parameter.grad).all())
                restored = decay_policy(option, warmup=1)
                restored.load_state_dict(schedule.state_dict(), strict=True)
                torch.testing.assert_close(restored(parameter), schedule(parameter))

    def test_failed_empty_and_eval_groups_do_not_advance_decay(self):
        schedule = decay_policy()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with grouped_forward_calls() as group:
                with group():
                    schedule(torch.ones(1))
                raise RuntimeError("failure")
        self.assertEqual(schedule.decay_step.item(), 0)
        with grouped_forward_calls():
            pass
        schedule.eval()
        with grouped_forward_calls() as group:
            with group():
                schedule(torch.ones(1))
        self.assertEqual(schedule.decay_step.item(), 0)
        schedule.train()
        schedule(torch.ones(1))
        self.assertEqual(schedule.decay_step.item(), 1)

    def test_nested_groups_and_uneven_expert_invocations(self):
        schedule, expert = decay_policy(), decay_policy()
        with grouped_forward_calls() as group:
            for calls in (1, 3, 2):
                with group():
                    schedule(torch.ones(1))
                    with grouped_forward_calls() as subgroup:
                        for _ in range(2):
                            with subgroup():
                                for _ in range(calls):
                                    expert(torch.ones(1))
        self.assertEqual(schedule.decay_step.item(), 1)
        self.assertEqual(expert.decay_step.item(), 3)

    def test_frozen_context_preserves_counters_and_restores_after_failure(self):
        schedule = decay_policy()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with freeze_forward_schedules():
                with grouped_forward_calls() as group:
                    with group():
                        schedule(torch.ones(1))
                raise RuntimeError("failure")
        self.assertEqual(schedule.decay_step.item(), 0)
        schedule(torch.ones(1))
        self.assertEqual(schedule.decay_step.item(), 1)

    def test_recurrent_adaptive_decay_uses_one_factor_per_run(self):
        from emperor.layers import LayerState, RecurrentLayerConfig
        from tests.support.hierarchical_embedding import adaptive_stack

        block = adaptive_stack(2, 2)
        decay = block.layer_config.layer_model_config.adaptive_augmentation_config.bias_config
        decay.decay_schedule = WeightDecayScheduleOptions.EXPONENTIAL
        decay.decay_rate = 0.2
        config = RecurrentLayerConfig(
            input_dim=2,
            output_dim=2,
            max_steps=4,
            initial_iterations=4,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            no_gradient_transition_count=0,
            gradient_transition_count=None,
            block_config=block,
        )
        for nested in (False, True):
            with self.subTest(nested=nested):
                active_config = config
                if nested:
                    active_config = RecurrentLayerConfig(
                        input_dim=2,
                        output_dim=2,
                        max_steps=2,
                        initial_iterations=2,
                        iteration_increment=1,
                        forward_calls_before_iteration_increment=1,
                        no_gradient_transition_count=0,
                        gradient_transition_count=None,
                        block_config=config,
                    )
                model = active_config.build().train()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(0.1)
                policy = next(
                    module
                    for module in model.modules()
                    if isinstance(module, DecayPolicy)
                )
                policy.decay_step.fill_(2)
                factors = []
                handle = policy.register_forward_hook(
                    lambda module, args, output, factors=factors: factors.append(
                        (output / args[0]).detach()
                    )
                )
                for progress in (2, 3):
                    factors.clear()
                    output = model(LayerState(hidden=torch.ones(1, 2)))
                    self.assertEqual(len(factors), 8 if nested else 4)
                    for factor in factors:
                        torch.testing.assert_close(
                            factor,
                            torch.full_like(
                                factor, torch.exp(torch.tensor(-0.2 * progress))
                            ),
                        )
                    self.assertEqual(policy.decay_step.item(), progress + 1)
                    output.hidden.sum().backward()
                    self.assertTrue(
                        all(
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()
                            if p.grad is not None
                        )
                    )
                handle.remove()

    def test_recurrent_scope_warmup_and_decay_are_independent_of_length_groups(self):
        for option in (
            WeightDecayScheduleOptions.EXPONENTIAL,
            WeightDecayScheduleOptions.LINEAR,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
        ):
            with self.subTest(schedule=option):
                policy = decay_policy(option, warmup=1)
                reference = decay_policy(option, warmup=1)
                for run in range(3):
                    expected = reference(torch.ones(2))
                    with grouped_forward_calls() as group:
                        for _ in range(3):
                            with group(), once_per_recurrent_run():
                                for _ in range(5):
                                    with once_per_recurrent_run():
                                        torch.testing.assert_close(
                                            policy(torch.ones(2)), expected
                                        )
                    self.assertEqual(policy.warmup_step.item(), 1)
                    self.assertEqual(policy.decay_step.item(), max(0, run))
                restored = decay_policy(option, warmup=1)
                restored.load_state_dict(policy.state_dict(), strict=True)
                torch.testing.assert_close(
                    restored(torch.ones(2)), reference(torch.ones(2))
                )

    def test_failed_recurrence_discards_decay_and_restores_scope(self):
        from emperor.layers import LayerState, RecurrentLayerConfig
        from tests.support.hierarchical_embedding import adaptive_stack

        block = adaptive_stack(2, 2)
        decay = block.layer_config.layer_model_config.adaptive_augmentation_config.bias_config
        decay.decay_schedule = WeightDecayScheduleOptions.LINEAR
        decay.decay_rate = 0.1
        model = RecurrentLayerConfig(
            input_dim=2,
            output_dim=2,
            max_steps=3,
            initial_iterations=3,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            block_config=block,
        ).build()
        policy = next(
            module for module in model.modules() if isinstance(module, DecayPolicy)
        )

        def fail(module, args, output):
            raise RuntimeError("recurrent failure")

        hook = policy.register_forward_hook(fail)
        with self.assertRaisesRegex(RuntimeError, "recurrent failure"):
            model(LayerState(hidden=torch.ones(1, 2)))
        self.assertEqual(policy.decay_step.item(), 0)
        hook.remove()
        model(LayerState(hidden=torch.ones(1, 2)))
        self.assertEqual(policy.decay_step.item(), 1)
        model.eval()
        model(LayerState(hidden=torch.ones(1, 2)))
        self.assertEqual(policy.decay_step.item(), 1)

    def test_provisional_branch_rolls_back_deferred_decay_updates(self):
        from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
            RecurrentRuntimeStateGuard,
        )

        committed, provisional = decay_policy(), decay_policy()
        modules = torch.nn.ModuleList([committed, provisional])
        guard = RecurrentRuntimeStateGuard()
        with grouped_forward_calls() as group:
            with group(), once_per_recurrent_run():
                committed(torch.ones(2))
                with guard.isolate_provisional_branch(modules):
                    committed(torch.ones(2))
                    provisional(torch.ones(2))
                committed(torch.ones(2))
        self.assertEqual(committed.decay_step.item(), 1)
        self.assertEqual(provisional.decay_step.item(), 0)
