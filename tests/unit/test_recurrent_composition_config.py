import ast
import importlib
import inspect
import pickle
import textwrap
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import torch

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    HierarchicalReasoningModelRecurrentConfig,
    InnerThinkingRecurrentConfig,
    LayerNormPositionOptions,
    LayerState,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentLayer,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.base import (
    RecurrentCompositionAbstract,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)
from emperor.layers._composition.recurrent.state import RecurrentState
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
    RecurrentIterationScheduleValidator,
    RecurrentLayerValidator,
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model import (
    HierarchicalReasoningModelRecurrent,
)
from emperor.layers._composition.recurrent.variants.tiny_recursive_model import (
    TinyRecursiveModelRecurrent,
)
from emperor.layers._pipeline import LayerProcessingPipeline
from support.layers import IdentityBlockConfig
from support.recurrent import inner_thinking_config


def _declared_protected_method_names(owner: type) -> set[str]:
    private_method_prefix = f"_{owner.__name__.lstrip('_')}__"
    return {
        name
        for name, member in vars(owner).items()
        if name.startswith("_")
        and not name.startswith("__")
        and not name.startswith(private_method_prefix)
        and (callable(member) or isinstance(member, (classmethod, staticmethod)))
    }


def _minimal_variant_configs():
    shared_config = dict(
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        recurrent_normalization=NormalizationOptions.LAYER_NORM,
        input_dim=2,
        output_dim=2,
        initial_iterations=1,
        iteration_increment=1,
        forward_calls_before_iteration_increment=1,
    )
    return (
        RecurrentLayerConfig(
            **shared_config, max_steps=1, block_config=IdentityBlockConfig()
        ),
        inner_thinking_config(max_steps=1, initial_iterations=1),
        TinyRecursiveModelRecurrentConfig(
            **shared_config,
            block_config=IdentityBlockConfig(),
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            initialization_standard_deviation=0.0,
        ),
        HierarchicalReasoningModelRecurrentConfig(
            **shared_config,
            high_block_config=IdentityBlockConfig(),
            low_block_config=IdentityBlockConfig(),
            high_cycles=1,
            low_cycles=1,
            initialization_standard_deviation=0.0,
        ),
    )


class TestRecurrentCompositionConfig(unittest.TestCase):
    def test_usage_monitoring_does_not_enter_base_or_variant_contracts(self) -> None:
        emperor_root = Path(__file__).parents[2] / "src" / "emperor"
        recurrent_root = emperor_root / "layers" / "_composition" / "recurrent"
        source_paths = (
            emperor_root / "halting" / "_base.py",
            *(emperor_root / "halting" / "_variants").glob("*.py"),
            recurrent_root / "base.py",
            *(recurrent_root / "variants").rglob("*.py"),
        )

        for source_path in source_paths:
            source = source_path.read_text(encoding="utf-8")
            for monitoring_term in (
                "HaltingUsageTracker",
                "_usage_tracker",
                "suppress_usage_tracking",
            ):
                with self.subTest(
                    source_path=source_path,
                    monitoring_term=monitoring_term,
                ):
                    self.assertNotIn(monitoring_term, source)

    def test_execution_framework_is_replaced_by_step_computation_and_handoff(
        self,
    ) -> None:
        recurrent_root = (
            Path(__file__).parents[2] / "src/emperor/layers/_composition/recurrent"
        )
        runtime_root = recurrent_root / "runtime"
        self.assertFalse(any((runtime_root / "execution").rglob("*.py")))
        self.assertTrue((runtime_root / "step.py").is_file())
        self.assertTrue((runtime_root / "runtime_state.py").is_file())
        handoff_source = (runtime_root / "handoff/core.py").read_text()
        self.assertIn("isolate_provisional_branch", handoff_source)
        self.assertIn("rollback_handoff_on_failure", handoff_source)
        for path in recurrent_root.rglob("*.py"):
            with self.subTest(path=path):
                source = path.read_text()
                for retired_name in (
                    "PreparedRecurrentTransition",
                    "RecurrentExecutionResult",
                    "RecurrentTransitionResult",
                    "RecurrentStateOperations",
                    "RecurrentExecutionAdapter",
                ):
                    self.assertNotIn(retired_name, source)
        handoff_package = importlib.import_module(
            "emperor.layers._composition.recurrent.runtime.handoff"
        )
        self.assertEqual(handoff_package.__all__, ["RecurrentDepthHandoff"])

    def test_shared_initialization_validates_before_reading_configuration(self) -> None:
        base_initializer_tree = ast.parse(
            textwrap.dedent(inspect.getsource(RecurrentCompositionAbstract.__init__))
        )
        base_initializer_calls = [
            node
            for node in ast.walk(base_initializer_tree)
            if isinstance(node, ast.Call)
        ]
        call_lines = {
            ast.unparse(call.func): call.lineno for call in base_initializer_calls
        }
        config_assignments = [
            node
            for node in ast.walk(base_initializer_tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and any(
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Attribute)
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id == "self"
                and value.value.attr == "cfg"
                for value in ast.walk(node.value)
            )
        ]
        self.assertTrue(config_assignments)
        self.assertLess(
            call_lines["self.VALIDATOR.validate"],
            min(assignment.lineno for assignment in config_assignments),
        )
        self.assertNotIn("RecurrentIterationSchedule", call_lines)
        self.assertNotIn("LayerProcessingPipeline", call_lines)
        for owner, step_builder in (
            (RecurrentLayer, "self._build_step"),
            (TinyRecursiveModelRecurrent, "self.__build_step"),
            (HierarchicalReasoningModelRecurrent, "self.__build_step"),
        ):
            with self.subTest(variant=owner.__name__):
                initializer_tree = ast.parse(
                    textwrap.dedent(inspect.getsource(owner.__init__))
                )
                initializer_calls = {
                    ast.unparse(node.func): node.lineno
                    for node in ast.walk(initializer_tree)
                    if isinstance(node, ast.Call)
                }
                self.assertLess(
                    initializer_calls["super().__init__"],
                    initializer_calls["RecurrentIterationSchedule"],
                )
                self.assertLess(
                    initializer_calls["RecurrentIterationSchedule"],
                    initializer_calls[step_builder],
                )
                self.assertNotIn("LayerProcessingPipeline", initializer_calls)
        self.assertNotIn(
            "_RecurrentCompositionAbstract__initialize_delegates",
            vars(RecurrentCompositionAbstract),
        )

    def test_every_variant_requires_explicit_normalization_settings(self) -> None:
        for config in _minimal_variant_configs():
            for field_name in (
                "recurrent_layer_norm_position",
                "recurrent_normalization",
            ):
                with self.subTest(variant=type(config).__name__, field=field_name):
                    missing_setting = replace(config, **{field_name: None})
                    with self.assertRaisesRegex(
                        ValueError, f"{field_name} is required.*received None"
                    ):
                        missing_setting.build()

    def test_every_variant_rejects_invalid_normalization_options(self) -> None:
        for config in _minimal_variant_configs():
            for field_name, wrong_enum in (
                ("recurrent_layer_norm_position", NormalizationOptions.LAYER_NORM),
                ("recurrent_normalization", LayerNormPositionOptions.DISABLED),
            ):
                for invalid_value in ("DISABLED", 0, wrong_enum):
                    with self.subTest(
                        variant=type(config).__name__,
                        field=field_name,
                        value=invalid_value,
                    ):
                        invalid_setting = replace(config, **{field_name: invalid_value})
                        with self.assertRaisesRegex(
                            TypeError, f"{field_name} must be a"
                        ):
                            invalid_setting.build()

    def test_every_variant_keeps_explicit_normalization_choices(self) -> None:
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                model = replace(
                    config,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    recurrent_normalization=NormalizationOptions.RMS_NORM,
                ).build()
                self.assertIs(
                    model.pipeline.normalization.position,
                    LayerNormPositionOptions.DISABLED,
                )
                self.assertIs(
                    model.pipeline.normalization.normalization,
                    NormalizationOptions.RMS_NORM,
                )
                self.assertIsNone(model.pipeline.normalization.module)

    def test_variants_own_the_ordinary_loop_without_execution_plans(self):
        for owner in (
            RecurrentLayer,
            TinyRecursiveModelRecurrent,
            HierarchicalReasoningModelRecurrent,
        ):
            with self.subTest(variant=owner.__name__):
                run_recurrence = getattr(
                    owner, f"_{owner.__name__}__run_recurrence_with_step_limit"
                )
                branch_source = inspect.getsource(run_recurrence)
                branch_tree = ast.parse(textwrap.dedent(branch_source))
                self.assertTrue(
                    any(isinstance(node, ast.While) for node in ast.walk(branch_tree))
                )
                self.assertIn("self.step(", branch_source)
                self.assertNotIn("execution_plan", branch_source)
                self.assertNotIn("execution_plan", inspect.getsource(owner.forward))

    def test_ordinary_forward_does_not_request_an_execution_plan(self):
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                model = config.build()
                state = LayerState(torch.ones(1, 2))
                with patch.object(
                    model.recurrent_iteration_schedule,
                    "execution_plan",
                    side_effect=AssertionError("ordinary recurrence requested a plan"),
                ):
                    self.assertIs(model(state), state)

    def test_variants_validate_before_running_shared_transition(self):
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                model = config.build()
                state = LayerState(torch.ones(1, 2))
                with patch.object(
                    model.step,
                    "forward",
                    wraps=model.step.forward,
                ) as run_transition:
                    self.assertIs(model(state), state)
                    self.assertGreater(run_transition.call_count, 0)
                    for call in run_transition.call_args_list:
                        self.assertTrue(
                            all(argument is not model for argument in call.args)
                        )
                        self.assertTrue(
                            all(
                                argument is not model
                                for argument in call.kwargs.values()
                            )
                        )
                    run_transition.reset_mock()
                    with self.assertRaises(ValueError):
                        model(LayerState(torch.ones(1, 3)))
                    run_transition.assert_not_called()

    def test_halting_floor_policy_does_not_live_in_the_recurrent_package(self) -> None:
        recurrent_package = (
            Path(__file__).parents[2]
            / "src"
            / "emperor"
            / "layers"
            / "_composition"
            / "recurrent"
        )
        prohibited_policy_names = (
            "min_steps",
            "is_update_eligible",
            "owner_step",
        )
        offenders = {}
        for path in sorted(recurrent_package.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            leaked_policy_names = tuple(
                policy_name
                for policy_name in prohibited_policy_names
                if policy_name in source
            )
            if leaked_policy_names:
                offenders[str(path.relative_to(recurrent_package))] = (
                    leaked_policy_names
                )

        self.assertEqual(offenders, {})

    def test_abstract_config_cannot_be_built(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "RecurrentCompositionConfig is abstract.*concrete recurrent config",
        ):
            RecurrentCompositionConfig().build()

    def test_standard_config_owns_the_canonical_standard_runtime(self) -> None:
        owner = RecurrentLayerConfig().registry_owner()

        self.assertIs(owner, RecurrentLayer)
        self.assertTrue(issubclass(owner, RecurrentCompositionAbstract))
        self.assertEqual(
            owner.__module__,
            "emperor.layers._composition.recurrent.variants.standard.core",
        )

    def test_variant_packages_export_named_classes_and_serialize(self):
        for variant, runtime_name, state_name in (
            ("standard", "RecurrentLayer", "StandardRecurrentState"),
            (
                "tiny_recursive_model",
                "TinyRecursiveModelRecurrent",
                "TinyRecursiveModelState",
            ),
            (
                "hierarchical_reasoning_model",
                "HierarchicalReasoningModelRecurrent",
                "HierarchicalReasoningModelState",
            ),
            ("inner_thinking", "InnerThinkingRecurrent", "InnerThinkingTokenSelection"),
        ):
            with self.subTest(variant=variant):
                package_name = (
                    f"emperor.layers._composition.recurrent.variants.{variant}"
                )
                package = importlib.import_module(package_name)
                core = importlib.import_module(f"{package_name}.core")
                state_module = importlib.import_module(f"{package_name}.state")
                runtime_class = getattr(core, runtime_name)
                self.assertEqual(runtime_class.__module__, f"{package_name}.core")
                self.assertEqual(package.__all__, (runtime_name, state_name))
                state_class = getattr(state_module, state_name)
                self.assertEqual(state_class.__module__, f"{package_name}.state")
                for owned_class in (runtime_class, state_class):
                    self.assertIs(getattr(package, owned_class.__name__), owned_class)
                    self.assertIs(pickle.loads(pickle.dumps(owned_class)), owned_class)
                    package_class_reference = (
                        f"c{package_name}\n{owned_class.__name__}\n."
                    ).encode()
                    self.assertIs(pickle.loads(package_class_reference), owned_class)

    def test_variant_forward_states_serialize_after_package_split(self):
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                model = config.build()
                state = model.step.initialize_recurrent_state(
                    LayerState(torch.tensor([[1.0, -2.0]]))
                )

                restored_state = pickle.loads(pickle.dumps(state))

                self.assertIs(type(restored_state), type(state))
                self.assertIs(
                    restored_state.fixed_input, restored_state.context_state.hidden
                )
                torch.testing.assert_close(
                    restored_state.output_hidden, state.output_hidden
                )
                self.assertEqual(
                    restored_state.transition_index, state.transition_index
                )

    def test_variants_build_keyword_only_recurrent_states(self) -> None:
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                layer_state = LayerState(torch.ones(1, 2))
                step = config.build().step
                recurrent_state = step.initialize_recurrent_state(layer_state)

                self.assertIsInstance(recurrent_state, RecurrentState)
                self.assertIs(recurrent_state.fixed_input, layer_state.hidden)
                self.assertIs(recurrent_state.context_state, layer_state)
                self.assertTrue(all(field.kw_only for field in fields(recurrent_state)))

    def test_all_recurrent_configs_own_the_shared_iteration_schedule(self) -> None:
        for config_type in (
            RecurrentLayerConfig,
            InnerThinkingRecurrentConfig,
            TinyRecursiveModelRecurrentConfig,
            HierarchicalReasoningModelRecurrentConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                config = config_type(
                    initial_iterations=2,
                    gradient_transition_count=3,
                    iteration_increment=4,
                    forward_calls_before_iteration_increment=5,
                    smooth_iteration_growth_flag=True,
                )

                self.assertEqual(config.initial_iterations, 2)
                self.assertEqual(config.gradient_transition_count, 3)
                self.assertEqual(config.iteration_increment, 4)
                self.assertEqual(config.forward_calls_before_iteration_increment, 5)
                self.assertTrue(config.smooth_iteration_growth_flag)

        self.assertIn(
            "smooth_iteration_growth_flag",
            RecurrentCompositionConfig.__annotations__,
        )
        self.assertNotIn(
            "smooth_iteration_growth_flag",
            RecurrentLayerConfig.__annotations__,
        )

        self.assertIs(RecurrentLayerConfig().registry_owner(), RecurrentLayer)

    def test_recurrent_config_module_contains_only_owner_backed_configs(self) -> None:
        config_module = importlib.import_module(
            "emperor.layers._composition.recurrent.config"
        )
        config_types = {
            value
            for value in vars(config_module).values()
            if isinstance(value, type)
            and issubclass(value, ConfigBase)
            and value.__module__ == config_module.__name__
        }

        self.assertEqual(
            config_types,
            {
                RecurrentCompositionConfig,
                RecurrentLayerConfig,
                InnerThinkingRecurrentConfig,
                TinyRecursiveModelRecurrentConfig,
                HierarchicalReasoningModelRecurrentConfig,
            },
        )
        for config_type in config_types - {RecurrentCompositionConfig}:
            with self.subTest(config_type=config_type.__name__):
                self.assertIsInstance(config_type().registry_owner(), type)

    def test_recurrent_validators_live_in_dedicated_validation_modules(self) -> None:
        cases = (
            (
                RecurrentLayerValidator,
                "emperor.layers._composition.recurrent.validation.standard",
            ),
            (
                TinyRecursiveModelRecurrentValidator,
                "emperor.layers._composition.recurrent.validation.tiny_recursive_model",
            ),
            (
                HierarchicalReasoningModelRecurrentValidator,
                "emperor.layers._composition.recurrent.validation."
                "hierarchical_reasoning_model",
            ),
            (
                RecurrentIterationScheduleValidator,
                "emperor.layers._composition.recurrent.validation.iteration_schedule",
            ),
        )

        for validator, expected_module in cases:
            with self.subTest(validator=validator.__name__):
                self.assertEqual(validator.__module__, expected_module)

    def test_step_registers_its_pipeline_and_transition_blocks_once(self):
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                model = config.build()
                self.assertIs(model.step, model.get_submodule("step"))
                self.assertNotIn("utilities", dict(model.named_children()))
                block_names = ("block_model",)
                if isinstance(config, HierarchicalReasoningModelRecurrentConfig):
                    block_names = ("high_model", "low_model")
                for name in ("pipeline", *block_names):
                    component = getattr(model.step, name)
                    self.assertIs(getattr(model, name), component)
                    component_paths = [
                        path
                        for path, module in model.named_modules(remove_duplicate=False)
                        if module is component
                    ]
                    self.assertEqual(component_paths, [f"step.{name}"])
                self.assertIs(model.step.cfg, model.cfg)
                self.assertIs(model.step.pipeline.config, model.step.cfg)
                self.assertFalse(hasattr(model, "execution"))

    def test_each_variant_runs_the_outer_pipeline_once_per_transition(self):
        for config in _minimal_variant_configs():
            with self.subTest(variant=type(config).__name__):
                model = config.build()
                pipeline = model.pipeline
                self.assertIsInstance(pipeline, LayerProcessingPipeline)
                transitions = model.recurrent_iteration_schedule.active_transition_count
                state = LayerState(torch.ones(1, 2))
                with (
                    patch.object(
                        pipeline,
                        "process_recurrent_transition",
                        wraps=pipeline.process_recurrent_transition,
                    ) as process_transition,
                    patch.object(
                        pipeline,
                        "process_normal_layer",
                        wraps=pipeline.process_normal_layer,
                    ) as process_layer,
                ):
                    self.assertIs(model(state), state)
                self.assertEqual(process_transition.call_count, transitions)
                process_layer.assert_not_called()

    def test_recurrent_controller_fields_belong_to_the_family_config(self) -> None:
        controller_field_names = {
            "recurrent_layer_norm_position",
            "gate_config",
            "residual_config",
            "halting_config",
            "memory_config",
        }

        self.assertTrue(
            controller_field_names
            <= {field.name for field in fields(RecurrentCompositionConfig)}
        )
        self.assertTrue(
            controller_field_names.isdisjoint(RecurrentLayerConfig.__annotations__)
        )

    def test_fixed_input_reinjection_control_belongs_only_to_standard_config(
        self,
    ) -> None:
        field_name = "reinject_original_hidden_flag"

        self.assertIn(
            field_name, {field.name for field in fields(RecurrentLayerConfig)}
        )
        for config_type in (
            RecurrentCompositionConfig,
            TinyRecursiveModelRecurrentConfig,
            HierarchicalReasoningModelRecurrentConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                self.assertNotIn(
                    field_name,
                    {field.name for field in fields(config_type)},
                )

    def test_minimum_steps_belongs_to_the_halting_contract(self) -> None:
        recurrent_field_names = {field.name for field in fields(RecurrentLayerConfig)}
        halting_field_names = [field.name for field in fields(HaltingConfig)]

        self.assertNotIn("min_steps", recurrent_field_names)
        self.assertGreater(
            halting_field_names.index("min_steps"),
            halting_field_names.index("halting_gate_config"),
        )

    def test_transition_seam_ignores_unconfigured_resources(self) -> None:
        config = RecurrentLayerConfig(block_config=None)

        config._map_transition_configs(
            lambda _transition: self.fail("None transition must not be mapped")
        )

        self.assertEqual(config._transition_config_items(), ())
        self.assertEqual(config._transition_configs(), ())
        self.assertEqual(config._missing_transition_config_fields(), ("block_config",))

    def test_only_subclass_extension_seams_are_protected(self) -> None:
        expected_protected_methods = {
            RecurrentCompositionAbstract: {
                "_new_recurrent_initial_buffer",
            },
            RecurrentCompositionConfig: {
                "_map_transition_configs",
                "_missing_transition_config_fields",
                "_registry_owner",
                "_transition_config_items",
                "_transition_configs",
            },
            RecurrentLayerConfig: {"_registry_owner"},
            TinyRecursiveModelRecurrentConfig: {"_registry_owner"},
            HierarchicalReasoningModelRecurrentConfig: {"_registry_owner"},
            RecurrentLayerValidator: {
                "_validate_integer_field",
                "_validate_residual_execution",
            },
            TinyRecursiveModelRecurrentValidator: set(),
            HierarchicalReasoningModelRecurrentValidator: set(),
            RecurrentIterationScheduleValidator: set(),
            RecurrentIterationSchedule: set(),
            RecurrentLayer: {"_build_step"},
            TinyRecursiveModelRecurrent: set(),
            HierarchicalReasoningModelRecurrent: set(),
        }

        for owner, expected_method_names in expected_protected_methods.items():
            with self.subTest(owner=owner.__name__):
                self.assertEqual(
                    _declared_protected_method_names(owner),
                    expected_method_names,
                )

    def test_old_private_runtime_and_validator_modules_are_retired(self) -> None:
        for module_name in (
            "emperor.layers._recurrent",
            "emperor.layers._validation.recurrent",
            "emperor.layers._composition.recurrent.variants.hrm",
            "emperor.layers._composition.recurrent.variants.trm",
            "emperor.layers._composition.recurrent.utilities",
        ):
            with (
                self.subTest(module_name=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
