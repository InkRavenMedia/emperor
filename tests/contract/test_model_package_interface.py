from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torch.nn import Module

from emperor.config import ModelConfig
from model_runtime.packages import ModelPackage
from model_runtime.runs import (
    ExperimentBase,
    PlanningBudget,
    RunRequest,
    SearchSpec,
    plan_runs,
)
from model_runtime.runs._handoff import require_run_experiment
from model_runtime.runs.artifacts import FilesystemRunArtifacts
from models.catalog import (
    MODEL_CATALOG,
    discover_model_ids,
    discover_model_packages,
    model_package,
)

_ORDERING_DIGEST_BY_PACKAGE = {
    "bert/linear": (
        "655c02799a5a8be562a9962f2b77376073e21054dadd805f4ed4c242c2b5329e",
        391,
        9,
    ),
    "bert/linear_adaptive": (
        "e32e2249062f14b8fdb93865a05abb36eba82170cd3c6156e36a1b34f31576ca",
        852,
        124,
    ),
    "bert/expert_linear": (
        "04f1a9b02c680032a5b5bdfae4aae2a57c613aa01d395666bc7deb94b5b916f7",
        531,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "19ba5431d901f0967e6267520722823f3d665a5f1b3f43799fb5f8f91d625d11",
        949,
        40,
    ),
    "experts/linear": (
        "7a63485a775991bde435a97fe7ace7ded5023c31a437edeba1e69ef71032ee65",
        287,
        7,
    ),
    "experts/linear_adaptive": (
        "e78713ab982fd82f450e8a0e833fc8f25af991d96db52019d11f1d766cbc3cf5",
        880,
        37,
    ),
    "gpt/linear": (
        "3ee162519034f20b9c71b5cb0367ce3372ab81a9c0a052a8cc977767a8f30eaa",
        387,
        8,
    ),
    "gpt/linear_adaptive": (
        "855b425dcc4f9c638dff0ea455c4a8d5eed24d57e62f40283830bca2a5b0953c",
        848,
        123,
    ),
    "gpt/expert_linear": (
        "c69ee0bf4d59a4643c2164c5f05f06c5c42d4ef8c83b2800d53aa937284ede0c",
        527,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "dc8bd440d888c3e410b84cb112b017e8940a85c28c9bfa0e8e8ecd2b4bbe5d08",
        952,
        39,
    ),
    "linears/linear": (
        "2093377908e48de0665bd7f4c63fb7b8ca03d3672d429fff85822baddea0d8fb",
        160,
        7,
    ),
    "linears/linear_adaptive": (
        "cf6bcbc299701d28e8d929876841fe48642fd07e9ddf11e62075dad530c09899",
        517,
        33,
    ),
    "mlp_mixer/linear": (
        "e7e7ff889887b2678f3b12c09efdb0613433213d10ad74ca1c0b3e50e64d39a0",
        411,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "c16cc0a4eedc4e8f03245afd60bad575f722b5cdd458b4fc0766a48d5a7f89aa",
        582,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "e716b060b706ce42ff425f8e197a5e780bdf00160f2a9f45f61f0cceae444ba0",
        537,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "235efc69eb8f68106de7cf9089fbb4f12e0e283834fb6e5cdd65dda221f48c92",
        708,
        12,
    ),
    "neuron/linear": (
        "6f98de2ee92999f4ef5907beacd91aae487a1e89ae350fdfeaa8c600f8b183f5",
        220,
        10,
    ),
    "neuron/linear_adaptive": (
        "583faece6a7290b937849ffd9a11ace0b5495954c461a2f40406290d5813f7c8",
        577,
        36,
    ),
    "neuron/expert_linear": (
        "ab457987d08c5f5a284c94eab7a0481916a60e8a1bdcb045ae5bf03a1de00af7",
        347,
        10,
    ),
    "neuron/expert_linear_adaptive": (
        "cbd39b8ba892d7900885c2088af942234c0ee61b1b0ba5820bbaf9f4ef452ac6",
        940,
        40,
    ),
    "parametric/parametric_generator": (
        "4b9e4fd713db817400c090d1fa07ced411cb574cf1582b288cb564e3c9ee7214",
        72,
        7,
    ),
    "parametric/parametric_matrix": (
        "fd56b87316fae8fa511c3cb3fbe4aedea7577382d81724472588ec37977ec7c8",
        66,
        7,
    ),
    "parametric/parametric_vector": (
        "4efd1de70bed3b66f5d45ab559b65adcaf3ddeb078c7ed84304a94b906c3fa97",
        67,
        6,
    ),
    "transformer/linear": (
        "18121c1a203a8cd76c1a198646f9d3ca47c2afceeeff51c0a60c81aa1c6c42cf",
        857,
        12,
    ),
    "transformer/linear_adaptive": (
        "e0a1d5d487a09820ab79aebb4e0b209989b11bab0c1fad5d67c5d1fc8fc62f2e",
        3209,
        20,
    ),
    "transformer/expert_linear": (
        "cc8ce55f6cb4cbbb93a44f17e9447a0f644c62b64c51c70187d0b2e7ac866b0f",
        1009,
        10,
    ),
    "transformer/expert_linear_adaptive": (
        "b047c6aea19f6895ec1bd337bc1fcbeef8855eeb98ddcb8574d1f2f98697490d",
        3585,
        30,
    ),
    "vit/linear": (
        "a948bb5470c490aa5b1d90fea5f3c3a823a18335c0f366f97782c4ecc8fd67ec",
        380,
        8,
    ),
    "vit/linear_adaptive": (
        "0b56e961af320d459917c5d1b5050ffe0074489140edd8b05656c6015b358dcb",
        841,
        122,
    ),
    "vit/expert_linear": (
        "f9fa829e495b4fcce3de73245903c7a1f85f1c7be3c72b074315da2f3e11aa97",
        520,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "18b32e3d232cdd4956ca8ecdf3b90fbe06c5b325100ff09aa62e51159a900fba",
        943,
        38,
    ),
}


_ORDERING_WITH_ATTENTION_SETTINGS_BY_PACKAGE = {
    "bert/linear": (
        "13babeab0263dd42e5eb9e9e29c88f522ce532e66480811dca203e6d25a791a4",
        431,
        9,
    ),
    "bert/linear_adaptive": (
        "d4901da81f712bf94ee2e5feb4036a9139963bb37170300d24f7f7e440e2354e",
        948,
        124,
    ),
    "bert/expert_linear": (
        "8ab75f3f0176a2701049a552af1190b67707309664654fe2218331d822a6b66a",
        585,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "762213fd6c1ea05b961710c91a441a6ae9a53ac452a94c5b96f4c93749af4643",
        1051,
        40,
    ),
    "experts/linear": (
        "1e79a6f80396d052246c1ff607971a96348d12cb0b4cc410d7adcd5fa569d31d",
        317,
        7,
    ),
    "experts/linear_adaptive": (
        "b8d955263821015c472b27c3e4c676e20cf9a5d53ae03e59c690219878f58a27",
        978,
        37,
    ),
    "gpt/linear": (
        "f8b248f9d6a0fbc267c6fd8da9d2b77760d1954124305b14a0e806c3ea6efc4e",
        427,
        8,
    ),
    "gpt/linear_adaptive": (
        "2e89117b99690c6d4abe0428740b45c87183c6c0b5cb24255f6fa5ed678d94c5",
        944,
        123,
    ),
    "gpt/expert_linear": (
        "a833b0489f5c33b4435750c12a0b4fea25044c43ea63ccb1eb1dfa0937ad242b",
        581,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "09d163539a333373e25b7c8025a9922ecb52558c378f66d664686b26c71598d7",
        1054,
        39,
    ),
    "linears/linear": (
        "c525e3226785d9b331406a013ec97b4fd09cc93a53b74e3b464d2bf9640618ff",
        176,
        7,
    ),
    "linears/linear_adaptive": (
        "9997be086f46c9e63555a52592f6c179b2a5591f58ad6cf33662d8925d64f295",
        573,
        33,
    ),
    "mlp_mixer/linear": (
        "41f2af887d9de28d133bbb126dbd0f173c57294678da282478fafe4491c0c621",
        461,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "1694fbc3777e64e04ffb81cc8e5c4e49a04623d04229fb6cf2d19407e557cdbc",
        652,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "7b8647ca676c79e41cc21da3b55c40e04d8af2baa23393fdbf61bd58d647f186",
        603,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "f748eaffe59128931975389756e955a8830f20d630f9e93770ad33ca514eba4d",
        794,
        12,
    ),
    "neuron/linear": (
        "7175e838ed89e0aef4980421b966b0a28745f609223d709995200d3a8a0f37a6",
        240,
        10,
    ),
    "neuron/linear_adaptive": (
        "09d76dd6c31179fb5bd7f6f15d8c736d4d809b495a30064ec78385368b7f9a4d",
        637,
        36,
    ),
    "neuron/expert_linear": (
        "f899de69833246528654dceb16161a5a1399c2499b30177a5b3f33929984f766",
        381,
        10,
    ),
    "neuron/expert_linear_adaptive": (
        "304078b24337cb79c89a284a54c127b8175d3ed31f4cfaed56a7fee807fc1002",
        1042,
        40,
    ),
    "parametric/parametric_generator": (
        "1b4c23232c677788252414093fe289736c5e63c3ebd7a6f81d9a76bf9aae387b",
        76,
        7,
    ),
    "parametric/parametric_matrix": (
        "707167f1705db3bc0c7992a2676f479be3410502debbc7acb7076ff9e83f0c5e",
        70,
        7,
    ),
    "parametric/parametric_vector": (
        "5eee857784ed8b5aae861e3d8b397f8cf21339182637c03316f0bbba23a7fe32",
        71,
        6,
    ),
    "transformer/linear": (
        "f8035d9b52a3389682321500034831d62b8e81f8dcde2c4509a6ff31519aefc2",
        947,
        12,
    ),
    "transformer/linear_adaptive": (
        "b37ffec3fdab41967f5ee2df98319d6bdecadbf47a9292dfd557f053d5499c42",
        3599,
        20,
    ),
    "transformer/expert_linear": (
        "2385c83e3cfb407b880334225de750527eb6c7a76f098d62c6865159c97fe03e",
        1123,
        10,
    ),
    "transformer/expert_linear_adaptive": (
        "b9d11a64c2f685434b0bd720ca15656c4c3c9f545bb2ff30c3cbec14d0cfbe51",
        4019,
        30,
    ),
    "vit/linear": (
        "c0ff9ed8dedf17cbb239f37522c73f26476185dc5b4f3dd235ac8d407fcdbee3",
        420,
        8,
    ),
    "vit/linear_adaptive": (
        "9467fcda0afeab300fd28e5cd24fa4accd2c12af7d0ec98b2d59d9b32c62d9bb",
        937,
        122,
    ),
    "vit/expert_linear": (
        "28b1ca9bf5d8831e1b071e53ebfefa61c6f9d56eee407669a6dbcd64177c5a14",
        574,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "46ae873ba2733c9351802d9a8f8d8149972b9aaeb12af51f77d9acbfa0d87c96",
        1047,
        38,
    ),
}


class TestModelPackageInterface(unittest.TestCase):
    def test_every_catalog_package_preserves_runtime_parameter_and_axis_order(self):
        actual = {}
        actual_with_attention_settings = {}
        for package in discover_model_packages():
            runtime_defaults = package.runtime_defaults_spec
            parameter_keys = list(runtime_defaults.ordered_configuration_keys())
            plan = plan_runs(
                package,
                RunRequest(
                    presets=(package.preset_name(package.default_preset),),
                    datasets=(package.resolve_dataset(None).__name__,),
                    search=SearchSpec(mode="random", random_samples=1),
                ),
                random_source=random.Random(0),
                budget=PlanningBudget.unlimited(),
            )
            search = plan.search
            assert search is not None
            axis_keys = [axis.key for axis in search.axes or ()]
            existing_parameter_keys = [
                key
                for key in parameter_keys
                if not key.endswith(
                    ("RESIDUAL_BLOCK_SIZE", "RESIDUAL_RMS_NORM_EPSILON")
                )
            ]
            for keys, signatures in (
                (existing_parameter_keys, actual),
                (parameter_keys, actual_with_attention_settings),
            ):
                payload = json.dumps(
                    {"parameters": keys, "axes": axis_keys},
                    separators=(",", ":"),
                )
                signatures[package.catalog_key] = (
                    hashlib.sha256(payload.encode()).hexdigest(),
                    len(keys),
                    len(axis_keys),
                )

        self.assertEqual(actual, _ORDERING_DIGEST_BY_PACKAGE)
        self.assertEqual(
            actual_with_attention_settings,
            _ORDERING_WITH_ATTENTION_SETTINGS_BY_PACKAGE,
        )

    def test_runtime_defaults_interpretation_is_authoritative_and_cached(self):
        package = model_package("linears/linear")
        assert package is not None

        first = package.runtime_defaults_spec
        second = package.runtime_defaults_spec
        supported_keys = set(first.supported_keys)

        self.assertIs(first, second)
        self.assertEqual(set(first.supported_keys), supported_keys)
        default_keys = tuple(key for key, _value in first.default_items())
        self.assertEqual(set(default_keys), supported_keys)
        self.assertEqual(len(default_keys), len(supported_keys))
        self.assertEqual(first.resolve_key("hidden-dim"), "HIDDEN_DIM")
        self.assertEqual(first.resolve_key("hidden_dim"), "HIDDEN_DIM")
        self.assertEqual(
            first.parse_overrides({"hidden-dim": "64"}),
            {"hidden_dim": 64},
        )
        self.assertEqual(
            first.serialize_overrides({"hidden-dim": "64"}),
            {"HIDDEN_DIM": 64},
        )
        self.assertEqual(
            tuple(first.search_values),
            tuple(package.search_metadata),
        )
        self.assertTrue(first.preset_locks("gating"))

        curated_package = model_package("gpt/expert_linear")
        assert curated_package is not None
        curated = curated_package.runtime_defaults_spec
        curated_supported_keys = set(curated.supported_keys)
        self.assertNotIn("HALTING_OUTPUT_DIM", curated_supported_keys)
        self.assertTrue(curated.skipped_schema_keys)
        self.assertTrue(curated.skipped_schema_keys <= curated_supported_keys)

    def test_model_metadata_sources_are_not_part_of_the_package_interface(self):
        package = model_package("linears/linear")
        assert package is not None

        self.assertFalse(hasattr(package, "runtime_defaults"))
        for source_name in (
            "runtime_defaults",
            "dataset_options",
            "monitor_options_source",
            "search_space",
        ):
            with self.subTest(source=source_name):
                self.assertFalse(hasattr(package.metadata, source_name))

    def test_every_catalog_package_builds_the_typed_run_experiment_port(self):
        with TemporaryDirectory() as temporary_directory:
            artifacts = FilesystemRunArtifacts(root=Path(temporary_directory))
            for package in discover_model_packages():
                with self.subTest(model_package=package.catalog_key):
                    experiment = package.build_experiment(
                        package.default_preset,
                        experiment_task=package.default_experiment_task,
                        run_artifacts=artifacts,
                    )
                    self.assertIs(
                        require_run_experiment(experiment, package.catalog_key),
                        experiment,
                    )
                    self.assertEqual(
                        experiment.num_epochs,
                        package.runtime_defaults_spec.current_value_or(
                            "NUM_EPOCHS",
                            10,
                        ),
                    )

    def test_every_catalog_package_preserves_its_public_experiment_identity(self):
        experiment_types: set[type[ExperimentBase]] = set()
        with TemporaryDirectory() as temporary_directory:
            artifacts = FilesystemRunArtifacts(root=Path(temporary_directory))
            for package in discover_model_packages():
                with self.subTest(model_package=package.catalog_key):
                    experiment = package.build_experiment(
                        package.default_preset,
                        experiment_task=package.default_experiment_task,
                        run_artifacts=artifacts,
                    )
                    experiment_type = type(experiment)
                    expected_module = (
                        f"models.{package.identity.model_type}."
                        f"{package.identity.model}.presets"
                    )

                    self.assertIsNot(experiment_type, ExperimentBase)
                    self.assertTrue(issubclass(experiment_type, ExperimentBase))
                    self.assertEqual(experiment_type.__name__, "Experiment")
                    self.assertEqual(experiment_type.__module__, expected_module)
                    self.assertEqual(
                        tuple(inspect.signature(experiment_type).parameters),
                        (
                            "experiment_preset",
                            "experiment_task",
                            "model_package",
                            "run_artifacts",
                        ),
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(experiment_type)),
                        experiment_type,
                    )
                    experiment_types.add(experiment_type)

        self.assertEqual(len(experiment_types), len(MODEL_CATALOG))

    def test_every_catalog_entry_uses_the_same_model_package_interface(self):
        packages = discover_model_packages()

        self.assertEqual(len(packages), len(MODEL_CATALOG))
        self.assertEqual(
            [package.catalog_key for package in packages],
            discover_model_ids(),
        )

        for package in packages:
            with self.subTest(model_package=package.catalog_key):
                package_module = (
                    f"models.{package.identity.model_type}.{package.identity.model}"
                )
                self.assertIsInstance(package, ModelPackage)
                self.assertIs(model_package(package.catalog_key), package)
                self.assertEqual(
                    package.identity.to_payload(),
                    {
                        "modelType": package.identity.model_type,
                        "model": package.identity.model,
                    },
                )
                self.assertIs(package.runtime_defaults_spec.package, package)
                runtime = package.bind_runtime_defaults()
                self.assertIs(type(runtime), package.runtime_options_type)
                self.assertEqual(
                    type(runtime).__module__,
                    f"{package_module}.runtime_options",
                )
                self.assertIn(
                    package.default_experiment_task,
                    package.dataset_metadata,
                )
                self.assertIs(package.metadata.identity, package.identity)
                self.assertIsInstance(package.monitor_metadata, list)
                self.assertIsInstance(package.search_metadata, dict)
                self.assertTrue(package.preset_type)
                self.assertEqual(
                    package.preset_type.__module__,
                    f"{package_module}.presets",
                )
                self.assertEqual(
                    type(package.presets).__module__,
                    f"{package_module}.presets",
                )
                selected_preset = next(iter(package.preset_type))
                self.assertIsInstance(package.preset_locks(selected_preset), dict)

                configuration = package.build_configuration()
                self.assertIsInstance(configuration, ModelConfig)
                model = package.build_model(configuration)
                self.assertIsInstance(model, Module)

    def test_checkpoint_reconstruction_is_an_explicit_optional_capability(self):
        tensor_shapes = {
            "input_model.model.weight_params": (784, 12),
            "main_model.layers.0.model.weight_params": (12, 12),
            "output_model.model.weight_params": (12, 10),
        }
        expected = {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 12,
            "stack_num_layers": 1,
        }

        for package in discover_model_packages():
            with self.subTest(model_package=package.catalog_key):
                overrides = package.checkpoint_config_overrides(tensor_shapes)
                if package.catalog_key == "linears/linear":
                    self.assertEqual(overrides, expected)
                else:
                    self.assertEqual(overrides, {})


if __name__ == "__main__":
    unittest.main()
