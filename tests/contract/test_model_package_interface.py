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
        "f85fb99a232cfe8f04a21e81b9d92158943a75b83d2062d26038c2a85fd79798",
        384,
        9,
    ),
    "bert/linear_adaptive": (
        "1d6bdb24e102336cf0feeb42086d18c8640cd0e4fd9bcebebabab6391d662f91",
        845,
        124,
    ),
    "bert/expert_linear": (
        "29d70fa0875098043b95e23130f99769a827765d014ed5847284267d3153cabc",
        512,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "771d03ba6cb45a3846001af04fcaf0abcd08df840522d7774061b849de461319",
        930,
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
        "47806b73709f898ea12cdbc50671447b8bbd79e7c67876d1cb19828d6dd1e873",
        379,
        8,
    ),
    "gpt/linear_adaptive": (
        "b45ad361aa440bc980f249acd65ccc22c7bf2580584449e91010d80bb91cd5cf",
        840,
        123,
    ),
    "gpt/expert_linear": (
        "3913eb5e8e27e18d83510bb054313aaf52ca6c7ee87367a2c6d7ce86de481f41",
        507,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "b7cfef8161a4d8cd01220e233051b779a3a5bc8f4a86a4fcca74bbaf81d22c4a",
        932,
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
        "790235750a90283510820b217d217e6af90cab2f7c6c635bd7f74900077d337e",
        387,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "f2ace8b7920745b7fa69ffc7520301e4d7b60cbe72667613aa1f0fb816e35c0d",
        558,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "01508148de34e024c08b01b6ff13239989bacac01cef168b480dcd85f373536f",
        513,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "286510429316cd053ecab422f5e79ef76fbeeb2ecb2485030d2bf32f655f09c4",
        684,
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
        "77d579e3e85e24f203c9ecfb14d0b61933ec40b4f56d59647d22f2fc18805b36",
        372,
        8,
    ),
    "vit/linear_adaptive": (
        "27249034213e0651bac8a3639f5313704bf621ff8994caebfff6843c666f8160",
        833,
        122,
    ),
    "vit/expert_linear": (
        "93e7a19f2510a203a11c0b0c94d3d3b0e848f607afb18154a6100ed000ed27e3",
        500,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "e46c550b22bd8b851cc9c6b9e822ade4dc7c6fec5d0344bc75ab390437f1f7e5",
        919,
        38,
    ),
}


_ORDERING_WITH_ATTENTION_SETTINGS_BY_PACKAGE = {
    "bert/linear": (
        "b2ae23648e0b6df0126eb575e611e5856c9265dcb5b74302ae4b0f095f1f9466",
        424,
        9,
    ),
    "bert/linear_adaptive": (
        "6d47ffb194b07d89347712ce2eac01858a7ebb4497a5ce5135c52f0c445525cd",
        941,
        124,
    ),
    "bert/expert_linear": (
        "f03a95af59b4e2f04b5bbf92ecc192d046f26d606c6355b9ba790a24502772e3",
        566,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "8cb239601407be03e90826b88180634ec66e027df6afefe17e74f9fa6c6ae957",
        1032,
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
        "13c637df0202b7669d243ac696aca4ca75a2dab5b9f24b0840a252c801800e52",
        419,
        8,
    ),
    "gpt/linear_adaptive": (
        "e26b404559098e176da19f303817f232c3fae6276ed80dda029db4130d7c8641",
        936,
        123,
    ),
    "gpt/expert_linear": (
        "b89c466523a5633f5b40c822d46384d2a8eb775ad11c6cc5a4bbd4fc601e4dda",
        561,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "bf2557785bd3464d0685ddb8cd99f00a2f6d1b26411b511a44c21bf40220f8e4",
        1034,
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
        "f565e447c6c27835ebf3d763ad060175006e8782c4d549a556e4e5b793b37fd3",
        437,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "0f9a42b7ab2fa56f73eebddb3d6c2c31ecb34a57988ecf65f6a0659e73719116",
        628,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "33f4cee7c616ade11eb9b088d365f0e58dab95134e05d63ca452bc4e5a0ab557",
        579,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "83ff16c395e1d83f2185dcd4fcf3d7a95ccfe9eb3b812045d105c61a3d9be695",
        770,
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
        "2fa929b1cd4c2adf59d280e93452eb6187e5164cb70a0d3f83269379e93e4d76",
        412,
        8,
    ),
    "vit/linear_adaptive": (
        "02942b2318eb5c72ff6a04beb4411737b5aee9db408699d9be3169ddcd4a8147",
        929,
        122,
    ),
    "vit/expert_linear": (
        "307d62ea7e588c0b887f29723789bf5c568e6b16c30df1be32a09049da9e38b1",
        554,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "563e8c7aedb51afbfc0ab71b70fbb1cac1f3990a83495dc6bd0975e1139ec8f1",
        1021,
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
