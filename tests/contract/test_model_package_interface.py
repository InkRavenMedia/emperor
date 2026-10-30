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
        "1f2b5cf47f0295b8adec0e77268c37e9056b4488ea4c04b8c5f90429a0e122b0",
        381,
        9,
    ),
    "bert/linear_adaptive": (
        "ebcad295ac34aac17cbdc27a83add8793a62d452f0c213a4e6cfa4de04407c34",
        842,
        124,
    ),
    "bert/expert_linear": (
        "e650fdb4a6eb51ef6ce22c96ed9c6bb72a24a1a9d8770b65c1737dc56fcb8f34",
        508,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "55271fcafd892ca41483669fc97cea1ef17d3cf85f129557e65d0695bd19a5ee",
        925,
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
        "446a0ac75723b7a142ac28c48e16885fc738f57a87c0eae62849f75a50f8d230",
        376,
        8,
    ),
    "gpt/linear_adaptive": (
        "c46eac95ffe77c373e82d0c8fadb787b0ce9c9b2ae15115cf6c1ef1030f5cb87",
        837,
        123,
    ),
    "gpt/expert_linear": (
        "9345679104fda935c9557e7db58a0088420f69cd62bd37092799cc0f44a1ecec",
        503,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "f67cb3777d788cb47d816bc90ac91270daf568f63c5bd93ba32c4a940be532e2",
        927,
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
        "69960da95452f7662778ac9bf203faaaf2b62f5957c3c0615d953f3d38d3eb2d",
        384,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "6034e64875e87910014183000f34b6782227adbfc755c61d639ac8454db058ff",
        555,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "26ae7cadefeec436cd352f9accc6e1c593487fb1acdb771bba6ad8a927800f1b",
        509,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "0a16f0d8f7807e7290a789427909371c941cbd5680cd6cfecc7028afc0532233",
        680,
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
        "2f8dad1f47e175704c6d3193bbae513cd97286a3919a5f96495930c382170384",
        850,
        12,
    ),
    "transformer/linear_adaptive": (
        "27fd94abd3ce53b18b8735449cb6059895dc1ae2010d0269b19ce7cdf2ba9b72",
        3202,
        20,
    ),
    "transformer/expert_linear": (
        "04bc7bb57986fa724a0cc9b694e77709324292b957c06a6b8481377deb37a265",
        1000,
        10,
    ),
    "transformer/expert_linear_adaptive": (
        "b6181634107560ab9514af8e9cefbdd461d74930587809a224b87ee27d2086de",
        3576,
        30,
    ),
    "vit/linear": (
        "9db38aa697e91bb975ef2c6dd7c0453c081f04f4b17a8261dee2dddb3f2334b6",
        369,
        8,
    ),
    "vit/linear_adaptive": (
        "1bf68a60b4f82c95febd4bbf26ac828e2c1059a61af7964a4647eeaf9b851631",
        830,
        122,
    ),
    "vit/expert_linear": (
        "d9fc395853f8b7562b8bed1c583b2655814964a86cf5d5ee3c4bc043919b2aa4",
        496,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "2e15e3a301469d43721900549982a4f2df9806804b119840f92e68ed9f6ddbf8",
        914,
        38,
    ),
}


_ORDERING_WITH_ATTENTION_SETTINGS_BY_PACKAGE = {
    "bert/linear": (
        "c96a71e0370d8f8433585b06714847160df56a98f3bc12f6e52db80ba85d2e99",
        421,
        9,
    ),
    "bert/linear_adaptive": (
        "301cca1f3fa2ac591e6a856ca7f18fcb1122f79a824c61211032faf25c24afc1",
        938,
        124,
    ),
    "bert/expert_linear": (
        "8401975e139ed10aa84af4e7a40147f56bc39f0c011f6802544a19fd809475d7",
        562,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "8f3a1fd990303cc7a9be87fd49b8ec4e58f8079d45471e9933d73352f941f34e",
        1027,
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
        "ba9c6e28801e34323e14429732dff8fe4d41f894211320324e004ef4aa4cbd12",
        416,
        8,
    ),
    "gpt/linear_adaptive": (
        "6b5fcc468c6d6193cdaa7ee4cdac12731ee40550fef2ca50adad3ac6a7b813be",
        933,
        123,
    ),
    "gpt/expert_linear": (
        "4644d74d42d1a945388fd61461c4e2718b4a47dc0d2ebca9929278f9966d1a02",
        557,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "2f72b1c1b76e6a325cc32e021f70eed832280a153f0a6f014fe76a21d66ea330",
        1029,
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
        "6e047edb971dfcf2a8af66c15d0231b53ae5a69c93c1a8c17711eb30d88b86db",
        434,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "9ca06fb398613119dcf8b210a93104939cdde4cbc1c049901527318a48b4b661",
        625,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "43e03818d43aaa01c85ec38f91a00cc72e76fc43ae897e3504a296aebd9f229a",
        575,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "f9dd9f01f084bb83e3eef2bd68155f9f6235277a61a4e1817f50c405b9d4676a",
        766,
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
        "9cb83951af14c1ecd35c3448b83c45aba708273ab67a56abd8f07f0262c3570c",
        940,
        12,
    ),
    "transformer/linear_adaptive": (
        "420b2fcb5c571033a84753a3b5e64c9467cde1667dad1aa16f06bda5a81e03bd",
        3592,
        20,
    ),
    "transformer/expert_linear": (
        "de2d83ff8cfdb5c835c5b4c2a29749c8d13f6ef74362f291f4dd4dd9e39697ac",
        1114,
        10,
    ),
    "transformer/expert_linear_adaptive": (
        "b8cfc17e21c694e013698174974330719c0e88d54fa7ff342bd176fd7b4fd4fc",
        4010,
        30,
    ),
    "vit/linear": (
        "dc380c8e06a71e3d869094d63882df1602c057844624290a2d191cdf0f95b098",
        409,
        8,
    ),
    "vit/linear_adaptive": (
        "b2f300905d622264967429b59961dfe579dfa66afad7b3f38d8c9d45ab899d6a",
        926,
        122,
    ),
    "vit/expert_linear": (
        "77f3729efde71ff8f79c9d7c6c54a95c9475b6bdd0291bb76d881d4b8bf08be4",
        550,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "c3df7587a93005a437063c28950b84ebc24aaaa03191d3f668c7112e96153e9c",
        1016,
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
