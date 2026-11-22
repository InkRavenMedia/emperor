from __future__ import annotations

import inspect
import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Annotated, Any, cast
from unittest.mock import patch

from model_runtime.inspection import (
    InspectionError,
    configuration_schema,
    search_space_schema,
)
from model_runtime.monitoring_catalog import monitors_for_model
from model_runtime.packages import (
    ModelMetadata,
    ModelPackage,
    RuntimeDefaultsError,
    configuration_field_metadata,
)
from model_runtime.runs import PlanningBudget, RunRequest, SearchSpec, plan_runs
from models.catalog import discover_model_packages, model_package
from models.linears.linear import dataset_options as linears_linear_datasets


class _MetadataAdapter:
    def __init__(self, package: ModelPackage, metadata: ModelMetadata) -> None:
        self._package = package
        self._metadata = metadata

    def load_metadata(self) -> ModelMetadata:
        return self._metadata

    def load_runtime_options_type(self) -> type[Any]:
        return self._package.runtime_options_type

    def bind_runtime_defaults(self, values: Mapping[str, object] | None) -> Any:
        return self._package.bind_runtime_defaults(values)

    def load_preset_type(self) -> type[Any]:
        return self._package.preset_type

    def load_presets(self) -> Any:
        return self._package.presets

    def build_configuration(
        self,
        presets: Any,
        preset: Any,
        dataset: type[Any],
        **kwargs: Any,
    ) -> Any:
        del presets
        return self._package.build_configuration(preset, dataset, **kwargs)

    def build_model(self, configuration: Any) -> Any:
        return self._package.build_model(configuration)

    def build_experiment(
        self,
        preset: Any,
        *,
        experiment_task: Any,
        model_package: ModelPackage,
        run_artifacts: Any,
    ) -> Any:
        del model_package
        return self._package.build_experiment(
            preset,
            experiment_task=experiment_task,
            run_artifacts=run_artifacts,
        )


def _linears_linear() -> ModelPackage:
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


def _plain_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {key: dict(entry) for key, entry in metadata.items()}


def _source_less_package() -> ModelPackage:
    from model_runtime.packages import RuntimeDefaultsSection

    base = _linears_linear()
    config = ModuleType("synthetic.source_less.config")
    config.ALPHA = 1
    config.BETA = 2
    config.__annotations__ = {"ALPHA": int, "BETA": int}
    search = ModuleType("synthetic.source_less.search_space")
    search.SEARCH_SPACE_BETA = [20, 21]
    search.SEARCH_SPACE_ALPHA = [10, 11]
    search.__annotations__ = {
        "SEARCH_SPACE_ALPHA": list[int],
        "SEARCH_SPACE_BETA": list[int],
    }
    metadata = ModelMetadata(
        base.identity,
        config,
        linears_linear_datasets,
        monitors_for_model,
        search,
        configuration_metadata_sections=(
            RuntimeDefaultsSection(
                ("Second",),
                (("BETA", 20, (20,)),),
            ),
            RuntimeDefaultsSection(
                ("First",),
                (("ALPHA", 10, (10,)),),
            ),
        ),
        search_metadata_sections=(
            RuntimeDefaultsSection((), (("SEARCH_SPACE_BETA", 20, (20,)),)),
            RuntimeDefaultsSection((), (("SEARCH_SPACE_ALPHA", 10, (10,)),)),
        ),
    )
    return ModelPackage(base.identity, _MetadataAdapter(base, metadata))


def _package_with_declarations(
    configuration_sections: Any,
    search_sections: Any,
    *,
    config_members: Mapping[str, Any] | None = None,
    search_members: Mapping[str, Any] | None = None,
) -> ModelPackage:
    base = _linears_linear()
    config = ModuleType("synthetic.validation.config")
    for key, value in (config_members or {"VALUE": 1}).items():
        setattr(config, key, value)
    search = ModuleType("synthetic.validation.search_space")
    for key, value in (search_members or {"SEARCH_SPACE_VALUE": [1]}).items():
        setattr(search, key, value)
    metadata = ModelMetadata(
        base.identity,
        config,
        linears_linear_datasets,
        monitors_for_model,
        search,
        configuration_metadata_sections=configuration_sections,
        search_metadata_sections=search_sections,
    )
    return ModelPackage(base.identity, _MetadataAdapter(base, metadata))


class RuntimeDefaultsExplicitMetadataTests(unittest.TestCase):
    def test_source_less_metadata_drives_schema_and_run_planning(self) -> None:
        package = _source_less_package()

        with patch(
            "model_runtime.packages.configuration_metadata.configuration_field_metadata",
            side_effect=AssertionError("source parser must not run"),
        ) as source_parser:
            spec = package.runtime_defaults_spec
            configuration = configuration_schema(package)
            search = search_space_schema(package)
            override_plan = plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    overrides={"beta": 8, "alpha": 9},
                ),
            )
            search_plan = plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    search=SearchSpec(mode="grid"),
                ),
                budget=PlanningBudget.unlimited(),
            )

        source_parser.assert_not_called()
        self.assertEqual(tuple(spec.configuration_metadata), ("BETA", "ALPHA"))
        self.assertEqual(
            _plain_metadata(spec.configuration_metadata),
            {
                "BETA": {
                    "line": 20,
                    "sortKey": [20],
                    "section": "Second",
                    "sectionPath": ["Second"],
                },
                "ALPHA": {
                    "line": 10,
                    "sortKey": [10],
                    "section": "First",
                    "sectionPath": ["First"],
                },
            },
        )
        self.assertEqual(
            [(field.key, field.section_path) for field in configuration.fields],
            [("ALPHA", ("First",)), ("BETA", ("Second",))],
        )
        self.assertEqual(
            [axis.search_key for axis in search.axes],
            ["SEARCH_SPACE_ALPHA", "SEARCH_SPACE_BETA"],
        )
        self.assertEqual(
            [parameter.key for parameter in override_plan.runs[0].parameters],
            ["ALPHA", "BETA"],
        )
        self.assertEqual(
            [parameter.key for parameter in search_plan.runs[0].parameters],
            ["ALPHA", "BETA"],
        )

    def test_every_catalog_declaration_matches_the_authoring_source(self) -> None:
        for catalog_package in discover_model_packages():
            with self.subTest(package=catalog_package.catalog_key):
                metadata = catalog_package._adapter.load_metadata()
                source_configuration = configuration_field_metadata(
                    metadata._runtime_defaults_source
                )
                source_search = configuration_field_metadata(
                    metadata._search_space_source,
                    include_search_space=True,
                )
                package = ModelPackage(
                    catalog_package.identity,
                    _MetadataAdapter(catalog_package, metadata),
                    catalog_package.inspection_construction_limits,
                )
                with patch(
                    "model_runtime.packages.configuration_metadata.configuration_field_metadata",
                    side_effect=AssertionError("catalog parser must not run"),
                ):
                    spec = package.runtime_defaults_spec

                def ordered_sections(entries, order):
                    return [(key, entries[key].get("sectionPath", [])) for key in order]

                self.assertIsNone(metadata._configuration_metadata_sections)
                self.assertIsNone(metadata._search_metadata_sections)
                self.assertEqual(
                    ordered_sections(
                        spec.configuration_metadata,
                        sorted(
                            spec.configuration_metadata,
                            key=lambda key: spec.configuration_metadata[key]["sortKey"],
                        ),
                    ),
                    ordered_sections(
                        source_configuration,
                        sorted(
                            source_configuration,
                            key=lambda key: source_configuration[key]["sortKey"],
                        ),
                    ),
                )
                self.assertEqual(
                    ordered_sections(
                        spec.search_metadata,
                        [key for key, _ in spec.ordered_search_items()],
                    ),
                    ordered_sections(
                        source_search,
                        sorted(
                            source_search, key=lambda key: source_search[key]["line"]
                        ),
                    ),
                )

    def test_annotated_linear_fields_keep_source_locations_in_authoring_only(
        self,
    ) -> None:
        from models.linears.linear import config

        runtime = _linears_linear().runtime_defaults_spec.configuration_metadata
        with patch.object(
            Path, "read_text", side_effect=AssertionError("unexpected source read")
        ):
            fresh = ModelPackage(_linears_linear().identity, _linears_linear()._adapter)
            self.assertEqual(
                fresh.runtime_defaults_spec.configuration_metadata["BATCH_SIZE"],
                runtime["BATCH_SIZE"],
            )
        self.assertNotIn("line", runtime["BATCH_SIZE"])
        source = configuration_field_metadata(config)
        source_lines = Path(config.__file__).read_text().splitlines()
        self.assertTrue(
            source_lines[source["BATCH_SIZE"]["line"] - 1].startswith("BATCH_SIZE:")
        )

    def test_annotated_owner_places_generated_bert_controls_in_catalog_order(
        self,
    ) -> None:
        from model_runtime.packages.configuration_metadata import (
            Section,
            annotated_configuration_metadata,
        )
        from model_runtime.packages.soft_moe import declare_soft_moe_runtime_defaults
        from models.bert.linear import config as bert

        section = Section(("Feed-Forward Stack Options",))
        config = ModuleType("test_annotated_bert")
        config.FF_NUM_LAYERS = bert.FF_NUM_LAYERS
        config.SOFT_MOE_CONFIG = bert.SOFT_MOE_CONFIG
        config.FF_BIAS_FLAG = bert.FF_BIAS_FLAG
        config.__annotations__ = {
            "FF_NUM_LAYERS": Annotated[int, section],
            "SOFT_MOE_CONFIG": Annotated[bert.SoftMoEConfig | None, section],
            "FF_BIAS_FLAG": Annotated[bool, section],
        }
        declare_soft_moe_runtime_defaults(vars(config))
        actual = annotated_configuration_metadata(config)
        package = model_package("bert/linear")
        assert package is not None
        reference = package.runtime_defaults_spec.configuration_metadata
        keys = [
            "FF_NUM_LAYERS",
            "SOFT_MOE_CONFIG",
            *config._CONFIG_FIELD_METADATA_ALIASES,
            "FF_BIAS_FLAG",
        ]
        self.assertEqual(sorted(actual, key=lambda key: actual[key]["sortKey"]), keys)
        self.assertEqual(sorted(keys, key=lambda key: reference[key]["sortKey"]), keys)
        for key in keys:
            self.assertEqual(actual[key]["sectionPath"], reference[key]["sectionPath"])

    def test_legacy_import_keeps_the_annotated_fields_section_and_order(self) -> None:
        config = ModuleType("legacy_import_fixture")
        with TemporaryDirectory() as directory:
            path = Path(directory, "config.py")
            path.write_text(
                "from models.linears.linear.config import BATCH_SIZE\n"
                "# Custom\nWIDTH: int = 10\n",
                encoding="utf-8",
            )
            config.__file__ = str(path)
            metadata = configuration_field_metadata(config)
        self.assertEqual(metadata["BATCH_SIZE"]["sectionPath"], ["Global"])
        self.assertEqual(metadata["WIDTH"]["sectionPath"], ["Custom"])
        self.assertLess(metadata["BATCH_SIZE"]["sortKey"], metadata["WIDTH"]["sortKey"])

    def test_explicit_declarations_are_authoritative(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        base = _linears_linear()
        config = ModuleType("synthetic.authoritative.config")
        config.VALUE = 1
        search = ModuleType("synthetic.authoritative.search_space")
        search.SEARCH_SPACE_VALUE = [1]
        common = (
            base.identity,
            config,
            linears_linear_datasets,
            monitors_for_model,
            search,
        )
        mixed = ModelMetadata(
            *common,
            configuration_metadata_sections=(
                RuntimeDefaultsSection(("General",), (("VALUE", 1, (1,)),)),
            ),
        )
        package = ModelPackage(base.identity, _MetadataAdapter(base, mixed))
        self.assertEqual(
            package.runtime_defaults_spec.ordered_search_items(),
            (("SEARCH_SPACE_VALUE", (1,)),),
        )

        metadata = ModelMetadata(
            *common,
            configuration_metadata_sections=(),
            search_metadata_sections=(),
        )
        package = ModelPackage(base.identity, _MetadataAdapter(base, metadata))
        with patch(
            "model_runtime.packages.configuration_metadata.configuration_field_metadata",
            side_effect=AssertionError("authoritative declarations cannot fallback"),
        ) as source_parser:
            with self.assertRaisesRegex(RuntimeDefaultsError, "missing.*VALUE"):
                _ = package.runtime_defaults_spec
        source_parser.assert_not_called()

        package = ModelPackage(base.identity, _MetadataAdapter(base, metadata))
        with self.assertRaises(InspectionError) as caught:
            configuration_schema(package)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_inferred_search_axis_requires_a_list_of_values(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        config = RuntimeDefaultsSection(("General",), (("VALUE", 1, (1,)),))
        for members in (
            {"SEARCH_SPACE_VALUE": 31},
            {"__annotations__": {"SEARCH_SPACE_VALUE": list[int]}},
        ):
            with self.subTest(members=members):
                package = _package_with_declarations(
                    (config,), None, search_members=members
                )
                with self.assertRaisesRegex(
                    RuntimeDefaultsError, "without values: SEARCH_SPACE_VALUE"
                ):
                    _ = package.runtime_defaults_spec

    def test_declaration_inputs_and_projections_are_defensive(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        path = ["Original"]
        sort_key = [7, 0]
        fields = [("VALUE", 7, sort_key)]
        section = RuntimeDefaultsSection(path, fields)
        path.append("Mutated")
        sort_key.append(999)
        fields.append(("OTHER", 8, [8]))

        self.assertEqual(section.path, ("Original",))
        self.assertEqual(section.fields, (("VALUE", 7, (7, 0)),))
        with self.assertRaises(FrozenInstanceError):
            cast(Any, section).path = ()

        config_section = [section]
        search_section = [
            RuntimeDefaultsSection((), (("SEARCH_SPACE_VALUE", 8, (8,)),))
        ]
        base = _linears_linear()
        config = ModuleType("synthetic.snapshot.config")
        config.VALUE = 1
        search = ModuleType("synthetic.snapshot.search_space")
        search.SEARCH_SPACE_VALUE = [1]
        metadata = ModelMetadata(
            base.identity,
            config,
            linears_linear_datasets,
            monitors_for_model,
            search,
            configuration_metadata_sections=config_section,
            search_metadata_sections=search_section,
        )
        config_section.clear()
        search_section.clear()
        package = ModelPackage(base.identity, _MetadataAdapter(base, metadata))
        spec = package.runtime_defaults_spec

        first = spec.configuration_metadata["VALUE"]
        cast(list[int], first["sortKey"]).append(999)
        cast(list[str], first["sectionPath"]).append("Mutated")
        second = spec.configuration_metadata["VALUE"]
        self.assertEqual(second["sortKey"], [7, 0])
        self.assertEqual(second["sectionPath"], ["Original"])

    def test_section_declaration_rejects_malformed_values(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        constructor = cast(Any, RuntimeDefaultsSection)
        cases = (
            (("General", (("VALUE", 1, (1,)),)), "path must be a sequence"),
            ((("",), (("VALUE", 1, (1,)),)), "path items"),
            (((" General",), (("VALUE", 1, (1,)),)), "path items"),
            ((("General",), "VALUE"), "fields must be a sequence"),
            ((("General",), ()), "require fields"),
            ((("General",), (("VALUE", 1),)), "require key, line, and sort key"),
            ((("General",), (("", 1, (1,)),)), "field keys"),
            ((("General",), (("VALUE", True, (1,)),)), "lines"),
            ((("General",), (("VALUE", 0, (1,)),)), "lines"),
            ((("General",), (("VALUE", 1, ()),)), "sort keys"),
            ((("General",), (("VALUE", 1, (-1,)),)), "sort keys"),
            ((("General",), (("VALUE", 1, (True,)),)), "sort keys"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    constructor(*arguments)

    def test_explicit_declaration_validation_is_complete_and_stable(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        config = RuntimeDefaultsSection(
            ("General",),
            (("VALUE", 1, (1,)),),
        )
        search = RuntimeDefaultsSection(
            (),
            (("SEARCH_SPACE_VALUE", 1, (1,)),),
        )
        cases = (
            (
                (
                    config,
                    RuntimeDefaultsSection(
                        ("Repeated",),
                        (("VALUE", 2, (2,)),),
                    ),
                ),
                (search,),
                {},
                "duplicate field 'VALUE'",
            ),
            (
                (
                    RuntimeDefaultsSection(
                        ("General",),
                        (("UNKNOWN", 1, (1,)),),
                    ),
                ),
                (search,),
                {},
                "unknown field 'UNKNOWN'",
            ),
            ((), (search,), {}, "missing required fields: VALUE"),
            (
                (RuntimeDefaultsSection((), (("VALUE", 1, (1,)),)),),
                (search,),
                {},
                "Configuration metadata sections require a non-empty path",
            ),
            ((config,), (), {}, "missing required fields: SEARCH_SPACE_VALUE"),
            (
                (config,),
                (
                    search,
                    RuntimeDefaultsSection(
                        (),
                        (("SEARCH_SPACE_UNUSED", 2, (2,)),),
                    ),
                ),
                {
                    "search_members": {
                        "SEARCH_SPACE_VALUE": [1],
                        "SEARCH_SPACE_UNUSED": 1,
                    }
                },
                "search metadata declares fields without values: SEARCH_SPACE_UNUSED",
            ),
        )
        for configuration, search_declaration, kwargs, message in cases:
            with self.subTest(message=message):
                package = _package_with_declarations(
                    configuration,
                    search_declaration,
                    **kwargs,
                )
                with self.assertRaisesRegex(RuntimeDefaultsError, message) as direct:
                    _ = package.runtime_defaults_spec
                self.assertIsInstance(direct.exception.__cause__, ValueError)

                package = _package_with_declarations(
                    configuration,
                    search_declaration,
                    **kwargs,
                )
                with self.assertRaisesRegex(InspectionError, message) as inspected:
                    configuration_schema(package)
                self.assertIsInstance(inspected.exception.__cause__, ValueError)

    def test_configuration_declaration_preserves_supported_extras(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        package = _package_with_declarations(
            (
                RuntimeDefaultsSection(
                    ("General",),
                    (
                        ("VALUE", 1, (1,)),
                        ("PRESENTATION_ONLY", 2, (2,)),
                    ),
                ),
            ),
            (
                RuntimeDefaultsSection(
                    (),
                    (("SEARCH_SPACE_VALUE", 1, (1,)),),
                ),
            ),
            config_members={"VALUE": 1, "PRESENTATION_ONLY": ["metadata"]},
        )

        self.assertEqual(
            tuple(package.runtime_defaults_spec.configuration_metadata),
            ("VALUE", "PRESENTATION_ONLY"),
        )

    def test_public_constructor_supports_optional_metadata_declarations(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection
        from model_runtime.packages.configuration_metadata import (
            RuntimeDefaultsSection as OwnerRuntimeDefaultsSection,
        )

        self.assertIs(RuntimeDefaultsSection, OwnerRuntimeDefaultsSection)
        parameters = inspect.signature(ModelMetadata).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "identity",
                "runtime_defaults",
                "dataset_options",
                "monitor_options_source",
                "search_space",
                "configuration_metadata_sections",
                "search_metadata_sections",
            ),
        )
        for name in tuple(parameters)[:5]:
            self.assertIs(
                parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        for name in tuple(parameters)[5:]:
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)


if __name__ == "__main__":
    unittest.main()
