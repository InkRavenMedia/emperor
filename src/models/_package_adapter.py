from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from model_runtime.packages import ModelIdentity, ModelMetadata

if TYPE_CHECKING:
    from torch.nn import Module

    from emperor.config import ModelConfig
    from emperor.experiments import ExperimentTask
    from model_runtime.packages import ExperimentPresetsBase, ModelPackage
    from model_runtime.runs._handoff import RunExperiment
    from model_runtime.runs.artifacts import RunArtifacts


class ModelPackageAdapter:
    package_name: ClassVar[str]
    identity: ClassVar[ModelIdentity]

    def load_metadata(self) -> ModelMetadata:
        from model_runtime.monitoring_catalog import monitors_for_model

        return ModelMetadata(
            identity=self.identity,
            runtime_defaults=import_module(".config", self.package_name),
            dataset_options=import_module(".dataset_options", self.package_name),
            monitor_options_source=monitors_for_model,
            search_space=import_module(".search_space", self.package_name),
        )

    def load_runtime_options_type(self) -> type[Any]:
        options = import_module(".runtime_options", self.package_name)
        return options.RuntimeOptions

    def bind_runtime_defaults(self, values: Mapping[str, object] | None) -> Any:
        defaults = import_module(".runtime_defaults", self.package_name)
        return defaults.runtime_from_flat(values)

    def load_preset_type(self) -> type[Any]:
        presets = import_module(".presets", self.package_name)
        return presets.ExperimentPreset

    def load_presets(self) -> ExperimentPresetsBase:
        presets = import_module(".presets", self.package_name)
        return presets.ExperimentPresets()

    def build_configuration(
        self,
        presets: ExperimentPresetsBase,
        preset: object,
        dataset: type[Any],
        **kwargs: Any,
    ) -> ModelConfig:
        return presets.get_config(preset, dataset, **kwargs)[0]

    def build_model(self, configuration: ModelConfig) -> Module:
        model = import_module(".model", self.package_name)
        return model.Model(configuration)

    def build_experiment(
        self,
        preset: object,
        *,
        experiment_task: ExperimentTask,
        model_package: ModelPackage,
        run_artifacts: RunArtifacts,
    ) -> RunExperiment:
        presets = import_module(".presets", self.package_name)
        return presets.Experiment(
            preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )
