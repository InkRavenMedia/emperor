from typing import TYPE_CHECKING

from models.linears.linear._hidden_model_config_factory import HiddenModelConfigFactory
from models.linears.linear._projection_config_factory import ProjectionConfigFactory
from models.linears.linear.experiment_config import ExperimentConfig
from models.linears.linear.runtime_defaults import DEFAULT_BUILD_SETTINGS
from models.linears.linear.runtime_options import LinearBuildSettings

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearConfigBuilder:
    def __init__(
        self,
        *,
        build_settings: LinearBuildSettings = DEFAULT_BUILD_SETTINGS,
        runtime: LinearBuildSettings = DEFAULT_BUILD_SETTINGS,
    ) -> None:
        if runtime is not DEFAULT_BUILD_SETTINGS:
            if build_settings is not DEFAULT_BUILD_SETTINGS:
                raise TypeError(
                    "Pass settings using either 'build_settings' or 'runtime'."
                )
            self.runtime = runtime
        else:
            self.build_settings = build_settings

    @property
    def build_settings(self) -> LinearBuildSettings:
        return self.__build_settings

    @build_settings.setter
    def build_settings(self, value: LinearBuildSettings) -> None:
        self.__set_settings(value, "build_settings", "LinearBuildSettings")

    @property
    def runtime(self) -> LinearBuildSettings:
        return self.__build_settings

    @runtime.setter
    def runtime(self, value: LinearBuildSettings) -> None:
        self.__set_settings(value, "runtime", "RuntimeOptions")

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        projection_factory = ProjectionConfigFactory(self.build_settings)
        hidden_factory = HiddenModelConfigFactory(self.build_settings)
        input_model_config = projection_factory.build_input_model_config()
        hidden_model_config = hidden_factory.build_hidden_model_config()
        output_model_config = projection_factory.build_output_model_config()
        experiment_config = ExperimentConfig(
            input_model_config=input_model_config,
            model_config=hidden_model_config,
            output_model_config=output_model_config,
        )
        return ModelConfig(
            learning_rate=self.build_settings.learning_rate,
            batch_size=self.build_settings.batch_size,
            input_dim=self.build_settings.input_dim,
            hidden_dim=self.build_settings.hidden_dim,
            output_dim=self.build_settings.output_dim,
            experiment_config=experiment_config,
        )

    def __set_settings(
        self, value: LinearBuildSettings, argument: str, expected_type: str
    ) -> None:
        if type(value) is not LinearBuildSettings:
            actual_type = type(value)
            raise TypeError(
                f"models.linears.linear LinearConfigBuilder '{argument}' must be "
                f"models.linears.linear.runtime_options.{expected_type}; got "
                f"{actual_type.__module__}.{actual_type.__qualname__}"
            )
        self.__build_settings = value
