from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import Any, Final, Protocol, TypeVar

from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayerConfig
from model_runtime.packages.settings_validation import BuildSettingsValidation


class _StackLayout(Protocol):
    @property
    def hidden_dim(self) -> int: ...

    @property
    def num_layers(self) -> int: ...

    @property
    def residual_model_flag(self) -> bool: ...

    @property
    def last_layer_bias_option(self) -> LastLayerBiasOptions: ...

    @property
    def apply_output_postprocessing_flag(self) -> bool: ...


class StackDefaults(_StackLayout, Protocol):
    @property
    def activation(self) -> ActivationOptions: ...

    @property
    def layer_norm_position(self) -> LayerNormPositionOptions: ...

    @property
    def normalization(self) -> NormalizationOptions: ...

    @property
    def residual_connection_option(self) -> type[ResidualConfig] | None: ...

    @property
    def residual_block_size(self) -> int | None: ...

    @property
    def residual_rms_norm_epsilon(self) -> float | None: ...

    @property
    def dropout_probability(self) -> float: ...

    @property
    def bias_flag(self) -> bool: ...


class ResidualStackSettings(StackDefaults, Protocol):
    @property
    def coefficient_activation(self) -> ActivationOptions: ...


_ValueT = TypeVar("_ValueT")
_Options = TypeVar("_Options", bound=_StackLayout)


class _ResidualSource(Protocol):
    coefficient_activation: Final[ActivationOptions]
    independent_flag: Final[bool]
    hidden_dim: Final[int | None]
    num_layers: Final[int | None]
    activation: Final[ActivationOptions | None]
    layer_norm_position: Final[LayerNormPositionOptions | None]
    normalization: Final[NormalizationOptions | None]
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_model_flag: Final[bool]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    dropout_probability: Final[float | None]
    last_layer_bias_option: Final[LastLayerBiasOptions | None]
    apply_output_postprocessing_flag: Final[bool | None]
    bias_flag: Final[bool | None]


_ResolvedStack = TypeVar("_ResolvedStack", covariant=True)


class _ResolvedStackFactory(Protocol[_ResolvedStack]):
    def __call__(
        self,
        *,
        coefficient_activation: ActivationOptions,
        hidden_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions,
        residual_connection_option: type[ResidualConfig] | None,
        residual_model_flag: bool,
        residual_block_size: int | None,
        residual_rms_norm_epsilon: float | None,
        dropout_probability: float,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_postprocessing_flag: bool,
        bias_flag: bool,
    ) -> _ResolvedStack: ...


class _FlatResidualSettings(Protocol):
    residual_coefficient_activation: Final[ActivationOptions]
    residual_stack_independent_flag: Final[bool]
    residual_stack_hidden_dim: Final[int | None]
    residual_stack_num_layers: Final[int | None]
    residual_stack_activation: Final[ActivationOptions | None]
    residual_stack_layer_norm_position: Final[LayerNormPositionOptions | None]
    residual_stack_normalization: Final[NormalizationOptions | None]
    residual_stack_residual_connection_option: Final[type[ResidualConfig] | None]
    residual_stack_residual_block_size: Final[int | None]
    residual_stack_residual_rms_norm_epsilon: Final[float | None]
    residual_stack_residual_model_flag: Final[bool]
    residual_stack_dropout_probability: Final[float | None]
    residual_stack_last_layer_bias_option: Final[LastLayerBiasOptions | None]
    residual_stack_apply_output_postprocessing_flag: Final[bool | None]
    residual_stack_bias_flag: Final[bool | None]


_Source = TypeVar("_Source", covariant=True)


class _SourceFactory(Protocol[_Source]):
    def __call__(
        self,
        *,
        coefficient_activation: ActivationOptions,
        independent_flag: bool,
        hidden_dim: int | None,
        num_layers: int | None,
        activation: ActivationOptions | None,
        layer_norm_position: LayerNormPositionOptions | None,
        normalization: NormalizationOptions | None,
        residual_connection_option: type[ResidualConfig] | None,
        residual_block_size: int | None,
        residual_rms_norm_epsilon: float | None,
        residual_model_flag: bool,
        dropout_probability: float | None,
        last_layer_bias_option: LastLayerBiasOptions | None,
        apply_output_postprocessing_flag: bool | None,
        bias_flag: bool | None,
    ) -> _Source: ...


def residual_source_from_overrides(
    config: ModuleType,
    read: Callable[[str, Any], Any],
    create_source: _SourceFactory[_Source],
    *,
    dimensions_first: bool = False,
) -> _Source:
    coefficient_activation = read(
        "residual_coefficient_activation", config.RESIDUAL_COEFFICIENT_ACTIVATION
    )
    independent_flag = read(
        "residual_stack_independent_flag", config.RESIDUAL_STACK_INDEPENDENT_FLAG
    )
    hidden_dim = read("residual_stack_hidden_dim", config.RESIDUAL_STACK_HIDDEN_DIM)
    if dimensions_first:
        num_layers = read("residual_stack_num_layers", config.RESIDUAL_STACK_NUM_LAYERS)
        activation = read("residual_stack_activation", config.RESIDUAL_STACK_ACTIVATION)
        layer_norm_position = read(
            "residual_stack_layer_norm_position",
            config.RESIDUAL_STACK_LAYER_NORM_POSITION,
        )
        normalization = read(
            "residual_stack_normalization", config.RESIDUAL_STACK_NORMALIZATION
        )
    else:
        layer_norm_position = read(
            "residual_stack_layer_norm_position",
            config.RESIDUAL_STACK_LAYER_NORM_POSITION,
        )
        normalization = read(
            "residual_stack_normalization", config.RESIDUAL_STACK_NORMALIZATION
        )
        num_layers = read("residual_stack_num_layers", config.RESIDUAL_STACK_NUM_LAYERS)
        activation = read("residual_stack_activation", config.RESIDUAL_STACK_ACTIVATION)
    return create_source(
        coefficient_activation=coefficient_activation,
        independent_flag=independent_flag,
        hidden_dim=hidden_dim,
        layer_norm_position=layer_norm_position,
        normalization=normalization,
        num_layers=num_layers,
        activation=activation,
        residual_connection_option=read(
            "residual_stack_residual_connection_option",
            config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION,
        ),
        residual_block_size=read(
            "residual_stack_residual_block_size",
            config.RESIDUAL_STACK_RESIDUAL_BLOCK_SIZE,
        ),
        residual_rms_norm_epsilon=read(
            "residual_stack_residual_rms_norm_epsilon",
            config.RESIDUAL_STACK_RESIDUAL_RMS_NORM_EPSILON,
        ),
        residual_model_flag=read(
            "residual_stack_residual_model_flag",
            config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG,
        ),
        dropout_probability=read(
            "residual_stack_dropout_probability",
            config.RESIDUAL_STACK_DROPOUT_PROBABILITY,
        ),
        last_layer_bias_option=read(
            "residual_stack_last_layer_bias_option",
            config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION,
        ),
        apply_output_postprocessing_flag=read(
            "residual_stack_apply_output_postprocessing_flag",
            config.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
        ),
        bias_flag=read("residual_stack_bias_flag", config.RESIDUAL_STACK_BIAS_FLAG),
    )


def residual_source_from_settings(
    settings: _FlatResidualSettings, create_source: _SourceFactory[_Source]
) -> _Source:
    return create_source(
        coefficient_activation=settings.residual_coefficient_activation,
        independent_flag=settings.residual_stack_independent_flag,
        hidden_dim=settings.residual_stack_hidden_dim,
        num_layers=settings.residual_stack_num_layers,
        activation=settings.residual_stack_activation,
        layer_norm_position=settings.residual_stack_layer_norm_position,
        normalization=settings.residual_stack_normalization,
        residual_connection_option=settings.residual_stack_residual_connection_option,
        residual_block_size=settings.residual_stack_residual_block_size,
        residual_rms_norm_epsilon=settings.residual_stack_residual_rms_norm_epsilon,
        residual_model_flag=settings.residual_stack_residual_model_flag,
        dropout_probability=settings.residual_stack_dropout_probability,
        last_layer_bias_option=settings.residual_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=settings.residual_stack_apply_output_postprocessing_flag,
        bias_flag=settings.residual_stack_bias_flag,
    )


def resolve_stack_options(
    source: _ResidualSource,
    defaults: StackDefaults,
    create_options: _ResolvedStackFactory[_ResolvedStack],
) -> _ResolvedStack:
    stack = defaults
    if source.independent_flag:
        stack = source
    return create_options(
        coefficient_activation=source.coefficient_activation,
        hidden_dim=inherit_if_unset(stack.hidden_dim, defaults.hidden_dim),
        num_layers=inherit_if_unset(stack.num_layers, defaults.num_layers),
        activation=inherit_if_unset(stack.activation, defaults.activation),
        layer_norm_position=inherit_if_unset(
            stack.layer_norm_position, defaults.layer_norm_position
        ),
        normalization=inherit_if_unset(stack.normalization, defaults.normalization),
        residual_connection_option=inherit_if_unset(
            stack.residual_connection_option, defaults.residual_connection_option
        ),
        residual_model_flag=source.residual_model_flag,
        residual_block_size=inherit_if_unset(
            source.residual_block_size, defaults.residual_block_size
        ),
        residual_rms_norm_epsilon=inherit_if_unset(
            source.residual_rms_norm_epsilon, defaults.residual_rms_norm_epsilon
        ),
        dropout_probability=inherit_if_unset(
            stack.dropout_probability, defaults.dropout_probability
        ),
        last_layer_bias_option=inherit_if_unset(
            stack.last_layer_bias_option, defaults.last_layer_bias_option
        ),
        apply_output_postprocessing_flag=inherit_if_unset(
            stack.apply_output_postprocessing_flag,
            defaults.apply_output_postprocessing_flag,
        ),
        bias_flag=inherit_if_unset(stack.bias_flag, defaults.bias_flag),
    )


def build_residual_stack(
    options: _Options, build_layer: Callable[[_Options], LayerConfig]
) -> LayerStackConfig:
    if options.residual_model_flag:
        raise ValueError(
            "RESIDUAL_STACK_RESIDUAL_MODEL_FLAG=True with "
            "RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION would recursively "
            "define the residual stack in itself."
        )

    layer_config = build_layer(options)
    return LayerStackConfig(
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=layer_config,
    )


@dataclass(frozen=True)
class ResidualBuilder:
    lazy_coefficient_activation: bool = False

    def build_residual_config(
        self,
        residual_connection_option: type[ResidualConfig] | None,
        residual_model_flag: bool,
        residual_stack_options: ResidualStackSettings | None = None,
        *,
        residual_block_size: int | None = None,
        residual_rms_norm_epsilon: float | None = None,
        selector_field: str = "residual_connection_option",
        model_flag_field: str = "residual_model_flag",
    ) -> ResidualConfig | None:
        coefficient_activation = ActivationOptions.SIGMOID
        if not self.lazy_coefficient_activation and residual_stack_options is not None:
            coefficient_activation = residual_stack_options.coefficient_activation
        self.__validate_selection(
            residual_connection_option,
            residual_model_flag,
            selector_field=selector_field,
            model_flag_field=model_flag_field,
        )
        if residual_connection_option is None:
            return None
        if (
            self.lazy_coefficient_activation
            and residual_stack_options is not None
            and issubclass(
                residual_connection_option,
                (WeightedResidualConfig, WeightedBlendResidualConfig),
            )
        ):
            coefficient_activation = residual_stack_options.coefficient_activation
        build_residual = self.__constructor(
            residual_connection_option,
            coefficient_activation,
            block_size=residual_block_size,
            rms_norm_epsilon=residual_rms_norm_epsilon,
        )
        if not residual_model_flag:
            residual_config = build_residual()
            return residual_config
        if residual_stack_options is None:
            raise ValueError(
                f"{model_flag_field}=True with {selector_field} requires resolved "
                "RESIDUAL_STACK options."
            )
        residual_stack_config = self.build_residual_stack_config(residual_stack_options)
        residual_config = build_residual(model_config=residual_stack_config)
        return residual_config

    def build_residual_stack_config(
        self,
        options: ResidualStackSettings,
    ) -> LayerStackConfig:
        return build_residual_stack(options, self.__build_stack_layer)

    def __build_stack_layer(self, options: ResidualStackSettings) -> LayerConfig:
        residual_config = self.build_residual_config(
            options.residual_connection_option,
            False,
            options,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            selector_field="RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION",
            model_flag_field="RESIDUAL_STACK_RESIDUAL_MODEL_FLAG",
        )
        linear_config = LinearLayerConfig(bias_flag=options.bias_flag)
        return LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            residual_config=residual_config,
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_config,
        )

    @staticmethod
    def __constructor(
        residual_type: type[ResidualConfig],
        coefficient_activation: ActivationOptions,
        *,
        block_size: int | None,
        rms_norm_epsilon: float | None,
    ) -> partial[ResidualConfig]:
        if issubclass(residual_type, AttentionResidualConfig):
            return partial(
                residual_type, block_size=block_size, rms_norm_epsilon=rms_norm_epsilon
            )
        if issubclass(
            residual_type, (WeightedResidualConfig, WeightedBlendResidualConfig)
        ):
            return partial(residual_type, coefficient_activation=coefficient_activation)
        return partial(residual_type)

    @staticmethod
    def __validate_selection(
        selector: object,
        model_flag: object,
        *,
        selector_field: str,
        model_flag_field: str,
    ) -> None:
        if type(model_flag) is not bool:
            raise TypeError(f"{model_flag_field} must be bool.")
        selector_type_name = type(selector).__name__
        if selector is not None and not BuildSettingsValidation.is_selectable_residual(
            selector
        ):
            raise TypeError(
                f"{selector_field} must be a ResidualConfig type or None; "
                f"received {selector_type_name}."
            )
        if model_flag and (
            selector is None
            or not issubclass(
                selector,
                (
                    AttentionResidualConfig,
                    WeightedResidualConfig,
                    WeightedBlendResidualConfig,
                ),
            )
        ):
            selected = "None"
            if selector is not None:
                selected = selector.__name__
            raise ValueError(
                f"{model_flag_field}=True requires {selector_field} to select "
                "WeightedResidualConfig, WeightedBlendResidualConfig, or "
                "AttentionResidualConfig; "
                f"received {selected}."
            )


def build_residual_config(
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    residual_stack_options: ResidualStackSettings | None = None,
    *,
    residual_block_size: int | None = None,
    residual_rms_norm_epsilon: float | None = None,
    selector_field: str = "residual_connection_option",
    model_flag_field: str = "residual_model_flag",
) -> ResidualConfig | None:
    builder = ResidualBuilder()
    return builder.build_residual_config(
        residual_connection_option,
        residual_model_flag,
        residual_stack_options,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        selector_field=selector_field,
        model_flag_field=model_flag_field,
    )


def build_residual_stack_config(options: ResidualStackSettings) -> LayerStackConfig:
    builder = ResidualBuilder()
    return builder.build_residual_stack_config(options)


def inherit_if_unset(value: _ValueT | None, default: _ValueT) -> _ValueT:
    if value is None:
        return default
    return value
