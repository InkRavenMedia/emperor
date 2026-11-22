from __future__ import annotations

from collections.abc import Callable
from dataclasses import Field, replace
from types import ModuleType
from typing import Any, ClassVar, Final, Generic, Protocol, TypeVar

from emperor.layers import ActivationOptions, LayerStackConfig, RecurrentLayerConfig
from models._linear_stack import LayerStackSettings, LinearStackFactory


class _Encoder(Protocol):
    hidden_dim: Final[int]
    activation: Final[ActivationOptions]
    dropout_probability: Final[float]


class _FeedForward(Protocol):
    num_layers: Final[int]
    bias_flag: Final[bool]


class _Attention(_FeedForward, Protocol):
    source_memory_flag: Final[bool]


class _Stack(LayerStackSettings, Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class _CoreInputs(Protocol):
    batch_size: Final[int]
    sequence_length: Final[int]
    encoder_options: Final[_Encoder | None]
    attention_options: Final[_Attention | None]
    feed_forward_options: Final[_FeedForward | None]
    stack_options: Final[object | None]
    attention_projection_stack_options: Final[_Stack | None]
    feed_forward_stack_options: Final[_Stack | None]
    submodule_stack_options: Final[_Stack | None]
    attention_projection_layer_controller_options: Final[object | None]
    attention_projection_dynamic_memory_options: Final[object | None]
    attention_projection_recurrent_controller_options: Final[object | None]
    feed_forward_layer_controller_options: Final[object | None]
    feed_forward_dynamic_memory_options: Final[object | None]
    feed_forward_recurrent_controller_options: Final[object | None]
    layer_controller_options: Final[object | None]
    dynamic_memory_options: Final[object | None]
    recurrent_controller_options: Final[object | None]
    linear_layer_config_factory: Final[LinearStackFactory]
    expert_config_factory: Final[object | None]


class _Defaults(Protocol):
    LinearRole: Final[type[Any]]

    def vit_encoder_options(self, config: ModuleType) -> _Encoder: ...
    def vit_feed_forward_options(self, config: ModuleType) -> _FeedForward: ...
    def vit_attention_options(self, config: ModuleType) -> _Attention: ...
    def main_layer_stack_options(self, config: ModuleType) -> object: ...
    def linears_submodule_stack_options(
        self, config: ModuleType, role: Any
    ) -> _Stack: ...


class _Builder(Protocol):
    def build_encoder_config(self) -> LayerStackConfig | RecurrentLayerConfig: ...


_Inputs = TypeVar("_Inputs", bound=_CoreInputs)
_Resolved = TypeVar("_Resolved")


class ConfiguredVisionEncoder(Generic[_Inputs, _Resolved]):
    _config: ModuleType
    _defaults: _Defaults
    _resolved_inputs: Callable[..., _Resolved]
    _encoder: Callable[[_Resolved], _Builder]

    def __init__(self, dependencies: _Inputs) -> None:
        self.dependencies = dependencies
        encoder_options = dependencies.encoder_options
        if encoder_options is None:
            encoder_options = self._defaults.vit_encoder_options(self._config)
        self.encoder_options = encoder_options
        attention_options = dependencies.attention_options
        if attention_options is None:
            attention_options = self._defaults.vit_attention_options(self._config)
        self.attention_options = attention_options
        feed_forward_options = dependencies.feed_forward_options
        if feed_forward_options is None:
            feed_forward_options = self._defaults.vit_feed_forward_options(self._config)
        self.feed_forward_options = feed_forward_options
        stack_options = dependencies.stack_options
        if stack_options is None:
            stack_options = self._defaults.main_layer_stack_options(self._config)
        self.stack_options = stack_options
        self.attention_projection_stack_options = (
            self.__default_attention_projection_stack_options(
                dependencies.attention_projection_stack_options
            )
        )
        self.feed_forward_stack_options = self.__default_feed_forward_stack_options(
            dependencies.feed_forward_stack_options
        )

    def build_encoder_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        core_dependencies = self.__core_dependencies()
        core_factory = self._encoder(core_dependencies)
        encoder_config = core_factory.build_encoder_config()
        return encoder_config

    def __default_attention_projection_stack_options(
        self,
        stack_options: _Stack | None,
    ) -> _Stack:
        if stack_options is not None:
            return stack_options
        defaults = self._defaults.linears_submodule_stack_options(
            self._config, self._defaults.LinearRole.ATTENTION
        )
        return replace(
            defaults,
            hidden_dim=self.encoder_options.hidden_dim,
            num_layers=self.attention_options.num_layers,
            activation=self.encoder_options.activation,
            bias_flag=self.attention_options.bias_flag,
        )

    def __default_feed_forward_stack_options(
        self,
        stack_options: _Stack | None,
    ) -> _Stack:
        if stack_options is not None:
            return stack_options
        defaults = self._defaults.linears_submodule_stack_options(
            self._config, self._defaults.LinearRole.FEED_FORWARD
        )
        return replace(
            defaults,
            hidden_dim=self.__scaled_feed_forward_hidden_dim(),
            num_layers=self.feed_forward_options.num_layers,
            activation=self.encoder_options.activation,
            dropout_probability=self.encoder_options.dropout_probability,
            bias_flag=self.feed_forward_options.bias_flag,
        )

    def __scaled_feed_forward_hidden_dim(self) -> int:
        if (
            self._config.HIDDEN_DIM > 0
            and self._config.FF_STACK_HIDDEN_DIM % self._config.HIDDEN_DIM == 0
        ):
            return self.encoder_options.hidden_dim * (
                self._config.FF_STACK_HIDDEN_DIM // self._config.HIDDEN_DIM
            )
        return self._config.FF_STACK_HIDDEN_DIM

    def __core_dependencies(self) -> _Resolved:
        dependencies = self.dependencies
        submodule_stack_options = dependencies.submodule_stack_options
        if (
            self.attention_options.source_memory_flag
            and submodule_stack_options is None
        ):
            submodule_stack_options = self._defaults.linears_submodule_stack_options(
                self._config, self._defaults.LinearRole.MAIN
            )
        attention_factory, feed_forward_factory = self._projection_factories()
        return self._resolved_inputs(
            batch_size=dependencies.batch_size,
            sequence_length=dependencies.sequence_length,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            attention_projection_stack_options=(
                self.attention_projection_stack_options
            ),
            attention_projection_layer_controller_options=(
                dependencies.attention_projection_layer_controller_options
            ),
            attention_projection_dynamic_memory_options=(
                dependencies.attention_projection_dynamic_memory_options
            ),
            attention_projection_recurrent_controller_options=(
                dependencies.attention_projection_recurrent_controller_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            feed_forward_layer_controller_options=(
                dependencies.feed_forward_layer_controller_options
            ),
            feed_forward_dynamic_memory_options=(
                dependencies.feed_forward_dynamic_memory_options
            ),
            feed_forward_recurrent_controller_options=(
                dependencies.feed_forward_recurrent_controller_options
            ),
            stack_options=self.stack_options,
            submodule_stack_options=submodule_stack_options,
            layer_controller_options=dependencies.layer_controller_options,
            dynamic_memory_options=dependencies.dynamic_memory_options,
            recurrent_controller_options=dependencies.recurrent_controller_options,
            linear_layer_config_factory=dependencies.linear_layer_config_factory,
            attention_projection_linear_layer_config_factory=attention_factory,
            feed_forward_linear_layer_config_factory=feed_forward_factory,
            expert_config_factory=dependencies.expert_config_factory,
        )

    def _projection_factories(
        self,
    ) -> tuple[LinearStackFactory | None, LinearStackFactory | None]:
        return None, None


class _ProjectionInputs(_CoreInputs, Protocol):
    attention_projection_linear_layer_config_factory: Final[LinearStackFactory | None]
    feed_forward_linear_layer_config_factory: Final[LinearStackFactory | None]


_Projection = TypeVar("_Projection", bound=_ProjectionInputs)


class ConfiguredProjectionEncoder(ConfiguredVisionEncoder[_Projection, _Resolved]):
    def _projection_factories(
        self,
    ) -> tuple[LinearStackFactory | None, LinearStackFactory | None]:
        dependencies = self.dependencies
        return (
            dependencies.attention_projection_linear_layer_config_factory,
            dependencies.feed_forward_linear_layer_config_factory,
        )
