from dataclasses import replace
from typing import Final, Generic, Protocol, TypeVar, cast

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    MirroredLayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    MemoryPositionOptions,
)
from emperor.patch import LinearPatchEmbeddingConfig
from models._recurrent_block import RecurrentSchedule, build_recurrent_block
from models.mlp_mixer._control_options import (
    ControlOptions,
    GateOptions,
    HaltingOptions,
    MemoryOptions,
    StackOptions,
)
from models.mlp_mixer._settings import ImageReadout


class _MixerSettings(Protocol):
    hidden_dim: Final[int]
    input_channels: Final[int]
    image_patch_size: Final[int]
    image_readout: Final[ImageReadout]
    patch_bias_flag: Final[bool]
    patch_dropout_probability: Final[float]

    token_mixer_source_memory_option: Final[type[DynamicMemoryConfig]]
    token_mixer_source_memory_test_time_training_learning_rate: Final[float | None]
    token_mixer_source_memory_test_time_training_num_inner_steps: Final[int | None]
    token_mixer_source_memory_num_memory_slots: Final[int]


_Settings = TypeVar("_Settings", bound=_MixerSettings)
_SettingsIn = TypeVar("_SettingsIn", contravariant=True)


class BuildResidual(Protocol[_SettingsIn]):
    def __call__(
        self,
        runtime: _SettingsIn,
        option: type[ResidualConfig] | None,
        model_flag: bool,
        /,
        *,
        residual_block_size: int | None,
        residual_rms_norm_epsilon: float | None,
    ) -> ResidualConfig | None: ...


class _LinearModel(Protocol):
    def __call__(self, *, bias_flag: bool) -> ConfigBase: ...


_StackConfigT = TypeVar("_StackConfigT", bound=ConfigBase)


class MixerComponentFactory(Generic[_Settings]):
    def __init__(
        self, residual: BuildResidual[_Settings], plain_layer: _LinearModel
    ) -> None:
        self._residual = residual
        self._plain_layer = plain_layer

    def build_affine_stack(
        self,
        runtime: _Settings,
        *,
        stack_options: StackOptions,
        input_dim: int | None,
        output_dim: int | None,
        mirrored: bool = False,
        build_layer: _LinearModel | None = None,
    ) -> LayerStackConfig:
        stack_type = LayerStackConfig
        stack_depth = stack_options.num_layers
        if mirrored:
            stack_type = MirroredLayerStackConfig
            if stack_options.num_layers <= 0 or stack_options.num_layers % 2:
                raise ValueError(
                    "channel_mixer_num_layers must be a positive even integer, got "
                    f"{stack_options.num_layers}"
                )
            stack_depth = stack_options.num_layers // 2
        if build_layer is None:
            build_layer = self._plain_layer
        linear_config = build_layer(bias_flag=stack_options.bias_flag)
        residual_config = self._residual(
            runtime,
            stack_options.residual_connection_option,
            stack_options.residual_model_flag,
            residual_block_size=stack_options.residual_block_size,
            residual_rms_norm_epsilon=stack_options.residual_rms_norm_epsilon,
        )
        affine_layer_config = LayerConfig(
            activation=stack_options.activation,
            residual_config=residual_config,
            dropout_probability=stack_options.dropout_probability,
            layer_norm_position=stack_options.layer_norm_position,
            normalization=stack_options.normalization,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_config,
        )
        stack = stack_type(
            input_dim=input_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=output_dim,
            num_layers=stack_depth,
            apply_output_postprocessing_flag=stack_options.apply_output_postprocessing_flag,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=None,
            layer_config=affine_layer_config,
        )
        return stack

    def build_patch(self, runtime: _Settings) -> LinearPatchEmbeddingConfig:
        patch_stack_options = StackOptions(
            hidden_dim=runtime.hidden_dim,
            num_layers=1,
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_connection_option=None,
            residual_block_size=None,
            residual_rms_norm_epsilon=None,
            residual_model_flag=False,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=False,
            bias_flag=runtime.patch_bias_flag,
        )
        projection = self.build_affine_stack(
            runtime,
            stack_options=patch_stack_options,
            input_dim=None,
            output_dim=None,
        )
        return LinearPatchEmbeddingConfig(
            embedding_dim=runtime.hidden_dim,
            num_input_channels=runtime.input_channels,
            patch_size=runtime.image_patch_size,
            dropout_probability=runtime.patch_dropout_probability,
            class_token_flag=runtime.image_readout == ImageReadout.CLS,
            stride=runtime.image_patch_size,
            padding=0,
            embedding_stack_config=projection,
        )

    def build_controller_stack(
        self,
        runtime: _Settings,
        *,
        defaults: StackOptions,
        controller: GateOptions | HaltingOptions | MemoryOptions | None = None,
        halting_gate: bool = False,
    ) -> LayerStackConfig:
        options = defaults
        if controller is not None:
            options = controller.stack.resolve(defaults)
        output_dim = None
        if halting_gate:
            output_dim = 2
            options = replace(
                options, last_layer_bias_option=LastLayerBiasOptions.DISABLED
            )
        controller_stack_config = self.build_affine_stack(
            runtime,
            stack_options=options,
            input_dim=None,
            output_dim=output_dim,
        )
        return controller_stack_config

    def build_source_memory(
        self, runtime: _Settings, defaults: StackOptions
    ) -> DynamicMemoryConfig:
        source_memory_model_config = self.build_controller_stack(
            runtime, defaults=defaults
        )
        memory = runtime.token_mixer_source_memory_option(
            memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
            model_config=source_memory_model_config,
            test_time_training_learning_rate=(
                runtime.token_mixer_source_memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                runtime.token_mixer_source_memory_test_time_training_num_inner_steps
            ),
        )
        if isinstance(memory, AttentionDynamicMemoryConfig):
            memory.num_memory_slots = runtime.token_mixer_source_memory_num_memory_slots
        return memory

    def build_gate(
        self,
        runtime: _Settings,
        *,
        options: GateOptions,
        defaults: StackOptions,
        model_dim: int | None,
    ) -> GateConfig | None:
        if not options.enabled:
            return None
        gate_model_config = self.build_controller_stack(
            runtime,
            controller=options,
            defaults=defaults,
        )
        return GateConfig(
            gate_dim=model_dim,
            option=options.option,
            activation=options.activation,
            model_config=gate_model_config,
        )

    def build_memory(
        self,
        runtime: _Settings,
        *,
        options: MemoryOptions | None,
        defaults: StackOptions,
        model_dim: int | None,
    ) -> DynamicMemoryConfig | None:
        if options is None or not options.enabled:
            return None
        memory_model_config = self.build_controller_stack(
            runtime,
            controller=options,
            defaults=defaults,
        )
        memory_config = options.implementation(
            input_dim=model_dim,
            output_dim=model_dim,
            memory_position_option=options.position,
            test_time_training_learning_rate=(options.test_time_training_learning_rate),
            test_time_training_num_inner_steps=(
                options.test_time_training_num_inner_steps
            ),
            model_config=memory_model_config,
        )
        if isinstance(memory_config, AttentionDynamicMemoryConfig):
            memory_config.num_memory_slots = options.num_memory_slots
        return memory_config

    def build_halting(
        self,
        runtime: _Settings,
        *,
        options: HaltingOptions,
        defaults: StackOptions,
        model_dim: int | None,
    ) -> HaltingConfig | None:
        if not options.enabled:
            return None
        halting_gate_config = self.build_controller_stack(
            runtime,
            controller=options,
            defaults=defaults,
            halting_gate=True,
        )
        return options.implementation(
            input_dim=model_dim,
            threshold=options.threshold,
            min_steps=self._halting_min_steps(options),
            ponder_cost_weight=self._halting_ponder_cost(options),
            dropout_probability=options.dropout_probability,
            hidden_state_mode=options.hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def configure_stack(
        self,
        runtime: _Settings,
        stack: LayerStackConfig,
        *,
        options: ControlOptions,
        defaults: StackOptions,
        model_dim: int | None,
        shared_halting: bool = True,
        shared_memory: bool = True,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        return self.configure_controls(
            runtime,
            options=options,
            model_config=stack,
            control_stack=stack,
            defaults=defaults,
            model_dim=model_dim,
            shared_halting=shared_halting,
            shared_memory=shared_memory,
        )

    def configure_controls(
        self,
        runtime: _Settings,
        *,
        options: ControlOptions,
        model_config: _StackConfigT,
        control_stack: LayerStackConfig,
        defaults: StackOptions,
        model_dim: int | None,
        shared_halting: bool = True,
        shared_memory: bool = True,
    ) -> _StackConfigT | RecurrentLayerConfig:
        cast(LayerConfig, control_stack.layer_config).gate_config = self.build_gate(
            runtime,
            options=options.gate,
            defaults=defaults,
            model_dim=model_dim,
        )
        halting_config = self.build_halting(
            runtime,
            options=options.halting,
            defaults=defaults,
            model_dim=model_dim,
        )
        if shared_halting:
            control_stack.shared_halting_config = halting_config
        else:
            cast(
                LayerConfig, control_stack.layer_config
            ).halting_config = halting_config
        memory_config = self.build_memory(
            runtime,
            options=options.memory,
            defaults=defaults,
            model_dim=model_dim,
        )
        if shared_memory:
            control_stack.shared_memory_config = memory_config
        else:
            cast(LayerConfig, control_stack.layer_config).memory_config = memory_config
        recurrent = options.recurrent
        if not recurrent.enabled:
            return model_config
        recurrent_gate_config = self.build_gate(
            runtime,
            options=recurrent.gate,
            defaults=defaults,
            model_dim=model_dim,
        )
        recurrent_residual_config = self._residual(
            runtime,
            recurrent.residual_connection_option,
            recurrent.residual_model_flag,
            residual_block_size=recurrent.residual_block_size,
            residual_rms_norm_epsilon=recurrent.residual_rms_norm_epsilon,
        )
        recurrent_halting_config = self.build_halting(
            runtime,
            options=recurrent.halting,
            defaults=defaults,
            model_dim=model_dim,
        )
        recurrent_memory_config = self.build_memory(
            runtime,
            options=recurrent.memory,
            defaults=defaults,
            model_dim=model_dim,
        )
        schedule = RecurrentSchedule.from_options(recurrent)
        return build_recurrent_block(
            schedule,
            model_config,
            input_dim=model_dim,
            output_dim=model_dim,
            smooth_iteration_growth_flag=recurrent.smooth_iteration_growth_flag,
            layer_norm_position=recurrent.layer_norm_position,
            normalization=recurrent.normalization,
            gate_config=recurrent_gate_config,
            residual_config=recurrent_residual_config,
            halting_config=recurrent_halting_config,
            memory_config=recurrent_memory_config,
        )

    @staticmethod
    def _halting_min_steps(options: HaltingOptions) -> int:
        return 1

    @staticmethod
    def _halting_ponder_cost(options: HaltingOptions) -> float:
        return 1.0
