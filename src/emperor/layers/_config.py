from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field
from emperor.layers._options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
)

if TYPE_CHECKING:
    from emperor.halting import HaltingConfig
    from emperor.layers._composition.residual.config import ResidualConfig
    from emperor.linears import LinearLayerConfig
    from emperor.memory import DynamicMemoryConfig


_TOKEN_PROCESSING_MODEL_FIELDS = (
    "feed_forward_config",
    "layer_config",
    "layer_model_config",
    "stack_config",
    "experts_config",
    "mixture_config",
    "attention_config",
    "self_attention_config",
    "cross_attention_config",
    "projection_model_config",
    "mixing_model_config",
    "model_config",
)


@dataclass
class GateConfig(ConfigBase):
    gate_dim: int | None = optional_field(
        "Gate feature dimension. Gate model input and output dimensions are both "
        "set to this value."
    )
    option: LayerGateOptions | None = optional_field(
        "Gate composition mode. Use MULTIPLIER to scale the current value or "
        "ADDITION to add gate values to the current value."
    )
    activation: ActivationOptions | None = optional_field(
        "Optional activation applied to gate logits before composition. "
        "Set to None for raw gate values."
    )
    model_config: "LayerStackConfig | None" = optional_field(
        "Gate stack model config. Required when GateConfig is provided."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.gate import LayerGate

        return LayerGate


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    activation: ActivationOptions | None = optional_field(
        "Activation applied to the layer output."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability. Use 0.0 to disable."
    )
    layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where layer normalization is applied."
    )
    residual_config: "ResidualConfig | None" = optional_field(
        "Optional concrete residual connection config. Set to None to disable. "
        "Residual connections require input_dim == output_dim."
    )
    gate_config: "GateConfig | None" = optional_field(
        "Optional layer gate config. Set to None to disable."
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Optional adaptive computation module. Set to None to disable halting."
    )
    memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional dynamic memory module. Set to None to disable memory."
    )
    layer_model_config: ConfigBase | None = optional_field(
        "Config for the wrapped module, such as LinearLayerConfig."
    )
    normalization: NormalizationOptions | None = optional_field(
        "Normalization or learned replacement applied at layer_norm_position, "
        "over the last feature dimension. Choose RMS_NORM, LAYER_NORM, "
        "DYNAMIC_TANH, DERF, or experimental DYISRU. Defaults to RMS_NORM when "
        "unspecified. RMS_NORM and LAYER_NORM use epsilon 1e-5. DYNAMIC_TANH "
        "and DERF start with scalar alpha=0.5; DERF shift starts at zero. "
        "DYISRU starts with beta=4, learned through softplus with a 1e-5 floor. "
        "Disable with layer_norm_position=DISABLED."
    )

    @classmethod
    def find_processing_linear_template(
        cls, config: ConfigBase
    ) -> "LinearLayerConfig | None":
        first_linear = None
        for template, is_expert in cls.__processing_templates(config):
            if is_expert:
                return template
            if first_linear is None:
                first_linear = template
        return first_linear

    @classmethod
    def __processing_templates(
        cls, config: ConfigBase, *, inside_expert: bool = False
    ) -> Iterator[tuple["LinearLayerConfig", bool]]:
        from emperor.layers._composition.recurrent.config import (
            RecurrentCompositionConfig,
        )
        from emperor.linears import LinearLayerConfig

        if isinstance(config, LinearLayerConfig):
            yield config, inside_expert
            return
        if isinstance(config, RecurrentCompositionConfig):
            children = config._transition_configs()
        else:
            expert = getattr(config, "expert_model_config", None)
            if isinstance(expert, ConfigBase):
                yield from cls.__processing_templates(expert, inside_expert=True)
                return
            children = (
                getattr(config, name, None) for name in _TOKEN_PROCESSING_MODEL_FIELDS
            )
        for child in children:
            if isinstance(child, ConfigBase):
                yield from cls.__processing_templates(
                    child, inside_expert=inside_expert
                )

    def _registry_owner(self) -> type:
        from emperor.layers._layer import Layer

        return Layer


@dataclass
class LayerStackConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    hidden_dim: int | None = optional_field("Hidden feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    num_layers: int | None = optional_field("Total number of layers in the stack.")
    apply_output_postprocessing_flag: bool | None = optional_field(
        "Apply activation, dropout, layer normalization, and residual processing "
        "to the final layer."
    )
    last_layer_bias_option: "LastLayerBiasOptions | None" = optional_field(
        "Bias behavior for the final layer."
    )
    shared_gate_config: "GateConfig | None" = optional_field(
        "Optional shared gate config used by all stack layers."
    )
    shared_halting_config: "HaltingConfig | None" = optional_field(
        "Optional halting config used to build one shared halting module for all "
        "stack layers."
    )
    shared_memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional memory config used to build one shared dynamic memory module for "
        "all stack layers."
    )
    layer_config: LayerConfig | None = optional_field(
        "Base layer config used to build each stack layer."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._stack import LayerStack

        return LayerStack


@dataclass
class MirroredLayerStackConfig(LayerStackConfig):
    def _registry_owner(self) -> type:
        from emperor.layers._stack import MirroredLayerStack

        return MirroredLayerStack
