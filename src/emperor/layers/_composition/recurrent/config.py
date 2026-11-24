from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar

from emperor.config import ConfigBase, optional_field
from emperor.layers._options import LayerNormPositionOptions, NormalizationOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from emperor.halting import HaltingConfig
    from emperor.layers._composition.residual.config import ResidualConfig
    from emperor.layers._config import GateConfig
    from emperor.linears import LinearLayerConfig
    from emperor.memory import DynamicMemoryConfig
    from emperor.sampler import TokenSamplerConfig


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


@dataclass(kw_only=True)
class RecurrentCompressionConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Feature width supplied by the recurrent owner."
    )
    max_slots: int | None = optional_field(
        "Maximum slot count for each inner block call."
    )
    scorer_config: LinearLayerConfig | None = optional_field(
        "Overrides the linear scorer inherited from the transition's expert or "
        "ordinary layer. A block without a linear template uses a linear layer with bias."
    )

    def _registry_owner(self) -> type:
        from .runtime.compression import RecurrentTokenCompressor

        return RecurrentTokenCompressor


@dataclass
class RecurrentCompositionConfig(ConfigBase):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = ()

    compression_config: RecurrentCompressionConfig | None = field(
        default=None,
        kw_only=True,
        metadata={"help": "Optional token compression around the inner block."},
    )
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    no_gradient_transition_count: int | None = optional_field(
        "Optional number of initial transition invocations executed without "
        "gradient tracking. Every subsequently executed transition uses gradients. "
        "With smooth iteration growth, explicit zero and an omitted gradient suffix "
        "select the single-chain full-gradient adjacent-depth handoff."
    )
    gradient_transition_count: int | None = optional_field(
        "Number of final active recurrent transitions in each forward call that "
        "track gradients. Earlier active transitions run under torch.no_grad(). "
        "This controls gradient tracking, not loop growth. Set to None to use the "
        "variant default. Mutually exclusive with no_gradient_transition_count."
    )
    initial_iterations: int | None = optional_field(
        "Starting number of complete outer-loop recurrent iterations executed per "
        "forward call. Required. Set it equal to the variant's maximum iteration "
        "count to use a fixed loop length without gradual growth."
    )
    iteration_increment: int | None = optional_field(
        "Number of complete outer-loop recurrent iterations added to the active loop "
        "length whenever the successful-forward threshold is reached. Growth stops "
        "at the variant's maximum iteration count. Required."
    )
    forward_calls_before_iteration_increment: int | None = optional_field(
        "Number of successful recurrent forward calls required before the active loop "
        "length increases by iteration_increment. The counter advances after each "
        "completed call; failed calls do not count. Required."
    )
    smooth_iteration_growth_flag: bool | None = optional_field(
        "Linearly hand off each one-iteration loop-length increase during the "
        "first half of its successful-forward interval. False or None preserves "
        "hard iteration growth. A positive gradient suffix uses the source/target branch "
        "handoff; an explicit zero no-gradient prefix uses the nested all-gradient "
        "handoff."
    )
    recurrent_layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Normalization position in the outer recurrent pipeline. TRM runs this "
        "pipeline on answer updates and HRM runs it on high-state updates. "
        "Choose a LayerNormPositionOptions value or DISABLED."
    )
    gate_config: GateConfig | None = optional_field(
        "Optional recurrent gate config. Set to None to disable."
    )
    residual_config: ResidualConfig | None = optional_field(
        "Optional residual connection applied by the outer recurrent pipeline. "
        "Set to None to disable recurrent residuals."
    )
    halting_config: HaltingConfig | None = optional_field(
        "Optional recurrent adaptive computation module. Set to None to disable."
    )
    memory_config: DynamicMemoryConfig | None = optional_field(
        "Optional dynamic memory module applied by the outer recurrent pipeline. "
        "Set to None to disable memory."
    )
    recurrent_normalization: NormalizationOptions | None = optional_field(
        "Required feature normalization used by the outer recurrent pipeline. "
        "Choose a NormalizationOptions value explicitly, including when "
        "recurrent_layer_norm_position is DISABLED."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "RecurrentCompositionConfig is abstract and has no registered recurrent "
            "composition; instantiate a concrete recurrent config instead."
        )

    def _transition_configs(self) -> tuple[ConfigBase, ...]:
        return tuple(
            transition_config
            for _, transition_config in self._transition_config_items()
        )

    def resolve_compression_config(self) -> RecurrentCompressionConfig | None:
        compression = self.compression_config
        if (
            not isinstance(compression, RecurrentCompressionConfig)
            or compression.scorer_config is not None
        ):
            return compression
        template = self.__compression_scorer_template()
        if template is None:
            return compression
        scorer = replace(
            deepcopy(template),
            input_dim=self.output_dim,
            output_dim=compression.max_slots,
            bias_flag=True if template.bias_flag is None else template.bias_flag,
        )
        return replace(compression, scorer_config=scorer)

    def __compression_scorer_template(self) -> LinearLayerConfig | None:
        first_linear = None
        for template, is_expert in self.__transition_linear_templates(self):
            if is_expert:
                return template
            if first_linear is None:
                first_linear = template
        return first_linear

    def __transition_linear_templates(
        self, config: ConfigBase, *, inside_expert: bool = False
    ) -> Iterator[tuple[LinearLayerConfig, bool]]:
        from emperor.linears import LinearLayerConfig

        if isinstance(config, LinearLayerConfig):
            yield config, inside_expert
            return
        if isinstance(config, RecurrentCompositionConfig):
            children = config._transition_configs()
        else:
            expert = getattr(config, "expert_model_config", None)
            if isinstance(expert, ConfigBase):
                yield from self.__transition_linear_templates(
                    expert, inside_expert=True
                )
                return
            children = (
                getattr(config, name, None) for name in _TOKEN_PROCESSING_MODEL_FIELDS
            )
        for child in children:
            if isinstance(child, ConfigBase):
                yield from self.__transition_linear_templates(
                    child, inside_expert=inside_expert
                )

    def _transition_config_items(self) -> tuple[tuple[str, ConfigBase], ...]:
        return tuple(
            (field_name, transition_config)
            for field_name in self._TRANSITION_CONFIG_FIELDS
            if (transition_config := getattr(self, field_name)) is not None
        )

    def _missing_transition_config_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in self._TRANSITION_CONFIG_FIELDS
            if getattr(self, field_name) is None
        )

    def _map_transition_configs(
        self,
        transform: Callable[[ConfigBase], ConfigBase],
    ) -> None:
        for field_name in self._TRANSITION_CONFIG_FIELDS:
            transition_config = getattr(self, field_name)
            if transition_config is not None:
                setattr(self, field_name, transform(transition_config))


@dataclass
class RecurrentLayerConfig(RecurrentCompositionConfig):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = ("block_config",)

    max_steps: int | None = optional_field("Maximum recurrent applications.")
    reinject_original_hidden_flag: bool | None = optional_field(
        "Add the original input hidden tensor to the evolving hidden tensor before "
        "every recurrent block invocation."
    )

    block_config: ConfigBase | None = optional_field(
        "ConfigBase block reused at every recurrent step. The built module must "
        "consume and return LayerState-compatible values and declare input_dim and "
        "output_dim fields."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.standard import (
            RecurrentLayer,
        )

        return RecurrentLayer


@dataclass
class InnerThinkingRecurrentConfig(RecurrentLayerConfig):
    """Token-selective recurrence with a full first pass and masked updates."""

    sampler_config: TokenSamplerConfig | None = optional_field(
        "Token sampler selecting and weighting positions for each pass after the "
        "first. Each extra step builds an independent sampler from this config. "
        "Inputs use [..., tokens, features]; blocks must accept variable token "
        "counts. max_steps includes the initial all-token pass."
    )
    thinking_step_scale: float | None = optional_field(
        "Non-negative scale of each selected residual update, before learned "
        "router and per-feature thinking-step weights. None uses 1.0."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.inner_thinking import (
            InnerThinkingRecurrent,
        )

        return InnerThinkingRecurrent


@dataclass
class TinyRecursiveModelRecurrentConfig(RecurrentCompositionConfig):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = ("block_config",)

    block_config: ConfigBase | None = optional_field(
        "ConfigBase transition block shared by latent and answer updates."
    )
    latent_updates_per_answer_update: int | None = optional_field(
        "Number of latent-state transition updates performed before each "
        "answer-state update."
    )
    answer_update_count: int | None = optional_field(
        "Number of answer-state updates performed in one recurrent forward."
    )
    initialization_standard_deviation: float | None = optional_field(
        "Standard deviation for persistent answer and latent initialization buffers."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.tiny_recursive_model import (
            TinyRecursiveModelRecurrent,
        )

        return TinyRecursiveModelRecurrent


@dataclass
class HierarchicalReasoningModelRecurrentConfig(RecurrentCompositionConfig):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = (
        "high_block_config",
        "low_block_config",
    )

    high_block_config: ConfigBase | None = optional_field(
        "ConfigBase transition block used for high-level state updates."
    )
    low_block_config: ConfigBase | None = optional_field(
        "ConfigBase transition block used for low-level state updates."
    )
    high_cycles: int | None = optional_field(
        "Number of high-level clock cycles in one recurrent forward."
    )
    low_cycles: int | None = optional_field(
        "Number of low-level updates in each high-level clock cycle."
    )
    initialization_standard_deviation: float | None = optional_field(
        "Standard deviation for persistent high- and low-state initialization buffers."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model import (
            HierarchicalReasoningModelRecurrent,
        )

        return HierarchicalReasoningModelRecurrent
