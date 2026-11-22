from typing import Final, Protocol, cast

import torch

from emperor.attention import (
    AttentionSourceMemoryConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsModelConfig, SoftMoEConfig
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    RecurrentCompositionConfig,
)
from emperor.memory import DynamicMemoryConfig
from emperor.transformer import FeedForwardConfig
from emperor.transformer._soft_moe_config import build_soft_moe_stack


class _Gating(Protocol):
    def build_gate_config(self) -> GateConfig | None: ...


class _Halting(Protocol):
    def build_halting_config(self) -> HaltingConfig | None: ...


class _Memory(Protocol):
    def build_memory_config(self) -> DynamicMemoryConfig | None: ...


class _LayerOptions(Protocol):
    shared_gate_config: Final[GateConfig | None]
    stack_gate_flag: Final[bool]
    stack_halting_flag: Final[bool]


class _MemoryOptions(Protocol):
    memory_flag: Final[bool]


def separate_self_attention(
    *,
    batch_size: int,
    num_heads: int,
    hidden_dim: int,
    sequence_length: int,
    dropout_probability: float,
    causal_attention_mask_flag: bool,
    add_key_value_bias_flag: bool,
    source_memory_flag: bool,
    source_memory_config: AttentionSourceMemoryConfig | None,
    projection_model_config: ConfigBase,
) -> SelfAttentionConfig:
    return SelfAttentionConfig(
        batch_size=batch_size,
        num_heads=num_heads,
        embedding_dim=hidden_dim,
        query_key_projection_dim=hidden_dim,
        value_projection_dim=hidden_dim,
        target_sequence_length=sequence_length,
        source_sequence_length=sequence_length,
        target_dtype=torch.float32,
        dropout_probability=dropout_probability,
        zero_attention_flag=False,
        causal_attention_mask_flag=causal_attention_mask_flag,
        add_key_value_bias_flag=add_key_value_bias_flag,
        source_memory_flag=source_memory_flag,
        source_memory_config=source_memory_config,
        average_attention_weights_flag=False,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=cast(
            LayerStackConfig | RecurrentCompositionConfig, projection_model_config
        ),
        projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
    )


def feed_forward_config(
    stack_config: ConfigBase,
    soft_moe_config: SoftMoEConfig | None,
    hidden_dim: int,
) -> FeedForwardConfig:
    configured = build_soft_moe_stack(stack_config, soft_moe_config, hidden_dim)
    return FeedForwardConfig(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        stack_config=cast(
            LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentCompositionConfig,
            configured,
        ),
    )


def feed_forward_control_stack(stack_config: ConfigBase) -> LayerStackConfig:
    if isinstance(stack_config, LayerStackConfig):
        return stack_config
    if isinstance(stack_config, MixtureOfExpertsModelConfig):
        return cast(LayerStackConfig, stack_config.stack_config)
    raise TypeError(
        "Feed-forward controls require a LayerStackConfig or "
        f"MixtureOfExpertsModelConfig, got {type(stack_config).__name__}."
    )


def apply_selected_controls(
    stack: LayerStackConfig,
    options: _LayerOptions | None,
    gate: _Gating | None,
    halting: _Halting | None,
    memory: _Memory | None,
) -> None:
    if gate is not None:
        cast(LayerConfig, stack.layer_config).gate_config = gate.build_gate_config()
    if halting is not None:
        cast(
            LayerConfig, stack.layer_config
        ).halting_config = halting.build_halting_config()
    if options is not None:
        stack.shared_gate_config = options.shared_gate_config
    if memory is not None:
        stack.shared_memory_config = memory.build_memory_config()


def apply_enabled_controls(
    stack: LayerStackConfig,
    options: _LayerOptions | None,
    memory: _MemoryOptions | None,
    gate_factory: _Gating | None,
    halting_factory: _Halting | None,
    memory_factory: _Memory | None,
) -> None:
    if gate_factory is not None and options is not None and options.stack_gate_flag:
        gate = gate_factory.build_gate_config()
        cast(LayerConfig, stack.layer_config).gate_config = gate
    if (
        halting_factory is not None
        and options is not None
        and options.stack_halting_flag
    ):
        halting = halting_factory.build_halting_config()
        cast(LayerConfig, stack.layer_config).halting_config = halting
    if options is not None and options.shared_gate_config is not None:
        stack.shared_gate_config = options.shared_gate_config
    if memory_factory is not None and memory is not None and memory.memory_flag:
        stack.shared_memory_config = memory_factory.build_memory_config()
