from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field
from emperor.experts._options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.layers import LayerConfig

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters import DepthMappingHandlerConfig
    from emperor.experts._layers.mixture import MixtureOfExperts
    from emperor.layers import LayerStackConfig, RecurrentCompositionConfig
    from emperor.sampler import SamplerConfig


@dataclass
class SoftMoEConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Routing feature dimension; supplied by the mixture owner."
    )
    num_experts: int | None = optional_field(
        "Expert count; supplied by the mixture owner."
    )
    slots_per_expert: int | None = optional_field(
        "Positive, fixed number of slots processed by each expert."
    )
    model_config: "DepthMappingHandlerConfig | None" = optional_field(
        "Required depth-mapped scoring stack; input/output dimensions are "
        "supplied by the router along with the expert count. "
        "Its inner stack's final layer must produce raw logits."
    )

    def _registry_owner(self) -> type:
        from emperor.experts._routing.soft import SoftMoERouter

        return SoftMoERouter


@dataclass
class MixtureOfExpertsModelConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input feature dimension for the mixture-of-experts model."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension for the mixture-of-experts model."
    )
    top_k: int | None = optional_field(
        "Top-k probabilities and indices selected from the routing distribution."
    )
    routing_initialization_mode: RoutingInitializationMode | None = optional_field(
        "SHARED for a single sampler/router across all expert layers; "
        "LAYER for one per layer."
    )
    sampler_config: "SamplerConfig | None" = optional_field(
        "Sampler configuration used when routing is SHARED. The router lives "
        "at sampler_config.router_config."
    )
    stack_config: "LayerStackConfig | None" = optional_field(
        "Stack of expert layers built and orchestrated by MixtureOfExpertsModel."
    )

    def _registry_owner(self) -> type:
        from emperor.experts._model import MixtureOfExpertsModel

        return MixtureOfExpertsModel


@dataclass
class MixtureOfExpertsConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension for each expert.")
    output_dim: int | None = optional_field("Output feature dimension for each expert.")
    top_k: int | None = optional_field(
        "Top-k probabilities and indices to be selected from a distribution"
    )
    num_experts: int | None = optional_field("Number of experts in the model")
    capacity_factor: float | None = optional_field(
        "Limits tokens per expert to prevent load imbalance. Tokens over capacity "
        "are dropped. 0.0=disabled, 1.0=fair share, >1.0=buffer."
    )
    dropped_token_behavior: DroppedTokenOptions | None = optional_field(
        "Controls dropped tokens. ZERO: become zero vectors (default). IDENTITY: "
        "retain original input."
    )
    compute_expert_mixture_flag: bool | None = optional_field(
        "When true computes the expert mixture for this layer."
    )
    weighted_parameters_flag: bool | None = optional_field(
        "When `True` the sepected parameters will be multiplied by their probs."
    )
    weighting_position_option: ExpertWeightingPositionOptions | None = optional_field(
        "Dictates if the weights are applided before or after the experts."
    )
    routing_initialization_mode: RoutingInitializationMode | None = optional_field(
        "Use `SHARED` for a single router and sampler across all layers, or "
        "`LAYER` for one per layer."
    )
    sampler_config: "SamplerConfig | None" = optional_field(
        "Sampler configuration used when the layer owns its sampler. The router "
        "lives at sampler_config.router_config."
    )
    expert_model_config: "LayerStackConfig | RecurrentCompositionConfig | None" = (
        optional_field("Expert stack configuration used to build each expert.")
    )
    soft_moe_config: SoftMoEConfig | None = optional_field(
        "Optional noncausal Soft MoE execution on [batch, tokens, features]. "
        "Requires LAYER routing, all experts, no sampler or capacity dropping, "
        "and weighted, reduced outputs. None preserves token routing."
    )

    def _registry_owner(self) -> type:
        from emperor.experts._layers.mixture import MixtureOfExperts

        return MixtureOfExperts


@dataclass
class _MixtureOfExpertsMapConfig(ConfigBase):
    mixture_config: MixtureOfExpertsConfig

    def build(
        self,
        overrides: MixtureOfExpertsConfig | None = None,
    ) -> "MixtureOfExperts":
        from emperor.experts._layers.map import MixtureOfExpertsMap

        return MixtureOfExpertsMap(self.mixture_config, overrides)


@dataclass
class _MixtureOfExpertsReduceConfig(ConfigBase):
    mixture_config: MixtureOfExpertsConfig

    def build(
        self,
        overrides: MixtureOfExpertsConfig | None = None,
    ) -> "MixtureOfExperts":
        from emperor.experts._layers.reduce import MixtureOfExpertsReduce

        return MixtureOfExpertsReduce(self.mixture_config, overrides)


@dataclass
class MixtureOfExpertsLayerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.experts._layers.layer import MixtureOfExpertsLayer

        return MixtureOfExpertsLayer
