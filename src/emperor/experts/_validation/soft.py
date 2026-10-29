"""Contracts for sequence-local Soft MoE and its pointwise expert stacks."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import (
    ValidatorBase,
    _adaptive_grouping_paths,  # pyright: ignore[reportPrivateUsage]
    _configuration_visits,  # pyright: ignore[reportPrivateUsage]
)
from emperor.experts._config import SoftMoEConfig
from emperor.experts._options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

if TYPE_CHECKING:
    from emperor.experts._layers.mixture import MixtureOfExperts
    from emperor.experts._routing.soft import SoftMoERouter


class SoftMoEValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "SoftMoERouter") -> None:
        cls.validate_config(model.cfg)

    @staticmethod
    def validate_config_type(cfg: object) -> None:
        if not isinstance(cfg, SoftMoEConfig):
            raise TypeError("soft_moe_config must be a SoftMoEConfig.")

    @staticmethod
    def validate_overrides_type(overrides: object) -> None:
        if overrides is not None and not isinstance(overrides, SoftMoEConfig):
            raise TypeError("Soft MoE overrides must be a SoftMoEConfig or None.")

    @classmethod
    def validate_config(cls, cfg: SoftMoEConfig) -> None:
        cls.validate_config_type(cfg)
        for name in ("input_dim", "num_experts", "slots_per_expert"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Soft MoE {name} must be a positive integer.")
        cls.validate_scoring_stack(cfg)

    @classmethod
    def validate_scoring_stack(cls, cfg: SoftMoEConfig) -> None:
        from emperor.augmentations.adaptive_parameters import DepthMappingHandlerConfig
        from emperor.layers import LayerConfig, LayerStackConfig
        from emperor.linears import LinearLayerConfig

        slot_logit_config = cfg.model_config
        if not isinstance(slot_logit_config, DepthMappingHandlerConfig):
            raise TypeError(
                "Soft MoE requires model_config=DepthMappingHandlerConfig(...)."
            )
        for name, expected in (
            ("input_dim", cfg.input_dim),
            ("output_dim", cfg.slots_per_expert),
            ("generator_depth", cfg.num_experts),
        ):
            supplied = getattr(slot_logit_config, name)
            if supplied is not None and (
                type(supplied) is not int or supplied != expected
            ):
                raise ValueError(f"Soft MoE scoring {name} must match the router.")
        stack = slot_logit_config.model_config
        if type(stack) is not LayerStackConfig:
            raise TypeError("Soft MoE scoring requires an ordinary LayerStackConfig.")
        if (
            type(stack.layer_config) is not LayerConfig
            or type(stack.layer_config.layer_model_config) is not LinearLayerConfig
        ):
            raise TypeError(
                "Soft MoE scoring requires ordinary LayerConfig and LinearLayerConfig."
            )
        for name, expected in (
            ("input_dim", cfg.input_dim),
            ("output_dim", cfg.slots_per_expert),
        ):
            supplied = getattr(stack, name)
            if supplied is not None and (
                isinstance(supplied, bool)
                or not isinstance(supplied, int)
                or supplied != expected
            ):
                raise ValueError(f"Soft MoE scoring {name} must match the router.")
        if stack.apply_output_postprocessing_flag is not False:
            raise ValueError(
                "Soft MoE scoring requires apply_output_postprocessing_flag=False "
                "for raw logits."
            )
        cls.validate_no_controllers(stack)

    @classmethod
    def validate_mixture(cls, model: "MixtureOfExperts") -> None:
        cfg = model.cfg
        if not isinstance(cfg.soft_moe_config, SoftMoEConfig):
            raise TypeError("soft_moe_config must be a SoftMoEConfig.")
        for name in ("input_dim", "num_experts"):
            supplied = getattr(cfg.soft_moe_config, name)
            if supplied is not None and (
                isinstance(supplied, bool)
                or not isinstance(supplied, int)
                or supplied <= 0
            ):
                raise ValueError(f"Soft MoE {name} must be a positive integer.")
            if supplied is not None and supplied != getattr(model, name):
                raise ValueError(f"Soft MoE {name} must match the mixture owner.")
        cls.validate_config(
            replace(
                cfg.soft_moe_config,
                input_dim=model.input_dim,
                num_experts=model.num_experts,
            )
        )
        if cfg.routing_initialization_mode != RoutingInitializationMode.LAYER:
            raise ValueError(
                "Soft MoE requires LAYER-owned routing; shared routes and Map/Reduce are unsupported."
            )
        if cfg.sampler_config is not None:
            raise ValueError(
                "Soft MoE requires sampler_config=None; it computes both routing distributions itself."
            )
        if cfg.top_k != cfg.num_experts or cfg.capacity_factor != 0:
            raise ValueError(
                "Soft MoE requires top_k=num_experts and capacity_factor=0; every slot is processed."
            )
        if cfg.dropped_token_behavior not in (None, DroppedTokenOptions.ZEROS):
            raise ValueError(
                "Soft MoE does not support dropped-token identity fallback."
            )
        if not cfg.compute_expert_mixture_flag or not cfg.weighted_parameters_flag:
            raise ValueError("Soft MoE requires weighted, reduced expert outputs.")
        if (
            cfg.weighting_position_option
            != ExpertWeightingPositionOptions.AFTER_EXPERTS
        ):
            raise ValueError(
                "Soft MoE requires AFTER_EXPERTS; dispatch already weights expert inputs."
            )
        cls.validate_expert_stack(cfg.expert_model_config)

    @classmethod
    def validate_expert_stack(cls, config: object) -> None:
        from emperor.layers import LayerConfig, LayerStackConfig
        from emperor.linears import LinearLayerConfig

        if (
            not isinstance(config, LayerStackConfig)
            or type(config.layer_config) is not LayerConfig
        ):
            raise ValueError(
                "Soft MoE requires a pointwise LayerStackConfig with ordinary LayerConfig experts."
            )
        if not isinstance(config.layer_config.layer_model_config, LinearLayerConfig):
            raise ValueError(
                "Soft MoE expert layers must be pointwise linear networks."
            )
        if _adaptive_grouping_paths(config, root="SoftMoE.experts"):
            raise ValueError(
                "Soft MoE experts cannot share adaptive grouping context across slots or samples."
            )
        cls.validate_no_controllers(config)

    @staticmethod
    def validate_no_controllers(config: object) -> None:
        for visit in _configuration_visits(config, "SoftMoE"):
            for name in (
                "gate_config",
                "halting_config",
                "memory_config",
                "shared_gate_config",
                "shared_halting_config",
                "shared_memory_config",
            ):
                if getattr(visit.config, name, None) is not None:
                    raise ValueError(
                        f"Soft MoE does not support context controllers at {visit.path}.{name}."
                    )

    @staticmethod
    def validate_inputs(inputs: object, input_dim: int, skip_mask: object) -> None:
        if not isinstance(inputs, Tensor) or inputs.ndim != 3:
            raise ValueError(
                "Soft MoE requires [batch, tokens, features]; flat rows cannot establish sequence boundaries."
            )
        if (
            inputs.shape[-1] != input_dim
            or inputs.shape[0] == 0
            or inputs.shape[1] == 0
        ):
            raise ValueError(
                "Soft MoE requires nonempty batch/tokens and the configured feature dimension."
            )
        if not inputs.is_floating_point():
            raise TypeError("Soft MoE inputs must be floating point.")
        if skip_mask is not None:
            if not isinstance(skip_mask, Tensor) or skip_mask.dtype != torch.bool:
                raise TypeError(
                    "Soft MoE skip_mask must be Boolean (True means active)."
                )
            if (
                skip_mask.shape != (*inputs.shape[:2], 1)
                or skip_mask.device != inputs.device
            ):
                raise ValueError(
                    "Soft MoE skip_mask must have shape [batch, tokens, 1] on the input device."
                )

    @staticmethod
    def validate_parameter_device(inputs: Tensor, parameter: Tensor) -> None:
        if inputs.device != parameter.device:
            raise ValueError(
                "Soft MoE inputs and routing parameters must be on the same device."
            )
