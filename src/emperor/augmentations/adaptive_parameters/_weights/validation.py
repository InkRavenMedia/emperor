from dataclasses import fields
from typing import TYPE_CHECKING

from torch.types import Tensor

from emperor._validation import ValidatorBase
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._weights.base import (
        DynamicWeightAbstract,
    )
    from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
        DepthMappingLayer,
    )
    from emperor.augmentations.adaptive_parameters._weights.normalization import (
        WeightNormalizationPolicy,
    )
    from emperor.layers import LayerStackConfig


class WeightNormalizationValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "WeightNormalizationPolicy") -> None:
        cls.validate_config(model.cfg)

    @classmethod
    def validate_config(cls, cfg) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationOptions,
        )

        from .config import WeightNormalizationConfig

        if not isinstance(cfg, WeightNormalizationConfig):
            raise TypeError(
                "WeightNormalizationPolicy requires a WeightNormalizationConfig."
            )
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        DepthMappingValidator.validate_generator_depth(cfg.generator_depth)
        cls.validate_normalization_option(cfg.normalization_option)
        cls.validate_normalization_position_option(cfg.normalization_position_option)
        cls.validate_normalization_learnable_scale_flag(
            cfg.normalization_learnable_scale_flag
        )
        cls.validate_learnable_scale_position(cfg)
        if cfg.normalization_option is WeightNormalizationOptions.DISABLED:
            raise ValueError(
                "Use normalization_config=None to disable normalization; "
                "a supplied config requires an enabled normalization_option."
            )

    @staticmethod
    def validate_learnable_scale_position(cfg) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationPositionOptions,
        )

        if (
            cfg.normalization_position_option
            is WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
            and cfg.normalization_learnable_scale_flag
        ):
            raise ValueError(
                "normalization_learnable_scale_flag must be False for AFTER_DEPTH_SUM; "
                "learnable normalization scales apply only before the depth sum."
            )

    @staticmethod
    def validate_depth_axis(
        vectors: Tensor, generator_depth: int, depth_axis: int
    ) -> None:
        if vectors.ndim < 2 or vectors.shape[depth_axis] != generator_depth:
            raise ValueError(
                f"Normalization expects {generator_depth} depths on axis {depth_axis}, "
                f"received shape {tuple(vectors.shape)}."
            )

    @staticmethod
    def validate_normalization_learnable_scale_flag(flag) -> None:
        if type(flag) is not bool:
            raise ValueError(
                f"Unsupported normalization_learnable_scale_flag value: {flag!r}."
            )

    @staticmethod
    def validate_normalization_option(normalization_option) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationOptions,
        )

        if not isinstance(normalization_option, WeightNormalizationOptions):
            raise ValueError(
                f"Unsupported normalization_option value: {normalization_option!r}."
            )

    @staticmethod
    def validate_normalization_position_option(normalization_position_option) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationPositionOptions,
        )

        if not isinstance(
            normalization_position_option, WeightNormalizationPositionOptions
        ):
            raise ValueError(
                "Unsupported normalization_position_option value: "
                f"{normalization_position_option!r}."
            )


class DynamicWeightValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"bank_expansion_factor", "normalization_config"}

    @classmethod
    def validate_generator_stack(cls, config):
        from emperor.layers import Layer, LayerStack, LayerStackConfig
        from emperor.linears import LinearLayer

        if not isinstance(config, LayerStackConfig):
            raise TypeError("Active generator must use a LayerStackConfig.")
        if config.registry_owner() is not LayerStack:
            raise ValueError("Active generator must use an ordinary LayerStack.")
        layer = config.layer_config
        if layer is None or layer.registry_owner() is not Layer:
            raise ValueError("Active generator must use ordinary feed-forward Layers.")
        if (
            layer.layer_model_config is None
            or layer.layer_model_config.registry_owner() is not LinearLayer
        ):
            raise ValueError("Active generator must wrap ordinary LinearLayer configs.")
        for owner in (config, layer):
            for name in (
                "gate_config",
                "halting_config",
                "memory_config",
                "shared_gate_config",
                "shared_halting_config",
                "shared_memory_config",
            ):
                if getattr(owner, name, None) is not None:
                    raise ValueError(f"Active generator does not support {name}.")
        if config.hidden_dim is not None and (
            type(config.hidden_dim) is not int or config.hidden_dim <= 0
        ):
            raise ValueError("Generator hidden_dim must be a positive integer.")

    @classmethod
    def validate(cls, model: "DynamicWeightAbstract") -> None:
        cls.validate_initialization_fields(model)
        cls.validate_variant_config(model)
        cls.validate_model_config(model.cfg)

    @classmethod
    def validate_initialization_fields(cls, model: "DynamicWeightAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        DepthMappingValidator.validate_generator_depth(model.cfg.generator_depth)
        cls.validate_decay_parameters(model.cfg)
        cls.validate_normalization_config(model.cfg)

    @staticmethod
    def validate_normalization_config(cfg) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationPositionOptions,
        )

        from .config import (
            BroadcastSumDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
            LayeredWeightedBankDynamicWeightConfig,
            LowRankDynamicWeightConfig,
            MatrixWeightsMixtureConfig,
            SoftWeightedBankDynamicWeightConfig,
            WeightNormalizationConfig,
        )

        normalization = cfg.normalization_config
        if normalization is None:
            if isinstance(cfg, BroadcastSumDynamicWeightConfig):
                raise ValueError(
                    "BroadcastSumDynamicWeight requires normalization_config "
                    "to normalize each depth matrix before summing."
                )
            return
        if not isinstance(normalization, WeightNormalizationConfig):
            raise TypeError(
                "normalization_config must be a WeightNormalizationConfig or None."
            )
        if isinstance(
            cfg,
            (
                LayeredWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeightConfig,
                MatrixWeightsMixtureConfig,
            ),
        ):
            raise ValueError(
                f"{type(cfg).__name__} does not support normalization_config "
                f"(normalization_option={normalization.normalization_option!r})."
            )
        if isinstance(
            cfg,
            (
                BroadcastSumDynamicWeightConfig,
                LowRankDynamicWeightConfig,
                HypernetworkDynamicWeightConfig,
            ),
        ):
            position = normalization.normalization_position_option
            if (
                position is not None
                and position is not WeightNormalizationPositionOptions.DISABLED
            ):
                raise ValueError(
                    f"{type(cfg).__name__} uses a fixed normalization stage; "
                    "normalization_position_option must be DISABLED."
                )

    @classmethod
    def validate_variant_config(cls, model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )
        from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
            SingleModelDynamicWeight,
        )
        from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        if isinstance(model, SingleModelDynamicWeight):
            cls.validate_square_dimensions(model)
        if isinstance(
            model,
            (
                LayeredWeightedBankDynamicWeight,
                SoftWeightedBankDynamicWeight,
            ),
        ):
            cls.validate_bank_expansion_factor(model)

    @staticmethod
    def validate_bank_expansion_factor(model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            BankExpansionFactorOptions,
        )

        factor = model.cfg.bank_expansion_factor
        if factor is None or not isinstance(factor, BankExpansionFactorOptions):
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor to be a "
                f"BankExpansionFactorOptions value, received {factor!r}."
            )
        if factor == BankExpansionFactorOptions.DISABLED:
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor > 0, "
                f"received {factor}. "
                f"Use FACTOR_OF_ONE, FACTOR_OF_TWO, FACTOR_OF_THREE, or "
                f"FACTOR_OF_FOUR."
            )

    @staticmethod
    def validate_square_dimensions(model: "DynamicWeightAbstract") -> None:
        if model.cfg.input_dim != model.cfg.output_dim:
            raise ValueError(
                f"{type(model).__name__} requires input_dim == output_dim, "
                f"received input_dim={model.cfg.input_dim}, "
                f"output_dim={model.cfg.output_dim}."
            )


class ModulatedLowRankValidator(DynamicWeightValidator):
    OPTIONAL_FIELDS = {
        "normalization_config",
        "model_config",
        "input_factor_model_config",
        "output_factor_model_config",
    }

    @staticmethod
    def validate_supported_fields(cfg) -> None:
        if vars(cfg).keys() - {field.name for field in fields(cfg)} - {"_passed_args"}:
            raise ValueError("Modulated low-rank config contains unsupported fields.")

    @classmethod
    def validate(cls, model) -> None:
        cls.validate_supported_fields(model.cfg)
        cls.validate_initialization_fields(model)
        cls.validate_active_models(model.cfg)

    @classmethod
    def validate_active_models(cls, cfg, fallback=None) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            LowRankFactorSourceOptions,
        )

        default_model = cfg.model_config if cfg.model_config is not None else fallback
        for side in ("input", "output"):
            source = getattr(cfg, f"{side}_factor_source")
            if source is None:
                source = LowRankFactorSourceOptions.GENERATED
            if not isinstance(source, LowRankFactorSourceOptions):
                raise TypeError(
                    f"{side}_factor_source must be a LowRankFactorSourceOptions."
                )
            override = getattr(cfg, f"{side}_factor_model_config")
            if source is LowRankFactorSourceOptions.SHARED_PARAMETER:
                if override is not None:
                    raise ValueError(
                        f"{side}_factor_model_config is invalid for SHARED_PARAMETER."
                    )
            else:
                cls.validate_generator_stack(
                    override if override is not None else default_model
                )
        cls.validate_generator_stack(
            cfg.coefficient_model_config
            if cfg.coefficient_model_config is not None
            else default_model
        )


class DepthMappingValidator(ValidatorBase):
    OPTIONAL_FIELDS: set[str] = set()

    @classmethod
    def validate(cls, model: "DepthMappingLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls.validate_generator_depth(model.cfg.generator_depth)

    @staticmethod
    def validate_generator_depth(generator_depth: object) -> None:
        if type(generator_depth) is not int or generator_depth <= 0:
            raise ValueError(
                f"generator_depth must be a positive integer, received {generator_depth}."
            )

    @staticmethod
    def validate_input_is_2d(
        input_batch: Tensor,
        input_dim: int | None = None,
    ) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "DepthMappingLayerStack input must be a Tensor, "
                f"received {type(input_batch).__name__}."
            )
        if not input_batch.dim() == 2:
            raise ValueError(
                "DepthMappingLayerStack expects a 2D input tensor "
                "(batch_size, features), "
                f"received a {input_batch.dim()}D tensor with shape "
                f"{tuple(input_batch.shape)}."
            )
        if input_dim is not None and input_batch.shape[-1] != input_dim:
            raise ValueError(
                "DepthMappingLayerStack input feature dimension must match "
                "input_dim, "
                f"received input_dim={input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_inner_model_is_linear_layer_config(
        model_config: "LayerStackConfig",
    ) -> None:
        from emperor.linears import LinearLayerConfig

        inner_config = model_config.layer_config.layer_model_config
        if not isinstance(inner_config, LinearLayerConfig):
            raise TypeError(
                "model_config.layer_config.layer_model_config must be a "
                "LinearLayerConfig, "
                f"received {type(inner_config).__name__}."
            )

    @staticmethod
    def validate_layer_config_has_no_gate_or_halting(
        model_config: "LayerStackConfig",
    ) -> None:
        layer_config = model_config.layer_config
        if layer_config.gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support gate_config. "
                "Set gate_config to None."
            )
        if model_config.shared_gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_gate_config. "
                "Set shared_gate_config to None."
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support halting_config. "
                "Set halting_config to None."
            )
        if model_config.shared_halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_halting_config. "
                "Set shared_halting_config to None."
            )
        if layer_config.memory_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support memory_config. "
                "Set memory_config to None."
            )
        if model_config.shared_memory_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_memory_config. "
                "Set shared_memory_config to None."
            )


class MatrixWeightsMixtureValidator(DynamicWeightValidator):
    @staticmethod
    def validate_matching_value(name, value, expected):
        if value is not None and (
            type(value) is not type(expected) or value != expected
        ):
            raise ValueError(f"{name} must match {expected!r}, received {value!r}.")

    @classmethod
    def validate(cls, model) -> None:
        cls.validate_initialization_config(model.cfg)

    @classmethod
    def validate_initialization_config(cls, cfg) -> None:
        if vars(cfg).keys() - {field.name for field in fields(cfg)} - {"_passed_args"}:
            raise ValueError(
                "Complete parameter bank config contains unsupported fields."
            )
        for name in ("input_dim", "output_dim", "num_experts"):
            value = getattr(cfg, name)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, received {value!r}."
                )
        cls.validate_field_types(cfg)
        cls.validate_normalization_config(cfg)
        cls.validate_decay_parameters(cfg)
        cls.validate_generator_stack(cfg.model_config)

    @classmethod
    def validate_forward_inputs(cls, model, weight_params, context):
        if (
            not isinstance(context, Tensor)
            or context.ndim != 2
            or context.shape[-1] != model.input_dim
        ):
            raise ValueError(
                "Matrix mixture context must have shape [contexts, input_dim]."
            )
        cls.validate_batched_weight_params(model, weight_params, context)
