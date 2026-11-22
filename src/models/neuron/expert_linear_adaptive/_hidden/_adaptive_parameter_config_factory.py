from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import LayerStackConfig
from models import _adaptive_parameters
from models._adaptive_parameters import (
    build_bias_config as build_bias_config,
)
from models._adaptive_parameters import (
    build_diagonal_config as build_diagonal_config,
)
from models._adaptive_parameters import (
    build_mask_config as build_mask_config,
)
from models.neuron.expert_linear_adaptive._generation import WeightGenerationConfig


def build_weight_config(
    weight_option: type[DynamicWeightConfig] | None,
    *,
    generation: WeightGenerationConfig | None = None,
    generator_depth: int,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    normalization_option: WeightNormalizationOptions,
    normalization_learnable_scale_flag: bool = False,
    normalization_position_option: WeightNormalizationPositionOptions,
    bank_expansion_factor: BankExpansionFactorOptions,
    model_config: LayerStackConfig | None = None,
) -> DynamicWeightConfig | None:
    if (
        weight_option is DiagonallyModulatedLowRankDynamicWeightConfig
        and generation is None
    ):
        raise KeyError("input_factor_source")
    return _adaptive_parameters.build_weight_config(
        weight_option,
        generation=generation,
        generator_depth=generator_depth,
        decay_schedule=decay_schedule,
        decay_rate=decay_rate,
        decay_warmup_batches=decay_warmup_batches,
        normalization_option=normalization_option,
        normalization_learnable_scale_flag=normalization_learnable_scale_flag,
        normalization_position_option=normalization_position_option,
        bank_expansion_factor=bank_expansion_factor,
        model_config=model_config,
    )
