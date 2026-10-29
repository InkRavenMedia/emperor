"""Small real configurations for parameter generation acceptance tests."""

from emperor.augmentations.adaptive_parameters import (
    DiagonallyModulatedLowRankDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
)
from emperor.sampler import RouterConfig, SamplerConfig
from support.adaptive_grouping import linear_stack_config


def matrix_sampler_config(*, input_dim=2, num_experts=3, top_k=2, **overrides):
    return SamplerConfig(
        num_experts=num_experts,
        top_k=top_k,
        **(
            dict(
                threshold=0.0,
                filter_above_threshold=False,
                num_topk_samples=0,
                noisy_topk_flag=False,
                normalize_probabilities_flag=top_k > 1,
                coefficient_of_variation_loss_weight=0.0,
                switch_loss_weight=0.0,
                zero_centred_loss_weight=0.0,
                mutual_information_loss_weight=0.0,
                router_config=RouterConfig(
                    input_dim=input_dim,
                    num_experts=num_experts,
                    noisy_topk_flag=overrides.get("noisy_topk_flag", False),
                    model_config=linear_stack_config(input_dim, num_experts),
                ),
            )
            | overrides
        ),
    )


def weight_mixture_config(**overrides):
    from emperor.augmentations.adaptive_parameters import MatrixWeightsMixtureConfig

    return MatrixWeightsMixtureConfig(
        **(
            dict(
                input_dim=2,
                output_dim=3,
                num_experts=3,
                top_k=2,
                sampler_config=matrix_sampler_config(
                    input_dim=overrides.get("input_dim", 2),
                    num_experts=overrides.get("num_experts", 3),
                    top_k=overrides.get("top_k") or 2,
                ),
                model_config=linear_stack_config(2, 3),
            )
            | overrides
        )
    )


def bias_mixture_config(**overrides):
    from emperor.augmentations.adaptive_parameters import MatrixBiasMixtureConfig

    return MatrixBiasMixtureConfig(
        **(
            dict(
                input_dim=2,
                output_dim=3,
                num_experts=3,
                top_k=2,
                sampler_config=matrix_sampler_config(
                    input_dim=overrides.get("input_dim", 2),
                    num_experts=overrides.get("num_experts", 3),
                    top_k=overrides.get("top_k") or 2,
                ),
                model_config=linear_stack_config(2, 3),
            )
            | overrides
        )
    )


def modulated_config(**overrides):
    values = dict(
        input_dim=3,
        output_dim=2,
        generator_depth=2,
        normalization_option=WeightNormalizationOptions.DISABLED,
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=linear_stack_config(3, 2),
    )
    return DiagonallyModulatedLowRankDynamicWeightConfig(**(values | overrides))
