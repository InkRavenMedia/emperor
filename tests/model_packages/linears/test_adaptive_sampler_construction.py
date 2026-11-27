from unittest.mock import patch

import pytest

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    DualModelDynamicWeightConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
)
from emperor.sampler import SamplerConfig
from models import _adaptive_parameters
from models.catalog import model_package


@pytest.mark.parametrize(
    ("overrides", "expected_expert_counts"),
    [
        ({}, []),
        (
            {
                "weight_option_flag": True,
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option_flag": True,
                "bias_option": AdditiveDynamicBiasConfig,
                "input_layer_weight_option": DualModelDynamicWeightConfig,
                "output_layer_bias_option": AdditiveDynamicBiasConfig,
            },
            [],
        ),
        (
            {
                "weight_option_flag": True,
                "weight_option": MatrixWeightsMixtureConfig,
                "weight_mixture_num_experts": 3,
                "input_layer_bias_option": MatrixBiasMixtureConfig,
                "input_layer_bias_mixture_num_experts": 5,
                "output_layer_weight_option": MatrixWeightsMixtureConfig,
                "output_layer_weight_mixture_num_experts": 7,
            },
            [5],
        ),
    ],
)
def test_only_sampled_bias_mixtures_build_a_sampler(overrides, expected_expert_counts):
    package = model_package("linears/linear_adaptive")
    with patch.object(
        _adaptive_parameters, "SamplerConfig", wraps=SamplerConfig
    ) as make_sampler:
        package.build_configuration(config_overrides=overrides)

    assert make_sampler.call_count == len(expected_expert_counts)
    built_expert_counts = [
        call.kwargs["num_experts"] for call in make_sampler.call_args_list
    ]
    assert sorted(built_expert_counts) == expected_expert_counts
