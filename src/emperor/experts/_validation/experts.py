"""Output contracts for the expert execution container."""

from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import _adaptive_grouping_paths

if TYPE_CHECKING:
    from emperor.experts._layers.experts import Experts


class ExpertsValidator:
    @classmethod
    def validate_output_rows(
        cls, experts: "Experts", expert_samples: Tensor, expert_output: Tensor
    ) -> None:
        if not cls.__requires_one_output_row_per_sample(experts):
            return
        expected_output_shape = (expert_samples.size(0), experts.output_dim)
        if cls.__has_expected_output_shape(expert_output, expected_output_shape):
            return
        expert_owner = cls.__resolve_expert_owner(experts)
        raise ValueError(
            f"{expert_owner} expert must return one row per retained assignment: "
            f"expected {expected_output_shape}."
        )

    @staticmethod
    def __requires_one_output_row_per_sample(experts: "Experts") -> bool:
        if experts.cfg.soft_moe_config is not None:
            return True
        grouping_paths = _adaptive_grouping_paths(
            experts.expert_model_config,
            root="MixtureOfExpertsConfig.expert_model_config",
        )
        return bool(grouping_paths)

    @staticmethod
    def __has_expected_output_shape(
        expert_output: Tensor, expected_output_shape: tuple[int, int]
    ) -> bool:
        if not isinstance(expert_output, Tensor):
            return False
        actual_output_shape = tuple(expert_output.shape)
        return actual_output_shape == expected_output_shape

    @staticmethod
    def __resolve_expert_owner(experts: "Experts") -> str:
        if experts.cfg.soft_moe_config is not None:
            return "Soft MoE"
        return "Grouped"
