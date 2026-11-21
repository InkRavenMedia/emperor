from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from emperor._validation import ValidatorBase
from emperor.layers._options import (
    ActivationOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingInterface, HaltingStateBase
    from emperor.layers._composition.gate import LayerGate
    from emperor.layers._composition.residual.base import ResidualConnectionAbstract
    from emperor.layers._pipeline.delegates import (
        LayerHaltingDelegate,
        LayerMemoryDelegate,
        LayerNormalizationDelegate,
        LayerPostprocessingDelegate,
        LayerResidualDelegate,
    )
    from emperor.layers._pipeline.delegates.residual_schedule import (
        DepthwiseRecurrentResidualSchedule,
        RecurrentResidualSchedule,
    )
    from emperor.memory import MemoryInterface
    from emperor.nn import Module


def _implements_halting_interface(
    model: object,
) -> TypeGuard[HaltingInterface[HaltingStateBase]]:
    return callable(getattr(model, "update_halting_state", None)) and callable(
        getattr(model, "finalize_weighted_accumulation", None)
    )


def _implements_memory_interface(model: object) -> TypeGuard[MemoryInterface]:
    return callable(model) and hasattr(model, "memory_position_option")


class LayerHaltingDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerHaltingDelegate) -> None:
        cls.validate_resolved_output_dim(model.cfg.output_dim)

    @staticmethod
    def validate_resolved_output_dim(output_dim: int | None) -> None:
        if output_dim is None:
            raise ValueError("Layer halting requires a resolved output_dim.")

    @staticmethod
    def validate_built_halting_model_interface(
        model: Module | None,
    ) -> HaltingInterface[HaltingStateBase] | None:
        if model is None:
            return None
        if not _implements_halting_interface(model):
            raise TypeError(
                "halting_config must build a model implementing HaltingInterface."
            )
        return model


class LayerMemoryDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerMemoryDelegate) -> None:
        cls.validate_resolved_dimensions(
            model.cfg.input_dim,
            model.cfg.output_dim,
        )

    @staticmethod
    def validate_resolved_dimensions(
        input_dim: int | None,
        output_dim: int | None,
    ) -> None:
        if input_dim is None or output_dim is None:
            raise ValueError(
                "Layer memory requires resolved input and output dimensions."
            )

    @staticmethod
    def validate_built_memory_model_interface(
        model: Module | None,
    ) -> MemoryInterface | None:
        if model is None:
            return None
        if not _implements_memory_interface(model):
            raise TypeError(
                "memory_config must build a model implementing MemoryInterface."
            )
        return model


class LayerNormalizationDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerNormalizationDelegate) -> None:
        cls.validate_normalization_option(model.cfg.normalization)
        cls.validate_resolved_position(model.cfg.layer_norm_position)
        cls.validate_resolved_dimensions(
            model.cfg.input_dim,
            model.cfg.output_dim,
        )

    @staticmethod
    def validate_normalization_option(
        normalization: NormalizationOptions | None,
    ) -> None:
        if normalization is not None and not isinstance(
            normalization, NormalizationOptions
        ):
            raise TypeError(
                "normalization must be a NormalizationOptions value or None, "
                f"got {type(normalization).__name__}."
            )

    @staticmethod
    def validate_resolved_position(
        position: LayerNormPositionOptions | None,
    ) -> None:
        if position is None:
            raise ValueError("Layer normalization requires a resolved position.")

    @staticmethod
    def validate_resolved_dimensions(
        input_dim: int | None,
        output_dim: int | None,
    ) -> None:
        if input_dim is None or output_dim is None:
            raise ValueError(
                "Layer normalization requires resolved input and output dimensions."
            )


class ElementwiseNormalizationValidator(ValidatorBase):
    @staticmethod
    def validate_forward_input(hidden: Tensor, dimension: int) -> None:
        if hidden.ndim == 0 or hidden.shape[-1] != dimension:
            raise ValueError(
                f"Normalization requires last dimension {dimension}, "
                f"got shape {tuple(hidden.shape)}."
            )
        if not hidden.is_floating_point():
            raise TypeError("Normalization requires floating-point inputs.")


class LayerPostprocessingDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerPostprocessingDelegate) -> None:
        cls.validate_resolved_activation(model.cfg.activation)
        cls.validate_resolved_output_dim(model.cfg.output_dim)
        cls.validate_resolved_dropout_probability(model.cfg.dropout_probability)

    @staticmethod
    def validate_resolved_activation(
        activation: ActivationOptions | None,
    ) -> None:
        if activation is None:
            raise ValueError("Layer postprocessing requires a resolved activation.")

    @staticmethod
    def validate_resolved_output_dim(output_dim: int | None) -> None:
        if output_dim is None:
            raise ValueError("Layer postprocessing requires a resolved output_dim.")

    @staticmethod
    def validate_resolved_dropout_probability(
        dropout_probability: float | None,
    ) -> None:
        if dropout_probability is None:
            raise ValueError(
                "Layer postprocessing requires a resolved dropout_probability."
            )

    @staticmethod
    def validate_built_gate_type(gate: Module | None) -> LayerGate | None:
        if gate is None:
            return None
        from emperor.layers._composition.gate import LayerGate

        if not isinstance(gate, LayerGate):
            raise TypeError("gate_config must build a LayerGate.")
        return gate


class LayerResidualDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerResidualDelegate) -> None:
        cls.validate_resolved_output_dim(model.cfg.output_dim)

    @staticmethod
    def validate_resolved_output_dim(output_dim: int | None) -> None:
        if output_dim is None:
            raise ValueError(
                "Layer residual processing requires a resolved output_dim."
            )

    @staticmethod
    def validate_built_residual_connection_type(
        connection: Module | None,
    ) -> ResidualConnectionAbstract | None:
        if connection is None:
            return None
        from emperor.layers._composition.residual.base import (
            ResidualConnectionAbstract,
        )

        if not isinstance(connection, ResidualConnectionAbstract):
            raise TypeError("residual_config must build a ResidualConnectionAbstract.")
        return connection

    @staticmethod
    def validate_forward_local_residual_runtime(
        residual_connection: ResidualConnectionAbstract | None,
        residual_schedule: RecurrentResidualSchedule | None,
    ) -> tuple[ResidualConnectionAbstract, RecurrentResidualSchedule]:
        if residual_connection is None or residual_schedule is None:
            raise RuntimeError(
                "forward-local residual state requires a recurrent residual schedule."
            )
        return residual_connection, residual_schedule


class RecurrentResidualScheduleValidator(ValidatorBase):
    @classmethod
    def validate(cls, schedule: RecurrentResidualSchedule) -> None:
        transition_count = schedule.transition_count
        if (
            isinstance(transition_count, bool)
            or not isinstance(transition_count, int)
            or transition_count <= 0
        ):
            raise ValueError("transition_count must be a positive integer.")

    @staticmethod
    def validate_transition_index(
        schedule: RecurrentResidualSchedule,
        transition_index: object,
    ) -> None:
        if (
            isinstance(transition_index, bool)
            or not isinstance(transition_index, int)
            or not 0 <= transition_index < schedule.transition_count
        ):
            raise IndexError(
                "transition_index must identify a configured recurrent transition."
            )

    @staticmethod
    def validate_subsequent_connections(
        schedule: DepthwiseRecurrentResidualSchedule,
    ) -> None:
        expected_connection_count = schedule.transition_count - 1
        if len(schedule.subsequent_connections) != expected_connection_count:
            raise ValueError(
                "subsequent_connections must contain one connection for every "
                "transition after transition zero."
            )
