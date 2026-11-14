"""Hierarchical Reasoning Model implementation and forward-local state exports."""

from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.core import (
    HierarchicalReasoningModelRecurrent,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.state import (
    HierarchicalReasoningModelState,
)

__all__ = ("HierarchicalReasoningModelRecurrent", "HierarchicalReasoningModelState")
