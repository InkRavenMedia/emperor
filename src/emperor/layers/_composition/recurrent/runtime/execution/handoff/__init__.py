"""Internal delegates for the two recurrent smooth depth handoff algorithms."""

from emperor.layers._composition.recurrent.runtime.execution.handoff.branched import (
    RecurrentBranchedHandoff,
)
from emperor.layers._composition.recurrent.runtime.execution.handoff.nested import (
    RecurrentNestedHandoff,
)

__all__ = ["RecurrentBranchedHandoff", "RecurrentNestedHandoff"]
