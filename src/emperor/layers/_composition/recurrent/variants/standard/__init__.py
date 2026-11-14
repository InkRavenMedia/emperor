"""Standard recurrent implementation and forward-local state exports."""

from emperor.layers._composition.recurrent.variants.standard.core import RecurrentLayer
from emperor.layers._composition.recurrent.variants.standard.state import (
    StandardRecurrentState,
)

__all__ = ("RecurrentLayer", "StandardRecurrentState")
