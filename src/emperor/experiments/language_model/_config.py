from dataclasses import dataclass

from emperor.config import ConfigBase


@dataclass
class HierarchicalObjectiveConfig(ConfigBase):
    """Equation 5 by default; normalization and native auxiliaries are explicit."""

    reduction: str = "sum"
    auxiliary_weight: float = 0.0

    def _registry_owner(self):
        from ._hierarchical import HierarchicalLanguageModelHandler

        return HierarchicalLanguageModelHandler
