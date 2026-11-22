"""Package-owned scalar grouping resolution."""

from types import ModuleType

from models import _learned_grouping
from models._adaptive_grouping import apply_grouping_options

__all__ = ["apply_grouping_options", "grouping_keys"]


def grouping_keys(config_module: ModuleType) -> set[str]:
    return {
        name.lower()
        for name in vars(config_module)
        if name.isupper()
        and any(
            name.lower().endswith(field) for field in _learned_grouping.GROUPING_FIELDS
        )
    }
