"""Register flat controls for optional components declared by a package."""

from models._learned_grouping import declare_grouping_model_defaults
from models._projection_model import declare_projection_model_defaults
from models._sequence_expansion_controls import declare_sequence_expansion_defaults


def declare_component_defaults(namespace: dict[str, object]) -> None:
    declare_projection_model_defaults(namespace)
    declare_grouping_model_defaults(namespace)
    declare_sequence_expansion_defaults(namespace)
