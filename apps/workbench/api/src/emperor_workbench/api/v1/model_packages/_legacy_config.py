"""Identify saved configuration objects with flat Workbench controls."""

from collections.abc import Mapping

from model_runtime.inspection import ConfigurationField

_REPLACEMENT_SUFFIXES = {
    "ATTN_SOURCE_MEMORY_CONFIG": (
        "FLAG",
        "CHUNK_SIZE",
        "GROUPING_METHOD",
        "MODEL_OPTION",
    ),
    "SEQUENCE_EXPANSION_CONFIG": ("FLAG", "MAX_NEW_TOKENS", "MODEL_OPTION"),
    "SOFT_MOE_CONFIG": ("FLAG", "SLOTS_PER_EXPERT"),
    "GROUPING_MODEL_CONFIG": ("OPTION", "BIAS_FLAG"),
    "ATTN_SOURCE_MEMORY_MODEL_CONFIG": ("OPTION", "BIAS_FLAG"),
    "SEQUENCE_EXPANSION_MODEL_CONFIG": ("OPTION", "BIAS_FLAG"),
    "SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG": ("OPTION", "NUM_EMBEDDINGS"),
    "HIERARCHICAL_EMBEDDING_CONFIG": ("FLAG",),
    "CONTEXTUAL_EMBEDDING_CONFIG": ("FLAG",),
}
_CONTROL_TYPES = {"bool", "class", "enum", "float", "int"}


def has_flat_config_controls(
    field: ConfigurationField, fields_by_key: Mapping[str, ConfigurationField]
) -> bool:
    if field.value_type != "json":
        return False
    stem = field.key.removesuffix("_CONFIG")
    for owner, suffixes in _REPLACEMENT_SUFFIXES.items():
        if field.key == owner or field.key.endswith(f"_{owner}"):
            return all(
                (control := fields_by_key.get(f"{stem}_{suffix}")) is not None
                and control.value_type in _CONTROL_TYPES
                and (
                    control.value_type not in {"class", "enum"} or bool(control.choices)
                )
                for suffix in suffixes
            )
    return False
