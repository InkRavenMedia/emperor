from __future__ import annotations

from model_runtime.inspection import ConfigurationField

_STACK_CONTROLS = (
    "stack_gate_flag",
    "gate_stack_independent_flag",
    "stack_halting_flag",
    "halting_stack_independent_flag",
    "memory_flag",
    "memory_stack_independent_flag",
    "recurrent_flag",
    "recurrent_stack_gate_flag",
    "recurrent_gate_stack_independent_flag",
    "recurrent_stack_halting_flag",
    "recurrent_halting_stack_independent_flag",
)
_SECTION_CONTROLS = {
    prefix + control
    for prefix in ("", "router_", "expert_", "ff_", "attn_")
    for control in _STACK_CONTROLS
} | {
    f"{prefix}{component}_{control}"
    for prefix in ("", "router_")
    for component in ("weight", "bias", "diagonal", "mask")
    for control in ("option_flag", "generator_stack_independent_flag")
} | {
    f"{prefix}attn_source_memory_flag"
    for prefix in ("", "encoder_", "decoder_self_", "decoder_cross_")
}

# These canonical keys identify sections with specialized Workbench presentation.
_SECTION_ANCHORS = (
    "stack_num_layers",
    "submodule_stack_residual_model_flag",
    "input_layer_weight_option",
    "output_layer_weight_option",
)


def configuration_section_metadata(
    fields: tuple[ConfigurationField, ...],
    *,
    model_type: str,
) -> dict[tuple[str, ...], tuple[list[str], str | None]]:
    """Resolve stable section anchors and controls before UI filtering."""

    section_keys: dict[tuple[str, ...], list[str]] = {}
    first_descendant_keys: dict[tuple[str, ...], str] = {}
    for field in fields:
        path = field.section_path
        key = field.key.lower()
        section_keys.setdefault(path, []).append(key)
        for depth in range(1, len(path) + 1):
            first_descendant_keys.setdefault(path[:depth], key)

    anchors: dict[tuple[str, ...], str] = {}
    controllers: dict[tuple[str, ...], str | None] = {}
    for path, keys in section_keys.items():
        candidates = [key for key in keys if key in _SECTION_CONTROLS]
        # Expert Transformer sections also contain router and expert overrides.
        feed_forward = next((key for key in candidates if key.startswith("ff_")), None)
        controller = candidates[0] if len(candidates) == 1 else feed_forward
        controllers[path] = controller
        anchors[path] = next(
            (key for key in _SECTION_ANCHORS if key in keys),
            controller or keys[0],
        )
        if (
            model_type != "transformer"
            and "expert_attention_use_kv_expert_models_flag" in keys
            and "num_experts" in keys
        ):
            anchors[path] = "mixture_of_experts_model"

    metadata: dict[tuple[str, ...], tuple[list[str], str | None]] = {}
    for path in section_keys:
        prefixes = (path[:depth] for depth in range(1, len(path) + 1))
        identity = [
            anchors.get(prefix, first_descendant_keys[prefix]) for prefix in prefixes
        ]
        metadata[path] = identity, controllers[path]
    return metadata
