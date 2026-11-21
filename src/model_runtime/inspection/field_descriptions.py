from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from model_runtime.inspection._field_help import (
    EXPLICIT_FIELD_DESCRIPTIONS,
    FLAG_DISABLED_PREFIXES,
    INHERITED_STACK_SECTIONS,
    SECTION_CONTEXT,
)


@dataclass(frozen=True, slots=True)
class _DescriptionRule:
    template: str
    adds_caveat: bool = True


class _Rule(Enum):
    INDEPENDENT = _DescriptionRule(
        "Controls whether the {context} uses its own stack settings instead of "
        "inheriting shared submodule settings."
    )
    RESIDUAL_MODEL = _DescriptionRule(
        "Uses the Residual Stack Options as a data-dependent query or coefficient "
        "model for the {context}. Attention residuals generate queries from the "
        "current output; weighted and weighted-blend residuals generate mixing "
        "coefficients from concatenated current and previous outputs."
    )
    RESIDUAL_BLOCK_SIZE = _DescriptionRule(
        "Required when the paired residual selector uses AttentionResidualConfig. "
        "Number of consecutive raw outputs combined into one depth source. "
        "Suggested starting value: 1 for full attention; larger integers enable "
        "block attention. Supply explicitly; no numeric default is applied.",
        adds_caveat=False,
    )
    RESIDUAL_RMS_NORM_EPSILON = _DescriptionRule(
        "Required when the paired residual selector uses AttentionResidualConfig. "
        "Finite positive float used to RMS-normalize routing keys. Suggested "
        "starting value: 1e-6. Supply explicitly; no numeric default is applied.",
        adds_caveat=False,
    )
    BIAS = _DescriptionRule(
        "Controls whether linear layers in the {context} include bias terms."
    )
    FLAG = _DescriptionRule("")
    HIDDEN_DIMENSION = _DescriptionRule(
        "Sets the hidden feature width used by the {context}. Larger values "
        "increase capacity and memory use."
    )
    LAYER_COUNT = _DescriptionRule(
        "Sets how many layers are built for the {context}. More layers add "
        "capacity and compute cost."
    )
    DROPOUT = _DescriptionRule(
        "Sets the dropout rate used by the {context}. Increase it for more "
        "regularization, or use 0 to disable dropout."
    )
    LAYER_NORM = _DescriptionRule(
        "Chooses where layer normalization is applied in the {context}. This "
        "changes training stability and the ordering of stack operations."
    )
    ACTIVATION = _DescriptionRule(
        "Selects the nonlinearity used by the {context}. The activation affects "
        "gradient flow and model expressiveness."
    )
    OPTION = _DescriptionRule(
        "Selects which implementation or behavior the {context} uses."
    )
    THRESHOLD = _DescriptionRule(
        "Sets the decision threshold used by the {context}. Higher values make "
        "activation or stopping conditions stricter."
    )
    TRAINER = _DescriptionRule(
        "Passes the {subject} setting through to the PyTorch Lightning trainer. "
        "Adjust it to control runtime training behavior rather than model "
        "architecture.",
        adds_caveat=False,
    )
    CALLBACK = _DescriptionRule(
        "Configures the {subject} callback setting used around training. These "
        "values affect monitoring, early stopping, and checkpoint behavior "
        "rather than the model forward pass.",
        adds_caveat=False,
    )
    DATA = _DescriptionRule(
        "Configures the {subject} data-loading setting for supported data "
        "modules. These values affect input pipeline behavior, not the model "
        "architecture.",
        adds_caveat=False,
    )
    NUMERIC = _DescriptionRule(
        "Sets the numeric {subject} value for the {context}. Tune it when "
        "changing capacity, regularization, or runtime limits."
    )
    BOOLEAN = _DescriptionRule(
        "Turns the {subject} behavior on or off for the {context}."
    )
    GENERIC = _DescriptionRule("Configures the {subject} setting for the {context}.")


_SUFFIX_DESCRIPTION_RULES = (
    ("_RESIDUAL_BLOCK_SIZE", _Rule.RESIDUAL_BLOCK_SIZE),
    ("_RESIDUAL_RMS_NORM_EPSILON", _Rule.RESIDUAL_RMS_NORM_EPSILON),
    ("_INDEPENDENT_FLAG", _Rule.INDEPENDENT),
    ("_RESIDUAL_MODEL_FLAG", _Rule.RESIDUAL_MODEL),
    ("_BIAS_FLAG", _Rule.BIAS),
    ("_FLAG", _Rule.FLAG),
    ("_HIDDEN_DIM", _Rule.HIDDEN_DIMENSION),
    ("_NUM_LAYERS", _Rule.LAYER_COUNT),
    ("_DROPOUT_PROBABILITY", _Rule.DROPOUT),
    ("_DROPOUT", _Rule.DROPOUT),
    ("_LAYER_NORM_POSITION", _Rule.LAYER_NORM),
    ("_ACTIVATION", _Rule.ACTIVATION),
    ("_OPTION", _Rule.OPTION),
    ("_THRESHOLD", _Rule.THRESHOLD),
)

_PREFIX_DESCRIPTION_RULES = (
    ("TRAINER_", _Rule.TRAINER),
    ("CALLBACK_", _Rule.CALLBACK),
    ("DATA_", _Rule.DATA),
)

_PREFIX_APPLICABILITY_CAVEATS = (
    ("FF_GATE_", "Only applies when the feed-forward gate feature is enabled."),
    (
        "FF_HALTING_",
        "Only applies when the feed-forward halting feature is enabled.",
    ),
    (
        "FF_MEMORY_",
        "Only applies when the feed-forward memory feature is enabled.",
    ),
    ("FF_RECURRENT_", "Only applies when feed-forward recurrence is enabled."),
    (
        "ATTN_GATE_",
        "Only applies when the attention projection gate feature is enabled.",
    ),
    (
        "ATTN_HALTING_",
        "Only applies when the attention projection halting feature is enabled.",
    ),
    (
        "ATTN_MEMORY_",
        "Only applies when the attention projection memory feature is enabled.",
    ),
    (
        "ATTN_RECURRENT_",
        "Only applies when attention projection recurrence is enabled.",
    ),
)


def _description_rule(key: str, kind: str) -> _Rule:
    for suffix, rule_key in _SUFFIX_DESCRIPTION_RULES:
        if key.endswith(suffix):
            return rule_key
    for prefix, rule_key in _PREFIX_DESCRIPTION_RULES:
        if key.startswith(prefix):
            return rule_key
    if kind in {"int", "float"}:
        return _Rule.NUMERIC
    if kind == "bool":
        return _Rule.BOOLEAN
    return _Rule.GENERIC


def _render_description(rule_key: _Rule, subject: str, context: str) -> str:
    if rule_key == _Rule.FLAG:
        if subject == context:
            return f"Enables or disables the {context}."
        return f"Enables or disables {subject} for the {context}."
    return rule_key.value.template.format(
        subject=subject,
        context=context,
    )


def config_field_description(
    key: str,
    *,
    section: str,
    kind: str,
    nullable: bool,
    default: Any,
) -> str:
    """Return practical help text for a public config field."""

    description_key = key
    for prefix in ("ENCODER_", "DECODER_", "DECODER_SELF_", "DECODER_CROSS_"):
        unscoped_key = key.removeprefix(prefix)
        if key != unscoped_key and unscoped_key.startswith(
            ("SEQUENCE_EXPANSION_", "ATTN_SOURCE_MEMORY_")
        ):
            description_key = unscoped_key
            break
    explicit_description = EXPLICIT_FIELD_DESCRIPTIONS.get(description_key)
    if explicit_description is not None:
        if description_key != key:
            return (
                explicit_description
                + " Overrides the common setting for this stack; None inherits it."
            )
        return explicit_description

    subject = _subject_for_key(key, section)
    context = SECTION_CONTEXT.get(section, _humanize_section(section))
    caveat = _applicability_caveat(key, section, nullable, default)
    rule_key = _description_rule(key, kind)
    rule = rule_key.value
    description = _render_description(rule_key, subject, context)
    return _with_caveat(description, caveat) if rule.adds_caveat else description


def _subject_for_key(key: str, section: str) -> str:
    section_words = set(_humanize_section(section).split())
    words = [
        word
        for word in key.lower().split("_")
        if word not in section_words and word not in {"flag", "option"}
    ]
    if not words:
        return SECTION_CONTEXT.get(section, _humanize_section(section))
    return " ".join(words)


def _humanize_section(section: str) -> str:
    return section.strip().lower() if section.strip() else "this config section"


def _applicability_caveat(
    key: str,
    section: str,
    nullable: bool,
    default: Any,
) -> str:
    if key.endswith("_INDEPENDENT_FLAG"):
        return ""
    if section in INHERITED_STACK_SECTIONS:
        return "Only matters when this stack is configured independently."
    for prefix, caveat in _PREFIX_APPLICABILITY_CAVEATS:
        if key.startswith(prefix) and not key.endswith("_FLAG"):
            return caveat
    if (
        any(key.startswith(prefix) for prefix in FLAG_DISABLED_PREFIXES)
        and not key.endswith("_FLAG")
        and not key.endswith("_INDEPENDENT_FLAG")
    ):
        root = key.split("_", 1)[0].lower()
        return f"Only applies when the {root} feature is enabled."
    if nullable and default is None:
        return "Use None to inherit the builder default when inheritance is supported."
    return ""


def _with_caveat(description: str, caveat: str) -> str:
    if not caveat:
        return description
    return f"{description} {caveat}"
