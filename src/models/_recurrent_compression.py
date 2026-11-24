"""Package controls for the recurrent sequence stacks that own token positions."""

from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from types import ModuleType
from typing import Any, cast

from emperor.augmentations import adaptive_parameters
from emperor.config import ConfigBase
from emperor.layers import (
    RecurrentCompositionConfig,
    RecurrentCompressionConfig,
)
from emperor.linears import LinearLayerConfig
from models._recurrent_variants import (
    RecurrentResidualOptions,
    RecurrentVariantOptions,
    declare_variant_options,
    variant_updates,
)
from models._runtime_binding import bind_runtime_options


@dataclass(frozen=True, kw_only=True)
class CompressionOptions:
    recurrent_compression_config: RecurrentCompressionConfig | None = None


@dataclass(frozen=True, kw_only=True)
class SequenceRecurrentOptions(CompressionOptions, RecurrentVariantOptions):
    pass


@dataclass(frozen=True, kw_only=True)
class SequenceResidualOptions(CompressionOptions, RecurrentResidualOptions):
    pass


@cache
def _scorer_types() -> dict[str, type]:
    """Canonical constructors for serialized adaptive generators and their options."""
    import emperor.experts as experts
    import emperor.halting as halting
    import emperor.layers as layers
    import emperor.memory as memory
    import emperor.sampler as sampler

    modules = (adaptive_parameters, experts, halting, layers, memory, sampler)
    return {
        name: cls
        for module in modules
        for name in module.__all__
        if isinstance(cls := getattr(module, name), type)
        and issubclass(cls, (ConfigBase, Enum))
    } | {"LinearLayerConfig": LinearLayerConfig}


def declare_compression_options(namespace: dict[str, Any]) -> None:
    fields = {
        "RECURRENT_COMPRESSION_CONFIG": RecurrentCompressionConfig,
        "RECURRENT_COMPRESSION_SLOTS": int,
        "RECURRENT_COMPRESSION_SCORER_CONFIG": LinearLayerConfig,
    }
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    for key, field_type in fields.items():
        namespace.setdefault(key, None)
        namespace["__annotations__"][key] = field_type | None
        aliases[key] = "RECURRENT_FLAG"
    object_types = namespace.setdefault("_CONFIG_OBJECT_TYPES", {})
    for key in ("RECURRENT_COMPRESSION_CONFIG", "RECURRENT_COMPRESSION_SCORER_CONFIG"):
        object_types[key] = _scorer_types()


def declare_sequence_options(
    namespace: dict[str, Any], *, include_residual: bool = False
) -> None:
    options = RecurrentResidualOptions if include_residual else RecurrentVariantOptions
    declare_variant_options(namespace, options)
    declare_compression_options(namespace)


def compression_from_config(source: ModuleType) -> RecurrentCompressionConfig | None:
    return _compression_from_values({}, source, None)


def _compression_from_values(
    values: dict[str, Any], source: ModuleType, base: CompressionOptions | None
) -> RecurrentCompressionConfig | None:
    compression = values.pop(
        "recurrent_compression_config",
        getattr(base, "recurrent_compression_config", None)
        or getattr(source, "RECURRENT_COMPRESSION_CONFIG", None),
    )
    if compression is not None and not isinstance(
        compression, RecurrentCompressionConfig
    ):
        raise TypeError(
            "recurrent_compression_config must be a RecurrentCompressionConfig."
        )
    slots = values.pop(
        "recurrent_compression_slots",
        getattr(source, "RECURRENT_COMPRESSION_SLOTS", None),
    )
    scorer = values.pop(
        "recurrent_compression_scorer_config",
        getattr(source, "RECURRENT_COMPRESSION_SCORER_CONFIG", None),
    )
    if slots is not None or scorer is not None:
        compression = compression or RecurrentCompressionConfig()
        compression = replace(
            compression,
            max_slots=compression.max_slots if slots is None else slots,
            scorer_config=compression.scorer_config if scorer is None else scorer,
        )
    if compression is not None:
        if type(compression.max_slots) is not int or compression.max_slots <= 0:
            raise ValueError("recurrent_compression_slots must be a positive integer.")
        if compression.scorer_config is not None and not isinstance(
            cast(object, compression.scorer_config), LinearLayerConfig
        ):
            raise TypeError(
                "recurrent_compression_scorer_config requires a linear or adaptive linear configuration."
            )
    return compression


def _compression_updates(
    values: dict[str, Any], source: ModuleType, base: CompressionOptions | None
) -> dict[str, Any]:
    return {
        "recurrent_compression_config": _compression_from_values(values, source, base)
    }


def bind_compression_options(config: ModuleType):
    return bind_runtime_options(config, _compression_updates)


def _sequence_updates(
    values: dict[str, Any],
    source: ModuleType,
    base: SequenceRecurrentOptions | SequenceResidualOptions | None,
) -> dict[str, Any]:
    return variant_updates(values, source, base) | _compression_updates(
        values, source, base
    )


def sequence_options_from_config(source: ModuleType) -> dict[str, Any]:
    return _sequence_updates({}, source, None)


def bind_sequence_options(config: ModuleType):
    return bind_runtime_options(config, _sequence_updates)


def configure_sequence_stacks(
    experiment: ConfigBase, compression: RecurrentCompressionConfig | None
) -> ConfigBase:
    if compression is None:
        return experiment
    changes = {}
    for name in ("encoder_config", "decoder_config"):
        stack = getattr(experiment, name, None)
        if isinstance(stack, RecurrentCompositionConfig):
            changes[name] = replace(stack, compression_config=compression)
    cluster = getattr(experiment, "neuron_cluster_config", None)
    if cluster is not None:
        neuron = cluster.neuron_config
        nucleus = neuron.nucleus_config
        if isinstance(nucleus.model_config, RecurrentCompositionConfig):
            encoder = replace(nucleus.model_config, compression_config=compression)
            nucleus = replace(nucleus, model_config=encoder)
            changes["neuron_cluster_config"] = replace(
                cluster, neuron_config=replace(neuron, nucleus_config=nucleus)
            )
    if not changes:
        raise ValueError(
            "Recurrent compression requires an enabled recurrent encoder or decoder stack."
        )
    return replace(experiment, **changes)
