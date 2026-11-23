"""Package controls for the recurrent sequence stacks that own token positions."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache, wraps
from types import ModuleType
from typing import ParamSpec, TypeVar

from emperor.augmentations import adaptive_parameters
from emperor.config import ConfigBase
from emperor.layers import RecurrentCompositionConfig, RecurrentCompressionConfig
from emperor.linears import LinearLayerConfig
from model_runtime.packages.runtime_values import runtime_default_mapping


@dataclass(frozen=True, kw_only=True)
class CompressionOptions:
    recurrent_compression_config: RecurrentCompressionConfig | None = None


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


def declare_compression_options(namespace: dict) -> None:
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


def compression_from_config(source: ModuleType) -> RecurrentCompressionConfig | None:
    return _compression_from_values({}, source, None)


def _compression_from_values(values, source, base) -> RecurrentCompressionConfig | None:
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
            compression.scorer_config, LinearLayerConfig
        ):
            raise TypeError(
                "recurrent_compression_scorer_config requires a linear or adaptive linear configuration."
            )
    return compression


_Args = ParamSpec("_Args")
_Runtime = TypeVar("_Runtime", bound=CompressionOptions)


def bind_compression_options(
    config: ModuleType,
) -> Callable[[Callable[_Args, _Runtime]], Callable[_Args, _Runtime]]:
    """Wrap each package's runtime binder, preserving its public call signature."""

    def decorate(bind: Callable[_Args, _Runtime]) -> Callable[_Args, _Runtime]:
        signature = inspect.signature(bind)
        values_parameter = next(iter(signature.parameters))

        @wraps(bind)
        def with_compression(*args: _Args.args, **kwargs: _Args.kwargs) -> _Runtime:
            bound = signature.bind(*args, **kwargs)
            source = bound.arguments.get("config_module", config)
            base = bound.arguments.get("base")
            values = runtime_default_mapping(
                bound.arguments.get(values_parameter), package=config.__package__
            )
            compression = _compression_from_values(values, source, base)
            bound.arguments[values_parameter] = values
            runtime = bind(*bound.args, **bound.kwargs)
            if runtime.recurrent_compression_config is compression:
                return runtime
            return replace(runtime, recurrent_compression_config=compression)

        return with_compression

    return decorate


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
