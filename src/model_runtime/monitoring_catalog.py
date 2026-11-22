from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from emperor.attention import AttentionMonitorCallback
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterMonitorCallback,
    WeightBankUtilizationMonitorCallback,
)
from emperor.halting import HaltingMonitorCallback
from emperor.layers import LayerControllerMonitorCallback, RecurrentLayerMonitorCallback
from emperor.linears import LinearMonitorCallback
from emperor.memory import MemoryMonitorCallback
from emperor.monitoring import MonitorOption, MonitorSettings
from emperor.neuron import NeuronClusterMonitorCallback
from emperor.parametric import ParametricLayerMonitorCallback
from emperor.sampler import SamplerMonitorCallback

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback


def _monitor_option(
    *,
    name: str,
    label: str,
    description: str,
    callback: Callable[[int], Callback],
    kinds: tuple[str, ...] = ("scalar",),
    default_enabled: bool = False,
) -> MonitorOption:
    def build_callback(settings: MonitorSettings) -> Callback:
        return callback(settings.log_every_n_steps)

    return MonitorOption(
        name=name,
        label=label,
        description=description,
        kinds=kinds,
        default_enabled=default_enabled,
        callback_factory=build_callback,
    )


_LINEAR = _monitor_option(
    name="linear",
    label="Linear layers",
    description=(
        "Logs activation, parameter, gradient, and dead-feature statistics "
        "for Emperor linear layers."
    ),
    callback=LinearMonitorCallback,
)
_RECURRENT_LAYER = _monitor_option(
    name="recurrent-layer",
    label="Recurrent layers",
    description=(
        "Logs recurrent step count, hidden-state convergence, recurrent gate "
        "openness, halted-state preservation, and step-delta visual summaries."
    ),
    callback=RecurrentLayerMonitorCallback,
    kinds=("scalar", "histogram", "image"),
)
_LAYER_CONTROLLER = _monitor_option(
    name="layer-controller",
    label="Layer controllers",
    description=(
        "Logs Layer gate, residual, dropout, layer-norm, and activation "
        "controller statistics without duplicating memory metrics."
    ),
    callback=LayerControllerMonitorCallback,
)
_HALTING = _monitor_option(
    name="halting",
    label="Halting (adaptive compute)",
    description=(
        "Logs recurrence depth, halting fraction, max-steps saturation, ponder "
        "loss, plus survival heatmap and ponder-cost histogram for "
        "stick-breaking / soft halting modules."
    ),
    callback=HaltingMonitorCallback,
    kinds=("scalar", "histogram", "image"),
)
_MEMORY = _monitor_option(
    name="memory",
    label="Memory modules",
    description=(
        "Logs gating, blend-weight, and state statistics for Emperor memory "
        "modules. Inactive until a memory config is enabled."
    ),
    callback=MemoryMonitorCallback,
)

_ADAPTIVE_PARAMETERS = _monitor_option(
    name="adaptive",
    label="Adaptive parameters",
    description=(
        "Logs dynamic weight, bias, diagonal, and mask parameter statistics, "
        "plus input-adaptivity (cross-sample variation / collapse detection)."
    ),
    callback=partial(
        AdaptiveParameterMonitorCallback, log_histograms=True, log_internal_stats=True
    ),
    kinds=("scalar", "histogram"),
)
_WEIGHT_BANK = _monitor_option(
    name="weight-bank",
    label="Weight bank utilization",
    description=(
        "Logs bank-slot selection entropy, utilization, and routing heatmaps "
        "for weighted-bank dynamic params."
    ),
    callback=WeightBankUtilizationMonitorCallback,
    kinds=("scalar", "histogram", "image"),
)

_ATTENTION = _monitor_option(
    name="attention",
    label="Attention",
    description=(
        "Logs Q/K/V norms, attention entropy, max probability, dropout and mask "
        "coverage, auxiliary loss, and attention head visual summaries."
    ),
    callback=AttentionMonitorCallback,
    kinds=("scalar", "histogram", "image"),
)
_ENCODER_ROUTING = _monitor_option(
    name="sampler",
    label="Expert routing",
    description=(
        "Logs top-k routing, load-balance losses, dropped-token behavior, and "
        "expert utilization for MoE-backed encoder sub-stacks."
    ),
    callback=SamplerMonitorCallback,
    kinds=("scalar", "histogram", "image"),
)
_SAMPLER = replace(
    _ENCODER_ROUTING,
    label="Sampler usage",
    description=(
        "Logs expert routing balance, capacity drop fraction, auxiliary "
        "load-balancing loss, usage histograms, and routing heatmaps."
    ),
)
_DECODER_ROUTING = replace(
    _ENCODER_ROUTING,
    description=(
        "Logs top-k routing, load-balance losses, dropped-token behavior, and "
        "expert utilization for MoE-backed decoder sub-stacks."
    ),
)
_ADAPTIVE_LAYER = _monitor_option(
    name="adaptive",
    label="Adaptive parameters",
    description=(
        "Logs adaptive weight, bias, diagonal, and mask controller statistics for "
        "adaptive linear layers."
    ),
    callback=AdaptiveParameterMonitorCallback,
    kinds=("scalar", "histogram"),
)
_ADAPTIVE_BANK = replace(
    _WEIGHT_BANK,
    label="Weight banks",
    description=(
        "Logs adaptive parameter bank usage and utilization for bank-backed "
        "dynamic weights and biases."
    ),
    kinds=("scalar", "histogram"),
)
_ADAPTIVE_EXPERT = replace(
    _ADAPTIVE_LAYER,
    description=(
        "Logs adaptive weight, bias, diagonal, and mask controller statistics for "
        "adaptive expert linear layers."
    ),
)
_EXPERT_BANK = replace(
    _ADAPTIVE_BANK,
    description=(
        "Logs adaptive parameter bank usage and utilization for bank-backed "
        "dynamic expert weights and biases."
    ),
)
_TOKEN_MIXER = replace(
    _ATTENTION,
    name="mixer",
    label="Token mixer",
    description=(
        "Logs mixer outputs and auxiliary-loss diagnostics without inventing "
        "attention heads or Q/K/V projections."
    ),
    kinds=("scalar", "histogram"),
)
_MIXER_CONTROLLER = replace(
    _LAYER_CONTROLLER,
    description="Logs Mixer block gates, residuals, normalization, and dropout.",
)
_MIXER_ADAPTIVE = replace(
    _ADAPTIVE_LAYER,
    description="Logs adaptive generator and generated-parameter diagnostics.",
)
_MIXER_BANK = replace(
    _ADAPTIVE_BANK,
    description="Logs utilization when an adaptive weight-bank option is selected.",
)
_MIXER_ROUTING = replace(
    _ENCODER_ROUTING,
    description="Logs top-k routing, expert utilization, capacity, and losses.",
)
_MIXER_ADAPTIVE_EXPERT = replace(
    _ADAPTIVE_LAYER,
    description="Logs adaptive generators inside selected experts.",
)
_MIXER_EXPERT_BANK = replace(
    _ADAPTIVE_BANK,
    description="Logs utilization when adaptive expert banks are selected.",
)
_NEURON_CLUSTER = _monitor_option(
    name="neuron_cluster",
    label="Neuron cluster growth",
    description=(
        "Logs cluster growth (count, capacity, fill, growth pressure) plus routing "
        "dynamics: route depth, escape/halt fractions, entry-routing entropy, "
        "survival curve, and per-neuron utilization heatmap."
    ),
    callback=NeuronClusterMonitorCallback,
    kinds=("scalar", "histogram", "image"),
    default_enabled=True,
)
_NEURON_ROUTING = replace(
    _SAMPLER,
    label="Routing samplers",
    description=(
        "Logs router/sampler internals for the cluster entry sampler and per-neuron "
        "terminal samplers: probability distributions, per-expert "
        "utilization, and auxiliary load-balancing loss components."
    ),
)
_VISION_CLUSTER = replace(
    _NEURON_CLUSTER,
    label="Neuron cluster",
    description=(
        "Logs whole-image routes, halting, per-neuron utilization, growth and pruning."
    ),
    default_enabled=False,
)
_PARAMETRIC = _monitor_option(
    name="parametric",
    label="Parametric layers",
    description=(
        "Logs generated parameter norms, affine deltas, router entropy, mixture "
        "utilization, skip/drop fraction, auxiliary loss, and utilization visual "
        "summaries."
    ),
    callback=ParametricLayerMonitorCallback,
    kinds=("scalar", "histogram", "image"),
)
_TRANSLATION_ATTENTION = replace(
    _ATTENTION,
    description="Logs encoder, decoder self-attention, and cross-attention health.",
)
_TRANSLATION_RECURRENCE = replace(
    _RECURRENT_LAYER,
    description="Logs recurrent encoder and decoder controller state.",
)
_TRANSLATION_CONTROLLER = replace(
    _LAYER_CONTROLLER,
    description="Logs gates, residual joins, halting, and normalization.",
)
_TRANSLATION_MEMORY = replace(
    _MEMORY,
    description="Logs optional encoder and decoder dynamic memory.",
)
_ADAPTIVE_ATTENTION = replace(
    _ATTENTION,
    description="Logs all Transformer attention paths.",
)
_TRANSLATION_ADAPTIVE = replace(
    _ADAPTIVE_LAYER,
    description="Logs active adaptive parameter generators.",
)
_TRANSLATION_BANK = replace(
    _ADAPTIVE_BANK,
    description="Logs adaptive weight-bank utilization.",
)
_TRANSFORMER_RECURRENCE = replace(
    _RECURRENT_LAYER,
    description="Logs recurrent stack state.",
)
_TRANSFORMER_CONTROLLER = replace(
    _LAYER_CONTROLLER,
    description="Logs gates, halting, and residuals.",
)
_TRANSFORMER_MEMORY = replace(
    _MEMORY,
    description="Logs optional dynamic memory.",
)
_EXPERT_ATTENTION = replace(
    _ATTENTION,
    description="Logs expert attention health.",
)
_EXPERT_ROUTING = replace(
    _ENCODER_ROUTING,
    description="Logs attention and feed-forward expert routing.",
)
_EXPERT_CONTROLLER = replace(
    _LAYER_CONTROLLER,
    description="Logs gates and halting.",
)
_ADAPTIVE_EXPERT_ATTENTION = replace(
    _ATTENTION,
    description="Logs adaptive expert attention health.",
)
_ADAPTIVE_EXPERT_ROUTING = replace(
    _ENCODER_ROUTING,
    description="Logs adaptive attention and feed-forward expert routing.",
)
_TRANSLATION_ADAPTIVE_EXPERT = replace(
    _ADAPTIVE_LAYER,
    description="Logs active expert, router, projection, and feed-forward adaptations.",
)
_EXPERT_RECURRENCE = replace(
    _RECURRENT_LAYER,
    description="Logs recurrent encoder and decoder state.",
)
_ADAPTIVE_EXPERT_CONTROLLER = replace(
    _LAYER_CONTROLLER,
    description="Logs gates, halting, residuals, and normalization.",
)

_LINEAR_MONITORS = (_LINEAR, _RECURRENT_LAYER, _LAYER_CONTROLLER, _HALTING, _MEMORY)
_ADAPTIVE_MONITORS = (_ADAPTIVE_PARAMETERS, _WEIGHT_BANK)
_EXPERT_MONITORS = (_LINEAR, _RECURRENT_LAYER, _LAYER_CONTROLLER, _SAMPLER, _MEMORY)
_ADAPTIVE_EXPERT_MONITORS = (
    _LINEAR,
    _RECURRENT_LAYER,
    _LAYER_CONTROLLER,
    _ADAPTIVE_PARAMETERS,
    _WEIGHT_BANK,
    _SAMPLER,
    _MEMORY,
)
_LANGUAGE_MONITORS = (_ATTENTION, _RECURRENT_LAYER, _LAYER_CONTROLLER, _MEMORY)
_VISION_MONITORS = (_ATTENTION, _LAYER_CONTROLLER)
_ADAPTIVE_LAYER_MONITORS = (_ADAPTIVE_LAYER, _ADAPTIVE_BANK)
_ADAPTIVE_EXPERT_LAYER_MONITORS = (_ADAPTIVE_EXPERT, _EXPERT_BANK)
_MIXER_MONITORS = (_TOKEN_MIXER, _MIXER_CONTROLLER)
_NEURON_MONITORS = (_NEURON_CLUSTER, _NEURON_ROUTING)
_PARAMETRIC_MONITORS = (_PARAMETRIC, _LAYER_CONTROLLER)

_MONITORS_BY_MODEL: dict[str, tuple[MonitorOption, ...]] = {
    "bert/expert_linear": (*_LANGUAGE_MONITORS, _ENCODER_ROUTING),
    "bert/expert_linear_adaptive": (
        *_LANGUAGE_MONITORS,
        _ENCODER_ROUTING,
        *_ADAPTIVE_EXPERT_LAYER_MONITORS,
    ),
    "bert/linear": _LANGUAGE_MONITORS,
    "bert/linear_adaptive": (*_LANGUAGE_MONITORS, *_ADAPTIVE_LAYER_MONITORS),
    "experts/linear": _EXPERT_MONITORS,
    "experts/linear_adaptive": _ADAPTIVE_EXPERT_MONITORS,
    "gpt/expert_linear": (*_LANGUAGE_MONITORS, _DECODER_ROUTING),
    "gpt/expert_linear_adaptive": (
        *_LANGUAGE_MONITORS,
        _DECODER_ROUTING,
        *_ADAPTIVE_EXPERT_LAYER_MONITORS,
    ),
    "gpt/linear": _LANGUAGE_MONITORS,
    "gpt/linear_adaptive": (*_LANGUAGE_MONITORS, *_ADAPTIVE_LAYER_MONITORS),
    "linears/linear": _LINEAR_MONITORS,
    "linears/linear_adaptive": (*_LINEAR_MONITORS, *_ADAPTIVE_MONITORS),
    "mlp_mixer/expert_linear": (*_MIXER_MONITORS, _MIXER_ROUTING),
    "mlp_mixer/expert_linear_adaptive": (
        *_MIXER_MONITORS,
        _MIXER_ROUTING,
        _MIXER_ADAPTIVE_EXPERT,
        _MIXER_EXPERT_BANK,
    ),
    "mlp_mixer/linear": _MIXER_MONITORS,
    "mlp_mixer/linear_adaptive": (*_MIXER_MONITORS, _MIXER_ADAPTIVE, _MIXER_BANK),
    "neuron/expert_linear": (*_EXPERT_MONITORS, _NEURON_CLUSTER),
    "neuron/expert_linear_adaptive": (*_ADAPTIVE_EXPERT_MONITORS, _NEURON_CLUSTER),
    "neuron/linear": (*_LINEAR_MONITORS, *_NEURON_MONITORS),
    "neuron/linear_adaptive": (
        *_LINEAR_MONITORS,
        *_ADAPTIVE_MONITORS,
        *_NEURON_MONITORS,
    ),
    "parametric/parametric_generator": _PARAMETRIC_MONITORS,
    "parametric/parametric_matrix": _PARAMETRIC_MONITORS,
    "parametric/parametric_vector": _PARAMETRIC_MONITORS,
    "transformer/expert_linear": (
        _EXPERT_ATTENTION,
        _EXPERT_ROUTING,
        _TRANSFORMER_RECURRENCE,
        _EXPERT_CONTROLLER,
        _TRANSFORMER_MEMORY,
    ),
    "transformer/expert_linear_adaptive": (
        _ADAPTIVE_EXPERT_ATTENTION,
        _ADAPTIVE_EXPERT_ROUTING,
        _TRANSLATION_ADAPTIVE_EXPERT,
        _TRANSLATION_BANK,
        _EXPERT_RECURRENCE,
        _ADAPTIVE_EXPERT_CONTROLLER,
        _TRANSFORMER_MEMORY,
    ),
    "transformer/linear": (
        _TRANSLATION_ATTENTION,
        _TRANSLATION_RECURRENCE,
        _TRANSLATION_CONTROLLER,
        _TRANSLATION_MEMORY,
    ),
    "transformer/linear_adaptive": (
        _ADAPTIVE_ATTENTION,
        _TRANSLATION_ADAPTIVE,
        _TRANSLATION_BANK,
        _TRANSFORMER_RECURRENCE,
        _TRANSFORMER_CONTROLLER,
        _TRANSFORMER_MEMORY,
    ),
    "vit/expert_linear": (*_VISION_MONITORS, _ENCODER_ROUTING),
    "vit/expert_linear_adaptive": (
        *_VISION_MONITORS,
        _ENCODER_ROUTING,
        *_ADAPTIVE_EXPERT_LAYER_MONITORS,
    ),
    "vit/expert_linear_adaptive_cluster": (
        _VISION_CLUSTER,
        *_VISION_MONITORS,
        _ENCODER_ROUTING,
        *_ADAPTIVE_EXPERT_LAYER_MONITORS,
    ),
    "vit/linear": _VISION_MONITORS,
    "vit/linear_adaptive": (*_VISION_MONITORS, *_ADAPTIVE_LAYER_MONITORS),
}


def monitors_for_model(catalog_key: str) -> list[MonitorOption]:
    return list(_MONITORS_BY_MODEL[catalog_key])
