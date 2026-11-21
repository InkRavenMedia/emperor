from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from torch.nn.parameter import is_lazy

from model_runtime.inspection._graph_accounting import GraphModule
from model_runtime.inspection._graph_component_semantics import (
    component_description,
    component_graph_role,
    display_graph_value,
    module_configuration,
)
from model_runtime.inspection._graph_neuron_details import NeuronDetails
from model_runtime.inspection._graph_recurrent_details import (
    LayerControllerDetails,
    RecurrentDetails,
)
from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.inspection.records import GraphConfiguration, GraphRole


@dataclass(frozen=True, slots=True)
class ModuleSemanticFacts:
    type_name: str
    description: str | None
    graph_role: GraphRole
    details: dict[str, Any]
    configuration: GraphConfiguration | None


@dataclass(slots=True)
class _SemanticDetailContext:
    module: GraphModule
    direct_parameters: dict[str, Any] | None
    limits: InspectionCaptureLimits
    details: dict[str, Any] = field(default_factory=dict[str, Any])


class _SemanticDetailsAdapter(Protocol):
    def apply(self, context: _SemanticDetailContext) -> None: ...


class _MissingAttribute:
    __slots__ = ()


_MISSING_ATTRIBUTE = _MissingAttribute()


def _shape_value(value: Any) -> str | None:
    if is_lazy(value):
        return None
    dimensions = tuple(value.shape)
    if not dimensions:
        return "scalar"
    return " x ".join(str(dimension) for dimension in dimensions)


@dataclass(frozen=True, slots=True)
class _ParameterShapeDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        direct_parameters = context.direct_parameters
        if direct_parameters is None:
            direct_parameters = dict(context.module.named_parameters(recurse=False))
        uninitialized_parameter_count = sum(
            is_lazy(parameter) for parameter in direct_parameters.values()
        )
        if uninitialized_parameter_count:
            context.details["uninitializedParameterCount"] = (
                uninitialized_parameter_count
            )
        for detail_key, parameter_names in (
            ("weightShape", ("weight", "weight_params", "weights")),
            ("biasShape", ("bias", "bias_params", "biases")),
            ("parameterBankShape", ("parameter_bank",)),
            ("inputFactorShape", ("input_factor",)),
            ("outputFactorShape", ("output_factor",)),
        ):
            for parameter_name in parameter_names:
                parameter = direct_parameters.get(parameter_name)
                if parameter is None:
                    continue
                shape = _shape_value(parameter)
                if shape is not None:
                    context.details[detail_key] = shape
                    break


@dataclass(frozen=True, slots=True)
class _DimensionDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        module = context.module
        input_dim = getattr(module, "input_dim", None)
        output_dim = getattr(module, "output_dim", None)
        hidden_dim = getattr(module, "hidden_dim", None)
        if input_dim is not None:
            context.details["inputDim"] = input_dim
        if hidden_dim is not None:
            context.details["hiddenDim"] = hidden_dim
        if output_dim is not None:
            context.details["outputDim"] = output_dim
        if input_dim is not None and output_dim is not None:
            context.details["dims"] = f"{input_dim} -> {output_dim}"


@dataclass(frozen=True, slots=True)
class _AttributeDetailsAdapter:
    fields: tuple[tuple[str, str], ...]

    def apply(self, context: _SemanticDetailContext) -> None:
        for source_attr, detail_key in self.fields:
            value = getattr(context.module, source_attr, None)
            if value is not None:
                context.details[detail_key] = display_graph_value(value)


def _first_detail_value(module: GraphModule, attr_paths: tuple[str, ...]) -> Any:
    for attr_path in attr_paths:
        value: Any = module
        for attr_name in attr_path.split("."):
            observed = getattr(value, attr_name, _MISSING_ATTRIBUTE)
            if observed is _MISSING_ATTRIBUTE:
                value = None
                break
            value = observed
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class _ExpertDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        for detail_key, attr_paths in (
            ("topK", ("top_k", "sampler_config.top_k", "cfg.top_k")),
            (
                "numExperts",
                ("num_experts", "sampler_config.num_experts", "cfg.num_experts"),
            ),
            (
                "routingMode",
                ("routing_initialization_mode", "cfg.routing_initialization_mode"),
            ),
        ):
            value = _first_detail_value(context.module, attr_paths)
            if value is not None:
                context.details[detail_key] = display_graph_value(value)


def _optional_model(
    module: object,
    attr_name: str,
) -> tuple[bool, Any | None]:
    model = getattr(module, attr_name, _MISSING_ATTRIBUTE)
    if model is _MISSING_ATTRIBUTE:
        return False, None
    return True, model


@dataclass(frozen=True, slots=True)
class _LayerBehaviorDetailsAdapter:
    def append(self, context: _SemanticDetailContext) -> LayerControllerDetails:
        module = context.module
        pipeline = getattr(module, "pipeline", None)
        postprocessing = getattr(pipeline, "postprocessing", None)
        if postprocessing is None:
            self._append_module_options(context)
            return LayerControllerDetails()

        dropout = getattr(postprocessing, "dropout_probability", None)
        if dropout is not None:
            context.details["dropout"] = dropout
        gate_present, gate = _optional_model(postprocessing, "gate")
        gate_option = getattr(gate, "option", None)
        option_name = (
            display_graph_value(gate_option) if gate_option is not None else None
        )
        if option_name is not None:
            context.details["gateOption"] = option_name
        if gate_present:
            context.details["gate"] = gate is not None
        halting_present, halting = _optional_model(
            getattr(pipeline, "halting", None), "model"
        )
        if halting_present:
            context.details["halting"] = halting is not None
        activation = getattr(postprocessing, "activation_function", None)
        if activation is not None:
            context.details["activation"] = display_graph_value(activation)
        position = getattr(getattr(pipeline, "normalization", None), "position", None)
        layer_norm = display_graph_value(position) if position is not None else None
        if layer_norm is not None:
            context.details["layerNorm"] = layer_norm
        return LayerControllerDetails(
            gate_active=getattr(gate, "model", None) is not None,
            gate_option=option_name,
            halting_model=halting,
            layer_norm=layer_norm,
        )

    @staticmethod
    def _append_module_options(context: _SemanticDetailContext) -> None:
        module = context.module
        gate = getattr(module, "gate", None)
        options = (
            ("dropout", getattr(module, "dropout_probability", None)),
            ("gateOption", getattr(gate, "option", None)),
            ("activation", getattr(module, "activation_function", None)),
            ("layerNorm", getattr(module, "layer_norm_position", None)),
        )
        for detail, value in options:
            if value is not None:
                context.details[detail] = display_graph_value(value)


@dataclass(frozen=True, slots=True)
class _CausalAttentionDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        causal = getattr(context.module, "causal_attention_mask_flag", None)
        if causal is not None:
            context.details["causalAttention"] = causal


_DETAIL_ADAPTERS: tuple[_SemanticDetailsAdapter, ...] = (
    _ParameterShapeDetailsAdapter(),
    _DimensionDetailsAdapter(),
    _AttributeDetailsAdapter(
        (
            ("embedding_dim", "embeddingDim"),
            ("num_heads", "numHeads"),
            ("num_layers", "numLayers"),
            ("source_sequence_length", "sourceSequenceLength"),
            ("target_sequence_length", "targetSequenceLength"),
        )
    ),
    _ExpertDetailsAdapter(),
)


def _snake_case_key(key: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _semantic_detail_value(value: Any) -> Any:
    if isinstance(value, dict):
        mapping_value = cast(dict[object, object], value)
        return {
            _snake_case_key(str(key)): _semantic_detail_value(item)
            for key, item in mapping_value.items()
        }
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return [_semantic_detail_value(item) for item in list_value]
    return value


@dataclass(frozen=True, slots=True)
class ModuleSemanticAdapter:
    module: GraphModule
    direct_parameters: dict[str, Any] | None = None
    limits: InspectionCaptureLimits | None = None

    def graph_role(self) -> GraphRole:
        return component_graph_role(self.module)

    def details(self) -> dict[str, Any]:
        if self.limits is None:
            raise RuntimeError("Semantic detail adaptation requires capture limits.")
        context = _SemanticDetailContext(
            module=self.module,
            direct_parameters=self.direct_parameters,
            limits=self.limits,
        )
        for adapter in _DETAIL_ADAPTERS:
            adapter.apply(context)
        controllers = _LayerBehaviorDetailsAdapter().append(context)
        cluster_present = NeuronDetails().append(
            context.module, context.limits, context.details
        )
        RecurrentDetails().append(
            context.module,
            context.details,
            cluster_present=cluster_present,
            controllers=controllers,
        )
        _CausalAttentionDetailsAdapter().apply(context)
        return context.details

    def facts(self) -> ModuleSemanticFacts:
        if self.limits is None:
            raise RuntimeError("Semantic fact adaptation requires capture limits.")
        return ModuleSemanticFacts(
            type_name=type(self.module).__name__,
            description=component_description(self.module),
            graph_role=self.graph_role(),
            details=_semantic_detail_value(self.details()),
            configuration=module_configuration(self.module, self.limits),
        )


__all__ = ["ModuleSemanticAdapter", "ModuleSemanticFacts"]
