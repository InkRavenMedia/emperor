from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

TensorShape = tuple[int, ...]

_LAYER_WEIGHT = re.compile(
    r"main_model\.(?P<recurrent>(?:step\.)?block_model\.)?"
    r"layers\.(?P<index>\d+)\.model\.weight_params"
)


def checkpoint_config_overrides(
    tensor_shapes: Mapping[str, TensorShape],
) -> dict[str, int]:
    """Recover the stable Linear package dimensions from its state-dict layout."""

    overrides, projection_widths = _read_projection_dimensions(tensor_shapes)
    stack = _read_stack_weights(tensor_shapes)
    _add_hidden_dimension(overrides, projection_widths, stack.matrix_dimensions)
    _add_stack_layer_count(overrides, stack.index_labels)
    return overrides


def _read_projection_dimensions(
    tensor_shapes: Mapping[str, TensorShape],
) -> tuple[dict[str, int], set[int]]:
    overrides: dict[str, int] = {}
    hidden_widths: set[int] = set()
    input_weight = _matrix_shape(tensor_shapes.get("input_model.model.weight_params"))
    if input_weight is not None:
        input_dim, hidden_dim = input_weight
        overrides["input_dim"] = input_dim
        hidden_widths.add(hidden_dim)

    output_weight = _matrix_shape(tensor_shapes.get("output_model.model.weight_params"))
    if output_weight is not None:
        hidden_dim, output_dim = output_weight
        overrides["output_dim"] = output_dim
        hidden_widths.add(hidden_dim)
    return overrides, hidden_widths


@dataclass
class _StackWeights:
    index_labels: set[str] = field(default_factory=set[str])
    matrix_dimensions: set[int] = field(default_factory=set[int])


def _read_stack_weights(tensor_shapes: Mapping[str, TensorShape]) -> _StackWeights:
    direct = _StackWeights()
    recurrent = _StackWeights()
    for name, shape in tensor_shapes.items():
        match = _LAYER_WEIGHT.fullmatch(name)
        if match is None:
            continue
        stack = recurrent if match["recurrent"] else direct
        stack.index_labels.add(match["index"])
        matrix = _matrix_shape(shape)
        if matrix is not None:
            stack.matrix_dimensions.update(matrix)
    return _StackWeights(
        index_labels=direct.index_labels or recurrent.index_labels,
        matrix_dimensions=direct.matrix_dimensions or recurrent.matrix_dimensions,
    )


def _add_hidden_dimension(
    overrides: dict[str, int], projection_widths: set[int], stack_widths: set[int]
) -> None:
    widths = projection_widths | stack_widths
    if len(widths) == 1:
        overrides["hidden_dim"] = widths.pop()


def _add_stack_layer_count(overrides: dict[str, int], index_labels: set[str]) -> None:
    indices = {int(label) for label in index_labels}
    layer_count = len(indices)
    if layer_count and indices == set(range(layer_count)):
        overrides["stack_num_layers"] = layer_count


def _matrix_shape(shape: TensorShape | None) -> tuple[int, int] | None:
    if shape is None or len(shape) != 2:
        return None
    return shape[0], shape[1]


__all__ = ["checkpoint_config_overrides"]
