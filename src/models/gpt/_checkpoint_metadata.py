"""Recover only dimensions that are unambiguous in hierarchical checkpoints."""

import re
from collections.abc import Mapping

_LAYER_COUNT_FIELDS = {
    "encoder.encoder.encoder_model": "hierarchical_byte_encoder_num_layers",
    "decoder.decoder.decoder_model": "hierarchical_byte_decoder_num_layers",
    "backbone": "stack_num_layers",
}
_LAYER_PARAMETER = re.compile(
    r"hierarchical_model\.(?P<stack>.+?)\.layers\.(?P<index>\d+)\."
)


def checkpoint_config_overrides(
    tensor_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    embedding_dim = _byte_embedding_dim(tensor_shapes)
    if embedding_dim is None:
        return {}
    overrides: dict[str, object] = {
        "hierarchical_language_model_flag": True,
        "lm_head_weight_tying_flag": False,
        "hierarchical_byte_embedding_dim": embedding_dim,
        "input_dim": 256,
        "output_dim": 256,
    }
    beginning = tensor_shapes.get("hierarchical_model.beginning_of_document")
    if beginning and len(beginning) == 1:
        overrides["hidden_dim"] = beginning[0]
    overrides.update(_stack_depths(tensor_shapes))
    return overrides


def _byte_embedding_dim(tensor_shapes: Mapping[str, tuple[int, ...]]) -> int | None:
    encoder = tensor_shapes.get("hierarchical_model.encoder.byte_embedding.weight")
    decoder = tensor_shapes.get("hierarchical_model.decoder.byte_embedding.weight")
    match encoder, decoder:
        case None, None:
            return None
        case (256, width), (256, decoder_width) if (
            width >= 1 and width == decoder_width
        ):
            return width
        case _:
            raise ValueError(
                "HAT byte embeddings must both have shape (256, embedding_dim) "
                "with the same positive embedding dimension."
            )


def _stack_depths(tensor_shapes: Mapping[str, tuple[int, ...]]) -> dict[str, int]:
    indices_by_stack = {stack: set[int]() for stack in _LAYER_COUNT_FIELDS}
    for name in tensor_shapes:
        match = _LAYER_PARAMETER.match(name)
        if match is not None and match["stack"] in indices_by_stack:
            indices_by_stack[match["stack"]].add(int(match["index"]))

    return {
        _LAYER_COUNT_FIELDS[stack]: len(indices)
        for stack, indices in indices_by_stack.items()
        if indices and max(indices) == len(indices) - 1
    }


__all__ = ["checkpoint_config_overrides"]
