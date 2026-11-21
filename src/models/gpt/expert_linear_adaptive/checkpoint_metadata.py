"""Recover only dimensions that are unambiguous in hierarchical checkpoints."""

import re
from collections.abc import Mapping


def checkpoint_config_overrides(
    tensor_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    prefix = "hierarchical_model."
    encoder = tensor_shapes.get(prefix + "encoder.byte_embedding.weight")
    decoder = tensor_shapes.get(prefix + "decoder.byte_embedding.weight")
    if encoder is None and decoder is None:
        return {}
    if (
        encoder is None
        or decoder is None
        or len(encoder) != 2
        or len(decoder) != 2
        or encoder[0] != 256
        or decoder[0] != 256
        or encoder[1] < 1
        or encoder[1] != decoder[1]
    ):
        raise ValueError(
            "HAT byte embeddings must both have shape (256, embedding_dim) "
            "with the same positive embedding dimension."
        )
    result = {
        "hierarchical_language_model_flag": True,
        "lm_head_weight_tying_flag": False,
        "hierarchical_byte_embedding_dim": encoder[1],
        "input_dim": 256,
        "output_dim": 256,
    }
    beginning = tensor_shapes.get(prefix + "beginning_of_document")
    if beginning and len(beginning) == 1:
        result["hidden_dim"] = beginning[0]
    for key, owner in (
        ("hierarchical_byte_encoder_num_layers", "encoder.encoder.encoder_model"),
        ("hierarchical_byte_decoder_num_layers", "decoder.decoder.decoder_model"),
        ("stack_num_layers", "backbone"),
    ):
        pattern = re.compile(re.escape(prefix + owner) + r"\.layers\.(\d+)\.")
        indices = sorted(
            {int(match[1]) for name in tensor_shapes if (match := pattern.match(name))}
        )
        if indices and indices == list(range(len(indices))):
            result[key] = len(indices)
    # Byte windows, routing and attention heads are owned by the run configuration.
    return result


__all__ = ["checkpoint_config_overrides"]
