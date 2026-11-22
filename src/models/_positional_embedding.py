from dataclasses import fields
from typing import NotRequired, TypedDict, Unpack, cast

from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig


class _PositionFields(TypedDict):
    num_embeddings: NotRequired[int | None]
    embedding_dim: int | None
    padding_idx: int | None
    auto_expand_flag: bool | None
    class_token_flag: NotRequired[bool]


def supported_position_fields(
    implementation: type[AbsolutePositionalEmbeddingConfig],
    **settings: Unpack[_PositionFields],
) -> dict[str, int | None]:
    supported = {field.name for field in fields(implementation)}
    return {
        name: cast(int | None, value)
        for name, value in settings.items()
        if name in supported
    }


def build_position_config(
    implementation: type[AbsolutePositionalEmbeddingConfig],
    **settings: Unpack[_PositionFields],
) -> AbsolutePositionalEmbeddingConfig:
    selected_fields = supported_position_fields(implementation, **settings)
    return implementation(**selected_fields)
