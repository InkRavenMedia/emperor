import copy

import torch
from torch import Tensor, nn

from emperor.embedding.hierarchical import BYTE_VALUES, OUTPUT_SYMBOLS
from emperor.layers import LayerState
from emperor.nn import Module, grouped_forward_calls

from ._config import HierarchicalByteDecoderConfig
from ._generation import HierarchicalByteGenerator
from ._records import (
    ByteGenerationOptions,
    ByteTokenGenerationOutput,
    HierarchicalByteDecoderOutput,
)
from ._validation import HierarchicalByteDecoderValidator


class HierarchicalByteDecoder(Module):
    VALIDATOR = HierarchicalByteDecoderValidator

    def __init__(
        self,
        cfg: HierarchicalByteDecoderConfig,
        overrides: HierarchicalByteDecoderConfig | None = None,
    ):
        self.VALIDATOR.validate_config_types(cfg, overrides)
        super().__init__()
        overridden_config = self._override_config(cfg, overrides)
        self.cfg = copy.deepcopy(overridden_config)
        self._max_group_tokens = self.VALIDATOR.resolve_config(self.cfg)
        self.byte_embedding = nn.Embedding(BYTE_VALUES, self.cfg.byte_embedding_dim)
        self.byte_position = (
            self.cfg.byte_position_config.build()
            if self.cfg.byte_position_config is not None
            else None
        )
        self.conditioning_projection = self.cfg.conditioning_projection_config.build()
        self.decoder = self.cfg.decoder_config.build()
        self.output_projection = self.cfg.output_projection_config.build()
        self.generation = HierarchicalByteGenerator(self)

    def generate_token(
        self,
        conditioning: Tensor,
        *,
        prefix: str | bytes = "",
        max_new_bytes: int | None = None,
        options: ByteGenerationOptions | None = None,
        generator: torch.Generator | None = None,
    ) -> ByteTokenGenerationOutput:
        generation_options = options or ByteGenerationOptions()
        return self.generation.generate(
            conditioning, prefix, max_new_bytes, generation_options, generator
        )

    def forward(
        self,
        conditioning: Tensor,
        byte_prefix_ids: Tensor,
        byte_lengths: Tensor,
        attention_mask: Tensor | None = None,
    ) -> HierarchicalByteDecoderOutput:
        byte_prefix_ids, byte_lengths, valid_tokens = (
            self.VALIDATOR.validate_forward_inputs(
                self.cfg,
                conditioning,
                byte_prefix_ids,
                byte_lengths,
                attention_mask,
                self.byte_embedding.weight.device,
            )
        )
        token_offsets = self.__build_token_offsets(byte_lengths, valid_tokens)
        total_positions = int(token_offsets[-1])
        if total_positions == 0:
            return self.__empty_predictions(conditioning, token_offsets)

        valid_token_indices = valid_tokens.reshape(-1).nonzero().flatten()
        selected_byte_lengths = byte_lengths.reshape(-1)[valid_token_indices]
        self.VALIDATOR.validate_prefix_group_sizes(
            selected_byte_lengths, self._max_group_tokens
        )
        conditioning_state = self.__project_conditioning(
            conditioning, valid_token_indices
        )
        decoded_state = self.__decode_prefix_groups(
            byte_prefix_ids,
            selected_byte_lengths,
            valid_token_indices,
            token_offsets,
            total_positions,
            conditioning_state,
        )
        return self.__project_predictions(decoded_state, token_offsets)

    def __build_token_offsets(self, byte_lengths, valid_tokens):
        flat_byte_lengths = byte_lengths.reshape(-1)
        flat_valid_tokens = valid_tokens.reshape(-1)
        unmasked_prediction_counts = flat_byte_lengths + 1
        prediction_counts = torch.where(
            flat_valid_tokens, unmasked_prediction_counts, 0
        )
        initial_offset = prediction_counts.new_zeros(1)
        cumulative_positions = prediction_counts.cumsum(0)
        token_offsets = torch.cat((initial_offset, cumulative_positions))
        return token_offsets

    def __empty_predictions(self, conditioning, token_offsets):
        empty_logits = conditioning.new_empty((0, OUTPUT_SYMBOLS))
        auxiliary_loss = conditioning.new_zeros(())
        return HierarchicalByteDecoderOutput(
            empty_logits, token_offsets, auxiliary_loss
        )

    def __project_conditioning(self, conditioning, valid_token_indices):
        flat_conditioning = conditioning.reshape(-1, self.cfg.conditioning_dim)
        selected_conditioning = flat_conditioning[valid_token_indices]
        projection_input = LayerState(hidden=selected_conditioning)
        return self.conditioning_projection(projection_input)

    def __decode_prefix_groups(
        self,
        byte_prefix_ids,
        selected_byte_lengths,
        valid_token_indices,
        token_offsets,
        total_positions,
        conditioning_state,
    ):
        auxiliary_loss = conditioning_state.loss
        if auxiliary_loss is None:
            auxiliary_loss = conditioning_state.hidden.new_zeros(())
        packed_features = None
        offset_count = token_offsets.numel()
        flat_token_count = offset_count - 1
        flat_prefixes = byte_prefix_ids.reshape(
            flat_token_count, byte_prefix_ids.shape[-1]
        )
        packed_shape = (total_positions, self.cfg.byte_embedding_dim)
        with grouped_forward_calls() as group_schedule:
            for byte_length in selected_byte_lengths.unique().tolist():
                with group_schedule():
                    compact_indices = (
                        (selected_byte_lengths == byte_length).nonzero().flatten()
                    )
                    original_indices = valid_token_indices[compact_indices]
                    byte_ids = flat_prefixes[original_indices, :byte_length]
                    encoded_features, group_loss = self.__decode_prefix_group(
                        conditioning_state.hidden, compact_indices, byte_ids
                    )
                    if packed_features is None:
                        packed_features = encoded_features.new_zeros(packed_shape)
                    packed_positions = self.__group_prediction_positions(
                        token_offsets, original_indices, byte_length
                    )
                    flat_positions = packed_positions.flatten()
                    flat_features = encoded_features.reshape(
                        -1, self.cfg.byte_embedding_dim
                    )
                    packed_features[flat_positions] = flat_features
                    group_position_count = packed_positions.numel()
                    group_weight = group_position_count / total_positions
                    weighted_group_loss = group_loss * group_weight
                    auxiliary_loss = auxiliary_loss + weighted_group_loss
        return LayerState(hidden=packed_features, loss=auxiliary_loss)

    def __decode_prefix_group(self, conditioning_features, compact_indices, byte_ids):
        conditioning_prefix = conditioning_features[compact_indices, None]
        byte_embeddings = self.byte_embedding(byte_ids)
        prefixed_embeddings = torch.cat((conditioning_prefix, byte_embeddings), dim=1)
        prediction_count = byte_ids.shape[-1] + 1
        position_shape = (len(compact_indices), prediction_count)
        position_ids = byte_ids.new_zeros(position_shape)
        positioned_embeddings = prefixed_embeddings
        if self.byte_position is not None:
            positioned_embeddings = prefixed_embeddings + self.byte_position(
                position_ids
            )
        return self.decoder(target_token_embeddings=positioned_embeddings)

    def __group_prediction_positions(
        self, token_offsets, original_indices, byte_length
    ):
        prediction_count = byte_length + 1
        relative_positions = torch.arange(
            prediction_count, device=original_indices.device
        )
        group_offsets = token_offsets[original_indices, None]
        packed_positions = group_offsets + relative_positions
        return packed_positions

    def __project_predictions(self, decoded_state, token_offsets):
        projection_input = LayerState(hidden=decoded_state.hidden)
        projected_state = self.output_projection(projection_input)
        auxiliary_loss = decoded_state.loss
        if projected_state.loss is not None:
            auxiliary_loss = auxiliary_loss + projected_state.loss
        scalar_loss = auxiliary_loss.reshape(())
        return HierarchicalByteDecoderOutput(
            projected_state.hidden, token_offsets, scalar_loss
        )
