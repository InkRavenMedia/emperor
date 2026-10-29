import copy

import torch
from torch import nn

from emperor.embedding.hierarchical import HAT_PROTOCOL_VERSION
from emperor.embedding.hierarchical.decoder import (
    ByteGenerationOptions,
    HierarchicalByteDecoderOutput,
)
from emperor.layers import LayerState
from emperor.nn import Module, freeze_forward_schedules, grouped_forward_calls
from emperor.transformer import TransformerDecoderLayerState

from ._batch import HierarchicalLanguageModelBatch
from ._records import HierarchicalTextGenerationOutput
from ._text import HierarchicalTextCodec
from ._validation import HierarchicalLanguageModelValidator


class HierarchicalLanguageModel(Module):
    """Compose independent token encoding, causal context, and byte prediction.

    Context/target shifting belongs to the batch codec; the backbone output at
    slot i predicts that slot's next-token target. BOS is a learned vector added
    by this integration. See docs/hierarchical-implementation-review.md for the
    v2 mapping and the differences from the released TFree-HAT models.
    """

    VALIDATOR = HierarchicalLanguageModelValidator

    def __init__(self, cfg, overrides=None):
        self.VALIDATOR.validate_config_types(cfg, overrides)
        super().__init__()
        overridden_config = self._override_config(cfg, overrides)
        self.cfg = copy.deepcopy(overridden_config)
        self._max_batch = self.VALIDATOR.resolve_config(self.cfg)
        self.encoder = self.cfg.embedding_config.build()
        self.backbone = self.cfg.backbone_config.build()
        self.decoder = self.cfg.decoding_config.build()
        # One C in HAT equations 1 and 4; standalone components still own a table.
        self.decoder.byte_embedding = self.encoder.byte_embedding
        self.register_buffer(
            "_hat_protocol_version", torch.tensor(HAT_PROTOCOL_VERSION)
        )
        self.positions = (
            self.cfg.position_config.build()
            if self.cfg.position_config is not None
            else None
        )
        self.input_normalization = self._build_normalization(
            self.cfg.embedding_normalization_config
        )
        self.output_normalization = self._build_normalization(
            self.cfg.output_normalization_config
        )
        self.dropout = nn.Dropout(self.cfg.dropout_probability)
        beginning_features = torch.empty(self.cfg.embedding_config.output_dim)
        self.beginning_of_document = nn.Parameter(beginning_features)
        beginning_feature_count = self.beginning_of_document.numel()
        initialization_scale = beginning_feature_count**-0.5
        nn.init.normal_(self.beginning_of_document, std=initialization_scale)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        version = state_dict.get(prefix + "_hat_protocol_version")
        if (
            version is None
            or version.numel() != 1
            or version.item() != HAT_PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "Incompatible HAT checkpoint protocol. Legacy independent embeddings "
                "and 258-symbol heads cannot be converted losslessly; use the original "
                "model or an explicit checkpoint conversion."
            )
        encoder = state_dict.get(prefix + "encoder.byte_embedding.weight")
        decoder = state_dict.get(prefix + "decoder.byte_embedding.weight")
        if (
            encoder is not None
            and decoder is not None
            and not torch.equal(encoder, decoder)
        ):
            raise RuntimeError(
                "Checkpoint contains conflicting shared character embeddings"
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def _build_normalization(config):
        return nn.Identity() if config is None else config.build_normalization()

    def _condition(self, batch):
        self.VALIDATOR.validate_batch(batch, self.cfg.sequence_length)
        device = self.beginning_of_document.device
        valid_tokens = batch.attention_mask.to(device)
        beginning_tokens = batch.bos_mask.to(device)
        encoder_mask = valid_tokens & ~beginning_tokens
        encoded_state = self.encoder(batch.context_texts, attention_mask=encoder_mask)
        conditioning_features = torch.where(
            beginning_tokens[..., None],
            self.beginning_of_document,
            encoded_state.hidden,
        )
        sequence_lengths = valid_tokens.sum(-1)
        return self.__condition_sequence_groups(
            conditioning_features, sequence_lengths, encoded_state.loss
        )

    def __condition_sequence_groups(
        self, token_features, sequence_lengths, auxiliary_loss
    ):
        valid_token_count = int(sequence_lengths.sum())
        conditioning_features = torch.zeros_like(token_features)
        with grouped_forward_calls() as group_schedule:
            for sequence_length in sequence_lengths.unique().tolist():
                with group_schedule():
                    if not sequence_length:
                        continue
                    row_indices = (
                        (sequence_lengths == sequence_length).nonzero().flatten()
                    )
                    group_row_count = len(row_indices)
                    self.VALIDATOR.validate_backbone_batch_size(
                        group_row_count, self._max_batch
                    )
                    group_features = token_features[row_indices, :sequence_length]
                    decoded_state = self.__condition_sequence_group(group_features)
                    conditioning_features[row_indices, :sequence_length] = (
                        decoded_state.hidden
                    )
                    if decoded_state.loss is not None:
                        group_token_count = group_row_count * sequence_length
                        group_weight = group_token_count / valid_token_count
                        weighted_group_loss = decoded_state.loss * group_weight
                        auxiliary_loss = auxiliary_loss + weighted_group_loss
        return LayerState(hidden=conditioning_features, loss=auxiliary_loss)

    def __condition_sequence_group(self, token_features):
        positioned_features = token_features
        if self.positions is not None:
            position_marker = (self.cfg.position_config.padding_idx or 0) + 1
            position_ids = torch.full(
                token_features.shape[:2],
                position_marker,
                dtype=torch.long,
                device=token_features.device,
            )
            positioned_features = token_features + self.positions(position_ids)
        normalized_features = self.input_normalization(positioned_features)
        backbone_features = self.dropout(normalized_features)
        backbone_input = TransformerDecoderLayerState(hidden=backbone_features)
        decoded_state = self.backbone(backbone_input)
        normalized_output = self.output_normalization(decoded_state.hidden)
        return LayerState(hidden=normalized_output, loss=decoded_state.loss)

    def forward(
        self, batch: HierarchicalLanguageModelBatch
    ) -> HierarchicalByteDecoderOutput:
        conditioning_state = self._condition(batch)
        decoded_output = self.decoder(
            conditioning_state.hidden,
            batch.byte_prefix_ids,
            batch.byte_lengths,
            batch.attention_mask,
        )
        auxiliary_loss = decoded_output.loss + conditioning_state.loss
        return HierarchicalByteDecoderOutput(
            decoded_output.logits, decoded_output.token_offsets, auxiliary_loss
        )

    def generate_text(
        self,
        prompt: str | HierarchicalTextGenerationOutput = "",
        *,
        max_new_tokens: int = 32,
        max_new_bytes: int = 256,
        options: ByteGenerationOptions | None = None,
        prompt_is_complete: bool = False,
        roll_context: bool = False,
    ) -> HierarchicalTextGenerationOutput:
        generation_options = options or ByteGenerationOptions()
        self.VALIDATOR.validate_options(generation_options)
        self.VALIDATOR.validate_text_generation_limits(
            max_new_tokens, max_new_bytes, prompt_is_complete
        )
        if type(roll_context) is not bool:
            raise TypeError("roll_context must be bool")
        continuation = (
            prompt if isinstance(prompt, HierarchicalTextGenerationOutput) else None
        )
        if continuation is not None:
            self.VALIDATOR.validate_continuation(continuation, prompt_is_complete)
            completed_tokens = list(continuation.completed_tokens)
            partial_token = continuation.partial_token
            prompt_bytes = continuation.data
        else:
            completed_tokens, partial_token = self.__split_prompt(
                prompt, prompt_is_complete
            )
            prompt_bytes = prompt.encode("utf-8")
        sampling_generator = None
        if (
            generation_options.do_sample
            or generation_options.seed is not None
            or (continuation is not None and continuation.generator_state is not None)
        ):
            sampling_generator = torch.Generator()
            if continuation is not None and continuation.generator_state is not None:
                sampling_generator.set_state(continuation.generator_state.cpu())
            else:
                seed = generation_options.seed
                sampling_generator.manual_seed(
                    seed if seed is not None else int(torch.randint(2**63 - 1, ()))
                )
        training_states = [(child, child.training) for child in self.modules()]
        try:
            self.eval()
            with torch.no_grad(), freeze_forward_schedules():
                return self.__generate_text_tokens(
                    prompt_bytes,
                    completed_tokens,
                    partial_token,
                    max_new_tokens,
                    max_new_bytes,
                    generation_options,
                    sampling_generator,
                    roll_context,
                )
        finally:
            for child, was_training in training_states:
                child.training = was_training

    def __split_prompt(self, prompt, prompt_is_complete):
        codec = HierarchicalTextCodec(self.cfg.embedding_config.max_token_bytes)
        completed = list(codec.split_text(prompt))
        partial = ""
        if completed and not prompt_is_complete and not completed[-1][-1].isspace():
            partial = completed.pop()
        return [word.encode("utf-8") for word in completed], partial.encode("utf-8")

    def __generate_text_tokens(
        self,
        prompt_bytes,
        completed_tokens,
        partial_token,
        max_new_tokens,
        max_new_bytes,
        options,
        sampling_generator,
        roll_context,
    ):
        generated_data = bytearray(prompt_bytes)
        generated_byte_count = 0
        generated_token_count = 0
        stop_reason = "token_budget"
        while generated_token_count < max_new_tokens:
            if generated_byte_count >= max_new_bytes:
                stop_reason = "byte_budget"
                break
            if (
                not roll_context
                and len(completed_tokens) + 1 > self.cfg.sequence_length
            ):
                stop_reason = "context_limit"
                break
            conditioning = self.__next_token_conditioning(completed_tokens)
            generated_token = self.decoder.generate_token(
                conditioning,
                prefix=partial_token,
                max_new_bytes=max_new_bytes - generated_byte_count,
                options=options,
                generator=sampling_generator,
            )
            generated_data.extend(generated_token.data[len(partial_token) :])
            generated_byte_count += generated_token.new_bytes
            partial_token = generated_token.data
            if generated_token.stop_reason != "end_of_token":
                stop_reason = generated_token.stop_reason
                break
            completed_tokens.append(partial_token)
            partial_token = b""
            generated_token_count += 1
        return HierarchicalTextGenerationOutput(
            bytes(generated_data),
            stop_reason,
            generated_byte_count,
            generated_token_count,
            tuple(completed_tokens),
            partial_token,
            sampling_generator.get_state() if sampling_generator is not None else None,
        )

    def __next_token_conditioning(self, completed_tokens):
        context_tokens = ([""] + completed_tokens)[-self.cfg.sequence_length :]
        context_token_count = len(context_tokens)
        includes_beginning = len(completed_tokens) < self.cfg.sequence_length
        remaining_context_count = context_token_count - 1
        beginning_mask = [includes_beginning] + [False] * remaining_context_count
        generation_example = {
            "context_texts": context_tokens,
            "target_texts": [None] * context_token_count,
            "bos_mask": beginning_mask,
        }
        generation_batch = HierarchicalLanguageModelBatch.collate([generation_example])
        conditioning_state = self._condition(generation_batch)
        return conditioning_state.hidden[0, -1]
