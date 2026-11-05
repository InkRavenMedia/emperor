import torch
from torch import Tensor, nn

from emperor.config import ConfigBase
from emperor.layers import Layer, LayerState
from emperor.linears import LinearAbstract
from emperor.nn import Module
from emperor.transformer.sequence_expansion._config import SequenceExpansionConfig
from emperor.transformer.sequence_expansion._layout import append_valid_tokens
from emperor.transformer.sequence_expansion._state import (
    SequenceExpansionRecord,
    SequenceExpansionResult,
)
from emperor.transformer.sequence_expansion._validation import (
    SequenceExpansionValidator,
)


class SequenceExpansion(Module):
    """Predict per-sample growth and place generated positions after valid tokens."""

    VALIDATOR = SequenceExpansionValidator

    def __init__(
        self,
        cfg: SequenceExpansionConfig,
        overrides: SequenceExpansionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: SequenceExpansionConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate_config(self.cfg)
        self.embedding_dim: int = self.cfg.embedding_dim
        self.max_new_tokens: int = self.cfg.max_new_tokens
        self.model_config: ConfigBase = self.cfg.model_config
        self.positional_embedding_config: ConfigBase | None = (
            self.cfg.positional_embedding_config
        )

        self.count_predictor = self.__build_count_predictor()
        self.VALIDATOR.validate_model(self.count_predictor)
        self.new_position_vector = nn.Parameter(torch.empty(self.embedding_dim))
        self.position_embedding = self.__build_position_embedding()
        nn.init.normal_(self.new_position_vector, std=0.02)

    def __build_count_predictor(self) -> Module:
        return self._build_from_config(
            self.model_config,
            input_dim=self.embedding_dim,
            output_dim=self.max_new_tokens,
        )

    def __build_position_embedding(self) -> Module | None:
        if self.positional_embedding_config is None:
            return None
        overrides = type(self.positional_embedding_config)(
            embedding_dim=self.embedding_dim
        )
        return self.positional_embedding_config.build(overrides=overrides)

    def forward(
        self,
        hidden: Tensor,
        key_padding_mask: Tensor | None = None,
        position_offset: Tensor | None = None,
    ) -> SequenceExpansionResult:
        self.VALIDATOR.validate_inputs(self.cfg, hidden, key_padding_mask)
        padding_mask = self.__normalize_padding_mask(hidden, key_padding_mask)
        self.VALIDATOR.validate_position_offset(hidden, padding_mask, position_offset)
        valid_token_counts, pooled_context = self.__pool_valid_context(
            hidden, padding_mask
        )
        count_prediction = self._predict_count(pooled_context)
        selected_counts, selected_probabilities = self.__select_counts(
            count_prediction.hidden
        )
        generated_mask, position_ids = self.__prepare_generated_positions(
            selected_counts, valid_token_counts, position_offset, hidden.device
        )
        generated_hidden = self.__initialize_generated_hidden(
            hidden, pooled_context, selected_probabilities, position_ids, generated_mask
        )
        expansion_record = SequenceExpansionRecord(
            input_padding_mask=padding_mask,
            selected_counts=selected_counts,
            count_logits=count_prediction.hidden,
            selected_probabilities=selected_probabilities,
            position_ids=position_ids.masked_fill(~generated_mask, -1),
        )
        expanded_hidden, expanded_padding_mask = append_valid_tokens(
            hidden, padding_mask, generated_hidden, ~generated_mask
        )
        return SequenceExpansionResult(
            hidden=expanded_hidden,
            key_padding_mask=expanded_padding_mask,
            expansion=expansion_record,
            loss=count_prediction.loss,
        )

    def __normalize_padding_mask(
        self, hidden: Tensor, padding_mask: Tensor | None
    ) -> Tensor:
        if padding_mask is None:
            return torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        if padding_mask.dtype == torch.bool:
            return padding_mask
        return padding_mask.isneginf()

    def __pool_valid_context(
        self, hidden: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        valid_token_counts = (~padding_mask).sum(dim=1)
        context_sum = hidden.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1)
        pooled_context = context_sum / valid_token_counts.unsqueeze(-1).to(hidden.dtype)
        return valid_token_counts, pooled_context

    def _predict_count(self, context: Tensor) -> LayerState:
        prediction = self.__run_count_predictor(context)
        self.VALIDATOR.validate_prediction(self.cfg, prediction, context)
        return prediction

    def __run_count_predictor(self, context: Tensor) -> LayerState:
        if isinstance(self.count_predictor, LinearAbstract):
            prediction = LayerState(hidden=self.count_predictor(context))
            get_auxiliary_loss = getattr(
                self.count_predictor, "get_auxiliary_loss", None
            )
            if callable(get_auxiliary_loss):
                prediction.loss = get_auxiliary_loss()
            return prediction
        return Layer.run_model_from_hidden(self.count_predictor, context)

    def __select_counts(self, count_logits: Tensor) -> tuple[Tensor, Tensor]:
        probability_dtype = (
            torch.float32
            if count_logits.dtype in (torch.float16, torch.bfloat16)
            else count_logits.dtype
        )
        probabilities = count_logits.softmax(dim=-1, dtype=probability_dtype)
        count_indices = probabilities.argmax(dim=-1)
        selected_counts = count_indices + 1
        selected_probabilities = probabilities.gather(
            1, count_indices.unsqueeze(-1)
        ).squeeze(-1)
        return selected_counts, selected_probabilities

    def __prepare_generated_positions(
        self,
        selected_counts: Tensor,
        valid_token_counts: Tensor,
        position_offset: Tensor | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        generated_sequence_length = int(selected_counts.max().item())
        relative_positions = torch.arange(generated_sequence_length, device=device)
        generated_mask = relative_positions.unsqueeze(0) < selected_counts.unsqueeze(1)
        position_starts = (
            valid_token_counts if position_offset is None else position_offset
        )
        position_ids = position_starts.unsqueeze(1) + relative_positions.unsqueeze(0)
        return generated_mask, position_ids

    def __initialize_generated_hidden(
        self,
        hidden: Tensor,
        pooled_context: Tensor,
        selected_probabilities: Tensor,
        position_ids: Tensor,
        generated_mask: Tensor,
    ) -> Tensor:
        generated_hidden = self.new_position_vector.to(hidden.dtype).expand(
            *position_ids.shape, -1
        )
        if self.position_embedding is not None:
            active_position_ids = position_ids.masked_fill(~generated_mask, 0)
            generated_hidden = generated_hidden + self.__encode_positions(
                active_position_ids
            ).to(hidden.dtype)
        generated_hidden = generated_hidden + (
            selected_probabilities.to(hidden.dtype)[:, None, None]
            * pooled_context.unsqueeze(1)
        )
        return generated_hidden.masked_fill(~generated_mask.unsqueeze(-1), 0)

    def __encode_positions(self, position_ids: Tensor) -> Tensor:
        position_count = int(position_ids.max().item()) + 1
        position_marker = (self.position_embedding.padding_idx or 0) + 1
        position_input = position_ids.new_full((1, position_count), position_marker)
        position_encodings = self.position_embedding(position_input).squeeze(0)
        return position_encodings[position_ids]
