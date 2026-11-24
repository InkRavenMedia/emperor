from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.embedding.hierarchical.decoder import (
    ByteGenerationOptions,
    HierarchicalByteDecoderOutput,
)
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.layers import (
    LayerConfig,
    LayerNormalizationDelegate,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    NormalizationOptions,
    RecurrentCompositionConfig,
)
from emperor.transformer import TransformerDecoderLayerState
from emperor.transformer.hierarchical import (
    HierarchicalLanguageModelBatch,
    HierarchicalLanguageModelConfig,
    HierarchicalTextGenerationOutput,
)
from emperor.transformer.sequence_expansion import SequenceExpansion
from models import _sequence_expansion

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _EmbeddingSettings(Protocol):
    hierarchical_language_model_flag: Final[bool]
    layer_norm_flag: Final[bool]
    normalization: Final[NormalizationOptions]
    dropout_probability: Final[float]


class _LmHeadSettings(Protocol):
    weight_tying_flag: Final[bool]
    bias_flag: Final[bool]


class _BoundarySettings(Protocol):
    embedding_options: Final[_EmbeddingSettings]
    lm_head_options: Final[_LmHeadSettings]


class _ExperimentSettings(Protocol):
    boundary_config: Final[object]
    positional_embedding_config: Final[AbsolutePositionalEmbeddingConfig | None]
    decoder_config: Final[LayerStackConfig | RecurrentCompositionConfig | None]
    decoder_output_normalization: Final[NormalizationOptions]
    hierarchical_language_model_config: Final[HierarchicalLanguageModelConfig | None]


_Experiment = TypeVar("_Experiment", bound=_ExperimentSettings)
_Boundary = TypeVar("_Boundary", bound=_BoundarySettings)


class GptBackbone(LanguageModelExperiment, Generic[_Experiment, _Boundary]):
    experiment_type: type[_Experiment]
    boundary_type: type[_Boundary]
    invalid_experiment_message: str
    token_embedding: nn.Module
    positional_embedding: nn.Module
    embedding_dropout: nn.Dropout
    embedding_layer_norm: LayerNormalizationDelegate
    transformer: nn.Module
    decoder_layer_norm: LayerNormalizationDelegate
    lm_head: nn.Linear
    hierarchical_model: nn.Module
    sequence_expansion_enabled: bool
    experiment_config: _Experiment
    boundary_config: _Boundary

    _INTEGER_DTYPES = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    @classmethod
    def _validate_experiment_config(cls, config: "ModelConfig") -> _Experiment:
        stages = config.experiment_config
        if stages is None:
            raise ValueError("experiment_config is required to build the model")
        if not isinstance(stages, cls.experiment_type):
            raise TypeError(cls.invalid_experiment_message)
        return stages

    @classmethod
    def _validate_boundary_config(cls, stages: _Experiment) -> _Boundary:
        boundary = stages.boundary_config
        if not isinstance(boundary, cls.boundary_type):
            if boundary is None:
                raise ValueError("experiment_config.boundary_config is required")
            raise TypeError(
                "config.experiment_config.boundary_config must be a resolved GptBoundaryConfig."
            )
        return boundary

    @staticmethod
    def _validate_tied_vocabulary_sizes(
        config: "ModelConfig",
        boundary_config: _Boundary,
    ) -> None:
        if (
            boundary_config.lm_head_options.weight_tying_flag
            and config.input_dim != config.output_dim
        ):
            raise ValueError(
                "GPT LM head weight tying requires input_dim to equal output_dim."
            )

    def _build_embedding_layer_norm(self) -> LayerNormalizationDelegate:
        layer_norm_position = LayerNormPositionOptions.DISABLED
        if self.boundary_config.embedding_options.layer_norm_flag:
            layer_norm_position = LayerNormPositionOptions.BEFORE
        normalization_config = LayerConfig(
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
            layer_norm_position=layer_norm_position,
            normalization=self.boundary_config.embedding_options.normalization,
        )
        return LayerNormalizationDelegate(normalization_config)

    def _build_embedding_dropout(self) -> nn.Dropout:
        return nn.Dropout(self.boundary_config.embedding_options.dropout_probability)

    def _build_decoder(self) -> nn.Module:
        decoder = self.experiment_config.decoder_config.build()
        return decoder

    def _build_decoder_layer_norm(self) -> LayerNormalizationDelegate:
        return LayerNormalizationDelegate(
            LayerConfig(
                input_dim=self.cfg.hidden_dim,
                output_dim=self.cfg.hidden_dim,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                normalization=self.experiment_config.decoder_output_normalization,
            )
        )

    def _build_lm_head(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.output_dim,
            bias=self.boundary_config.lm_head_options.bias_flag,
        )

    def _tie_lm_head_weights(self) -> None:
        if not self.boundary_config.lm_head_options.weight_tying_flag:
            return
        self.lm_head.weight = self.token_embedding.weight

    def forward_for_input_tokens(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        if not self.sequence_expansion_enabled:
            return super().forward_for_input_tokens(input_ids)
        return self(input_ids, align_to_input=True)

    def _run_decoder(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        align_to_input: bool = False,
    ) -> tuple[Tensor, Tensor]:
        decoder_state = self.transformer(
            TransformerDecoderLayerState(
                hidden=hidden,
                target_key_padding_mask=attention_mask == 0,
            )
        )
        sequence_output = self.decoder_layer_norm.before_model(
            LayerState(hidden=decoder_state.hidden)
        ).hidden
        if align_to_input:
            sequence_output = decoder_state.gather_input_tokens(sequence_output)
        auxiliary_loss = (
            decoder_state.loss
            if decoder_state.loss is not None
            else sequence_output.new_zeros(())
        )
        return sequence_output, auxiliary_loss

    def _prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        input_ids = self.__prepare_input_ids(input_ids)
        attention_mask = self.__prepare_attention_mask(attention_mask, input_ids)
        return input_ids, attention_mask

    def __prepare_input_ids(self, input_ids: Tensor) -> Tensor:
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 [batch, sequence] tensor.")
        if input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must contain a non-empty prompt per batch.")
        if input_ids.size(1) > self.cfg.sequence_length:
            raise ValueError(
                "input sequence length exceeds the configured context length "
                f"of {self.cfg.sequence_length}."
            )
        if input_ids.dtype not in self._INTEGER_DTYPES:
            raise TypeError("input_ids must contain integer token IDs.")
        input_ids = input_ids.to(self.device, dtype=torch.long)
        if bool(torch.any(input_ids < 0).item()) or bool(
            torch.any(input_ids >= self.cfg.input_dim).item()
        ):
            raise ValueError(
                f"input_ids must be in the range [0, {self.cfg.input_dim})."
            )
        return input_ids

    def __prepare_attention_mask(
        self,
        attention_mask: Tensor | None,
        input_ids: Tensor,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids)
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor or None.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, "
                f"received {tuple(attention_mask.shape)} and "
                f"{tuple(input_ids.shape)}."
            )
        return attention_mask.to(self.device)

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
        if not hasattr(self, "hierarchical_model"):
            raise ValueError(
                "generate_text requires hierarchical_language_model_flag=True"
            )
        return self.hierarchical_model.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            max_new_bytes=max_new_bytes,
            options=options,
            prompt_is_complete=prompt_is_complete,
            roll_context=roll_context,
        )

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if hasattr(self, "hierarchical_model"):
            raise ValueError("Use generate_text for hierarchical language modeling")
        self.__validate_max_new_tokens(max_new_tokens)
        input_ids, _ = self._prepare_inputs(input_ids, attention_mask=None)
        self.__validate_generation_length(input_ids, max_new_tokens)
        return self.__generate_greedily(input_ids, max_new_tokens)

    @staticmethod
    def __validate_max_new_tokens(max_new_tokens: int) -> None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

    def __validate_generation_length(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> None:
        if input_ids.size(1) + max_new_tokens > self.cfg.sequence_length:
            raise ValueError(
                "prompt plus max_new_tokens exceeds the configured context length "
                f"of {self.cfg.sequence_length}."
            )

    def __generate_greedily(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        was_training = self.training
        generated = input_ids.clone()
        try:
            self.eval()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits, _auxiliary_loss = self.forward_for_input_tokens(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)
        return generated


class GptModel(GptBackbone[_Experiment, _Boundary]):
    def __init__(
        self,
        config: "ModelConfig",
    ) -> None:
        experiment_config = self._validate_experiment_config(config)
        boundary_config = self._validate_boundary_config(experiment_config)
        self._validate_tied_vocabulary_sizes(config, boundary_config)
        hierarchical = experiment_config.hierarchical_language_model_config
        enabled = boundary_config.embedding_options.hierarchical_language_model_flag
        if enabled != (hierarchical is not None):
            raise ValueError(
                "hierarchical_language_model_flag must match its supplied configuration"
            )
        if not enabled:
            if experiment_config.positional_embedding_config is None:
                raise ValueError(
                    "experiment_config.positional_embedding_config is required"
                )
            if experiment_config.decoder_config is None:
                raise ValueError("experiment_config.decoder_config is required")
        super().__init__(config)
        self.experiment_config = experiment_config
        self.sequence_expansion_enabled = (
            _sequence_expansion.expanded_sequence_capacity(
                self.experiment_config.decoder_config, 0
            )
            > 0
        )
        self.boundary_config = boundary_config
        if enabled:
            self._validate_hierarchical_output(experiment_config, boundary_config)
            self.hierarchical_model = hierarchical.build()
            return
        self.token_embedding = self.__build_token_embedding()
        self.positional_embedding = self.__build_positional_embedding()
        self.embedding_layer_norm = self._build_embedding_layer_norm()
        self.embedding_dropout = self._build_embedding_dropout()
        self.transformer = self._build_decoder()
        self.decoder_layer_norm = self._build_decoder_layer_norm()
        self.lm_head = self._build_lm_head()
        self._tie_lm_head_weights()

    @staticmethod
    def _validate_hierarchical_output(
        experiment_config: _Experiment, boundary_config: _Boundary
    ) -> None:
        if boundary_config.lm_head_options.weight_tying_flag:
            raise ValueError(
                "Hierarchical language modeling requires lm_head_weight_tying_flag=False"
            )
        SequenceExpansion.VALIDATOR.validate_fixed_token_output(
            experiment_config.decoder_config, "Hierarchical language modeling"
        )

    def __build_token_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.cfg.input_dim, self.cfg.hidden_dim)

    def __build_positional_embedding(self) -> nn.Module:
        positional_embedding = (
            self.experiment_config.positional_embedding_config.build()
        )
        return positional_embedding

    def forward(
        self,
        input_ids: Tensor | HierarchicalLanguageModelBatch,
        attention_mask: Tensor | None = None,
        align_to_input: bool = False,
    ) -> tuple[Tensor, Tensor] | HierarchicalByteDecoderOutput:
        """Return all token logits, or restore input positions with align_to_input."""
        if hasattr(self, "hierarchical_model"):
            if attention_mask is not None:
                raise ValueError("hierarchical batches carry their own attention_mask")
            return self.hierarchical_model(input_ids)
        if isinstance(input_ids, HierarchicalLanguageModelBatch):
            raise ValueError(
                "HierarchicalLanguageModelBatch requires hierarchical_language_model_flag=True"
            )
        input_ids, attention_mask = self._prepare_inputs(input_ids, attention_mask)
        hidden = self.__build_input_embeddings(input_ids)
        sequence_output, auxiliary_loss = self._run_decoder(
            hidden, attention_mask, align_to_input
        )
        logits = self.lm_head(sequence_output)
        return logits, LayerState.sum_losses(
            auxiliary_loss, LayerState.module_loss(self.positional_embedding)
        ).reshape(())

    def __build_input_embeddings(self, input_ids: Tensor) -> Tensor:
        token_embedding = self.token_embedding(input_ids)
        positional_embedding = self.positional_embedding(input_ids)
        hidden = token_embedding + positional_embedding
        hidden = self.embedding_layer_norm.before_model(
            LayerState(hidden=hidden)
        ).hidden
        return self.embedding_dropout(hidden)
