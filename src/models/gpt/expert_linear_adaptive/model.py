from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.embedding._token_text import TokenTextAdapter
from emperor.embedding.contextual import ByteContextualEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.embedding.hierarchical.decoder import (
    HierarchicalByteDecoderOutput,
)
from emperor.layers import (
    LayerState,
)
from emperor.transformer.hierarchical import (
    HierarchicalLanguageModelBatch,
)
from emperor.transformer.sequence_expansion import SequenceExpansion
from models import _sequence_expansion
from models.gpt.expert_linear_adaptive._boundary_config_factory import GptBoundaryConfig
from models.gpt.expert_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig

from models.gpt._model import GptBackbone


class Model(GptBackbone[ExperimentConfig, GptBoundaryConfig]):
    experiment_type = ExperimentConfig
    boundary_type = GptBoundaryConfig
    invalid_experiment_message = "config.experiment_config must be a GPT Expert Linear Adaptive ExperimentConfig."

    def __init__(
        self,
        config: "ModelConfig",
    ) -> None:
        experiment_config = self._validate_experiment_config(config)
        boundary_config = self._validate_boundary_config(experiment_config)
        self.__validate_contextual_embedding(config, experiment_config, boundary_config)
        self.__validate_hierarchical_embedding(
            config, experiment_config, boundary_config
        )
        self._validate_tied_vocabulary_sizes(config, boundary_config)
        hierarchical = experiment_config.hierarchical_language_model_config
        enabled = boundary_config.embedding_options.hierarchical_language_model_flag
        if enabled != (hierarchical is not None):
            raise ValueError(
                "hierarchical_language_model_flag must match its supplied configuration"
            )
        if not enabled:
            if (
                experiment_config.positional_embedding_config is None
                and not boundary_config.embedding_options.contextual_flag
            ):
                raise ValueError(
                    "experiment_config.positional_embedding_config is required"
                )
            if experiment_config.decoder_config is None:
                raise ValueError("experiment_config.decoder_config is required")
        super().__init__(config)
        self.experiment_config: ExperimentConfig = experiment_config
        self.sequence_expansion_enabled = (
            _sequence_expansion.expanded_sequence_capacity(
                self.experiment_config.decoder_config, 0
            )
            > 0
        )
        self.boundary_config: GptBoundaryConfig = boundary_config
        if enabled:
            if boundary_config.lm_head_options.weight_tying_flag:
                raise ValueError(
                    "Hierarchical language modeling requires lm_head_weight_tying_flag=False"
                )
            SequenceExpansion.VALIDATOR.validate_fixed_token_output(
                experiment_config.decoder_config, "Hierarchical language modeling"
            )
            self.hierarchical_model = hierarchical.build()
            return
        embedding_options = boundary_config.embedding_options
        self.token_text_adapter = None
        if embedding_options.contextual_flag or embedding_options.hierarchical_flag:
            self.token_text_adapter = TokenTextAdapter(config.input_dim)
        self.token_embedding = self.__build_token_embedding()
        self.positional_embedding = self.__build_positional_embedding()
        self.embedding_layer_norm = self._build_embedding_layer_norm()
        self.embedding_dropout = self._build_embedding_dropout()
        self.transformer = self._build_decoder()
        self.decoder_layer_norm = self._build_decoder_layer_norm()
        self.lm_head = self._build_lm_head()
        self._tie_lm_head_weights()

    @staticmethod
    def __validate_contextual_embedding(
        config: "ModelConfig",
        experiment_config: ExperimentConfig,
        boundary_config: GptBoundaryConfig,
    ) -> None:
        contextual = experiment_config.contextual_embedding_config
        if boundary_config.embedding_options.contextual_flag:
            if not isinstance(contextual, ByteContextualEmbeddingConfig):
                raise TypeError(
                    "Contextual embedding requires ByteContextualEmbeddingConfig."
                )
            if contextual.hidden_dim != config.hidden_dim:
                raise ValueError(
                    "Contextual embedding hidden_dim must equal GPT hidden_dim."
                )
            if boundary_config.lm_head_options.weight_tying_flag:
                raise ValueError(
                    "Contextual embedding requires lm_head_weight_tying_flag=False."
                )
        elif contextual is not None:
            raise ValueError(
                "contextual_embedding_config requires contextual_embedding_flag=True."
            )

    @staticmethod
    def __validate_hierarchical_embedding(
        config: "ModelConfig",
        experiment_config: ExperimentConfig,
        boundary_config: GptBoundaryConfig,
    ) -> None:
        hierarchical = experiment_config.hierarchical_embedding_config
        if boundary_config.embedding_options.hierarchical_flag:
            if not isinstance(hierarchical, HierarchicalByteEmbeddingConfig):
                raise TypeError(
                    "Hierarchical embedding requires HierarchicalByteEmbeddingConfig."
                )
            if hierarchical.output_dim != config.hidden_dim:
                raise ValueError(
                    "Hierarchical embedding output_dim must equal GPT hidden_dim."
                )
            if boundary_config.lm_head_options.weight_tying_flag:
                raise ValueError(
                    "Hierarchical embedding requires lm_head_weight_tying_flag=False."
                )
        elif hierarchical is not None:
            raise ValueError(
                "hierarchical_embedding_config requires "
                "hierarchical_embedding_flag=True."
            )

    def set_token_vocabulary(self, token_texts: Sequence[str]) -> None:
        """Bind exact vocabulary spellings in token-ID order for standalone inference."""
        if getattr(self, "token_text_adapter", None) is None:
            raise ValueError(
                "Token text vocabulary is only used with contextual or hierarchical "
                "embedding."
            )
        self.token_text_adapter.bind(token_texts)
        self.__validate_token_byte_lengths()

    def setup(self, stage: str) -> None:
        if getattr(self, "token_text_adapter", None) is not None:
            self.token_text_adapter.bind_datamodule(self.trainer.datamodule)
            self.__validate_token_byte_lengths()

    def __validate_token_byte_lengths(self) -> None:
        embedding_config = self.experiment_config.hierarchical_embedding_config
        if self.token_text_adapter is None or embedding_config is None:
            return
        token_texts = self.token_text_adapter.token_texts
        limit = embedding_config.max_token_bytes
        if token_texts is None or limit is None:
            return
        longest = max(
            token_texts, key=lambda text: len(text.encode("utf-8", "surrogatepass"))
        )
        longest_bytes = len(longest.encode("utf-8", "surrogatepass"))
        if longest_bytes > limit:
            raise ValueError(
                f"Vocabulary token {longest!r} has {longest_bytes} UTF-8 bytes, but "
                f"hierarchical_embedding_max_token_bytes is {limit}; raise it to at "
                f"least {longest_bytes}."
            )

    def __build_token_embedding(self) -> nn.Module:
        if self.boundary_config.embedding_options.contextual_flag:
            token_embedding = self.experiment_config.contextual_embedding_config.build()
            return token_embedding
        if self.boundary_config.embedding_options.hierarchical_flag:
            token_embedding = (
                self.experiment_config.hierarchical_embedding_config.build()
            )
            return token_embedding
        return nn.Embedding(self.cfg.input_dim, self.cfg.hidden_dim)

    def __build_positional_embedding(self) -> nn.Module:
        if self.boundary_config.embedding_options.contextual_flag:
            return nn.Identity()
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
        hidden, embedding_loss = self.__build_input_embeddings(
            input_ids, attention_mask
        )
        sequence_output, auxiliary_loss = self._run_decoder(
            hidden, attention_mask, align_to_input
        )
        logits = self.lm_head(sequence_output)
        return logits, LayerState.sum_losses(
            auxiliary_loss,
            embedding_loss,
            LayerState.module_loss(self.positional_embedding),
        ).reshape(())

    def __build_input_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        if getattr(self, "token_text_adapter", None) is not None:
            state = self.token_embedding(
                self.token_text_adapter(input_ids), attention_mask=attention_mask != 0
            )
            hidden, loss = state.hidden, state.loss
            if self.boundary_config.embedding_options.hierarchical_flag:
                # Byte positions are token-local; add sentence positions here.
                hidden = hidden + self.positional_embedding(input_ids)
        else:
            hidden = self.token_embedding(input_ids) + self.positional_embedding(
                input_ids
            )
            loss = hidden.new_zeros(())
        hidden = self.embedding_layer_norm.before_model(
            LayerState(hidden=hidden)
        ).hidden
        hidden = self.embedding_dropout(hidden)
        if getattr(self, "token_text_adapter", None) is not None:
            hidden = hidden * (attention_mask != 0).unsqueeze(-1)
        return hidden, loss
