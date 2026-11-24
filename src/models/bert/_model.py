from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.embedding.contextual import ByteContextualEmbeddingConfig
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.layers import (
    ActivationOptions,
    HierarchicalReasoningModelRecurrentConfig,
    LayerConfig,
    LayerNormalizationDelegate,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    NormalizationOptions,
    RecurrentCompositionConfig,
)
from emperor.transformer import TransformerEncoderLayerState
from models import _sequence_expansion

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _EmbeddingSettings(Protocol):
    token_type_vocab_size: Final[int]
    layer_norm_flag: Final[bool]
    normalization: Final[NormalizationOptions]
    dropout_probability: Final[float]
    contextual_embedding_config: Final[ByteContextualEmbeddingConfig | None]


class _MlmHeadSettings(Protocol):
    activation: Final[ActivationOptions]
    dense_bias_flag: Final[bool]
    layer_norm_flag: Final[bool]
    normalization: Final[NormalizationOptions]
    decoder_bias_flag: Final[bool]
    decoder_weight_tying_flag: Final[bool]


class _NspHeadSettings(Protocol):
    pooler_activation: Final[ActivationOptions]
    pooler_bias_flag: Final[bool]
    output_dim: Final[int]
    head_bias_flag: Final[bool]


class _BoundarySettings(Protocol):
    embedding_options: Final[_EmbeddingSettings]
    mlm_head_options: Final[_MlmHeadSettings]
    nsp_head_options: Final[_NspHeadSettings]


class _ExperimentSettings(Protocol):
    boundary_config: Final[object]
    positional_embedding_config: Final[AbsolutePositionalEmbeddingConfig | None]
    encoder_config: Final[LayerStackConfig | RecurrentCompositionConfig | None]
    encoder_output_normalization: Final[NormalizationOptions]


_Experiment = TypeVar("_Experiment", bound=_ExperimentSettings)
_Boundary = TypeVar("_Boundary", bound=_BoundarySettings)


class BertBackbone(BertPretrainingExperiment, Generic[_Experiment, _Boundary]):
    experiment_type: type[_Experiment]
    boundary_type: type[_Boundary]
    invalid_experiment_message: str
    token_embedding: nn.Module
    token_type_embedding: nn.Embedding
    positional_embedding: nn.Module
    embedding_dropout: nn.Dropout
    embedding_layer_norm: LayerNormalizationDelegate
    encoder_layer_norm: LayerNormalizationDelegate
    transformer: nn.Module
    mlm_dense: nn.Linear
    mlm_activation: nn.Module
    mlm_layer_norm: LayerNormalizationDelegate
    mlm_decoder: nn.Linear
    mlm_decoder_bias: nn.Parameter | None
    pooler: nn.Linear
    pooler_activation: nn.Module
    nsp_head: nn.Linear
    sequence_expansion_enabled: bool
    experiment_config: _Experiment
    boundary_config: _Boundary

    _ACTIVATION_MODULE_TYPES: dict[ActivationOptions, type[nn.Module]] = {
        ActivationOptions.DISABLED: nn.Identity,
        ActivationOptions.RELU: nn.ReLU,
        ActivationOptions.GELU: nn.GELU,
        ActivationOptions.SIGMOID: nn.Sigmoid,
        ActivationOptions.TANH: nn.Tanh,
        ActivationOptions.LEAKY_RELU: nn.LeakyReLU,
        ActivationOptions.ELU: nn.ELU,
        ActivationOptions.SELU: nn.SELU,
        ActivationOptions.SOFTPLUS: nn.Softplus,
        ActivationOptions.SOFTSIGN: nn.Softsign,
        ActivationOptions.SILU: nn.SiLU,
        ActivationOptions.MISH: nn.Mish,
    }

    @staticmethod
    def _build_activation_module(activation: ActivationOptions) -> nn.Module:
        try:
            module_type = BertBackbone._ACTIVATION_MODULE_TYPES[activation]
        except KeyError as error:
            raise ValueError(
                f"Unsupported BERT head activation: {activation!r}."
            ) from error
        return module_type()

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
                "config.experiment_config.boundary_config must be a resolved BertBoundaryConfig."
            )
        return boundary

    def _validate_pretraining_dimensions(self) -> None:
        if not self.boundary_config.mlm_head_options.decoder_weight_tying_flag:
            return
        if self.cfg.input_dim != self.cfg.output_dim:
            raise ValueError(
                "BERT MLM decoder weight tying requires config.input_dim to equal "
                "config.output_dim."
            )

    def _build_token_type_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            self.boundary_config.embedding_options.token_type_vocab_size,
            self.cfg.hidden_dim,
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

    def _build_encoder_model(self) -> nn.Module:
        encoder_model = self.experiment_config.encoder_config.build()
        return encoder_model

    def _build_encoder_layer_norm(self) -> LayerNormalizationDelegate:
        encoder_config = self.experiment_config.encoder_config
        if isinstance(encoder_config, HierarchicalReasoningModelRecurrentConfig):
            stack_config = encoder_config.high_block_config
        else:
            stack_config = getattr(encoder_config, "block_config", encoder_config)
        layer_config = stack_config.layer_config.layer_model_config
        layer_norm_position = LayerNormPositionOptions.DISABLED
        if layer_config.layer_norm_position == LayerNormPositionOptions.BEFORE:
            layer_norm_position = LayerNormPositionOptions.BEFORE
        normalization_config = LayerConfig(
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
            layer_norm_position=layer_norm_position,
            normalization=self.experiment_config.encoder_output_normalization,
        )
        return LayerNormalizationDelegate(normalization_config)

    def _build_mlm_dense(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.hidden_dim,
            bias=self.boundary_config.mlm_head_options.dense_bias_flag,
        )

    def _build_mlm_activation(self) -> nn.Module:
        return self._build_activation_module(
            self.boundary_config.mlm_head_options.activation
        )

    def _build_mlm_layer_norm(self) -> LayerNormalizationDelegate:
        layer_norm_position = LayerNormPositionOptions.DISABLED
        if self.boundary_config.mlm_head_options.layer_norm_flag:
            layer_norm_position = LayerNormPositionOptions.BEFORE
        normalization_config = LayerConfig(
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
            layer_norm_position=layer_norm_position,
            normalization=self.boundary_config.mlm_head_options.normalization,
        )
        return LayerNormalizationDelegate(normalization_config)

    def _build_mlm_decoder(self) -> nn.Linear:
        return nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim, bias=False)

    def _tie_mlm_decoder_weights(self) -> None:
        if not self.boundary_config.mlm_head_options.decoder_weight_tying_flag:
            return
        self.mlm_decoder.weight = self.token_embedding.weight

    def _build_mlm_decoder_bias(self) -> nn.Parameter | None:
        if not self.boundary_config.mlm_head_options.decoder_bias_flag:
            return None
        return nn.Parameter(torch.zeros(self.cfg.output_dim))

    def _build_pooler(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.hidden_dim,
            bias=self.boundary_config.nsp_head_options.pooler_bias_flag,
        )

    def _build_pooler_activation(self) -> nn.Module:
        return self._build_activation_module(
            self.boundary_config.nsp_head_options.pooler_activation
        )

    def _build_nsp_head(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.boundary_config.nsp_head_options.output_dim,
            bias=self.boundary_config.nsp_head_options.head_bias_flag,
        )

    def forward_for_input_tokens(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not self.sequence_expansion_enabled:
            return super().forward_for_input_tokens(
                input_ids, attention_mask, token_type_ids
            )
        return self(input_ids, attention_mask, token_type_ids, align_to_input=True)

    def _run_encoder(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        align_to_input: bool = False,
    ) -> tuple[Tensor, Tensor]:
        encoder_state = self.transformer(
            TransformerEncoderLayerState(
                hidden=hidden,
                key_padding_mask=attention_mask == 0,
            )
        )
        sequence_output = self.encoder_layer_norm.before_model(
            LayerState(hidden=encoder_state.hidden)
        ).hidden
        if align_to_input:
            sequence_output = encoder_state.gather_input_tokens(sequence_output)
        auxiliary_loss = (
            encoder_state.loss
            if encoder_state.loss is not None
            else sequence_output.new_zeros(())
        )
        return sequence_output, auxiliary_loss

    def _build_mlm_logits(self, sequence_output: Tensor) -> Tensor:
        mlm_hidden = self.mlm_dense(sequence_output)
        mlm_hidden = self.mlm_activation(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm.before_model(
            LayerState(hidden=mlm_hidden)
        ).hidden
        mlm_logits = self.mlm_decoder(mlm_hidden)
        if self.mlm_decoder_bias is not None:
            mlm_logits = mlm_logits + self.mlm_decoder_bias
        return mlm_logits

    def _build_nsp_logits(self, sequence_output: Tensor) -> Tensor:
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        return self.nsp_head(pooled_output)


class BertModel(BertBackbone[_Experiment, _Boundary]):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        experiment_config = self._validate_experiment_config(config)
        boundary_config = self._validate_boundary_config(experiment_config)
        if (
            experiment_config.positional_embedding_config is None
            and boundary_config.embedding_options.contextual_embedding_config is None
        ):
            raise ValueError(
                "experiment_config.positional_embedding_config is required"
            )
        if experiment_config.encoder_config is None:
            raise ValueError("experiment_config.encoder_config is required")
        super().__init__(config)
        self.experiment_config = experiment_config
        self.sequence_expansion_enabled = (
            _sequence_expansion.expanded_sequence_capacity(
                self.experiment_config.encoder_config, 0
            )
            > 0
        )
        self.boundary_config = boundary_config
        self._validate_pretraining_dimensions()
        self.token_embedding = self._build_token_embedding()
        self.token_type_embedding = self._build_token_type_embedding()
        self.positional_embedding = self._build_positional_embedding()
        self.embedding_layer_norm = self._build_embedding_layer_norm()
        self.embedding_dropout = self._build_embedding_dropout()
        self.transformer = self._build_encoder_model()
        self.encoder_layer_norm = self._build_encoder_layer_norm()
        self.mlm_dense = self._build_mlm_dense()
        self.mlm_activation = self._build_mlm_activation()
        self.mlm_layer_norm = self._build_mlm_layer_norm()
        self.mlm_decoder = self._build_mlm_decoder()
        self._tie_mlm_decoder_weights()
        self.mlm_decoder_bias = self._build_mlm_decoder_bias()
        self.pooler = self._build_pooler()
        self.pooler_activation = self._build_pooler_activation()
        self.nsp_head = self._build_nsp_head()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        align_to_input: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return all token logits, or restore input positions with align_to_input."""
        input_ids, attention_mask, token_type_ids = self._prepare_model_inputs(
            input_ids,
            attention_mask,
            token_type_ids,
            token_type_vocab_size=self.token_type_embedding.num_embeddings,
        )
        hidden, embedding_loss = self.__build_input_embeddings(
            input_ids, token_type_ids, attention_mask
        )
        sequence_output, auxiliary_loss = self._run_encoder(
            hidden, attention_mask, align_to_input
        )
        mlm_logits = self._build_mlm_logits(sequence_output)
        nsp_logits = self._build_nsp_logits(sequence_output)
        total_auxiliary_loss = LayerState.sum_losses(
            auxiliary_loss,
            embedding_loss,
            LayerState.module_loss(self.positional_embedding),
        )
        return mlm_logits, nsp_logits, total_auxiliary_loss

    def __build_input_embeddings(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        embedding_state = self._build_token_embedding_state(input_ids, attention_mask)
        token_embedding = embedding_state.hidden
        positioned_token_embeddings = self._add_sentence_positions(
            token_embedding, input_ids
        )
        segment_embedding = self.token_type_embedding(token_type_ids)
        hidden = positioned_token_embeddings + segment_embedding
        hidden = self.embedding_layer_norm.before_model(
            LayerState(hidden=hidden)
        ).hidden
        processed_embeddings = self.embedding_dropout(hidden)
        return processed_embeddings, embedding_state.loss
