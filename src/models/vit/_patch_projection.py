from types import ModuleType
from typing import Final, Generic, Protocol, TypeVar

from emperor.layers import LayerNormPositionOptions, NormalizationOptions
from emperor.patch import LinearPatchEmbeddingConfig
from models._linear_stack import LinearStackFactory


class PatchSettings(Protocol):
    patch_size: Final[int]
    image_height: Final[int]
    input_channels: Final[int]
    bias_flag: Final[bool]
    dropout_probability: Final[float]


class EncoderDropout(Protocol):
    dropout_probability: Final[float]


class PatchProjectionFactory:
    def __init__(
        self,
        hidden_dim: int,
        patch_options: PatchSettings,
        encoder_options: EncoderDropout,
        linear_layer_config_factory: LinearStackFactory,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.patch_options = patch_options
        self.encoder_options = encoder_options
        self.linear_layer_config_factory = linear_layer_config_factory

    @property
    def sequence_length(self) -> int:
        if self.patch_options.patch_size <= 0:
            raise ValueError(
                "image_patch_size must be positive, "
                f"received {self.patch_options.patch_size}."
            )
        if self.patch_options.image_height % self.patch_options.patch_size != 0:
            raise ValueError(
                "image_height must be divisible by image_patch_size, "
                f"received image_height={self.patch_options.image_height} and "
                f"image_patch_size={self.patch_options.patch_size}."
            )
        patches_per_axis = (
            self.patch_options.image_height // self.patch_options.patch_size
        )
        return patches_per_axis * patches_per_axis + 1

    def build_patch_config(self) -> LinearPatchEmbeddingConfig:
        options = self.patch_options
        embedding_stack_config = (
            self.linear_layer_config_factory.build_plain_linear_stack_config(
                input_dim=options.input_channels * options.patch_size**2,
                output_dim=self.hidden_dim,
                num_layers=1,
                bias_flag=options.bias_flag,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                normalization=NormalizationOptions.RMS_NORM,
                dropout_probability=self.encoder_options.dropout_probability,
                apply_output_postprocessing_flag=False,
            )
        )
        return LinearPatchEmbeddingConfig(
            embedding_dim=self.hidden_dim,
            num_input_channels=options.input_channels,
            patch_size=options.patch_size,
            stride=options.patch_size,
            padding=0,
            dropout_probability=options.dropout_probability,
            embedding_stack_config=embedding_stack_config,
        )


class PatchInputs(Protocol):
    hidden_dim: Final[int]
    patch_options: Final[PatchSettings | None]
    encoder_options: Final[EncoderDropout | None]
    linear_layer_config_factory: Final[LinearStackFactory]


class _PatchDefaults(Protocol):
    def vit_patch_options(self, config: ModuleType) -> PatchSettings: ...
    def vit_encoder_options(self, config: ModuleType) -> EncoderDropout: ...


_Inputs = TypeVar("_Inputs", bound=PatchInputs)


class ConfiguredPatchProjection(PatchProjectionFactory, Generic[_Inputs]):
    _config: ModuleType
    _defaults: _PatchDefaults

    def __init__(self, dependencies: _Inputs) -> None:
        patch_options = dependencies.patch_options
        if patch_options is None:
            patch_options = self._defaults.vit_patch_options(self._config)
        encoder_options = dependencies.encoder_options
        if encoder_options is None:
            encoder_options = self._defaults.vit_encoder_options(self._config)
        super().__init__(
            dependencies.hidden_dim,
            patch_options,
            encoder_options,
            dependencies.linear_layer_config_factory,
        )
