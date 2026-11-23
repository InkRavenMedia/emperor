from enum import StrEnum

from emperor.convs import Conv2dLayerConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.patch import ConvPatchEmbeddingConfig


class ImagePatchEmbedding(StrEnum):
    LINEAR = "linear"
    CNN = "cnn"


def image_patch_stride(
    image_height: int, patch_size: int, configured_stride: int | None
) -> int:
    if patch_size <= 0:
        raise ValueError(f"image_patch_size must be positive, received {patch_size}.")
    if patch_size > image_height:
        raise ValueError("image_patch_size must fit within image_height.")
    if configured_stride is not None:
        if type(configured_stride) is not int or configured_stride <= 0:
            raise ValueError("image_patch_stride must be a positive integer.")
        return configured_stride
    if image_height % patch_size != 0:
        raise ValueError(
            "image_height must be divisible by image_patch_size, "
            f"received image_height={image_height} and image_patch_size={patch_size}."
        )
    return patch_size


def image_patch_count(image_height: int, patch_size: int, stride: int | None) -> int:
    stride = image_patch_stride(image_height, patch_size, stride)
    patches_per_axis = (image_height - patch_size) // stride + 1
    return patches_per_axis**2


def build_conv_patch_config(
    *,
    embedding_dim: int,
    input_channels: int,
    patch_size: int,
    stride: int,
    bias_flag: bool,
    dropout_probability: float,
    class_token_flag: bool | None = None,
    layer_dropout_probability: float = 0.0,
) -> ConvPatchEmbeddingConfig:
    return ConvPatchEmbeddingConfig(
        embedding_dim=embedding_dim,
        num_input_channels=input_channels,
        patch_size=patch_size,
        dropout_probability=dropout_probability,
        class_token_flag=class_token_flag,
        conv_stack_config=LayerStackConfig(
            input_dim=input_channels,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=False,
            layer_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                normalization=NormalizationOptions.RMS_NORM,
                dropout_probability=layer_dropout_probability,
                layer_model_config=Conv2dLayerConfig(
                    kernel_size=patch_size,
                    stride=stride,
                    padding=0,
                    bias_flag=bias_flag,
                ),
            ),
        ),
    )
