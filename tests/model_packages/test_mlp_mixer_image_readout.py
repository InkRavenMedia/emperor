import pytest
import torch

from emperor.layers import LayerState
from emperor.patch import PatchEmbeddingConv, PatchEmbeddingLinear
from model_runtime.inspection import configuration_schema
from models.catalog import model_package
from tests.model_packages.test_mlp_mixer_source_memory import PACKAGES, configuration
from tests.model_packages.test_soft_moe_cli import cli_values


@pytest.mark.parametrize("name", PACKAGES)
@pytest.mark.parametrize(
    ("embedding", "patch_type"),
    [("linear", PatchEmbeddingLinear), ("cnn", PatchEmbeddingConv)],
)
@pytest.mark.parametrize("readout", ["mean", "cls"])
def test_overlapping_image_patch_embeddings(name, embedding, patch_type, readout):
    package = model_package(name)
    options = cli_values(
        package,
        [
            "--image-patch-embedding",
            embedding,
            "--image-patch-size",
            "5",
            "--image-patch-stride",
            "3",
            "--image-readout",
            readout,
        ],
    )
    config = configuration(name, input_dim=1024, image_height=32, **options)
    model = package.build_model(config).eval()
    token_count = 100 + int(readout == "cls")
    assert config.sequence_length == token_count
    assert isinstance(model.patch, patch_type)
    assert model.patch.class_token_flag is (readout == "cls")

    encoder_inputs = []
    model.transformer.register_forward_pre_hook(
        lambda _, args: encoder_inputs.append(args[0].hidden.shape)
    )
    with torch.no_grad():
        result = model(torch.randn(2, 1, 32, 32))
    logits = result[0] if isinstance(result, tuple) else result
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    assert encoder_inputs == [(2, token_count, 8)]


@pytest.mark.parametrize("name", PACKAGES)
@pytest.mark.parametrize("readout", ["mean", "cls"])
def test_image_readout_matches_the_encoder_tokens(name, readout):
    torch.manual_seed(43)
    package = model_package(name)
    options = cli_values(package, ["--image-readout", readout])
    config = configuration(name, **options)
    model = package.build_model(config).eval()
    field = next(
        f for f in configuration_schema(package).fields if f.key == "IMAGE_READOUT"
    )
    assert field.default == "MEAN" and field.choices == ("MEAN", "CLS")
    has_cls = readout == "cls"
    assert config.sequence_length == 4 + int(has_cls)
    assert model.patch.class_token_flag is has_cls

    inputs, outputs, head_inputs = [], [], []

    model.transformer.register_forward_pre_hook(
        lambda _, args: inputs.append(args[0].hidden.detach().clone())
    )
    model.transformer.register_forward_hook(
        lambda _, args, result: outputs.append(result.hidden)
    )
    model.output.register_forward_pre_hook(
        lambda _, args: head_inputs.append(args[0].hidden)
    )
    images = torch.randn(2, 1, 8, 8)
    result = model(images)
    logits = result[0] if isinstance(result, tuple) else result
    assert inputs[0].shape == (2, config.sequence_length, 8)
    assert torch.isfinite(logits).all() and logits.shape == (2, 3)
    normalized = model.encoder_layer_norm.before_model(LayerState(outputs[0])).hidden
    expected = normalized[:, 0] if has_cls else normalized.mean(dim=1)
    torch.testing.assert_close(head_inputs[0], expected)

    if has_cls:
        torch.testing.assert_close(
            inputs[0][:, :1], model.patch.class_token.expand(2, -1, -1)
        )
        logits.square().mean().backward()
        grad = model.patch.class_token.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    else:
        default = package.build_model(configuration(name)).eval()
        assert not hasattr(default.patch, "class_token")
        default.load_state_dict(model.state_dict(), strict=True)
        torch.manual_seed(84)
        actual = default(images)
        torch.manual_seed(84)
        explicit = model(images)
        torch.testing.assert_close(actual, explicit)


def test_recurrent_mixer_carries_one_cls_position_across_compressed_passes():
    name = "mlp_mixer/expert_linear_adaptive"
    package = model_package(name)
    options = cli_values(package, ["--image-readout", "cls"])
    config = configuration(
        name,
        **options,
        recurrent_flag=True,
        recurrent_max_steps=2,
        recurrent_initial_iterations=2,
        recurrent_compression_slots=3,
    )
    model = package.build_model(config).eval()
    step = model.transformer.step
    boundaries, block_lengths = [], []
    step.compressor.register_forward_hook(
        lambda _, args, out: boundaries.append(
            (args[1].hidden.size(1), out.hidden.size(1))
        )
    )
    step.block_model.register_forward_pre_hook(
        lambda _, args: block_lengths.append(args[0].hidden.size(1))
    )
    result = model(torch.randn(2, 1, 8, 8))
    logits = result[0] if isinstance(result, tuple) else result
    assert boundaries == [(5, 5), (5, 5)] and block_lengths == [3, 3]
    logits.square().mean().backward()
    assert model.patch.class_token.grad is not None
