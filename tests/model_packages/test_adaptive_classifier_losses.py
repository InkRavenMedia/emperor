import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
)
from models.catalog import model_package
from tests.model_packages.test_attention_residual_options import walk


@pytest.mark.parametrize(
    "name",
    (
        "linears/linear_adaptive",
        "experts/linear_adaptive",
        "neuron/linear_adaptive",
        "neuron/expert_linear_adaptive",
    ),
)
@pytest.mark.parametrize("boundary", ("input", "output"))
def test_selected_adaptive_boundary_contributes_once_with_its_gradient(name, boundary):
    torch.manual_seed(45)
    package = model_package(name)
    values = {
        "input_dim": 4,
        "hidden_dim": 4,
        "output_dim": 2,
        "stack_num_layers": 1,
        f"{boundary}_layer_bias_option": MatrixBiasMixtureConfig,
        f"{boundary}_layer_bias_mixture_num_experts": 3,
        f"{boundary}_layer_bias_mixture_top_k": 1,
    }
    if name.startswith("neuron/"):
        for axis in ("x", "y", "z"):
            values[f"cluster_{axis}_axis_total_neurons"] = 1
            values[f"cluster_initial_{axis}_axis_total_neurons"] = 1
        values.update(
            cluster_beam_width=1,
            cluster_max_steps=2,
            cluster_growth_threshold=None,
            cluster_pruning_threshold=None,
            cluster_terminal_sampler_num_topk_samples=0,
            cluster_terminal_sampler_noisy_topk_flag=False,
        )
    configuration = package.build_configuration(config_overrides=values)
    adapter = getattr(configuration.experiment_config, boundary + "_model_config")
    (mixture,) = [
        item for item in walk(adapter) if isinstance(item, MatrixBiasMixtureConfig)
    ]
    mixture.sampler_config.switch_loss_weight = 0.2
    model = package.build_model(configuration).eval()
    models = {
        "input": model.input_model,
        "output": model.output_model,
        "main": model.neuron_cluster
        if name.startswith("neuron/")
        else model.main_model,
    }
    losses = {}

    def record(stage):
        def capture(module, args, output):
            losses[stage] = output[1] if isinstance(output, tuple) else output.loss

        return capture

    handles = [
        module.register_forward_hook(record(stage)) for stage, module in models.items()
    ]
    try:
        _, total = model(torch.tensor([[0.5, -0.5, 1.0, 0.0], [-0.3, 0.7, 0.1, 1.0]]))
    finally:
        for handle in handles:
            handle.remove()
    assert losses[boundary].item() > 0
    torch.testing.assert_close(
        total, sum(loss for loss in losses.values() if loss is not None)
    )
    torch.testing.assert_close(
        torch.autograd.grad(total, losses[boundary])[0],
        torch.ones_like(losses[boundary]),
    )
    gradients = torch.autograd.grad(
        total,
        [p for p in models[boundary].parameters() if p.requires_grad],
        allow_unused=True,
    )
    assert any(grad is not None and grad.abs().sum() > 0 for grad in gradients)


def test_no_auxiliary_term_and_zero_valued_term_keep_distinct_public_results():
    package = model_package("linears/linear_adaptive")
    dimensions = {
        "input_dim": 4,
        "hidden_dim": 4,
        "output_dim": 2,
        "stack_num_layers": 1,
    }
    inputs = torch.randn(2, 4)
    plain = package.build_model(
        package.build_configuration(config_overrides=dimensions)
    )
    assert isinstance(plain(inputs), torch.Tensor)
    dense = package.build_configuration(
        config_overrides={
            **dimensions,
            "input_layer_weight_option": MatrixWeightsMixtureConfig,
            "input_layer_weight_mixture_num_experts": 3,
        }
    )
    dense_logits = package.build_model(dense)(inputs)
    assert isinstance(dense_logits, torch.Tensor) and dense_logits.shape == (2, 2)
    configured = package.build_configuration(
        config_overrides={
            **dimensions,
            "input_layer_bias_option": MatrixBiasMixtureConfig,
            "input_layer_bias_mixture_num_experts": 3,
            "input_layer_bias_mixture_top_k": 1,
        }
    )
    (mixture,) = [
        item
        for item in walk(configured.experiment_config.input_model_config)
        if isinstance(item, MatrixBiasMixtureConfig)
    ]
    mixture.sampler_config.switch_loss_weight = 0.0
    logits, loss = package.build_model(configured)(inputs)
    assert logits.shape == (2, 2) and loss.shape == torch.Size([])
    assert loss.item() == 0.0
