from dataclasses import replace
from importlib import import_module
from unittest.mock import patch

import pytest
import torch

from emperor.augmentations.adaptive_parameters import MatrixBiasMixtureConfig
from emperor.layers import AdditiveResidualConfig, LayerState, ResidualConfig
from emperor.linears import LinearLayerConfig
from emperor.sampler import SamplerConfig
from models.catalog import model_package

PACKAGES = (
    "experts/linear",
    "experts/linear_adaptive",
    "mlp_mixer/linear",
    "mlp_mixer/expert_linear",
    "mlp_mixer/linear_adaptive",
    "mlp_mixer/expert_linear_adaptive",
    "neuron/linear",
    "neuron/expert_linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear_adaptive",
    "parametric/parametric_matrix",
    "parametric/parametric_generator",
    "parametric/parametric_vector",
)


@pytest.mark.parametrize("name", PACKAGES)
def test_builders_use_replaced_settings_and_survive_a_rejected_assignment(name):
    package = model_package(name)
    module = import_module(f"models.{name.replace('/', '.')}.config_builder")
    (builder_type,) = (
        obj
        for attr, obj in vars(module).items()
        if attr.endswith("ConfigBuilder") and obj.__module__ == module.__name__
    )
    builder = builder_type()
    replacement = package.bind_runtime_defaults({"batch_size": 7})
    builder.runtime = replacement
    with pytest.raises(TypeError, match="runtime"):
        builder.runtime = object()
    assert builder.runtime is replacement
    assert builder.build().batch_size == 7


class ConcreteResidual(AdditiveResidualConfig):
    pass


class UnimplementedResidual(ResidualConfig):
    pass


@pytest.mark.parametrize("name", PACKAGES)
def test_concrete_residual_selections_and_build_dimensions(name):
    package = model_package(name)
    key = "stack_residual_connection_option"
    config = package.build_configuration(config_overrides={key: ConcreteResidual})
    assert config.experiment_config is not None
    for bad in (ResidualConfig, UnimplementedResidual):
        with pytest.raises((TypeError, ValueError), match=key):
            package.bind_runtime_defaults({key: bad})
    for key, invalid in (("learning_rate", float("inf")), ("output_dim", 0)):
        with pytest.raises(ValueError, match=key.upper() + "|" + key):
            package.bind_runtime_defaults({key: invalid})


@pytest.mark.parametrize("name", PACKAGES)
def test_missing_stage_names_identify_the_package_role(name):
    package = model_package(name)
    config = package.build_configuration()
    if name.startswith("mlp_mixer"):
        fields = ("patch_config", "encoder_config", "output_config")
    elif name.startswith("neuron"):
        fields = ("input_model_config", "neuron_cluster_config", "output_model_config")
    else:
        fields = ("input_model_config", "model_config", "output_model_config")
    for field in fields:
        with pytest.raises(ValueError, match=rf"\b{field} is required"):
            package.build_model(
                replace(
                    config,
                    experiment_config=replace(
                        config.experiment_config, **{field: None}
                    ),
                )
            )


@pytest.mark.parametrize(
    "name",
    [
        "experts/linear_adaptive",
        "neuron/expert_linear_adaptive",
        "neuron/linear_adaptive",
    ],
)
def test_only_selected_adaptive_mixtures_construct_samplers(name):
    from models import _adaptive_parameters

    package = model_package(name)
    with patch.object(
        _adaptive_parameters, "SamplerConfig", wraps=SamplerConfig
    ) as make_sampler:
        package.build_configuration()
        make_sampler.assert_not_called()
        package.build_configuration(
            config_overrides={
                "input_layer_bias_option": MatrixBiasMixtureConfig,
                "input_layer_bias_mixture_num_experts": 3,
            }
        )
    assert [call.kwargs["num_experts"] for call in make_sampler.call_args_list] == [3]


@pytest.mark.parametrize(
    "name", [name for name in PACKAGES if name.startswith("neuron/")]
)
def test_hidden_block_state_uses_custom_forward_and_torch_hooks(name):
    owner = import_module(f"models.{name.replace('/', '.')}._hidden_block")

    class CustomAdapter(owner.HiddenBlockAdapter):
        def forward(self, inputs):
            return 2 * inputs

        def get_auxiliary_loss(self):
            return self.penalty

    cfg = owner.HiddenBlockConfig(
        input_dim=2, output_dim=2, model_config=LinearLayerConfig(bias_flag=True)
    )
    adapter = CustomAdapter(cfg)
    adapter.penalty = torch.nn.Parameter(torch.tensor(0.0))
    inputs = torch.ones(1, 2)
    with adapter.register_forward_hook(lambda _module, _args, output: output + 1):
        state = adapter.forward_state(inputs)
    assert isinstance(state, LayerState)
    torch.testing.assert_close(state.hidden, 2 * inputs + 1)
    assert state.loss is adapter.penalty


@pytest.mark.parametrize(
    "name", [name for name in PACKAGES if name.startswith("neuron/")]
)
def test_hidden_block_and_nucleus_hooks_receive_tensors_during_loss_collection(name):
    package = model_package(name)
    config = package.build_configuration(
        config_overrides={"hidden_dim": 4, "stack_num_layers": 1}
    )
    model = package.build_model(config).eval()
    neuron = next(iter(model.neuron_cluster.cluster.values()))
    inputs = torch.ones(2, 4)
    with (
        neuron.nucleus.model.register_forward_hook(
            lambda _m, _args, tensor: tensor + 1
        ),
        neuron.nucleus.register_forward_hook(lambda _m, _args, tensor: tensor * 2),
    ):
        expected = neuron.process_signal(inputs)
        state = neuron.process_signal_state(inputs)
    torch.testing.assert_close(state.hidden, expected)


@pytest.mark.parametrize(
    "name",
    [
        "experts/linear",
        "neuron/linear",
        "neuron/expert_linear",
        "parametric/parametric_matrix",
        "parametric/parametric_generator",
        "parametric/parametric_vector",
        "mlp_mixer/linear",
        "mlp_mixer/expert_linear",
        "mlp_mixer/linear_adaptive",
        "mlp_mixer/expert_linear_adaptive",
    ],
)
def test_classification_heads_contribute_their_differentiable_loss_once(name):
    from emperor.linears import LinearLayer

    class PenalizedLinear(LinearLayer):
        def __init__(self, cfg, overrides=None):
            super().__init__(cfg, overrides)
            self.penalty = torch.nn.Parameter(torch.tensor(0.0))

        def get_auxiliary_loss(self):
            return self.penalty

    class PenalizedConfig(LinearLayerConfig):
        def _registry_owner(self):
            return PenalizedLinear

    package = model_package(name)
    overrides = dict(input_dim=4, hidden_dim=4, output_dim=2, stack_num_layers=1)
    inputs = torch.ones(2, 4)
    if name.startswith("mlp_mixer"):
        overrides.update(
            image_height=4,
            image_patch_size=2,
            input_channels=1,
            token_mixer_num_layers=2,
            channel_mixer_num_layers=2,
        )
        inputs = torch.ones(2, 1, 4, 4)
    if name.startswith("neuron"):
        for axis in ("x", "y", "z"):
            overrides[f"cluster_{axis}_axis_total_neurons"] = 1
            overrides[f"cluster_initial_{axis}_axis_total_neurons"] = 1
        overrides.update(
            cluster_beam_width=1,
            cluster_max_steps=2,
            cluster_growth_threshold=None,
            cluster_pruning_threshold=None,
            cluster_terminal_sampler_num_topk_samples=0,
        )
    config = package.build_configuration(config_overrides=overrides)
    stages = config.experiment_config
    boundaries = (
        (stages.output_config,)
        if name.startswith("mlp_mixer")
        else (stages.input_model_config, stages.output_model_config)
    )
    for layer in boundaries:
        layer.layer_model_config = PenalizedConfig(bias_flag=True)
    model = package.build_model(config)
    logits, loss = model(inputs)
    assert logits.shape == (2, 2)
    penalties = [m.penalty for m in model.modules() if isinstance(m, PenalizedLinear)]
    assert len(penalties) == len(boundaries)
    for gradient in torch.autograd.grad(loss, penalties):
        assert gradient.item() == 1.0
    if name.startswith("mlp_mixer"):
        with torch.no_grad():
            eval_logits, eval_loss = model.eval()(inputs)
        assert eval_logits.shape == logits.shape
        assert eval_loss.ndim == 0


@pytest.mark.parametrize(
    "name", ["mlp_mixer/linear_adaptive", "mlp_mixer/expert_linear_adaptive"]
)
def test_flat_and_typed_mixer_settings_share_scoped_controller_rules(name):
    package = model_package(name)
    settings = package.bind_runtime_defaults({})
    for invalid in (
        {"token_mixer_gate_stack_num_layers": 0},
        {
            "channel_mixer_recurrent_stack_gate_flag": True,
            "channel_mixer_recurrent_gate_option": None,
        },
        {"token_mixer_halting_threshold": 0.0},
        {"bias_decay_warmup_batches": 0.5},
    ):
        with pytest.raises((TypeError, ValueError)) as flat_error:
            package.bind_runtime_defaults(invalid)
        with pytest.raises(type(flat_error.value)):
            replace(settings, **invalid)


def test_masked_word_mixer_retains_the_positional_embedding_penalty():
    from emperor.datasets.text.bert_pretraining import (
        BertMaskedWordBatch,
        WikiText103BertHierarchical,
    )

    package = model_package("mlp_mixer/expert_linear_adaptive")
    config = package.build_configuration(
        dataset=WikiText103BertHierarchical,
        config_overrides={
            "hierarchical_masked_word_flag": True,
            "sequence_length": 8,
            "hidden_dim": 8,
            "hierarchical_byte_embedding_dim": 8,
            "hierarchical_byte_feed_forward_dim": 8,
            "hierarchical_max_token_bytes": 8,
            "stack_num_layers": 1,
            "num_experts": 2,
        },
    )
    model = package.build_model(config).eval()
    position = model.positional_embedding
    position.penalty = torch.nn.Parameter(torch.tensor(0.0))
    batch = BertMaskedWordBatch.collate(
        [(("a",), ("b",), 0)], sequence_length=8, max_token_bytes=8, mlm_probability=1.0
    )
    with patch.object(
        position, "get_auxiliary_loss", create=True, return_value=position.penalty
    ):
        _, _, loss = model(batch)
    torch.testing.assert_close(
        torch.autograd.grad(loss, position.penalty)[0],
        torch.ones_like(position.penalty),
    )


@pytest.mark.parametrize(
    ("module", "group", "key", "typed_key"),
    [
        (
            "models.experts.linear_adaptive",
            "hidden_adaptive_bias_options",
            "bias_decay_warmup_batches",
            "decay_warmup_batches",
        ),
        (
            "models.neuron.expert_linear_adaptive._hidden",
            "input_boundary_options",
            "input_layer_weight_decay_warmup_batches",
            "weight_decay_warmup_batches",
        ),
        (
            "models.neuron.linear_adaptive._hidden",
            "output_projection",
            "output_layer_bias_decay_warmup_batches",
            "bias_decay_warmup_batches",
        ),
    ],
)
def test_flat_and_typed_adaptive_roles_require_integral_warmup(
    module, group, key, typed_key
):
    read = import_module(module + ".runtime_defaults").runtime_from_flat
    with pytest.raises(TypeError):
        read({key: 1.5})
    with pytest.raises(TypeError, match=typed_key):
        replace(getattr(read(), group), **{typed_key: 1.5})


def test_neuron_adaptive_hidden_residuals_keep_integer_probability_inputs():
    from models.neuron.linear_adaptive._hidden.runtime_defaults import runtime_from_flat

    settings = runtime_from_flat(
        {
            "residual_stack_independent_flag": True,
            "residual_stack_dropout_probability": 0,
        }
    )
    assert settings.residual_stack.dropout_probability == 0
