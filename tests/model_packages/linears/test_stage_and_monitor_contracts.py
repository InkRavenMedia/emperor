from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from emperor.linears import LinearLayer, LinearLayerConfig
from emperor.monitoring import MonitorSettings
from models.catalog import model_package

PACKAGES = ("linears/linear", "linears/linear_adaptive")


class _LinearWithPenalty(LinearLayer):
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        self.penalty = nn.Parameter(torch.tensor(1.25))

    def get_auxiliary_loss(self) -> Tensor:
        return self.penalty


class _PenalizedLinearConfig(LinearLayerConfig):
    def _registry_owner(self) -> type:
        return _LinearWithPenalty


@pytest.mark.parametrize("name", PACKAGES)
def test_each_stage_contributes_its_auxiliary_loss_and_gradient(name):
    package = model_package(name)
    config = package.build_configuration(
        config_overrides={"hidden_dim": 8, "stack_num_layers": 1}
    )
    stages = config.experiment_config
    for layer in (
        stages.input_model_config,
        stages.model_config.layer_config,
        stages.output_model_config,
    ):
        layer.layer_model_config = _PenalizedLinearConfig(bias_flag=True)

    model = package.build_model(config)
    logits, auxiliary_loss = model(torch.ones(2, config.input_dim))
    assert logits.shape == (2, config.output_dim)
    assert auxiliary_loss.item() == pytest.approx(3.75)
    auxiliary_loss.backward()
    penalties = [
        layer.penalty
        for layer in model.modules()
        if isinstance(layer, _LinearWithPenalty)
    ]
    assert len(penalties) == 3
    assert all(penalty.grad.item() == 1.0 for penalty in penalties)


@pytest.mark.parametrize("name", PACKAGES)
@pytest.mark.parametrize(
    "field",
    ["experiment_config", "input_model_config", "model_config", "output_model_config"],
)
def test_incomplete_experiment_names_the_missing_stage(name, field):
    package = model_package(name)
    config = package.build_configuration()
    stages = None
    if field != "experiment_config":
        stages = replace(config.experiment_config, **{field: None})
    with pytest.raises(ValueError, match=rf"\b{field} is required"):
        package.build_model(replace(config, experiment_config=stages))


def test_linear_models_share_immutable_monitor_definitions_and_fresh_callbacks():
    plain = model_package(PACKAGES[0]).monitor_options()
    adaptive = model_package(PACKAGES[1]).monitor_options()
    assert adaptive[: len(plain)] == plain
    assert [option.name for option in adaptive[len(plain) :]] == [
        "adaptive",
        "weight-bank",
    ]
    assert "condition" not in plain[0].description
    for option in adaptive:
        assert isinstance(option.kinds, tuple)
        exported = option.to_api()
        exported["kinds"].clear()
        assert option.to_api()["kinds"] == list(option.kinds)
        assert option.kinds
        settings = MonitorSettings(log_every_n_steps=7)
        assert option.build_callback(settings) is not option.build_callback(settings)
