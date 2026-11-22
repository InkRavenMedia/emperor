from dataclasses import replace

import pytest

from emperor.layers import AdditiveResidualConfig, ResidualConfig
from models.linears.linear._residual import build_residual_stack_config
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.runtime_defaults import build_settings_from_flat
from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linears.linear_adaptive.runtime_defaults import runtime_from_flat

READERS = (build_settings_from_flat, runtime_from_flat)


def test_plain_residual_stack_reads_coefficient_activation_only_for_weighted_layers(
    monkeypatch,
):
    defaults = build_settings_from_flat().residual_stack
    stack = replace(defaults, residual_connection_option=AdditiveResidualConfig)

    def unexpected_coefficient_read(_stack):
        pytest.fail("An additive residual requested a coefficient activation")

    monkeypatch.setattr(
        type(stack), "coefficient_activation", property(unexpected_coefficient_read)
    )
    layer = build_residual_stack_config(stack).layer_config
    assert isinstance(layer.residual_config, AdditiveResidualConfig)


@pytest.mark.parametrize("read", READERS)
@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", 0.0),
        ("batch_size", 0),
        ("input_dim", -1),
        ("hidden_dim", 0),
        ("output_dim", 0),
        ("batch_size", False),
    ],
)
def test_flat_reader_and_typed_settings_reject_invalid_dimensions(read, field, invalid):
    settings = read()
    with pytest.raises((TypeError, ValueError)) as flat_error:
        read({field: invalid})
    with pytest.raises(type(flat_error.value)):
        replace(settings, **{field: invalid})


@pytest.mark.parametrize("read", READERS)
@pytest.mark.parametrize("prefix", ["", "recurrent_"])
def test_gate_and_halting_validation_at_configuration_time(read, prefix):
    settings = read()
    controls = settings.recurrence if prefix else settings
    with pytest.raises(ValueError, match=f"{prefix}gate_option"):
        read({f"{prefix}stack_gate_flag": True, f"{prefix}gate_option": None})
    with pytest.raises(ValueError, match="gate_option"):
        replace(controls.gate, enabled=True, option=None)
    assert replace(controls.gate, enabled=False, option=None).option is None
    with pytest.raises(ValueError, match=f"{prefix}halting_threshold"):
        read({f"{prefix}halting_threshold": 0.0})
    with pytest.raises(ValueError, match="halting_threshold"):
        replace(controls.halting, threshold=0.0)
    assert replace(controls.halting, threshold=1.0).threshold == 1.0


@pytest.mark.parametrize("read", READERS)
def test_typed_resolved_stacks_validate_depth_and_dropout(read):
    settings = read()
    for stack in (settings.stack, settings.gate.stack, settings.residual_stack):
        with pytest.raises(ValueError, match="num_layers"):
            replace(stack, num_layers=0)
        with pytest.raises(ValueError, match="dropout"):
            replace(stack, dropout_probability=-1.0)


class InheritedAdditiveResidual(AdditiveResidualConfig):
    pass


class UnimplementedResidual(ResidualConfig):
    pass


@pytest.mark.parametrize(
    ("read", "builder"),
    [
        (build_settings_from_flat, LinearConfigBuilder),
        (runtime_from_flat, LinearAdaptiveConfigBuilder),
    ],
)
def test_concrete_residual_selections_follow_inherited_implementations(read, builder):
    key = "stack_residual_connection_option"
    settings = read({key: InheritedAdditiveResidual})
    configuration = builder(runtime=settings).build()
    residual = configuration.experiment_config.model_config.layer_config.residual_config
    assert type(residual) is InheritedAdditiveResidual
    for invalid in (ResidualConfig, UnimplementedResidual):
        with pytest.raises(TypeError, match=key):
            read({key: invalid})
        with pytest.raises(TypeError, match="residual_connection_option"):
            replace(settings.stack, residual_connection_option=invalid)


@pytest.mark.parametrize("attribute", ["runtime", "build_settings"])
def test_plain_builder_settings_aliases_validate_before_assignment(attribute):
    builder = LinearConfigBuilder()
    replacement = replace(builder.build_settings, batch_size=7)
    setattr(builder, attribute, replacement)
    with pytest.raises(TypeError, match=attribute):
        setattr(builder, attribute, runtime_from_flat())
    assert builder.build_settings is builder.runtime is replacement
    assert builder.build().batch_size == 7


@pytest.mark.parametrize(
    "group, field, flat_key, invalid",
    [
        ("weight", "decay_warmup_batches", "weight_decay_warmup_batches", 0.5),
        ("bias", "decay_rate", "bias_decay_rate", True),
        ("mask", "surrogate_scale", "mask_surrogate_scale", True),
        (
            "input_projection",
            "weight_decay_warmup_batches",
            "input_layer_weight_decay_warmup_batches",
            1.5,
        ),
        (
            "output_projection",
            "mask_transition_width",
            "output_layer_mask_transition_width",
            True,
        ),
    ],
)
def test_adaptive_flat_and_typed_numeric_controls_require_the_same_type(
    group, field, flat_key, invalid
):
    options = getattr(runtime_from_flat(), group)
    with pytest.raises(TypeError):
        runtime_from_flat({flat_key: invalid})
    with pytest.raises(TypeError, match=field):
        replace(options, **{field: invalid})


def test_adaptive_builder_validates_runtime_replacements():
    builder = LinearAdaptiveConfigBuilder()
    replacement = replace(builder.runtime, batch_size=7)
    builder.runtime = replacement
    with pytest.raises(TypeError, match="RuntimeOptions"):
        builder.runtime = build_settings_from_flat()
    assert builder.runtime is replacement
    assert builder.build().batch_size == 7
