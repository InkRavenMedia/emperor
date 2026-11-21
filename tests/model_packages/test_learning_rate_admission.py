import pytest

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.overrides import validate_typed_overrides
from model_runtime.packages import RuntimeDefaultsError
from model_runtime.runs import InvalidRunRequest, RunRequest, plan_runs
from models.catalog import discover_model_packages


@pytest.mark.parametrize(
    "package", discover_model_packages(), ids=lambda p: p.catalog_key
)
def test_learning_rate_is_finite_at_each_public_entry_point(package):
    dataset = package.resolve_dataset(None).__name__
    preset = package.preset_name(package.default_preset)
    spec = package.runtime_defaults_spec
    search_key = "SEARCH_SPACE_LEARNING_RATE"
    for invalid in (float("nan"), float("inf"), float("-inf")):
        values = {"learning_rate": invalid}
        with pytest.raises(ValueError, match="finite"):
            spec.parse_value("LEARNING_RATE", str(invalid))
        if spec.has_search_value(search_key):
            with pytest.raises(ValueError, match="finite"):
                spec.parse_search_value("LEARNING_RATE", invalid, search_key=search_key)
            with pytest.raises(ValueError, match="finite"):
                spec.parse_search_key_value(search_key, str(invalid))
        with pytest.raises((ValueError, RuntimeDefaultsError), match="finite"):
            package.bind_runtime_defaults(values)
        with pytest.raises((ValueError, RuntimeDefaultsError), match="finite"):
            package.build_configuration(config_overrides=values)
        with pytest.raises(InspectionError, match="finite"):
            validate_typed_overrides(package, values)
        with pytest.raises(InvalidRunRequest, match="finite"):
            plan_runs(
                package,
                RunRequest(presets=(preset,), datasets=(dataset,), overrides=values),
            )
    values = {"learning_rate": 0.002}
    assert spec.parse_value("LEARNING_RATE", "0.002") == 0.002
    if spec.has_search_value(search_key):
        assert spec.parse_search_key_value(search_key, "0.002") == 0.002
    assert package.build_configuration(config_overrides=values).learning_rate == 0.002
    assert plan_runs(
        package, RunRequest(presets=(preset,), datasets=(dataset,), overrides=values)
    ).runs
