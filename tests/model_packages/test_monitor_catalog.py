import pytest

from emperor.monitoring import MonitorSettings
from model_runtime.monitoring_catalog import monitors_for_model
from models.catalog import discover_model_packages


@pytest.mark.parametrize(
    "package", discover_model_packages(), ids=lambda p: p.catalog_key
)
def test_monitor_metadata_and_training_share_immutable_definitions(package):
    options = package.monitor_options()
    catalog_options = monitors_for_model(package.catalog_key)
    assert options == catalog_options
    assert len({option.name for option in options}) == len(options)
    settings = MonitorSettings(log_every_n_steps=19)
    for option in options:
        assert isinstance(option.kinds, tuple)
        exported = option.to_api()
        exported["kinds"].clear()
        assert option.to_api()["kinds"] == list(option.kinds)
        callback = option.build_callback(settings)
        assert callback.log_every_n_steps == settings.log_every_n_steps
        assert option.build_callback(settings) is not callback
    catalog_options.clear()
    assert package.monitor_options() == options
