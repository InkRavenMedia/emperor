from model_runtime.packages.settings_validation import BuildSettingsValidation

SETTINGS_VALIDATION = BuildSettingsValidation(
    "models.neuron.linear_adaptive._hidden",
    allow_integer_floats=True,
    allow_full_dropout=True,
    invalid_value_format="runtime key {key!r} {requirement}",
)
