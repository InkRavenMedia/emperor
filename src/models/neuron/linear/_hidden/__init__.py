from model_runtime.packages.settings_validation import BuildSettingsValidation

SETTINGS_VALIDATION = BuildSettingsValidation(
    "models.neuron.linear._hidden", allow_integer_floats=False, allow_full_dropout=False
)
