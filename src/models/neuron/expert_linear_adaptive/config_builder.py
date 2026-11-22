from models.neuron.expert_linear_adaptive._neuron_config_builder import (
    NeuronConfigBuilder,
)
from models.neuron.expert_linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.neuron.expert_linear_adaptive.runtime_options import RuntimeOptions


class NeuronExpertLinearAdaptiveConfigBuilder(NeuronConfigBuilder):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self.runtime = runtime

    @property
    def runtime(self) -> RuntimeOptions:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.neuron.expert_linear_adaptive "
                "NeuronExpertLinearAdaptiveConfigBuilder runtime must be "
                "RuntimeOptions"
            )
        super().__init__(**runtime._as_construction_kwargs())
        self._runtime = runtime


__all__ = ["NeuronExpertLinearAdaptiveConfigBuilder"]
