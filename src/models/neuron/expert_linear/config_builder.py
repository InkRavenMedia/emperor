from models.neuron.expert_linear._neuron_config_builder import NeuronConfigBuilder
from models.neuron.expert_linear.runtime_defaults import DEFAULT_RUNTIME
from models.neuron.expert_linear.runtime_options import RuntimeOptions


class NeuronExpertLinearConfigBuilder(NeuronConfigBuilder):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self.runtime = runtime

    @property
    def runtime(self) -> RuntimeOptions:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.neuron.expert_linear NeuronExpertLinearConfigBuilder "
                "runtime must be RuntimeOptions"
            )
        super().__init__(**runtime._as_construction_kwargs())
        self._runtime = runtime


__all__ = ["NeuronExpertLinearConfigBuilder"]
