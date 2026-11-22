from models import _attention_controls
from models.gpt.linear_adaptive._control_factory_dependencies import (
    GptControlFactoryDependencies,
)
from models.gpt.linear_adaptive._gate_config_factory import GateConfigFactory
from models.gpt.linear_adaptive._halting_config_factory import HaltingConfigFactory
from models.gpt.linear_adaptive._memory_config_factory import MemoryConfigFactory
from models.gpt.linear_adaptive._recurrent_config_factory import (
    RecurrentConfigFactory,
)


class GptGateConfigFactory(
    _attention_controls.DecoderGateFactories[
        GptControlFactoryDependencies, GateConfigFactory
    ]
):
    _factory = GateConfigFactory


class GptHaltingConfigFactory(
    _attention_controls.DecoderHaltingFactories[
        GptControlFactoryDependencies, HaltingConfigFactory
    ]
):
    _factory = HaltingConfigFactory


class GptMemoryConfigFactory(
    _attention_controls.DecoderMemoryFactories[
        GptControlFactoryDependencies, MemoryConfigFactory
    ]
):
    _factory = MemoryConfigFactory


class GptRecurrentConfigFactory(
    _attention_controls.DecoderRecurrentFactories[
        GptControlFactoryDependencies,
        RecurrentConfigFactory,
        GptGateConfigFactory,
        GptHaltingConfigFactory,
    ]
):
    _factory = RecurrentConfigFactory
    _gate_factory = GptGateConfigFactory
    _halting_factory = GptHaltingConfigFactory
