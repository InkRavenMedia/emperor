from models import _attention_controls
from models.vit.linear._control_factory_dependencies import (
    VitControlFactoryDependencies,
)
from models.vit.linear._gate_config_factory import GateConfigFactory
from models.vit.linear._halting_config_factory import HaltingConfigFactory
from models.vit.linear._memory_config_factory import MemoryConfigFactory
from models.vit.linear._recurrent_config_factory import RecurrentConfigFactory


class VitGateConfigFactory(
    _attention_controls.EncoderGateFactories[
        VitControlFactoryDependencies, GateConfigFactory
    ]
):
    _factory = GateConfigFactory


class VitHaltingConfigFactory(
    _attention_controls.EncoderHaltingFactories[
        VitControlFactoryDependencies, HaltingConfigFactory
    ]
):
    _factory = HaltingConfigFactory


class VitMemoryConfigFactory(
    _attention_controls.EncoderMemoryFactories[
        VitControlFactoryDependencies, MemoryConfigFactory
    ]
):
    _factory = MemoryConfigFactory


class VitRecurrentConfigFactory(
    _attention_controls.EncoderRecurrentFactories[
        VitControlFactoryDependencies,
        RecurrentConfigFactory,
        VitGateConfigFactory,
        VitHaltingConfigFactory,
    ]
):
    _factory = RecurrentConfigFactory
    _gate_factory = VitGateConfigFactory
    _halting_factory = VitHaltingConfigFactory
