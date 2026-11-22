from models import _attention_controls
from models.bert.linear_adaptive._control_factory_dependencies import (
    BertControlFactoryDependencies,
)
from models.bert.linear_adaptive._gate_config_factory import GateConfigFactory
from models.bert.linear_adaptive._halting_config_factory import HaltingConfigFactory
from models.bert.linear_adaptive._memory_config_factory import MemoryConfigFactory
from models.bert.linear_adaptive._recurrent_config_factory import (
    RecurrentConfigFactory,
)


class BertGateConfigFactory(
    _attention_controls.EncoderGateFactories[
        BertControlFactoryDependencies, GateConfigFactory
    ]
):
    _factory = GateConfigFactory


class BertHaltingConfigFactory(
    _attention_controls.EncoderHaltingFactories[
        BertControlFactoryDependencies, HaltingConfigFactory
    ]
):
    _factory = HaltingConfigFactory


class BertMemoryConfigFactory(
    _attention_controls.EncoderMemoryFactories[
        BertControlFactoryDependencies, MemoryConfigFactory
    ]
):
    _factory = MemoryConfigFactory


class BertRecurrentConfigFactory(
    _attention_controls.EncoderRecurrentFactories[
        BertControlFactoryDependencies,
        RecurrentConfigFactory,
        BertGateConfigFactory,
        BertHaltingConfigFactory,
    ]
):
    _factory = RecurrentConfigFactory
    _gate_factory = BertGateConfigFactory
    _halting_factory = BertHaltingConfigFactory
