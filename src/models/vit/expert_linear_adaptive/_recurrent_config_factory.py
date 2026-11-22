from models._recurrent_block import (
    ProjectedResidualRecurrentFactory,
)
from models.vit.expert_linear_adaptive._gate_config_factory import GateConfigFactory
from models.vit.expert_linear_adaptive._halting_config_factory import (
    HaltingConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import RecurrentControllerOptions


class RecurrentConfigFactory(
    ProjectedResidualRecurrentFactory[
        RecurrentControllerOptions, GateConfigFactory, HaltingConfigFactory
    ]
):
    pass
