from dataclasses import dataclass, field

import models.gpt.linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    GroupingConfig,
)
from models._adaptive_hidden_model import AdaptiveHiddenModelFactory
from models.gpt.linear_adaptive import _config_defaults as config_defaults
from models.gpt.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.gpt.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.gpt.linear_adaptive._gate_config_factory import GateConfigFactory
from models.gpt.linear_adaptive._halting_config_factory import HaltingConfigFactory
from models.gpt.linear_adaptive._memory_config_factory import MemoryConfigFactory
from models.gpt.linear_adaptive._recurrent_config_factory import (
    RecurrentConfigFactory,
)
from models.gpt.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    DynamicMemoryOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
)


@dataclass(frozen=True)
class HiddenModelConfigDependencies:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    hidden_dim: int
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    output_dim: int


class HiddenModelConfigFactory(
    AdaptiveHiddenModelFactory[
        SubmoduleStackOptions,
        DynamicMemoryOptions,
        RecurrentControllerOptions,
        AdaptiveGeneratorStackOptions,
    ]
):
    layer_template_type = AdaptiveLinearLayerTemplateConfig
    gate_type = GateConfigFactory
    halting_type = HaltingConfigFactory
    memory_type = MemoryConfigFactory
    recurrent_type = RecurrentConfigFactory
    generator_type = AdaptiveGeneratorStackConfigFactory

    def __init__(self, dependencies: HiddenModelConfigDependencies) -> None:
        super().__init__(
            dependencies, config, config_defaults, config_defaults.LinearRole.MAIN
        )
