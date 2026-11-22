from models._adaptive_generator_stack import GeneratorStackFactory
from models.neuron.expert_linear_adaptive._hidden.runtime_options import (
    AdaptiveGeneratorStackOptions,
)


class AdaptiveGeneratorStackConfigFactory(
    GeneratorStackFactory[AdaptiveGeneratorStackOptions]
):
    options_type = AdaptiveGeneratorStackOptions
