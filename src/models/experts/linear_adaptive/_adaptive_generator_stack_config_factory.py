from models._adaptive_generator_stack import GeneratorStackFactory
from models.experts.linear_adaptive.runtime_options import AdaptiveGeneratorStackOptions


class AdaptiveGeneratorStackConfigFactory(
    GeneratorStackFactory[AdaptiveGeneratorStackOptions]
):
    options_type = AdaptiveGeneratorStackOptions
