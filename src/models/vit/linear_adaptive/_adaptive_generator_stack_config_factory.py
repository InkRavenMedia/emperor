from models._adaptive_generator_stack import GeneratorStackFactory
from models.vit.linear_adaptive.runtime_options import AdaptiveGeneratorStackOptions


class AdaptiveGeneratorStackConfigFactory(
    GeneratorStackFactory[AdaptiveGeneratorStackOptions]
):
    options_type = AdaptiveGeneratorStackOptions
