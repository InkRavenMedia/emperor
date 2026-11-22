from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from models.linears.linear._control_config_factory import ControlConfigFactory
from models.linears.linear.runtime_options import LinearBuildSettings


class HiddenModelConfigFactory:
    def __init__(self, build_settings: LinearBuildSettings) -> None:
        self.build_settings = build_settings
        self.control_factory = ControlConfigFactory(build_settings)

    def build_hidden_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        stack_config = self.__build_hidden_stack()
        hidden_model_config = self.control_factory.wrap_recurrent(stack_config)
        return hidden_model_config

    def __build_hidden_stack(self) -> LayerStackConfig:
        options = self.build_settings.stack
        layer_config, memory_config = (
            self.control_factory.build_hidden_layer_and_shared_memory()
        )
        return LayerStackConfig(
            hidden_dim=self.build_settings.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
            shared_gate_config=self.build_settings.gate.shared_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )
