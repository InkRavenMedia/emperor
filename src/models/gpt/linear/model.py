from emperor.transformer.sequence_expansion import SequenceExpansion
from models.gpt._model import GptModel
from models.gpt.linear._boundary_config_factory import GptBoundaryConfig
from models.gpt.linear.experiment_config import ExperimentConfig


class Model(GptModel[ExperimentConfig, GptBoundaryConfig]):
    experiment_type = ExperimentConfig
    boundary_type = GptBoundaryConfig
    invalid_experiment_message = (
        "config.experiment_config must be a GPT Linear ExperimentConfig."
    )

    @staticmethod
    def _validate_hierarchical_output(
        experiment_config: ExperimentConfig, boundary_config: GptBoundaryConfig
    ) -> None:
        SequenceExpansion.VALIDATOR.validate_fixed_token_output(
            experiment_config.decoder_config, "Hierarchical language modeling"
        )
        if boundary_config.lm_head_options.weight_tying_flag:
            raise ValueError(
                "Hierarchical language modeling requires lm_head_weight_tying_flag=False"
            )
