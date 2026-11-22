from models.gpt._model import GptModel
from models.gpt.expert_linear._boundary_config_factory import GptBoundaryConfig
from models.gpt.expert_linear.experiment_config import ExperimentConfig


class Model(GptModel[ExperimentConfig, GptBoundaryConfig]):
    experiment_type = ExperimentConfig
    boundary_type = GptBoundaryConfig
    invalid_experiment_message = (
        "config.experiment_config must be a GPT Expert Linear ExperimentConfig."
    )
