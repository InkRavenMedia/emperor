from models.gpt._model import GptModel
from models.gpt.linear_adaptive._boundary_config_factory import GptBoundaryConfig
from models.gpt.linear_adaptive.experiment_config import ExperimentConfig


class Model(GptModel[ExperimentConfig, GptBoundaryConfig]):
    experiment_type = ExperimentConfig
    boundary_type = GptBoundaryConfig
    invalid_experiment_message = (
        "config.experiment_config must be a GPT Linear Adaptive ExperimentConfig."
    )
