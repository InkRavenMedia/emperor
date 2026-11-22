from models.vit._classifier import ExpertHeadClassifier

from .experiment_config import ExperimentConfig


class Model(ExpertHeadClassifier):
    experiment_type = ExperimentConfig
    invalid_experiment_message = (
        "config.experiment_config must be a ViT Expert Linear ExperimentConfig."
    )
