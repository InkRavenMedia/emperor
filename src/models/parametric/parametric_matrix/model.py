from models.parametric._classifier import ParametricClassifier

from .experiment_config import ExperimentConfig


class Model(ParametricClassifier):
    experiment_type = ExperimentConfig
    invalid_experiment_message = (
        "config.experiment_config must be a parametric_matrix ExperimentConfig."
    )
