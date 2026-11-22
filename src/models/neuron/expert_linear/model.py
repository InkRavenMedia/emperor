from models.neuron._classifier import NeuronClassifier

from .experiment_config import ExperimentConfig


class Model(NeuronClassifier):
    experiment_type = ExperimentConfig
    invalid_experiment_message = (
        "config.experiment_config must be a Neuron Expert Linear ExperimentConfig."
    )
