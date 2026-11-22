from models.bert._model import BertModel
from models.bert.linear_adaptive._boundary_config_factory import BertBoundaryConfig
from models.bert.linear_adaptive.experiment_config import ExperimentConfig


class Model(BertModel[ExperimentConfig, BertBoundaryConfig]):
    experiment_type = ExperimentConfig
    boundary_type = BertBoundaryConfig
    invalid_experiment_message = (
        "config.experiment_config must be a BERT Linear Adaptive ExperimentConfig."
    )
