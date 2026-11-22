from emperor.datasets.image.classification import (
    Cifar10,
    Cifar100,
    FashionMNIST,
    JointImageClassification,
    Mnist,
)
from emperor.experiments import ExperimentTask

DEFAULT_EXPERIMENT_TASK: ExperimentTask = ExperimentTask.IMAGE_CLASSIFICATION
DATASET_OPTIONS_BY_TASK: dict[ExperimentTask, list[type]] = {
    DEFAULT_EXPERIMENT_TASK: [Mnist, FashionMNIST, Cifar10, Cifar100],
    ExperimentTask.JOINT_IMAGE_CLASSIFICATION: [JointImageClassification],
}
