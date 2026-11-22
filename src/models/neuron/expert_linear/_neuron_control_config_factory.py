from models.neuron._cluster_builder import NeuronControlConfigFactory as _ClusterBuilder

from ._hidden_block import HiddenBlockConfig


class NeuronControlConfigFactory(_ClusterBuilder):
    hidden_block_type = HiddenBlockConfig
