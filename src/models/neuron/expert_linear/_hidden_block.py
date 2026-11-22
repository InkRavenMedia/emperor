from models.neuron import _hidden_block


class HiddenBlockAdapter(_hidden_block.HiddenBlockAdapter):
    pass


class HiddenBlockConfig(_hidden_block.HiddenBlockConfig):
    def _registry_owner(self) -> type:
        return HiddenBlockAdapter
