from models import _router_controller_config


class RouterControllerModel(_router_controller_config.RouterControllerModel):
    pass


class RouterControllerModelConfig(
    _router_controller_config.RouterControllerModelConfig
):
    def _registry_owner(self) -> type:
        return RouterControllerModel
