from __future__ import annotations

from model_runtime.packages import ModelIdentity, ModelPackage
from models._package_adapter import ModelPackageAdapter

_IDENTITY = ModelIdentity("mlp_mixer", "expert_linear_adaptive")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY

    def build_model(self, configuration):
        from .experiment_config import MaskedWordExperimentConfig

        if isinstance(configuration.experiment_config, MaskedWordExperimentConfig):
            from ._text_model import MaskedWordModel

            return MaskedWordModel(configuration)
        from .model import Model

        return Model(configuration)


MODEL_PACKAGE = ModelPackage(_IDENTITY, _ModelPackageAdapter())

__all__ = ["MODEL_PACKAGE"]
