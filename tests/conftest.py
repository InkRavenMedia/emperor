from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def require_training_marker(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Enforce the training marker before a real fit or parameter update can start."""
    if request.node.get_closest_marker("training") is not None:
        yield
        return

    from lightning import Trainer
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def reject_training(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "A real Trainer.fit or optimizer step requires @pytest.mark.training"
        )

    monkeypatch.setattr(Trainer, "fit", reject_training)
    handle = register_optimizer_step_pre_hook(reject_training)
    try:
        yield
    finally:
        handle.remove()
