"""The continuation loader consumes the current recurrent module hierarchy."""

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch

from model_runtime.runs import CheckpointContinuation, InvalidCheckpointContinuation
from model_runtime.runs.checkpoints import CheckpointContinuationLifecycle
from models.catalog import model_package


@pytest.mark.parametrize(
    "controller_path", ["step.pipeline", "pipeline", "utilities", "utilities.pipeline"]
)
def test_recurrent_continuation_requires_the_registered_tensor_owners(
    tmp_path, controller_path
):
    package = model_package("linears/linear")
    config = package.build_configuration(
        config_overrides={
            "input_dim": 4,
            "hidden_dim": 4,
            "output_dim": 2,
            "stack_num_layers": 1,
            "recurrent_flag": True,
            "recurrent_max_steps": 2,
            "recurrent_initial_iterations": 2,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
        }
    )
    expected = package.build_model(config).eval()
    state = expected.state_dict()
    controller = "main_model.step.pipeline."
    assert any(name.startswith(controller + "postprocessing.gate.") for name in state)
    assert any(name.startswith(controller + "halting.model.") for name in state)
    if controller_path == "pipeline":
        state = {
            name.replace("main_model.step.", "main_model.", 1): tensor
            for name, tensor in state.items()
        }
    elif controller_path != "step.pipeline":
        state = {
            name.replace(controller, f"main_model.{controller_path}.", 1): tensor
            for name, tensor in state.items()
        }
    path = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": version("lightning"),
            "epoch": 0,
            "global_step": 1,
            "optimizer_states": [{}],
            "state_dict": state,
        },
        path,
    )
    restored = package.build_model(config).eval()
    with CheckpointContinuationLifecycle.admit(
        CheckpointContinuation(path), SimpleNamespace(runs=(object(),))
    ) as lifecycle:
        preload = lifecycle.bind_training_runs(
            [SimpleNamespace(num_epochs=2)]
        ).strict_model_preloader
        assert preload is not None
        if controller_path not in ("step.pipeline", "pipeline"):
            with pytest.raises(InvalidCheckpointContinuation, match="(?i)missing"):
                preload(restored)
            return
        preload(restored)
    tensor_shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    assert package.checkpoint_config_overrides(tensor_shapes)["stack_num_layers"] == 1
    inputs = torch.randn(2, 4)
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), expected(inputs))
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
