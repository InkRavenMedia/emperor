"""Compare current checkpoint names, dimensions and dtypes with their declared layout."""

import json
from pathlib import Path

import pytest

from models.catalog import discover_model_packages

_LAYOUT = json.loads(
    (Path(__file__).parent / "fixtures" / "checkpoint_layout.json").read_text()
)


def _expected_tensors(package):
    pending = [("", _LAYOUT["packages"][package])]
    tensors = {}
    while pending:
        prefix, module = pending.pop()
        for name, value in _LAYOUT["modules"][module].items():
            path = prefix + name
            if isinstance(value, str):
                pending.append((path + ".", value))
            else:
                tensors[path] = value
    return tensors


def test_checkpoint_layout_covers_the_catalog():
    assert set(_LAYOUT["packages"]) == {
        package.catalog_key for package in discover_model_packages()
    }


@pytest.mark.parametrize(
    "package", discover_model_packages(), ids=lambda p: p.catalog_key
)
def test_current_default_checkpoint_names_shapes_and_dtypes(package):
    model = package.build_model(package.build_configuration())
    actual = {
        name: [str(tensor.dtype).removeprefix("torch."), *tensor.shape]
        for name, tensor in model.state_dict().items()
    }
    # Mapping equality shows the exact added, missing or reshaped tensor on failure.
    assert actual == _expected_tensors(package.catalog_key)
    tensor_owners = {name.rpartition(".")[0] for name in actual}
    registered_modules = dict(model.named_modules(remove_duplicate=False))
    assert tensor_owners <= registered_modules.keys()
