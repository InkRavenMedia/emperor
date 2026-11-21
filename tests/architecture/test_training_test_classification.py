from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / "tests"


@dataclass(frozen=True)
class _FunctionRecord:
    qualified_name: str
    class_name: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef


def _is_training_operation(
    node: ast.Call,
    *,
    include_run_execution: bool,
) -> bool:
    if (
        include_run_execution
        and isinstance(node.func, ast.Name)
        and node.func.id == "execute_runs"
    ):
        return True
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "fit":
        return True
    if node.func.attr != "step":
        return False
    return "optimizer" in ast.unparse(node.func.value).lower()


def _function_records(
    tree: ast.Module,
) -> tuple[dict[str, _FunctionRecord], set[str]]:
    records: dict[str, _FunctionRecord] = {}
    class_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            records[node.name] = _FunctionRecord(node.name, None, node)
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        class_names.add(node.name)
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified_name = f"{node.name}.{member.name}"
                records[qualified_name] = _FunctionRecord(
                    qualified_name,
                    node.name,
                    member,
                )
    return records, class_names


def _local_references(
    record: _FunctionRecord,
    records: dict[str, _FunctionRecord],
    functions_by_class: dict[str, set[str]],
) -> set[str]:
    references: set[str] = set()
    for node in ast.walk(record.node):
        if isinstance(node, ast.Name):
            if node.id in records:
                references.add(node.id)
            references.update(functions_by_class.get(node.id, ()))
            continue
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"self", "cls"}
            and record.class_name is not None
        ):
            target = f"{record.class_name}.{node.attr}"
            if target in records:
                references.add(target)
    return references


def _training_calls(
    scope: ast.AST,
    local_fit_owners: set[str],
    include_run_execution: bool,
    local_super_fit: bool = False,
) -> bool:
    if isinstance(scope, ast.ClassDef):
        local_super_fit = (
            len(scope.bases) == 1
            and isinstance(scope.bases[0], ast.Name)
            and scope.bases[0].id in local_fit_owners
        )
    if isinstance(scope, ast.Call) and _is_training_operation(
        scope,
        include_run_execution=include_run_execution,
    ):
        method = scope.func
        if not (
            local_super_fit
            and isinstance(method, ast.Attribute)
            and method.attr == "fit"
            and isinstance(method.value, ast.Call)
            and isinstance(method.value.func, ast.Name)
            and method.value.func.id == "super"
        ):
            return True
    return any(
        _training_calls(child, local_fit_owners, include_run_execution, local_super_fit)
        for child in ast.iter_child_nodes(scope)
    )


def _training_tests(
    tree: ast.Module,
    *,
    include_run_execution: bool,
) -> tuple[list[_FunctionRecord], list[_FunctionRecord]]:
    records, class_names = _function_records(tree)
    functions_by_class = {
        class_name: {
            name for name, record in records.items() if record.class_name == class_name
        }
        for class_name in class_names
    }
    references = {
        name: _local_references(record, records, functions_by_class)
        for name, record in records.items()
    }
    # A nested recording double may call its locally defined parent's fit.
    # The reference graph still inspects that parent's method for real operations.
    local_fit_owners = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and not node.bases
        and f"{node.name}.fit" in records
    }
    training_functions = {
        name
        for name, record in records.items()
        if _training_calls(record.node, local_fit_owners, include_run_execution)
    }

    changed = True
    while changed:
        changed = False
        for name, targets in references.items():
            if name not in training_functions and targets & training_functions:
                training_functions.add(name)
                changed = True

    training_tests = [
        record
        for name, record in records.items()
        if name in training_functions and record.node.name.startswith("test")
    ]
    referenced_functions = set().union(*references.values(), set())
    unresolved_helpers = [
        record
        for name, record in records.items()
        if name in training_functions
        and not record.node.name.startswith("test")
        and name not in referenced_functions
    ]
    return training_tests, unresolved_helpers


def _has_training_marker(record: _FunctionRecord) -> bool:
    return any(
        ast.unparse(decorator).removesuffix("()") == "pytest.mark.training"
        for decorator in record.node.decorator_list
    )


def test_training_marker_is_registered() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    markers = project["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.partition(":")[0].strip() == "training" for marker in markers)


def test_training_operations_are_statically_classified() -> None:
    missing_markers: list[str] = []
    unresolved_helpers: list[str] = []
    for path in sorted(TEST_ROOT.rglob("*.py")):
        relative_path = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        training_tests, unresolved = _training_tests(
            tree,
            include_run_execution=relative_path.parts[:2] == ("tests", "integration"),
        )
        missing_markers.extend(
            f"{relative_path}:{record.node.lineno}: {record.qualified_name}"
            for record in training_tests
            if not _has_training_marker(record)
        )
        unresolved_helpers.extend(
            f"{relative_path}:{record.node.lineno}: {record.qualified_name}"
            for record in unresolved
        )

    assert not unresolved_helpers, (
        "Training helpers are not statically reachable from a test:\n"
        + "\n".join(unresolved_helpers)
    )
    assert not missing_markers, (
        "Tests that fit models or step optimizers require @pytest.mark.training:\n"
        + "\n".join(missing_markers)
    )


def test_classifier_follows_local_super_fit_and_real_optimizer_steps() -> None:
    tree = ast.parse("""
class Recorder:
    def fit(self): self.called = True
def test_fixture():
    class RecordingTrainer(Recorder):
        def fit(self): super().fit()
def test_real_fit():
    class Instrumented(ExternalTrainer):
        def fit(self): super().fit()
def test_optimizer():
    optimizer.step()
""")
    selected, _ = _training_tests(tree, include_run_execution=False)
    assert {item.qualified_name for item in selected} == {
        "test_real_fit",
        "test_optimizer",
    }


def test_nontraining_lane_blocks_fit_and_step_before_they_run(tmp_path) -> None:
    import os
    import subprocess
    import sys

    (tmp_path / "conftest.py").write_bytes((TEST_ROOT / "conftest.py").read_bytes())
    (tmp_path / "test_guard.py").write_text(
        """import lightning, pytest, torch
def test_training_guards():
    parameter = torch.nn.Parameter(torch.tensor(4.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(pytest.fail.Exception, match="requires @pytest.mark.training"):
        optimizer.step()
    assert parameter.item() == 4.0
    with pytest.raises(pytest.fail.Exception, match="requires @pytest.mark.training"):
        lightning.Trainer.fit(None)
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not training",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
