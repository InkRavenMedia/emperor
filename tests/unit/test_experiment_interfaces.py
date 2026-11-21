import importlib
import unittest

TASK_EXPORTS = {
    "emperor.experiments.bert_pretraining": (
        "BertPretrainingExperiment",
        "emperor.experiments.bert_pretraining._experiment",
    ),
    "emperor.experiments.classifier": (
        "ClassifierExperiment",
        "emperor.experiments.classifier._experiment",
    ),
    "emperor.experiments.language_model": (
        "LanguageModelExperiment",
        "emperor.experiments.language_model._experiment",
    ),
    "emperor.experiments.masked_language_model": (
        "MaskedLanguageModelExperiment",
        "emperor.experiments.masked_language_model._experiment",
    ),
    "emperor.experiments.sequence_classifier": (
        "SequenceClassifierExperiment",
        "emperor.experiments.sequence_classifier._experiment",
    ),
    "emperor.experiments.translation": (
        "TranslationExperiment",
        "emperor.experiments.translation._experiment",
    ),
}

TASK_NAMESPACES = tuple(TASK_EXPORTS)

class ExperimentInterfaceTests(unittest.TestCase):
    def test_root_and_task_packages_eagerly_export_their_interfaces(self) -> None:
        root = importlib.import_module("emperor.experiments")
        self.assertEqual(
            root.__all__,
            (
                "ExperimentTask",
                "experiment_task_label",
                "experiment_task_name",
                "resolve_experiment_task",
            ),
        )
        self.assertFalse(hasattr(root, "__getattr__"))
        self.assertFalse(hasattr(root, "_LAZY_EXPORTS"))
        for namespace_name, (export_name, _owner_name) in TASK_EXPORTS.items():
            namespace = importlib.import_module(namespace_name)
            with self.subTest(namespace=namespace_name):
                required = {export_name}
                if namespace_name == "emperor.experiments.language_model":
                    required.add("HierarchicalObjectiveConfig")
                self.assertTrue(required.issubset(namespace.__all__))
                for name in namespace.__all__:
                    self.assertTrue(hasattr(namespace, name))
                self.assertTrue(hasattr(namespace, export_name))
                self.assertFalse(hasattr(namespace, "__getattr__"))
                self.assertFalse(hasattr(namespace, "_LAZY_EXPORTS"))

    def test_task_identity_uses_the_original_owner_module(self) -> None:
        tasks = importlib.import_module("emperor.experiments._tasks")
        expected = (
            "ExperimentTask",
            "experiment_task_label",
            "experiment_task_name",
            "resolve_experiment_task",
        )
        self.assertFalse(hasattr(tasks, "__all__"))
        for name in expected:
            self.assertEqual(getattr(tasks, name).__module__, tasks.__name__)

    def test_experiment_task_classes_use_their_original_owner_modules(self) -> None:
        for namespace_name, (export_name, owner_name) in TASK_EXPORTS.items():
            with self.subTest(namespace=namespace_name):
                owner = importlib.import_module(owner_name)
                exported = getattr(owner, export_name)
                self.assertFalse(hasattr(owner, "__all__"))
                self.assertEqual(exported.__module__, owner_name)
                namespace = importlib.import_module(namespace_name)
                self.assertIs(getattr(namespace, export_name), exported)

    def test_mistaken_renamed_experiment_owner_modules_are_absent(self) -> None:
        retired_modules = ("emperor.experiments.tasks",) + tuple(
            f"{namespace}.experiment" for namespace in TASK_NAMESPACES
        )
        for module_name in retired_modules:
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
