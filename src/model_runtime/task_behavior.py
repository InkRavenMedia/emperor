from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

from emperor.experiments import ExperimentTask

if TYPE_CHECKING:
    from emperor.transformer.hierarchical import HierarchicalLanguageModelBatch


class _FixedTokenOutputValidator(Protocol):
    def validate_fixed_token_output(self, config: object, operation: str) -> None: ...


class _HierarchicalExample(TypedDict):
    context_texts: list[str]
    target_texts: list[str]
    bos_mask: list[bool]


class _HierarchicalBatchCollator(Protocol):
    def collate(
        self, examples: Sequence[_HierarchicalExample]
    ) -> HierarchicalLanguageModelBatch: ...


class SyntheticInputError(ValueError):
    """An Experiment Task cannot create a safe synthetic input."""


@dataclass(frozen=True, slots=True)
class DatasetArgument:
    name: str
    configuration_path: tuple[str, ...]

    def value_from(self, configuration: Any) -> Any:
        value = configuration
        for attribute in self.configuration_path:
            value = getattr(value, attribute)
        return value


@dataclass(frozen=True, slots=True)
class RankingMetric:
    tier: float
    key: str
    direction: float = 1.0


@dataclass(frozen=True, slots=True)
class ExperimentTaskBehavior:
    task: ExperimentTask
    synthetic_input_builder: Callable[[type, Any], tuple[Any, ...]]
    dataset_arguments: tuple[DatasetArgument, ...]
    ranking_metrics: tuple[RankingMetric, ...]
    core_result_metric_keys: tuple[str, ...]

    def synthetic_inputs(self, dataset: type, configuration: Any) -> tuple[Any, ...]:
        return self.synthetic_input_builder(dataset, configuration)

    def validate_dataset(self, dataset: type, configuration: Any) -> None:
        self.validate_configuration(configuration)
        if self.task == ExperimentTask.BERT_PRETRAINING:
            experiment = getattr(configuration, "experiment_config", None)
            hierarchical = getattr(experiment, "masked_word_decoder_config", None)
            raw_words = bert_dataset_uses_masked_words(dataset)
            if (hierarchical is not None) != raw_words:
                raise ValueError(
                    "Masked-word BERT requires a compatible dataset such as WikiText103BertHierarchical."
                )
        if self.task != ExperimentTask.CAUSAL_LANGUAGE_MODELING:
            return
        experiment = getattr(configuration, "experiment_config", None)
        hierarchical = getattr(experiment, "hierarchical_language_model_config", None)
        validate_language_model_dataset(dataset, hierarchical=hierarchical is not None)

    def validate_configuration(self, configuration: Any) -> None:
        from emperor.transformer.sequence_expansion import SequenceExpansion

        experiment = getattr(configuration, "experiment_config", None)
        if getattr(experiment, "hierarchical_language_model_config", None) is None:
            return
        validator = cast(_FixedTokenOutputValidator, SequenceExpansion.VALIDATOR)
        validator.validate_fixed_token_output(
            getattr(experiment, "decoder_config", None),
            "Hierarchical language modeling",
        )

    def dataset_constructor_kwargs(self, configuration: Any) -> dict[str, Any]:
        arguments = {
            argument.name: argument.value_from(configuration)
            for argument in self.dataset_arguments
        }
        experiment = getattr(configuration, "experiment_config", None)
        hierarchical = getattr(experiment, "hierarchical_language_model_config", None)
        if (
            self.task == ExperimentTask.CAUSAL_LANGUAGE_MODELING
            and hierarchical is not None
        ):
            arguments["max_token_bytes"] = hierarchical.embedding_config.max_token_bytes
        if self.task == ExperimentTask.BERT_PRETRAINING:
            decoder = getattr(experiment, "masked_word_decoder_config", None)
            if decoder is not None:
                arguments["max_token_bytes"] = decoder.max_token_bytes
        return arguments

    def ranking_score(self, result: Mapping[str, Any]) -> tuple[float, float] | None:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        typed_metrics = cast(Mapping[str, Any], metrics)
        for preference in self.ranking_metrics:
            if (
                self.task is ExperimentTask.BERT_PRETRAINING
                and result.get("rankingMetric") != preference.key
            ):
                continue
            score = _finite_score(typed_metrics.get(preference.key))
            if score is not None:
                return preference.tier, preference.direction * score
        return None


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        score = float(value)
    except OverflowError:
        return None
    return score if math.isfinite(score) else None


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SyntheticInputError(
            f"Cannot build a shape-trace input because {label} is not a positive "
            f"integer: {value!r}."
        )
    return value


def _token_batch(length: int, vocabulary_size: int, token_id: int = 1) -> Any:
    import torch

    if not 0 <= token_id < vocabulary_size:
        token_id = 1 if vocabulary_size > 1 else 0
    return torch.full((1, length), token_id, dtype=torch.long)


def _image_inputs(dataset: type, _configuration: Any) -> tuple[Any, ...]:
    import torch

    channels = _positive_integer(
        getattr(dataset, "num_channels", None),
        f"{dataset.__name__}.num_channels",
    )
    width = _positive_integer(
        getattr(dataset, "default_width", None),
        f"{dataset.__name__}.default_width",
    )
    height = _positive_integer(
        getattr(dataset, "default_height", None),
        f"{dataset.__name__}.default_height",
    )
    return (torch.zeros((1, channels, height, width), dtype=torch.float32),)


def _joint_image_inputs(dataset: type, configuration: Any) -> tuple[Any, ...]:
    return (*_image_inputs(dataset, configuration), "mnist")


def _token_inputs(_dataset: type, configuration: Any) -> tuple[Any, ...]:
    sequence_length = _positive_integer(
        getattr(configuration, "sequence_length", None),
        "configuration.sequence_length",
    )
    experiment = getattr(configuration, "experiment_config", None)
    masked_word_decoder = getattr(experiment, "masked_word_decoder_config", None)
    if masked_word_decoder is not None:
        from emperor.datasets.text.bert_pretraining import BertMaskedWordBatch

        return (
            BertMaskedWordBatch.collate(
                [(("a",), ("b",), 0)],
                sequence_length=sequence_length,
                max_token_bytes=masked_word_decoder.max_token_bytes,
                mlm_probability=1.0,
            ),
        )
    hierarchical = getattr(experiment, "hierarchical_language_model_config", None)
    if hierarchical is not None:
        from emperor.transformer.hierarchical import HierarchicalLanguageModelBatch

        collator = cast(_HierarchicalBatchCollator, HierarchicalLanguageModelBatch)
        return (
            collator.collate(
                [
                    {
                        "context_texts": [""] + ["a"] * (sequence_length - 1),
                        "target_texts": ["a"] * sequence_length,
                        "bos_mask": [True] + [False] * (sequence_length - 1),
                    }
                ]
            ),
        )
    vocabulary_size = _positive_integer(
        getattr(configuration, "input_dim", None),
        "configuration.input_dim",
    )
    return (_token_batch(sequence_length, vocabulary_size),)


def _translation_inputs(_dataset: type, configuration: Any) -> tuple[Any, ...]:
    experiment_config = getattr(configuration, "experiment_config", None)
    source_length = _positive_integer(
        getattr(experiment_config, "source_sequence_length", None),
        "configuration.experiment_config.source_sequence_length",
    )
    target_length = _positive_integer(
        getattr(experiment_config, "target_sequence_length", None),
        "configuration.experiment_config.target_sequence_length",
    )
    vocabulary_size = _positive_integer(
        getattr(experiment_config, "vocab_size", None),
        "configuration.experiment_config.vocab_size",
    )
    source_token = int(getattr(experiment_config, "bos_token_id", 1))
    return (
        _token_batch(source_length, vocabulary_size, source_token),
        _token_batch(max(1, target_length - 1), vocabulary_size, source_token),
    )


_BATCH_SIZE = DatasetArgument("batch_size", ("batch_size",))
_SEQUENCE_LENGTH = DatasetArgument("sequence_length", ("sequence_length",))
_SOURCE_SEQUENCE_LENGTH = DatasetArgument(
    "source_sequence_length",
    ("experiment_config", "source_sequence_length"),
)
_TARGET_SEQUENCE_LENGTH = DatasetArgument(
    "target_sequence_length",
    ("experiment_config", "target_sequence_length"),
)
_VALIDATION_ACCURACY = RankingMetric(
    tier=1.0,
    key="validation/accuracy",
)
_VALIDATION_LOSS = RankingMetric(
    tier=1.0,
    key="validation/loss",
    direction=-1.0,
)
_VALIDATION_BLEU = RankingMetric(
    tier=2.0,
    key="validation/bleu",
)

_BERT_EXACT_WORD_ACCURACY = RankingMetric(1.0, "validation/mlm/exact_word_accuracy")
_BERT_MASKED_TOKEN_ACCURACY = RankingMetric(1.0, "validation/mlm/masked_accuracy")


def validate_language_model_dataset(dataset: type, *, hierarchical: bool) -> None:
    raw_text = bool(getattr(dataset, "hierarchical_language_model_flag", False))
    if hierarchical and not raw_text:
        raise ValueError(
            "Hierarchical language modeling requires a raw-text dataset such as "
            f"WikiText103Hierarchical. Received {dataset.__name__}."
        )
    if raw_text and not hierarchical:
        raise ValueError(
            f"{dataset.__name__} is a raw-text dataset. Token-level language "
            "modeling requires a token-ID dataset such as WikiText2."
        )


def bert_dataset_uses_masked_words(dataset_type: type[object]) -> bool:
    from emperor.datasets.text.bert_pretraining import (
        PennTreebankBertPretraining,
        WikiText2BertPretraining,
    )

    token_adapters = (PennTreebankBertPretraining, WikiText2BertPretraining)
    if issubclass(dataset_type, token_adapters) and not hasattr(
        dataset_type, "hierarchical_masked_word_flag"
    ):
        return False
    flag = getattr(dataset_type, "hierarchical_masked_word_flag", None)
    if not isinstance(flag, bool):
        raise ValueError(
            f"BERT dataset {dataset_type.__name__} must declare "
            "hierarchical_masked_word_flag as a boolean."
        )
    return flag


def bert_validation_metric(dataset_type: type) -> str:
    """Choose the measured unit for this dataset when writing its Run result."""
    if bert_dataset_uses_masked_words(dataset_type):
        return _BERT_EXACT_WORD_ACCURACY.key
    return _BERT_MASKED_TOKEN_ACCURACY.key


def _stage_metric_keys(*metric_names: str) -> tuple[str, ...]:
    return tuple(
        f"{stage}/{metric_name}"
        for stage in ("train", "validation", "test")
        for metric_name in metric_names
    )


_IMAGE_CLASSIFICATION_RESULT_METRIC_KEYS = (
    *_stage_metric_keys("loss", "accuracy", "f1_score"),
    "train/loss_epoch",
    "train/accuracy_epoch",
    "validation/loss_epoch",
    "validation/accuracy_epoch",
    "gap/accuracy",
    "gap/loss",
    "best_validation/accuracy",
    "best_validation/loss",
    "best_validation/epoch",
    "best_validation/accuracy_epoch",
    "best_validation/loss_epoch",
)
_BERT_PRETRAINING_RESULT_METRIC_KEYS = (
    *_stage_metric_keys(
        "loss",
        "mlm/loss",
        "mlm/perplexity",
        "mlm/masked_accuracy",
        "mlm/masked_top_5_accuracy",
        "mlm/byte_reconstruction_loss",
        "mlm/oversized_words",
        "nsp/loss",
        "nsp/accuracy",
        "auxiliary/loss",
    ),
    "validation/mlm/exact_word_accuracy",
    "test/mlm/exact_word_accuracy",
)
_TEXT_TRANSLATION_RESULT_METRIC_KEYS = (
    *_stage_metric_keys(
        "loss",
        "nll",
        "perplexity",
        "token_accuracy",
        "auxiliary_loss",
    ),
    "validation/bleu",
    "test/bleu",
)
_CAUSAL_LANGUAGE_MODELING_RESULT_METRIC_KEYS = _stage_metric_keys(
    "loss",
    "cross_entropy",
    "perplexity",
    "auxiliary_loss",
    "symbol_cross_entropy",
    "bits_per_byte",
)

_DECLARED_BEHAVIORS = (
    ExperimentTaskBehavior(
        task=ExperimentTask.IMAGE_CLASSIFICATION,
        synthetic_input_builder=_image_inputs,
        dataset_arguments=(_BATCH_SIZE,),
        ranking_metrics=(_VALIDATION_ACCURACY,),
        core_result_metric_keys=_IMAGE_CLASSIFICATION_RESULT_METRIC_KEYS,
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.JOINT_IMAGE_CLASSIFICATION,
        synthetic_input_builder=_joint_image_inputs,
        dataset_arguments=(_BATCH_SIZE,),
        ranking_metrics=(_VALIDATION_ACCURACY,),
        core_result_metric_keys=(
            *_stage_metric_keys("accuracy", "loss"),
            *(
                f"{stage}/{name}/{metric}"
                for stage in ("validation", "test")
                for name in ("mnist", "fashion-mnist", "cifar10", "cifar100")
                for metric in ("loss", "accuracy")
            ),
            "validation/macro_accuracy",
            "test/macro_accuracy",
        ),
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.BERT_PRETRAINING,
        synthetic_input_builder=_token_inputs,
        dataset_arguments=(_BATCH_SIZE, _SEQUENCE_LENGTH),
        ranking_metrics=(_BERT_EXACT_WORD_ACCURACY, _BERT_MASKED_TOKEN_ACCURACY),
        core_result_metric_keys=_BERT_PRETRAINING_RESULT_METRIC_KEYS,
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.TEXT_TRANSLATION,
        synthetic_input_builder=_translation_inputs,
        dataset_arguments=(
            _BATCH_SIZE,
            _SOURCE_SEQUENCE_LENGTH,
            _TARGET_SEQUENCE_LENGTH,
        ),
        ranking_metrics=(_VALIDATION_BLEU, _VALIDATION_LOSS),
        core_result_metric_keys=_TEXT_TRANSLATION_RESULT_METRIC_KEYS,
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.CAUSAL_LANGUAGE_MODELING,
        synthetic_input_builder=_token_inputs,
        dataset_arguments=(_BATCH_SIZE, _SEQUENCE_LENGTH),
        ranking_metrics=(_VALIDATION_LOSS,),
        core_result_metric_keys=_CAUSAL_LANGUAGE_MODELING_RESULT_METRIC_KEYS,
    ),
)


def _compile_registry() -> Mapping[ExperimentTask, ExperimentTaskBehavior]:
    task_counts: Counter[ExperimentTask] = Counter(
        behavior.task for behavior in _DECLARED_BEHAVIORS
    )
    duplicates = sorted(task.name for task, count in task_counts.items() if count > 1)
    missing = sorted(task.name for task in ExperimentTask if task not in task_counts)
    if duplicates or missing:
        details: list[str] = []
        if duplicates:
            details.append(f"duplicates: {', '.join(duplicates)}")
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        raise RuntimeError(
            "Invalid Experiment Task behavior registry (" + "; ".join(details) + ")"
        )
    return MappingProxyType(
        {behavior.task: behavior for behavior in _DECLARED_BEHAVIORS}
    )


EXPERIMENT_TASK_BEHAVIORS = _compile_registry()
CORE_RESULT_METRIC_KEYS = frozenset(
    key
    for behavior in EXPERIMENT_TASK_BEHAVIORS.values()
    for key in (
        *behavior.core_result_metric_keys,
        *(metric.key for metric in behavior.ranking_metrics),
    )
)


def experiment_task_behavior(task: ExperimentTask) -> ExperimentTaskBehavior:
    try:
        return EXPERIMENT_TASK_BEHAVIORS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported Experiment Task behavior: {task!r}.") from exc


__all__ = [
    "CORE_RESULT_METRIC_KEYS",
    "EXPERIMENT_TASK_BEHAVIORS",
    "ExperimentTaskBehavior",
    "SyntheticInputError",
    "bert_dataset_uses_masked_words",
    "bert_validation_metric",
    "experiment_task_behavior",
    "validate_language_model_dataset",
]
