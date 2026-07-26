from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from agent_defense.artifacts import DetectorArtifact
from agent_defense.detectors import fit_direction_artifact, fit_linear_probe_artifact, load_detector
from agent_defense.types import CandidateToolCall, DetectionContext

_ACTIVATION_COMPATIBILITY_KEYS = (
    "checkpoint_content_id",
    "revision",
    "model_dtype",
    "quantization_config_hash",
    "tokenizer_class",
    "state_kind",
    "module_path",
    "chat_template_hash",
    "tool_schema_hash",
    "system_message_hash",
    "render_mode",
)


@dataclass(frozen=True)
class ActivationSample:
    sample_id: str
    label: int
    split: Literal["train", "calibration", "test"]
    activation: tuple[float, ...]
    model_id: str
    layer: int
    position: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ActivationSample:
        label = int(data["label"])
        if label not in (0, 1):
            raise ValueError("Activation labels must be 0 (benign) or 1 (hijacked/violating)")
        split = str(data["split"])
        if split not in {"train", "calibration", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        activation = tuple(float(value) for value in data["activation"])
        if not activation:
            raise ValueError("Activation vectors must not be empty")
        if not np.all(np.isfinite(np.asarray(activation, dtype=np.float64))):
            raise ValueError("Activation vectors must contain only finite values")
        return cls(
            sample_id=str(data["sample_id"]),
            label=label,
            split=split,  # type: ignore[arg-type]
            activation=activation,
            model_id=str(data["model_id"]),
            layer=int(data["layer"]),
            position=str(data["position"]),
            metadata=dict(data.get("metadata", {})),
        )


def load_activation_samples(path: str | Path) -> list[ActivationSample]:
    samples: list[ActivationSample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                samples.append(ActivationSample.from_mapping(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid activation sample at line {line_number}: {error}") from error
    if not samples:
        raise ValueError("The activation dataset is empty")
    _validate_compatibility(samples)
    return samples


def _load_label_manifest(path: str | Path) -> dict[str, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((item["sample_id"], item["label"]) for item in raw)
    else:
        raise ValueError("Label manifest must be a JSON object or a list of sample_id/label objects")
    labels: dict[str, int] = {}
    for sample_id, value in items:
        label = int(value)
        if label not in {0, 1}:
            raise ValueError(f"Invalid label for {sample_id!r}: {value!r}")
        key = str(sample_id)
        if key in labels:
            raise ValueError(f"Duplicate sample_id in label manifest: {key}")
        labels[key] = label
    if not labels:
        raise ValueError("Label manifest is empty")
    return labels


def apply_label_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    require_all: bool = True,
) -> dict[str, int]:
    """Apply reviewed call-level labels while leaving activation values unchanged."""

    source = Path(dataset_path)
    target = Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("Write labeled activations to a new file; in-place overwrite is not allowed")
    labels = _load_label_manifest(manifest_path)
    seen: set[str] = set()
    output_lines: list[str] = []
    labeled = 0
    unlabeled = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            sample_id = str(payload["sample_id"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid activation sample at line {line_number}: {error}") from error
        if sample_id in seen:
            raise ValueError(f"Duplicate activation sample_id: {sample_id}")
        seen.add(sample_id)
        if sample_id in labels:
            payload["label"] = labels[sample_id]
            metadata = dict(payload.get("metadata", {}))
            metadata["label_requires_review"] = False
            metadata["label_source"] = "reviewed_manifest"
            payload["metadata"] = metadata
            labeled += 1
        else:
            unlabeled += 1
            if require_all:
                raise ValueError(f"No reviewed label for activation sample {sample_id!r}")
        output_lines.append(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    unknown = set(labels) - seen
    if unknown:
        raise ValueError(f"Label manifest contains unknown sample IDs: {sorted(unknown)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return {"samples": len(output_lines), "labeled": labeled, "unlabeled": unlabeled}


def _validate_compatibility(samples: Sequence[ActivationSample]) -> None:
    if not samples:
        raise ValueError("At least one activation sample is required")
    reference = samples[0]
    dimension = len(reference.activation)
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    for sample in samples:
        if sample.sample_id in sample_ids:
            raise ValueError(f"Duplicate activation sample_id: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        group_value = sample.metadata.get("group_id", sample.metadata.get("user_task_id"))
        if group_value is not None:
            group_splits.setdefault(str(group_value), set()).add(sample.split)
    for sample in samples[1:]:
        if len(sample.activation) != dimension:
            raise ValueError("All activation vectors must have the same dimension")
        if (sample.model_id, sample.layer, sample.position) != (
            reference.model_id,
            reference.layer,
            reference.position,
        ):
            raise ValueError("Do not mix model, layer, or token position in one detector artifact")
    leaked_groups = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked_groups:
        raise ValueError(f"Activation groups cross train/calibration/test splits: {leaked_groups}")


def _shared_activation_metadata(samples: Sequence[ActivationSample]) -> dict[str, Any]:
    shared: dict[str, Any] = {}
    for key in _ACTIVATION_COMPATIBILITY_KEYS:
        present = [sample.metadata[key] for sample in samples if key in sample.metadata]
        if not present:
            continue
        if len(present) != len(samples) or any(value != present[0] for value in present[1:]):
            raise ValueError(f"Do not mix activation metadata {key!r} in one detector artifact")
        shared[key] = present[0]
    return shared


def _validate_artifact_compatibility(
    artifact: DetectorArtifact,
    samples: Sequence[ActivationSample],
) -> None:
    _validate_compatibility(samples)
    reference = samples[0]
    if artifact.dimension != len(reference.activation):
        raise ValueError(
            f"Artifact dimension {artifact.dimension} does not match activation dimension "
            f"{len(reference.activation)}"
        )
    expected = (reference.model_id, reference.layer, reference.position)
    actual = (artifact.model_id, artifact.layer, artifact.position)
    if actual != expected:
        raise ValueError(
            "Artifact model, layer, or token position does not match the activation dataset: "
            f"artifact={actual!r}, samples={expected!r}"
        )
    artifact_metadata = artifact.metadata.get("activation_compatibility")
    if artifact_metadata is not None:
        sample_metadata = _shared_activation_metadata(samples)
        if dict(artifact_metadata) != sample_metadata:
            raise ValueError(
                "Artifact activation metadata does not match the activation dataset: "
                f"artifact={dict(artifact_metadata)!r}, samples={sample_metadata!r}"
            )


def _matrix(samples: Iterable[ActivationSample]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    selected = list(samples)
    if not selected:
        raise ValueError("The requested dataset split is empty")
    matrix = np.asarray([sample.activation for sample in selected], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Activation samples must form a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Activation samples contain NaN or infinity")
    return matrix, np.asarray([sample.label for sample in selected], dtype=np.int64)


def fit_artifact_from_samples(
    samples: Sequence[ActivationSample],
    *,
    kind: Literal["direction", "linear_probe"],
    max_false_positive_rate: float = 0.05,
    direction_score_mode: Literal["cosine", "projection"] = "cosine",
) -> DetectorArtifact:
    _validate_compatibility(samples)
    compatibility_metadata = _shared_activation_metadata(samples)
    train = [sample for sample in samples if sample.split == "train"]
    calibration_negative = [
        sample for sample in samples if sample.split == "calibration" and sample.label == 0
    ]
    train_matrix, train_labels = _matrix(train)
    calibration_matrix, _ = _matrix(calibration_negative)
    reference = samples[0]
    if kind == "direction":
        positive = train_matrix[train_labels == 1]
        negative = train_matrix[train_labels == 0]
        if not len(positive) or not len(negative):
            raise ValueError("Direction fitting requires both positive and negative train samples")
        artifact = fit_direction_artifact(
            positive,
            negative,
            model_id=reference.model_id,
            layer=reference.layer,
            position=reference.position,
            calibration_negative=calibration_matrix,
            max_false_positive_rate=max_false_positive_rate,
            score_mode=direction_score_mode,
        )
    elif kind == "linear_probe":
        artifact = fit_linear_probe_artifact(
            train_matrix,
            train_labels,
            model_id=reference.model_id,
            layer=reference.layer,
            position=reference.position,
            calibration_negative=calibration_matrix,
            max_false_positive_rate=max_false_positive_rate,
        )
    else:
        raise ValueError(f"Unsupported detector kind: {kind}")
    return replace(
        artifact,
        metadata={
            **artifact.metadata,
            "activation_compatibility": compatibility_metadata,
        },
    )


def evaluate_artifact(artifact: DetectorArtifact, samples: Sequence[ActivationSample]) -> dict[str, Any]:
    _validate_artifact_compatibility(artifact, samples)
    test = [sample for sample in samples if sample.split == "test"]
    matrix, labels = _matrix(test)
    detector = load_detector(artifact)
    scores: list[float] = []
    predicted: list[int] = []
    for activation, sample in zip(matrix, test, strict=True):
        observation = detector.inspect(
            DetectionContext(
                candidate=CandidateToolCall("evaluation_only", {}),
                activation=activation,
                metadata={
                    "model_id": sample.model_id,
                    "layer": sample.layer,
                    "position": sample.position,
                    **sample.metadata,
                },
            )
        )
        if not observation.valid:
            raise ValueError(f"Detector failed on a test activation: {observation.details}")
        scores.append(observation.score)
        predicted.append(int(observation.triggered))
    labels_list = labels.tolist()
    true_positive = sum(p == 1 and y == 1 for p, y in zip(predicted, labels_list, strict=True))
    true_negative = sum(p == 0 and y == 0 for p, y in zip(predicted, labels_list, strict=True))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predicted, labels_list, strict=True))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predicted, labels_list, strict=True))
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    auc = roc_auc_score(labels, scores) if len(set(labels_list)) == 2 else None
    average_precision = average_precision_score(labels, scores) if len(set(labels_list)) == 2 else None
    return {
        "test_samples": len(test),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "false_positive_rate": None if negative == 0 else false_positive / negative,
        "false_negative_rate": None if positive == 0 else false_negative / positive,
        "true_positive_rate": None if positive == 0 else true_positive / positive,
        "precision": (
            None if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        ),
        "roc_auc": auc,
        "average_precision": average_precision,
        "threshold": artifact.threshold,
    }
