from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from agent_defense.artifacts import DetectorArtifact
from agent_defense.training import (
    ActivationSample,
    apply_label_manifest,
    evaluate_artifact,
    fit_artifact_from_samples,
    load_activation_samples,
)

MODEL_ID = "model-a"
LAYER = 3
POSITION = "generation_prefill_last_nonpad"


def _sample(
    sample_id: str,
    label: int,
    split: str,
    activation: tuple[float, ...],
) -> ActivationSample:
    return ActivationSample(
        sample_id=sample_id,
        label=label,
        split=split,  # type: ignore[arg-type]
        activation=activation,
        model_id=MODEL_ID,
        layer=LAYER,
        position=POSITION,
        metadata={},
    )


def _separable_samples() -> list[ActivationSample]:
    return [
        _sample("train-negative-1", 0, "train", (-2.0, -1.0)),
        _sample("train-negative-2", 0, "train", (-1.0, -1.0)),
        _sample("train-positive-1", 1, "train", (1.0, 1.0)),
        _sample("train-positive-2", 1, "train", (2.0, 1.0)),
        _sample("calibration-negative-1", 0, "calibration", (-1.5, -1.0)),
        _sample("calibration-negative-2", 0, "calibration", (-0.5, -1.0)),
        _sample("test-negative", 0, "test", (-1.0, -1.0)),
        _sample("test-positive", 1, "test", (2.0, 1.0)),
    ]


def test_linear_probe_scaler_is_fit_on_train_only_and_round_trips_in_json(tmp_path) -> None:
    samples = [
        _sample("train-negative-1", 0, "train", (0.0, 0.0)),
        _sample("train-negative-2", 0, "train", (2.0, 0.0)),
        _sample("train-positive-1", 1, "train", (4.0, 2.0)),
        _sample("train-positive-2", 1, "train", (6.0, 2.0)),
        # This calibration outlier must not influence scaler_mean/scaler_scale.
        _sample("calibration-negative", 0, "calibration", (100.0, 100.0)),
        _sample("test-negative", 0, "test", (1.0, 0.0)),
        _sample("test-positive", 1, "test", (5.0, 2.0)),
    ]

    artifact = fit_artifact_from_samples(samples, kind="linear_probe")

    assert artifact.metadata["feature_scaler"] == "standard"
    assert artifact.metadata["scaler_mean"] == pytest.approx([3.0, 1.0])
    assert artifact.metadata["scaler_scale"] == pytest.approx([math.sqrt(5.0), 1.0])

    path = tmp_path / "linear-probe.json"
    artifact.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    restored = DetectorArtifact.load(path)

    assert raw["metadata"]["scaler_mean"] == pytest.approx([3.0, 1.0])
    assert raw["metadata"]["scaler_scale"] == pytest.approx([math.sqrt(5.0), 1.0])
    assert restored.metadata["scaler_mean"] == pytest.approx([3.0, 1.0])
    assert restored.metadata["scaler_scale"] == pytest.approx([math.sqrt(5.0), 1.0])


@pytest.mark.parametrize("score_mode", ["cosine", "projection"])
def test_direction_training_threads_score_mode_into_artifact(score_mode: str) -> None:
    artifact = fit_artifact_from_samples(
        _separable_samples(),
        kind="direction",
        direction_score_mode=score_mode,  # type: ignore[arg-type]
    )

    assert artifact.metadata["score_mode"] == score_mode


def test_training_freezes_shared_model_revision_and_chat_template_metadata() -> None:
    samples = [
        replace(
            sample,
            metadata={
                "checkpoint_content_id": "local-checkpoint-v1:abc123",
                "revision": "revision-a",
                "state_kind": "resid_pre",
                "module_path": "model.layers.3",
                "chat_template_hash": "template-a",
                "render_mode": "native_tools",
            },
        )
        for sample in _separable_samples()
    ]

    artifact = fit_artifact_from_samples(samples, kind="linear_probe")

    assert artifact.metadata["activation_compatibility"] == {
        "checkpoint_content_id": "local-checkpoint-v1:abc123",
        "revision": "revision-a",
        "state_kind": "resid_pre",
        "module_path": "model.layers.3",
        "chat_template_hash": "template-a",
        "render_mode": "native_tools",
    }


def test_training_rejects_mixed_model_revision_metadata() -> None:
    samples = _separable_samples()
    samples = [replace(sample, metadata={"revision": "revision-a"}) for sample in samples]
    samples[-1] = replace(samples[-1], metadata={"revision": "revision-b"})

    with pytest.raises(ValueError, match="revision"):
        fit_artifact_from_samples(samples, kind="linear_probe")


def test_training_rejects_group_leakage_across_splits() -> None:
    samples = _separable_samples()
    samples[0] = replace(samples[0], metadata={"group_id": "same-task"})
    samples[-1] = replace(samples[-1], metadata={"group_id": "same-task"})

    with pytest.raises(ValueError, match="cross train/calibration/test"):
        fit_artifact_from_samples(samples, kind="linear_probe")


def test_training_rejects_duplicate_sample_ids() -> None:
    samples = _separable_samples()
    samples[-1] = replace(samples[-1], sample_id=samples[0].sample_id)

    with pytest.raises(ValueError, match="Duplicate activation sample_id"):
        fit_artifact_from_samples(samples, kind="linear_probe")


def test_evaluate_artifact_reports_separable_probe_results() -> None:
    samples = _separable_samples()
    artifact = fit_artifact_from_samples(samples, kind="linear_probe")

    result = evaluate_artifact(artifact, samples)

    assert result["test_samples"] == 2
    assert result["true_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["true_positive_rate"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_sample",
    [
        replace(_separable_samples()[-1], activation=(2.0, 1.0, 0.0)),
        replace(_separable_samples()[-1], model_id="model-b"),
        replace(_separable_samples()[-1], layer=4),
        replace(_separable_samples()[-1], position="function_call_last_token"),
    ],
)
def test_training_rejects_dimension_or_metadata_mismatch(bad_sample: ActivationSample) -> None:
    samples = _separable_samples()
    samples[-1] = bad_sample

    with pytest.raises(ValueError):
        fit_artifact_from_samples(samples, kind="linear_probe")


@pytest.mark.parametrize(
    "artifact",
    [
        DetectorArtifact(
            kind="linear_probe",
            weights=(1.0,),
            bias=0.0,
            threshold=0.5,
            model_id=MODEL_ID,
            layer=LAYER,
            position=POSITION,
        ),
        DetectorArtifact(
            kind="linear_probe",
            weights=(1.0, 1.0),
            bias=0.0,
            threshold=0.5,
            model_id="model-b",
            layer=LAYER,
            position=POSITION,
        ),
        DetectorArtifact(
            kind="linear_probe",
            weights=(1.0, 1.0),
            bias=0.0,
            threshold=0.5,
            model_id=MODEL_ID,
            layer=4,
            position=POSITION,
        ),
    ],
)
def test_evaluation_rejects_artifact_dimension_or_metadata_mismatch(
    artifact: DetectorArtifact,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        evaluate_artifact(artifact, _separable_samples())


def test_load_activation_samples_rejects_nan(tmp_path) -> None:
    payload = {
        "sample_id": "nan-sample",
        "label": 0,
        "split": "train",
        "activation": [0.0, math.nan],
        "model_id": MODEL_ID,
        "layer": LAYER,
        "position": POSITION,
    }
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(payload, allow_nan=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite values"):
        load_activation_samples(path)


def test_fit_rejects_nan_in_manually_constructed_sample() -> None:
    samples = _separable_samples()
    samples[0] = replace(samples[0], activation=(math.nan, -1.0))

    with pytest.raises(ValueError, match="NaN or infinity"):
        fit_artifact_from_samples(samples, kind="linear_probe")


def test_apply_label_manifest_completes_pending_call_level_review(tmp_path) -> None:
    dataset = tmp_path / "pending.jsonl"
    manifest = tmp_path / "labels.json"
    output = tmp_path / "labeled.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "sample_id": "run:0",
                "label": None,
                "split": "train",
                "activation": [1.0, 2.0],
                "model_id": MODEL_ID,
                "layer": LAYER,
                "position": POSITION,
                "metadata": {"label_requires_review": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"run:0": 1}), encoding="utf-8")

    result = apply_label_manifest(dataset, manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result == {"samples": 1, "labeled": 1, "unlabeled": 0}
    assert payload["label"] == 1
    assert payload["activation"] == [1.0, 2.0]
    assert payload["metadata"]["label_requires_review"] is False
    assert payload["metadata"]["label_source"] == "reviewed_manifest"
    assert load_activation_samples(output)[0].label == 1


def test_apply_label_manifest_refuses_in_place_or_incomplete_review(tmp_path) -> None:
    dataset = tmp_path / "pending.jsonl"
    dataset.write_text(
        json.dumps({"sample_id": "run:0", "label": None}) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"another-id": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="in-place"):
        apply_label_manifest(dataset, manifest, dataset)
    with pytest.raises(ValueError, match="No reviewed label"):
        apply_label_manifest(dataset, manifest, tmp_path / "output.jsonl")
