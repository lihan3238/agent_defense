from __future__ import annotations

import math

import numpy as np
import pytest

from agent_defense.artifacts import DetectorArtifact
from agent_defense.detectors import (
    DirectionDetector,
    LinearProbeDetector,
    calibrate_upper_threshold,
    fit_direction_artifact,
    fit_linear_probe_artifact,
)
from agent_defense.types import CandidateToolCall, DetectionContext


def _context(activation: list[float], **metadata: object) -> DetectionContext:
    return DetectionContext(
        candidate=CandidateToolCall("evaluation_only", {}),
        activation=np.asarray(activation, dtype=np.float64),
        metadata=metadata,
    )


def _linear_artifact(**overrides: object) -> DetectorArtifact:
    values: dict[str, object] = {
        "kind": "linear_probe",
        "weights": (1.0, 0.0),
        "bias": 0.0,
        "threshold": 0.5,
        "model_id": "model-a",
        "layer": 3,
        "position": "generation_prefill_last_nonpad",
        "metadata": {
            "scaler_mean": [10.0, 20.0],
            "scaler_scale": [2.0, 4.0],
        },
    }
    values.update(overrides)
    return DetectorArtifact(**values)  # type: ignore[arg-type]


def test_threshold_calibration_is_conservative_and_has_cross_implementation_margin() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    threshold = calibrate_upper_threshold(scores, max_false_positive_rate=0.2)

    assert np.mean(scores > threshold) <= 0.2
    assert threshold - 0.5 >= 0.9e-12
    # Regression: a benign score differing by a few floating-point ULPs must
    # remain below a threshold calibrated by another sigmoid implementation.
    assert math.nextafter(math.nextafter(0.5, math.inf), math.inf) < threshold


@pytest.mark.parametrize("rate", [-0.01, 1.0, math.nan])
def test_threshold_calibration_rejects_invalid_false_positive_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="max_false_positive_rate"):
        calibrate_upper_threshold([0.1, 0.2], rate)


def test_threshold_calibration_rejects_nan_scores() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        calibrate_upper_threshold([0.1, math.nan])


def test_direction_detector_distinguishes_cosine_and_projection_scores() -> None:
    common = {
        "kind": "direction",
        "weights": (1.0, 0.0),
        "bias": 0.0,
        "threshold": 0.75,
        "model_id": "model-a",
        "layer": 3,
        "position": "generation_prefill_last_nonpad",
    }
    projection = DirectionDetector(
        DetectorArtifact(**common, metadata={"score_mode": "projection"})  # type: ignore[arg-type]
    )
    cosine = DirectionDetector(
        DetectorArtifact(**common, metadata={"score_mode": "cosine"})  # type: ignore[arg-type]
    )
    context = _context(
        [3.0, 4.0],
        model_id="model-a",
        layer=3,
        position="generation_prefill_last_nonpad",
    )

    projection_result = projection.inspect(context)
    cosine_result = cosine.inspect(context)

    assert projection_result.score == pytest.approx(3.0)
    assert projection_result.triggered
    assert projection_result.details["score_mode"] == "projection"
    assert cosine_result.score == pytest.approx(0.6)
    assert not cosine_result.triggered
    assert cosine_result.details["score_mode"] == "cosine"


@pytest.mark.parametrize("score_mode", ["cosine", "projection"])
def test_direction_fitting_records_and_calibrates_the_selected_score_mode(score_mode: str) -> None:
    positive = np.asarray([[3.0, 1.0], [4.0, 1.0]])
    negative = np.asarray([[1.0, 1.0], [1.0, 2.0]])
    calibration = np.asarray([[0.8, 1.0], [1.0, 1.5]])

    artifact = fit_direction_artifact(
        positive,
        negative,
        model_id="model-a",
        layer=3,
        position="generation_prefill_last_nonpad",
        calibration_negative=calibration,
        score_mode=score_mode,  # type: ignore[arg-type]
    )

    assert artifact.metadata["score_mode"] == score_mode
    detector = DirectionDetector(artifact)
    for activation in calibration:
        assert not detector.inspect(_context(activation.tolist())).triggered


def test_cosine_direction_rejects_zero_activation_at_runtime() -> None:
    artifact = DetectorArtifact(
        kind="direction",
        weights=(1.0, 0.0),
        bias=0.0,
        threshold=0.5,
        model_id="model-a",
        layer=3,
        position="generation_prefill_last_nonpad",
        metadata={"score_mode": "cosine"},
    )

    result = DirectionDetector(artifact).inspect(_context([0.0, 0.0]))

    assert not result.valid
    assert not result.triggered
    assert "non-zero activation" in result.details["error"]


def test_linear_probe_applies_scaler_stored_in_artifact() -> None:
    detector = LinearProbeDetector(_linear_artifact())

    centered = detector.inspect(_context([10.0, 20.0]))
    shifted = detector.inspect(_context([12.0, 20.0]))

    assert centered.score == pytest.approx(0.5)
    assert not centered.triggered
    assert shifted.score == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert shifted.triggered
    assert shifted.details["standardized"] is True


def test_linear_probe_calibration_sample_is_not_blocked_after_online_rescoring() -> None:
    matrix = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    calibration = np.asarray([[-0.75]], dtype=np.float64)
    artifact = fit_linear_probe_artifact(
        matrix,
        labels,
        model_id="model-a",
        layer=3,
        position="generation_prefill_last_nonpad",
        calibration_negative=calibration,
        max_false_positive_rate=0.0,
    )

    result = LinearProbeDetector(artifact).inspect(_context(calibration[0].tolist()))

    assert result.valid
    assert result.score < artifact.threshold
    assert not result.triggered


@pytest.mark.parametrize(
    ("activation", "metadata", "error"),
    [
        ([math.nan, 0.0], {}, "NaN or infinity"),
        ([1.0], {}, "does not match artifact dimension"),
        ([1.0, 0.0], {"model_id": "model-b"}, "does not match artifact"),
        ([1.0, 0.0], {"layer": 4}, "does not match artifact"),
        ([1.0, 0.0], {"position": "function_call"}, "does not match artifact"),
    ],
)
def test_artifact_detector_fails_closed_on_activation_or_metadata_mismatch(
    activation: list[float],
    metadata: dict[str, object],
    error: str,
) -> None:
    result = LinearProbeDetector(_linear_artifact()).inspect(_context(activation, **metadata))

    assert not result.valid
    assert not result.triggered
    assert error in result.details["error"]


def test_missing_activation_preserves_capture_failure_reason() -> None:
    result = LinearProbeDetector(_linear_artifact()).inspect(
        DetectionContext(
            candidate=CandidateToolCall("send_money", {}),
            activation=None,
            metadata={"activation_error": "multiple_tool_calls_require_per_call_activations"},
        )
    )

    assert not result.valid
    assert "multiple_tool_calls_require_per_call_activations" in result.details["error"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"scaler_mean": [0.0, 0.0]},
        {"scaler_mean": [0.0], "scaler_scale": [1.0, 1.0]},
        {"scaler_mean": [0.0, 0.0], "scaler_scale": [1.0, 0.0]},
        {"scaler_mean": [0.0, math.nan], "scaler_scale": [1.0, 1.0]},
    ],
)
def test_linear_probe_rejects_invalid_scaler_metadata(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LinearProbeDetector(_linear_artifact(metadata=metadata))


def test_direction_detector_rejects_unknown_score_mode() -> None:
    artifact = DetectorArtifact(
        kind="direction",
        weights=(1.0, 0.0),
        bias=0.0,
        threshold=0.5,
        model_id="model-a",
        layer=3,
        position="generation_prefill_last_nonpad",
        metadata={"score_mode": "euclidean"},
    )

    with pytest.raises(ValueError, match="Unsupported direction score mode"):
        DirectionDetector(artifact)


def test_detector_requires_frozen_revision_and_template_metadata() -> None:
    artifact = _linear_artifact(
        metadata={
            "scaler_mean": [0.0, 0.0],
            "scaler_scale": [1.0, 1.0],
            "activation_compatibility": {
                "revision": "revision-a",
                "chat_template_hash": "template-a",
            },
        }
    )
    detector = LinearProbeDetector(artifact)

    missing = detector.inspect(_context([1.0, 0.0]))
    mismatched = detector.inspect(
        _context(
            [1.0, 0.0],
            revision="revision-b",
            chat_template_hash="template-a",
        )
    )
    matching = detector.inspect(
        _context(
            [1.0, 0.0],
            revision="revision-a",
            chat_template_hash="template-a",
        )
    )

    assert not missing.valid
    assert "missing required" in missing.details["error"]
    assert not mismatched.valid
    assert "does not match" in mismatched.details["error"]
    assert matching.valid


def test_detector_requires_symmetric_local_checkpoint_identity() -> None:
    local_context = _context(
        [1.0, 0.0],
        checkpoint_content_id="local-checkpoint-v1:checkpoint-a",
    )
    legacy = LinearProbeDetector(_linear_artifact()).inspect(local_context)
    matching = LinearProbeDetector(
        _linear_artifact(
            metadata={
                "scaler_mean": [0.0, 0.0],
                "scaler_scale": [1.0, 1.0],
                "activation_compatibility": {"checkpoint_content_id": "local-checkpoint-v1:checkpoint-a"},
            }
        )
    ).inspect(local_context)

    assert not legacy.valid
    assert "missing checkpoint_content_id" in legacy.details["error"]
    assert matching.valid
