from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from agent_defense.artifacts import DetectorArtifact
from agent_defense.types import DetectionContext, ProbeObservation


class Detector(Protocol):
    name: str

    def inspect(self, context: DetectionContext) -> ProbeObservation: ...


def _vector(value: ArrayLike, *, dimension: int | None = None) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1-D activation, got shape {vector.shape}")
    if dimension is not None and vector.shape[0] != dimension:
        raise ValueError(
            f"Activation dimension {vector.shape[0]} does not match artifact dimension {dimension}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Activation contains NaN or infinity")
    return vector


def calibrate_upper_threshold(negative_scores: ArrayLike, max_false_positive_rate: float = 0.05) -> float:
    """Choose a conservative threshold using only benign calibration scores."""

    if not 0 <= max_false_positive_rate < 1:
        raise ValueError("max_false_positive_rate must be in [0, 1)")
    scores = _vector(negative_scores)
    if scores.size == 0:
        raise ValueError("At least one benign calibration score is required")
    quantile = float(np.quantile(scores, 1 - max_false_positive_rate, method="higher"))
    # A single nextafter step is insufficient when calibration and online scoring
    # use different, but equivalent, NumPy/math implementations (for example,
    # two sigmoid implementations can differ by several ULPs). Keep the margin
    # negligible at detector-score scale while making the allow/block boundary
    # stable across those implementations.
    ulp_margin = 8.0 * abs(float(np.spacing(quantile)))
    relative_margin = 1e-12 * max(1.0, abs(quantile))
    threshold = quantile + max(ulp_margin, relative_margin)
    if not math.isfinite(threshold):
        raise ValueError("Could not construct a finite calibration threshold")
    return threshold


class NoDefenseDetector:
    name = "none"

    def inspect(self, context: DetectionContext) -> ProbeObservation:
        del context
        return ProbeObservation(
            detector=self.name,
            score=0.0,
            threshold=math.inf,
            triggered=False,
            valid=True,
            latency_ms=0.0,
        )


class _ArtifactDetector:
    expected_kind: str

    def __init__(self, artifact: DetectorArtifact) -> None:
        if artifact.kind != self.expected_kind:
            raise ValueError(f"Expected a {self.expected_kind} artifact, got {artifact.kind}")
        self.artifact = artifact
        self.weights = _vector(artifact.weights)
        if not math.isfinite(artifact.bias):
            raise ValueError("Artifact bias must be finite")
        if not math.isfinite(artifact.threshold):
            raise ValueError("Artifact threshold must be finite")
        compatibility = artifact.metadata.get("activation_compatibility", {})
        if not isinstance(compatibility, Mapping):
            raise ValueError("Artifact activation_compatibility must be a mapping")
        self.activation_compatibility = dict(compatibility)
        self.name = artifact.kind

    def _validate_context_metadata(self, context: DetectionContext) -> None:
        if (
            context.metadata.get("checkpoint_content_id") is not None
            and "checkpoint_content_id" not in self.activation_compatibility
        ):
            raise ValueError("Artifact is missing checkpoint_content_id required by the local runtime")
        expected = {
            "model_id": self.artifact.model_id,
            "layer": self.artifact.layer,
            "position": self.artifact.position,
        }
        for key, expected_value in expected.items():
            if key in context.metadata and context.metadata[key] != expected_value:
                raise ValueError(
                    f"Activation metadata {key}={context.metadata[key]!r} does not match "
                    f"artifact {key}={expected_value!r}"
                )
        for key, expected_value in self.activation_compatibility.items():
            if key not in context.metadata:
                raise ValueError(f"Activation metadata is missing required compatibility key {key!r}")
            if context.metadata[key] != expected_value:
                raise ValueError(
                    f"Activation metadata {key}={context.metadata[key]!r} does not match "
                    f"artifact {key}={expected_value!r}"
                )

    def _raw_score(self, activation: NDArray[np.float64]) -> float:
        return float(np.dot(self.weights, activation) + self.artifact.bias)

    def _final_score(self, raw_score: float) -> float:
        return raw_score

    def _observation_details(self) -> dict[str, object]:
        return {
            "model_id": self.artifact.model_id,
            "layer": self.artifact.layer,
            "position": self.artifact.position,
            "activation_compatibility": self.activation_compatibility,
        }

    def inspect(self, context: DetectionContext) -> ProbeObservation:
        started = perf_counter()
        try:
            if context.activation is None:
                capture_error = context.metadata.get("activation_error")
                suffix = f"; capture_error={capture_error}" if capture_error else ""
                raise ValueError(f"No activation was provided at the tool-call boundary{suffix}")
            self._validate_context_metadata(context)
            activation = _vector(context.activation, dimension=self.artifact.dimension)
            raw_score = self._raw_score(activation)
            score = self._final_score(raw_score)
            valid = math.isfinite(score)
            if not valid:
                raise ValueError("Detector produced a non-finite score")
            return ProbeObservation(
                detector=self.name,
                score=score,
                threshold=self.artifact.threshold,
                triggered=score > self.artifact.threshold,
                valid=True,
                latency_ms=(perf_counter() - started) * 1000,
                details=self._observation_details(),
            )
        except ValueError as error:
            return ProbeObservation(
                detector=self.name,
                score=math.nan,
                threshold=self.artifact.threshold,
                triggered=False,
                valid=False,
                latency_ms=(perf_counter() - started) * 1000,
                details={"error": str(error)},
            )


class DirectionDetector(_ArtifactDetector):
    """Difference-in-means detector with an explicit cosine or projection score."""

    expected_kind = "direction"

    def __init__(self, artifact: DetectorArtifact) -> None:
        super().__init__(artifact)
        score_mode = artifact.metadata.get("score_mode", "projection")
        if score_mode not in {"cosine", "projection"}:
            raise ValueError(f"Unsupported direction score mode: {score_mode!r}")
        self.score_mode: Literal["cosine", "projection"] = score_mode

    def _raw_score(self, activation: NDArray[np.float64]) -> float:
        projection = float(np.dot(self.weights, activation))
        if self.score_mode == "projection":
            return projection + self.artifact.bias
        activation_norm = float(np.linalg.norm(activation))
        if not math.isfinite(activation_norm) or activation_norm <= 1e-12:
            raise ValueError("Cosine direction scoring requires a non-zero activation")
        return projection / activation_norm + self.artifact.bias

    def _observation_details(self) -> dict[str, object]:
        return {**super()._observation_details(), "score_mode": self.score_mode}


class LinearProbeDetector(_ArtifactDetector):
    """Single-layer logistic probe evaluated without loading a pickle."""

    expected_kind = "linear_probe"

    def __init__(self, artifact: DetectorArtifact) -> None:
        super().__init__(artifact)
        mean_value = artifact.metadata.get("scaler_mean")
        scale_value = artifact.metadata.get("scaler_scale")
        if (mean_value is None) != (scale_value is None):
            raise ValueError("Linear-probe artifact must contain both scaler_mean and scaler_scale")
        if mean_value is None:
            # Backward compatibility for schema-v1 artifacts created before feature scaling.
            self.scaler_mean = np.zeros(artifact.dimension, dtype=np.float64)
            self.scaler_scale = np.ones(artifact.dimension, dtype=np.float64)
            self.standardized = False
            return
        self.scaler_mean = _vector(mean_value, dimension=artifact.dimension)
        self.scaler_scale = _vector(scale_value, dimension=artifact.dimension)
        if np.any(self.scaler_scale <= 0):
            raise ValueError("Linear-probe scaler_scale values must be positive")
        self.standardized = True

    def _raw_score(self, activation: NDArray[np.float64]) -> float:
        standardized = (activation - self.scaler_mean) / self.scaler_scale
        return float(np.dot(self.weights, standardized) + self.artifact.bias)

    def _final_score(self, raw_score: float) -> float:
        if raw_score >= 0:
            return 1.0 / (1.0 + math.exp(-raw_score))
        exp_score = math.exp(raw_score)
        return exp_score / (1.0 + exp_score)

    def _observation_details(self) -> dict[str, object]:
        return {**super()._observation_details(), "standardized": self.standardized}


def _direction_scores(
    activations: ArrayLike,
    direction: NDArray[np.float64],
    score_mode: Literal["cosine", "projection"],
) -> NDArray[np.float64]:
    matrix = np.asarray(activations, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != direction.shape[0]:
        raise ValueError("Activations have an incompatible shape for direction scoring")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Activations contain NaN or infinity")
    projections = matrix @ direction
    if score_mode == "projection":
        return projections
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Cosine direction scoring requires non-zero activations")
    return projections / norms


def fit_direction_artifact(
    positive: ArrayLike,
    negative: ArrayLike,
    *,
    model_id: str,
    layer: int,
    position: str,
    calibration_negative: ArrayLike | None = None,
    max_false_positive_rate: float = 0.05,
    score_mode: Literal["cosine", "projection"] = "cosine",
) -> DetectorArtifact:
    if score_mode not in {"cosine", "projection"}:
        raise ValueError(f"Unsupported direction score mode: {score_mode!r}")
    positive_array = np.asarray(positive, dtype=np.float64)
    negative_array = np.asarray(negative, dtype=np.float64)
    if positive_array.ndim != 2 or negative_array.ndim != 2:
        raise ValueError("Positive and negative activations must be 2-D matrices")
    if positive_array.shape[1] != negative_array.shape[1]:
        raise ValueError("Positive and negative activations must have the same dimension")
    if positive_array.shape[0] == 0 or negative_array.shape[0] == 0:
        raise ValueError("Positive and negative activations must not be empty")
    if not np.all(np.isfinite(positive_array)) or not np.all(np.isfinite(negative_array)):
        raise ValueError("Training activations contain NaN or infinity")
    direction = positive_array.mean(axis=0) - negative_array.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot extract a direction from identical class means")
    direction /= norm
    benign = (
        negative_array if calibration_negative is None else np.asarray(calibration_negative, dtype=np.float64)
    )
    if benign.ndim != 2 or benign.shape[1] != direction.shape[0]:
        raise ValueError("Calibration activations have an incompatible shape")
    threshold = calibrate_upper_threshold(
        _direction_scores(benign, direction, score_mode),
        max_false_positive_rate,
    )
    return DetectorArtifact(
        kind="direction",
        weights=tuple(float(value) for value in direction),
        bias=0.0,
        threshold=threshold,
        model_id=model_id,
        layer=layer,
        position=position,
        metadata={
            "method": "difference_in_means",
            "score_mode": score_mode,
            "positive_samples": int(positive_array.shape[0]),
            "negative_samples": int(negative_array.shape[0]),
            "max_false_positive_rate": max_false_positive_rate,
        },
    )


def fit_linear_probe_artifact(
    activations: ArrayLike,
    labels: ArrayLike,
    *,
    model_id: str,
    layer: int,
    position: str,
    calibration_negative: ArrayLike | None = None,
    max_false_positive_rate: float = 0.05,
    random_state: int = 0,
) -> DetectorArtifact:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(activations, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != target.shape[0]:
        raise ValueError("Expected activations [samples, hidden] and one label per sample")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Training activations must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Training activations contain NaN or infinity")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("The training labels must contain both 0 and 1")
    scaler = StandardScaler()
    standardized_matrix = scaler.fit_transform(matrix)
    estimator = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)
    estimator.fit(standardized_matrix, target)
    weights = estimator.coef_[0].astype(np.float64)
    bias = float(estimator.intercept_[0])
    benign = (
        matrix[target == 0]
        if calibration_negative is None
        else np.asarray(calibration_negative, dtype=np.float64)
    )
    if benign.ndim != 2 or benign.shape[1] != weights.shape[0]:
        raise ValueError("Calibration activations have an incompatible shape")
    if benign.shape[0] == 0:
        raise ValueError("Calibration activations must not be empty")
    if not np.all(np.isfinite(benign)):
        raise ValueError("Calibration activations contain NaN or infinity")
    standardized_benign = scaler.transform(benign)
    logits = standardized_benign @ weights + bias
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
    threshold = calibrate_upper_threshold(probabilities, max_false_positive_rate)
    return DetectorArtifact(
        kind="linear_probe",
        weights=tuple(float(value) for value in weights),
        bias=bias,
        threshold=threshold,
        model_id=model_id,
        layer=layer,
        position=position,
        metadata={
            "method": "logistic_regression",
            "feature_scaler": "standard",
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "training_samples": int(matrix.shape[0]),
            "max_false_positive_rate": max_false_positive_rate,
            "random_state": random_state,
        },
    )


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_@.:-]+")


class HashingToolCallEmbedder:
    """Small deterministic local embedder for an auditable MELON smoke path."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 16:
            raise ValueError("Embedding dimension must be at least 16")
        self.dimension = dimension

    def embed(self, text: str) -> NDArray[np.float64]:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0 else vector / norm


class MelonToolCallDetector:
    """MELON's tool-call comparison, with masked re-execution supplied separately."""

    name = "melon"

    def __init__(self, threshold: float = 0.8, embedder: HashingToolCallEmbedder | None = None) -> None:
        self.threshold = threshold
        self.embedder = embedder or HashingToolCallEmbedder()

    def inspect(self, context: DetectionContext) -> ProbeObservation:
        started = perf_counter()
        if melon_error := context.metadata.get("melon_error"):
            return ProbeObservation(
                detector=self.name,
                score=math.nan,
                threshold=self.threshold,
                triggered=False,
                valid=False,
                latency_ms=(perf_counter() - started) * 1000,
                details={"error": str(melon_error)},
            )
        if not context.masked_candidates:
            return ProbeObservation(
                detector=self.name,
                score=0.0,
                threshold=self.threshold,
                triggered=False,
                valid=True,
                latency_ms=(perf_counter() - started) * 1000,
                details={"masked_candidates": 0},
            )
        current = self.embedder.embed(context.candidate.canonical_text())
        best_score = -1.0
        best_match = ""
        for candidate in context.masked_candidates:
            score = float(np.dot(current, self.embedder.embed(candidate.canonical_text())))
            if score > best_score:
                best_score = score
                best_match = candidate.canonical_text()
        return ProbeObservation(
            detector=self.name,
            score=best_score,
            threshold=self.threshold,
            triggered=best_score > self.threshold,
            valid=True,
            latency_ms=(perf_counter() - started) * 1000,
            details={"masked_candidates": len(context.masked_candidates), "best_match": best_match},
        )


def load_detector(artifact: DetectorArtifact) -> DirectionDetector | LinearProbeDetector:
    if artifact.kind == "direction":
        return DirectionDetector(artifact)
    return LinearProbeDetector(artifact)


def with_threshold(artifact: DetectorArtifact, threshold: float) -> DetectorArtifact:
    return replace(artifact, threshold=threshold)
