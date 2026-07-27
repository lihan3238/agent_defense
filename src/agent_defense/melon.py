"""Independent MELON tool-call representation and comparison primitives.

This module implements the representation and similarity portion of MELON from
the paper description.  Masked model execution remains an orchestration concern:
callers provide the original calls and the calls cached by each masking prompt.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from agent_defense.types import CandidateToolCall, DetectionContext, ProbeObservation

NO_TOOL_CALLS_TEXT = "No tool calls"
DEFAULT_CALL_SIMILARITY_THRESHOLD = 0.8
DEFAULT_ENSEMBLE_THRESHOLD = 0.5

_SECURITY_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "send_email": ("recipients",),
    "send_money": ("recipient", "amount"),
}
_FUNCTION_SEPARATOR_PATTERN = re.compile(r"[\s-]+")


class SemanticEmbedder(Protocol):
    """Backend contract for mapping text into one shared semantic vector space."""

    def embed(self, texts: Sequence[str]) -> NDArray[np.floating[Any]]:
        """Return one non-zero finite embedding row for every input text."""


@dataclass(frozen=True)
class ProjectedToolCall:
    """Auditable natural-language view after security-specific projection."""

    function: str
    argument_names: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ToolCallMatch:
    """The highest-similarity original/cache pair in one comparison."""

    original_index: int
    masked_index: int
    original_text: str
    masked_text: str
    similarity: float


@dataclass(frozen=True)
class ToolCallComparison:
    """All pairwise scores and the MELON decision for one masked-call cache."""

    original_calls: tuple[ProjectedToolCall, ...]
    masked_calls: tuple[ProjectedToolCall, ...]
    similarity_matrix: tuple[tuple[float, ...], ...]
    threshold: float
    best_match: ToolCallMatch | None

    @property
    def max_similarity(self) -> float:
        """Return the action-level similarity used by the paper's detector."""

        return 0.0 if self.best_match is None else self.best_match.similarity

    @property
    def triggered(self) -> bool:
        """Use the paper's strict ``similarity > threshold`` decision rule."""

        return self.best_match is not None and self.max_similarity > self.threshold


@dataclass(frozen=True)
class MaskedPromptRun:
    """Tool-call cache produced by one task-neutral masking prompt."""

    prompt_id: str
    masking_prompt: str
    cached_calls: tuple[CandidateToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not self.masking_prompt.strip():
            raise ValueError("masking_prompt must not be empty")
        object.__setattr__(self, "cached_calls", tuple(self.cached_calls))


@dataclass(frozen=True)
class PromptDetectionResult:
    """Single weak detector result for one masking prompt."""

    run: MaskedPromptRun
    comparison: ToolCallComparison

    @property
    def score(self) -> float:
        return self.comparison.max_similarity

    @property
    def triggered(self) -> bool:
        return self.comparison.triggered


@dataclass(frozen=True)
class EnsembleDetectionResult:
    """Mean-similarity ensemble described in MELON section 3.4."""

    members: tuple[PromptDetectionResult, ...]
    mean_similarity: float
    threshold: float
    triggered: bool

    @property
    def member_scores(self) -> tuple[float, ...]:
        return tuple(member.score for member in self.members)


class PaperMelonToolCallDetector:
    """Pre-action detector using MELON's projected semantic call comparison."""

    name = "melon_paper"

    def __init__(
        self,
        embedder: SemanticEmbedder,
        *,
        threshold: float = DEFAULT_CALL_SIMILARITY_THRESHOLD,
        projector: Callable[[CandidateToolCall], ProjectedToolCall] | None = None,
    ) -> None:
        self.embedder = embedder
        self.threshold = _validate_threshold(threshold, name="threshold")
        self.projector = projector or official_project_tool_call

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
        try:
            comparison = compare_tool_calls(
                [context.candidate],
                context.masked_candidates,
                embedder=self.embedder,
                threshold=self.threshold,
                projector=self.projector,
            )
        except Exception as error:
            return ProbeObservation(
                detector=self.name,
                score=math.nan,
                threshold=self.threshold,
                triggered=False,
                valid=False,
                latency_ms=(perf_counter() - started) * 1000,
                details={"error": f"{type(error).__name__}: {error}"},
            )
        best_match = comparison.best_match
        return ProbeObservation(
            detector=self.name,
            score=comparison.max_similarity,
            threshold=self.threshold,
            triggered=comparison.triggered,
            valid=True,
            latency_ms=(perf_counter() - started) * 1000,
            details={
                "masked_candidates": len(context.masked_candidates),
                "projected_candidate": (
                    comparison.original_calls[0].text if comparison.original_calls else ""
                ),
                "best_match": best_match.masked_text if best_match is not None else "",
            },
        )


def _normalized_function_name(function: str) -> str:
    return _FUNCTION_SEPARATOR_PATTERN.sub("_", function.strip().casefold())


def _selected_argument_names(call: CandidateToolCall) -> tuple[str, ...]:
    normalized_name = _normalized_function_name(call.function)
    if normalized_name in _SECURITY_ARGUMENTS:
        return tuple(name for name in _SECURITY_ARGUMENTS[normalized_name] if name in call.args)
    if any(not isinstance(name, str) for name in call.args):
        raise TypeError("Tool-call argument names must be strings")
    return tuple(sorted(call.args))


def _render_argument(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError("Tool-call arguments must be JSON-serializable") from error


def project_tool_call(call: CandidateToolCall) -> ProjectedToolCall:
    """Apply Appendix A.3 argument projection and render a stable description.

    ``send_email`` retains only ``recipients``; ``send_money`` retains only
    ``recipient`` and ``amount``; every other function retains all arguments.
    """

    function = call.function.strip()
    if not function:
        raise ValueError("Tool-call function must not be empty")
    argument_names = _selected_argument_names(call)
    rendered_arguments = ", ".join(f"{name} = {_render_argument(call.args[name])}" for name in argument_names)
    return ProjectedToolCall(
        function=function,
        argument_names=argument_names,
        text=f"{function}({rendered_arguments})",
    )


def describe_tool_call(call: CandidateToolCall) -> str:
    """Return the projected natural-language description embedded by MELON."""

    return project_tool_call(call).text


def official_project_tool_call(call: CandidateToolCall) -> ProjectedToolCall:
    """Match the released implementation's insertion-order ``str(value)`` rendering."""

    function = call.function.strip()
    if not function:
        raise ValueError("Tool-call function must not be empty")
    normalized_name = _normalized_function_name(function)
    if normalized_name in _SECURITY_ARGUMENTS:
        allowed = set(_SECURITY_ARGUMENTS[normalized_name])
        argument_names = tuple(name for name in call.args if name in allowed)
    else:
        if any(not isinstance(name, str) for name in call.args):
            raise TypeError("Tool-call argument names must be strings")
        argument_names = tuple(call.args)
    rendered_arguments = ", ".join(f"{name} = {call.args[name]}" for name in argument_names)
    return ProjectedToolCall(
        function=function,
        argument_names=argument_names,
        text=f"{function}({rendered_arguments})",
    )


def official_describe_tool_call(call: CandidateToolCall) -> str:
    """Return the exact call string consumed by the released MELON embedder path."""

    return official_project_tool_call(call).text


def describe_tool_calls(calls: Sequence[CandidateToolCall]) -> str:
    """Describe an action's calls, including the paper's no-call sentinel."""

    projected = tuple(describe_tool_call(call) for call in calls)
    return NO_TOOL_CALLS_TEXT if not projected else "\n".join(projected)


def _validate_threshold(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _embedding_matrix(
    embedder: SemanticEmbedder,
    texts: Sequence[str],
) -> NDArray[np.float64]:
    matrix = np.asarray(embedder.embed(texts), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Embedder output must be a two-dimensional matrix")
    if matrix.shape[0] != len(texts):
        raise ValueError("Embedder must return one row per input text")
    if matrix.shape[1] == 0:
        raise ValueError("Embedding dimension must not be zero")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embeddings must not contain NaN or infinity")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Embeddings must be non-zero")
    return matrix / norms[:, None]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute a validated cosine similarity for two embedding vectors."""

    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    if left_vector.ndim != 1 or right_vector.ndim != 1:
        raise ValueError("Cosine inputs must be one-dimensional")
    if left_vector.shape != right_vector.shape or left_vector.size == 0:
        raise ValueError("Cosine inputs must have the same non-zero dimension")
    if not np.all(np.isfinite(left_vector)) or not np.all(np.isfinite(right_vector)):
        raise ValueError("Cosine inputs must not contain NaN or infinity")
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0.0:
        raise ValueError("Cosine inputs must be non-zero")
    return float(np.clip(np.dot(left_vector, right_vector) / denominator, -1.0, 1.0))


def compare_tool_calls(
    original_calls: Sequence[CandidateToolCall],
    masked_calls: Sequence[CandidateToolCall],
    *,
    embedder: SemanticEmbedder,
    threshold: float = DEFAULT_CALL_SIMILARITY_THRESHOLD,
    projector: Callable[[CandidateToolCall], ProjectedToolCall] = project_tool_call,
) -> ToolCallComparison:
    """Compare every original call against every cached masked-run call."""

    threshold = _validate_threshold(threshold, name="threshold")
    original = tuple(projector(call) for call in original_calls)
    masked = tuple(projector(call) for call in masked_calls)
    if not original or not masked:
        return ToolCallComparison(
            original_calls=original,
            masked_calls=masked,
            similarity_matrix=tuple(() for _ in original),
            threshold=threshold,
            best_match=None,
        )

    texts = tuple(call.text for call in original + masked)
    embeddings = _embedding_matrix(embedder, texts)
    original_embeddings = embeddings[: len(original)]
    masked_embeddings = embeddings[len(original) :]
    scores = np.clip(original_embeddings @ masked_embeddings.T, -1.0, 1.0)
    flat_best = int(np.argmax(scores))
    original_index, masked_index = np.unravel_index(flat_best, scores.shape)
    best_score = float(scores[original_index, masked_index])
    best_match = ToolCallMatch(
        original_index=int(original_index),
        masked_index=int(masked_index),
        original_text=original[original_index].text,
        masked_text=masked[masked_index].text,
        similarity=best_score,
    )
    return ToolCallComparison(
        original_calls=original,
        masked_calls=masked,
        similarity_matrix=tuple(tuple(float(value) for value in row) for row in scores),
        threshold=threshold,
        best_match=best_match,
    )


def compare_masked_prompt(
    original_calls: Sequence[CandidateToolCall],
    run: MaskedPromptRun,
    *,
    embedder: SemanticEmbedder,
    threshold: float = DEFAULT_CALL_SIMILARITY_THRESHOLD,
    projector: Callable[[CandidateToolCall], ProjectedToolCall] = project_tool_call,
) -> PromptDetectionResult:
    """Evaluate one task-neutral prompt as a MELON weak detector."""

    return PromptDetectionResult(
        run=run,
        comparison=compare_tool_calls(
            original_calls,
            run.cached_calls,
            embedder=embedder,
            threshold=threshold,
            projector=projector,
        ),
    )


def aggregate_prompt_results(
    members: Sequence[PromptDetectionResult],
    *,
    threshold: float = DEFAULT_ENSEMBLE_THRESHOLD,
) -> EnsembleDetectionResult:
    """Average weak-detector similarity scores and apply the ensemble threshold."""

    threshold = _validate_threshold(threshold, name="ensemble threshold")
    frozen_members = tuple(members)
    if not frozen_members:
        raise ValueError("MELON ensemble requires at least one masking prompt")
    prompt_ids = tuple(member.run.prompt_id for member in frozen_members)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("MELON ensemble prompt_id values must be unique")
    scores = np.asarray([member.score for member in frozen_members], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("MELON ensemble scores must be finite")
    mean_similarity = float(np.mean(scores))
    return EnsembleDetectionResult(
        members=frozen_members,
        mean_similarity=mean_similarity,
        threshold=threshold,
        triggered=mean_similarity > threshold,
    )


def evaluate_ensemble(
    original_calls: Sequence[CandidateToolCall],
    runs: Sequence[MaskedPromptRun],
    *,
    embedder: SemanticEmbedder,
    call_threshold: float = DEFAULT_CALL_SIMILARITY_THRESHOLD,
    ensemble_threshold: float = DEFAULT_ENSEMBLE_THRESHOLD,
    projector: Callable[[CandidateToolCall], ProjectedToolCall] = project_tool_call,
) -> EnsembleDetectionResult:
    """Evaluate and aggregate several task-neutral masking prompts."""

    members = tuple(
        compare_masked_prompt(
            original_calls,
            run,
            embedder=embedder,
            threshold=call_threshold,
            projector=projector,
        )
        for run in runs
    )
    return aggregate_prompt_results(members, threshold=ensemble_threshold)
