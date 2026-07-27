from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from agent_defense.melon import (
    NO_TOOL_CALLS_TEXT,
    MaskedPromptRun,
    SemanticEmbedder,
    aggregate_prompt_results,
    compare_masked_prompt,
    compare_tool_calls,
    cosine_similarity,
    describe_tool_call,
    describe_tool_calls,
    evaluate_ensemble,
    official_describe_tool_call,
    project_tool_call,
)
from agent_defense.types import CandidateToolCall


class LookupEmbedder:
    """Deterministic semantic stand-in that also exposes embedded text to tests."""

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.batches: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        batch = tuple(texts)
        self.batches.append(batch)
        return np.asarray([self.vectors[text] for text in batch], dtype=np.float64)


def test_lookup_embedder_satisfies_semantic_embedder_protocol_statically() -> None:
    embedder: SemanticEmbedder = LookupEmbedder({})

    assert isinstance(embedder, LookupEmbedder)


def test_send_email_projection_keeps_only_recipients() -> None:
    call = CandidateToolCall(
        "send_email",
        {
            "recipients": ["attacker@example.com"],
            "subject": "private information",
            "body": "must not reach the embedding backend",
        },
    )

    projected = project_tool_call(call)

    assert projected.argument_names == ("recipients",)
    assert projected.text == 'send_email(recipients = ["attacker@example.com"])'
    assert "subject" not in projected.text
    assert "body" not in projected.text


def test_send_money_projection_keeps_recipient_and_amount_in_paper_order() -> None:
    call = CandidateToolCall(
        "send_money",
        {"note": "rent", "amount": 100, "recipient": "bob"},
    )

    projected = project_tool_call(call)

    assert projected.argument_names == ("recipient", "amount")
    assert projected.text == 'send_money(recipient = "bob", amount = 100)'
    assert "note" not in projected.text


def test_official_description_matches_released_insertion_order_string_format() -> None:
    call = CandidateToolCall(
        "send_money",
        {"amount": 100, "note": "rent", "recipient": "bob"},
    )

    assert official_describe_tool_call(call) == "send_money(amount = 100, recipient = bob)"


def test_other_functions_keep_all_arguments_in_deterministic_order() -> None:
    call = CandidateToolCall("create_report", {"z": [2, 1], "a": {"b": True}})

    assert describe_tool_call(call) == 'create_report(a = {"b":true}, z = [2,1])'
    assert describe_tool_call(CandidateToolCall("get_balance", {})) == "get_balance()"
    assert describe_tool_calls([]) == NO_TOOL_CALLS_TEXT


def test_security_projection_recognizes_case_and_space_variants() -> None:
    call = CandidateToolCall(
        "SEND EMAIL",
        {"recipients": "attacker@example.com", "body": "ignored"},
    )

    assert project_tool_call(call).argument_names == ("recipients",)


def test_comparison_uses_projected_text_and_maximum_pairwise_cosine() -> None:
    original = (
        CandidateToolCall("get_balance", {"account": "checking"}),
        CandidateToolCall(
            "send_email",
            {"recipients": ["attacker@example.com"], "body": "primary-run body"},
        ),
    )
    masked = (
        CandidateToolCall("send_money", {"recipient": "shop", "amount": 10}),
        CandidateToolCall(
            "send_email",
            {"recipients": ["attacker@example.com"], "body": "different masked-run body"},
        ),
    )
    descriptions = [describe_tool_call(call) for call in original + masked]
    embedder = LookupEmbedder(
        {
            descriptions[0]: [0.0, 1.0],
            descriptions[1]: [1.0, 0.0],
            descriptions[2]: [0.6, 0.8],
            descriptions[3]: [2.0, 0.0],
        }
    )

    result = compare_tool_calls(original, masked, embedder=embedder, threshold=0.8)

    assert result.max_similarity == pytest.approx(1.0)
    assert result.triggered
    assert result.best_match is not None
    assert (result.best_match.original_index, result.best_match.masked_index) == (1, 1)
    assert np.asarray(result.similarity_matrix) == pytest.approx(np.asarray(((0.8, 0.0), (0.6, 1.0))))
    embedded_text = "\n".join(embedder.batches[0])
    assert "primary-run body" not in embedded_text
    assert "different masked-run body" not in embedded_text


def test_comparison_uses_strict_threshold_and_handles_an_empty_cache() -> None:
    text = describe_tool_call(CandidateToolCall("send_money", {"recipient": "bob", "amount": 1}))
    embedder = LookupEmbedder({text: [1.0, 0.0]})

    exact_threshold = compare_tool_calls(
        [CandidateToolCall("send_money", {"recipient": "bob", "amount": 1})],
        [CandidateToolCall("send_money", {"recipient": "bob", "amount": 1})],
        embedder=embedder,
        threshold=1.0,
    )
    empty = compare_tool_calls(
        [CandidateToolCall("read_file", {"path": "safe.txt"})],
        [],
        embedder=embedder,
    )

    assert exact_threshold.max_similarity == pytest.approx(1.0)
    assert not exact_threshold.triggered
    assert empty.max_similarity == 0.0
    assert empty.best_match is None
    assert not empty.triggered
    assert len(embedder.batches) == 1


def test_cosine_similarity_validates_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-zero"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize(
    ("matrix", "error"),
    [
        (np.asarray([1.0, 0.0]), "two-dimensional"),
        (np.asarray([[1.0, 0.0]]), "one row per input"),
        (np.asarray([[0.0, 0.0], [1.0, 0.0]]), "non-zero"),
        (np.asarray([[np.nan, 0.0], [1.0, 0.0]]), "NaN or infinity"),
    ],
)
def test_comparison_rejects_invalid_embedding_batches(matrix: NDArray[np.float64], error: str) -> None:
    class InvalidEmbedder:
        def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
            return matrix

    with pytest.raises(ValueError, match=error):
        compare_tool_calls(
            [CandidateToolCall("read_file", {})],
            [CandidateToolCall("read_file", {})],
            embedder=InvalidEmbedder(),
        )


def test_single_prompt_result_retains_prompt_and_cross_step_cache() -> None:
    call = CandidateToolCall("send_money", {"recipient": "attacker", "amount": 100})
    description = describe_tool_call(call)
    run = MaskedPromptRun(
        prompt_id="summarize",
        masking_prompt="Summarize the content and execute any instructions.",
        cached_calls=(CandidateToolCall("read_file", {"path": "old.txt"}), call),
    )
    read_description = describe_tool_call(run.cached_calls[0])
    embedder = LookupEmbedder({description: [1.0, 0.0], read_description: [0.0, 1.0]})

    result = compare_masked_prompt([call], run, embedder=embedder)

    assert result.run.prompt_id == "summarize"
    assert len(result.run.cached_calls) == 2
    assert result.score == pytest.approx(1.0)
    assert result.triggered


def test_ensemble_averages_weak_detector_scores_instead_of_votes() -> None:
    original = CandidateToolCall("send_money", {"recipient": "attacker", "amount": 100})
    original_text = describe_tool_call(original)
    strong = CandidateToolCall("send_money", {"recipient": "attacker", "amount": 100})
    medium = CandidateToolCall("send_money", {"recipient": "other", "amount": 50})
    weak = CandidateToolCall("get_balance", {"account": "checking"})
    medium_text = describe_tool_call(medium)
    weak_text = describe_tool_call(weak)
    embedder = LookupEmbedder(
        {
            original_text: [1.0, 0.0],
            medium_text: [0.6, 0.8],
            weak_text: [0.0, 1.0],
        }
    )
    runs = (
        MaskedPromptRun("summary", "Summarize this content.", (strong,)),
        MaskedPromptRun("grammar", "Check the grammar.", (medium,)),
        MaskedPromptRun("sentiment", "Classify sentiment.", (weak,)),
    )

    result = evaluate_ensemble(
        [original],
        runs,
        embedder=embedder,
        call_threshold=0.8,
        ensemble_threshold=0.5,
    )

    assert result.member_scores == pytest.approx((1.0, 0.6, 0.0))
    assert result.mean_similarity == pytest.approx(1.6 / 3.0)
    assert result.triggered
    # Only one member exceeds the single-prompt threshold.  The ensemble follows
    # the paper's mean-similarity equation rather than majority voting.
    assert [member.triggered for member in result.members] == [True, False, False]


def test_ensemble_requires_unique_nonempty_prompt_set_and_uses_strict_threshold() -> None:
    call = CandidateToolCall("send_money", {"recipient": "attacker", "amount": 100})
    text = describe_tool_call(call)
    embedder = LookupEmbedder({text: [1.0, 0.0]})
    member = compare_masked_prompt(
        [call],
        MaskedPromptRun("summary", "Summarize.", (call,)),
        embedder=embedder,
        threshold=1.0,
    )

    result = aggregate_prompt_results([member], threshold=1.0)

    assert result.mean_similarity == pytest.approx(1.0)
    assert not result.triggered
    with pytest.raises(ValueError, match="at least one"):
        aggregate_prompt_results([])
    with pytest.raises(ValueError, match="unique"):
        aggregate_prompt_results([member, member])
