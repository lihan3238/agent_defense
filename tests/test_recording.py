from __future__ import annotations

import json

import numpy as np
import pytest

from agent_defense.recording import JsonlActivationRecorder
from agent_defense.types import (
    CandidateToolCall,
    DecisionAction,
    DetectionContext,
    PolicyDecision,
    ProbeObservation,
    RiskLevel,
)


def test_recorder_keeps_group_metadata_but_never_prompt_text(tmp_path) -> None:
    recorder = JsonlActivationRecorder(
        path=tmp_path / "activations.jsonl",
        run_id="banking-user-task-1",
        label=None,
        split="train",
    )
    context = DetectionContext(
        candidate=CandidateToolCall("update_password", {"password": "not-recorded-by-test-policy"}),
        activation=np.asarray([1.0, 2.0]),
        metadata={
            "model_id": "model-a",
            "checkpoint_content_id": "local-checkpoint-v1:abc123",
            "layer": 3,
            "position": "generation_prefill_last_nonpad",
            "user_task_id": "user_task_1",
            "group_id": "user_task_1",
            "prompt": "private prompt must not be persisted",
        },
    )
    decision = PolicyDecision(
        action=DecisionAction.BLOCK,
        reason="score_above_threshold",
        risk=RiskLevel.CRITICAL,
        observation=ProbeObservation("probe", 0.9, 0.5, True, True, 0.1),
    )

    sample_id = recorder.record(context, decision)
    raw = recorder.path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert recorder.count == 1
    assert sample_id == "banking-user-task-1:0"
    assert payload["label"] is None
    assert payload["metadata"]["user_task_id"] == "user_task_1"
    assert payload["metadata"]["checkpoint_content_id"] == "local-checkpoint-v1:abc123"
    assert payload["metadata"]["label_requires_review"] is True
    assert payload["metadata"]["trajectory_id"] == "banking-user-task-1"
    assert payload["metadata"]["step_index"] == 0
    assert "private prompt" not in raw
    assert "not-recorded-by-test-policy" not in raw


def test_recorder_rejects_reusing_a_run_id_in_an_append_only_file(tmp_path) -> None:
    path = tmp_path / "activations.jsonl"
    path.write_text('{"sample_id":"existing-run:0"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        JsonlActivationRecorder(path=path, run_id="existing-run", label=0, split="train")


def test_recorder_rejects_activation_without_identity_metadata(tmp_path) -> None:
    recorder = JsonlActivationRecorder(tmp_path / "activations.jsonl", "run", 0, "train")
    context = DetectionContext(
        candidate=CandidateToolCall("read_file", {}),
        activation=np.asarray([1.0, 2.0]),
        metadata={},
    )
    decision = PolicyDecision(
        action=DecisionAction.ALLOW,
        reason="detector_clear",
        risk=RiskLevel.LOW,
        observation=ProbeObservation("probe", 0.1, 0.5, False, True, 0.1),
    )

    with pytest.raises(ValueError, match="required recorder keys"):
        recorder.record(context, decision)
