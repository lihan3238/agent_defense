from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.functions_runtime import FunctionCall

from agent_defense.artifacts import DetectorArtifact
from agent_defense.experiments import (
    HfBackendCache,
    _attach_cached_backend,
    _completion_status,
    _fixed_reference_summary,
    _matches_attack_reference,
    _store_loaded_backend,
    build_hf_experiment_pipeline,
    run_hf_agentdojo_case,
)
from agent_defense.hf_llm import HuggingFaceToolCallingLLM
from agent_defense.melon import PaperMelonToolCallDetector
from agent_defense.melon_agentdojo import AgentDojoPaperMaskedReexecutionProvider
from agent_defense.recording import JsonlActivationRecorder
from agent_defense.semantic_embeddings import TransformersMeanPoolingEmbedder
from agent_defense.types import RiskLevel


def test_builtin_agentdojo_comparator_builds_without_loading_model() -> None:
    pipeline, executor = build_hf_experiment_pipeline(
        "org/model-not-loaded-during-construction",
        defense="repeat_user_prompt",
        local_files_only=True,
    )

    assert isinstance(pipeline, AgentPipeline)
    assert pipeline.name.endswith("repeat_user_prompt")
    assert executor is None


def test_none_pipeline_keeps_auditable_executor_but_skips_activation_capture() -> None:
    pipeline, executor = build_hf_experiment_pipeline(
        "org/model-not-loaded-during-construction",
        defense="none",
        local_files_only=True,
    )

    assert pipeline.name.endswith("-none")
    assert "/" not in pipeline.name
    assert executor is not None
    assert executor.gate.name == "none"


def test_paper_melon_pipeline_builds_batch_abort_path_without_loading_models() -> None:
    pipeline, executor = build_hf_experiment_pipeline(
        "org/model-not-loaded-during-construction",
        defense="melon_paper",
        melon_embedding_backend="hf",
        melon_embedding_model="org/embedder-not-loaded-during-construction",
        local_files_only=True,
    )

    assert isinstance(pipeline, AgentPipeline)
    assert executor is not None
    assert executor.batch_preflight is True
    assert executor.abort_episode_on_block is True
    assert executor.gate.minimum_block_risk == RiskLevel.LOW
    assert isinstance(executor.gate.detector, PaperMelonToolCallDetector)
    assert isinstance(executor.gate.detector.embedder, TransformersMeanPoolingEmbedder)
    assert isinstance(executor.masked_call_provider, AgentDojoPaperMaskedReexecutionProvider)


def test_activation_recording_enables_capture_on_no_defense_pipeline(tmp_path) -> None:
    recorder = JsonlActivationRecorder(tmp_path / "calls.jsonl", "run", 0, "train")
    pipeline, _ = build_hf_experiment_pipeline(
        "model-not-loaded-during-construction",
        defense="none",
        local_files_only=True,
        recorder=recorder,
    )
    llm = next(element for element in pipeline.elements if isinstance(element, HuggingFaceToolCallingLLM))

    assert llm.capture_activation is True


def test_builtin_comparator_rejects_activation_recording_until_it_has_a_guarded_executor(
    tmp_path,
) -> None:
    recorder = JsonlActivationRecorder(tmp_path / "calls.jsonl", "run", 0, "train")

    with pytest.raises(ValueError, match="guarded custom-defense"):
        build_hf_experiment_pipeline(
            "model-not-loaded-during-construction",
            defense="repeat_user_prompt",
            recorder=recorder,
        )


def test_attacked_collection_requires_post_run_call_level_labels(tmp_path) -> None:
    with pytest.raises(ValueError, match="call-level post-run labeling"):
        run_hf_agentdojo_case(
            "model-not-loaded",
            defense="none",
            attacked=True,
            record_activations=tmp_path / "calls.jsonl",
            activation_label=1,
        )


def test_collection_rejects_defense_selected_trajectories(tmp_path) -> None:
    with pytest.raises(ValueError, match="defense=none"):
        run_hf_agentdojo_case(
            "model-not-loaded",
            defense="melon",
            record_activations=tmp_path / "calls.jsonl",
        )


def test_transformers_detector_honors_local_only_without_leaking_environment(monkeypatch) -> None:
    observed: dict[str, str | None] = {}

    def fake_from_config(config):
        del config
        observed["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        observed["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return SimpleNamespace(name="stub-transformers-detector")

    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(AgentPipeline, "from_config", fake_from_config)

    pipeline, executor = build_hf_experiment_pipeline(
        "model-not-loaded",
        defense="transformers_pi_detector",
        local_files_only=True,
    )

    assert pipeline.name == "stub-transformers-detector"
    assert executor is None
    assert observed == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_artifact_is_rejected_for_unrelated_defense(tmp_path) -> None:
    artifact = tmp_path / "unused.json"
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="only valid"):
        build_hf_experiment_pipeline("model-not-loaded", defense="none", artifact_path=artifact)


@pytest.mark.parametrize(
    ("artifact_layer", "artifact_position", "message"),
    [
        (7, "generation_prefill_last_nonpad", "layer"),
        (22, "function_call_end", "position"),
    ],
)
def test_probe_artifact_cannot_override_requested_capture_config(
    tmp_path,
    artifact_layer: int,
    artifact_position: str,
    message: str,
) -> None:
    artifact_path = tmp_path / "probe.json"
    DetectorArtifact(
        kind="linear_probe",
        weights=(1.0,),
        bias=0.0,
        threshold=0.5,
        model_id="model-not-loaded",
        layer=artifact_layer,
        position=artifact_position,
    ).save(artifact_path)

    with pytest.raises(ValueError, match=message):
        build_hf_experiment_pipeline(
            "model-not-loaded",
            defense="activation_probe",
            artifact_path=artifact_path,
            layer=22,
            position="tool_input",
        )


def test_matching_probe_artifact_preserves_requested_capture_config(tmp_path) -> None:
    artifact_path = tmp_path / "probe.json"
    DetectorArtifact(
        kind="linear_probe",
        weights=(1.0,),
        bias=0.0,
        threshold=0.5,
        model_id="model-not-loaded",
        layer=22,
        position="generation_prefill_last_nonpad",
    ).save(artifact_path)

    pipeline, _ = build_hf_experiment_pipeline(
        "model-not-loaded",
        defense="activation_probe",
        artifact_path=artifact_path,
        layer=22,
        position="tool_input",
    )
    llm = next(element for element in pipeline.elements if isinstance(element, HuggingFaceToolCallingLLM))

    assert llm.layer == 22
    assert llm.position == "tool_input"


@pytest.mark.parametrize(
    ("detector_errors", "runtime_errors", "tool_parse_errors", "invalid_tool_errors", "expected"),
    [
        (0, 0, 0, 0, ("model_run_completed", True, None)),
        (1, 0, 0, 0, ("model_run_completed_with_detector_errors", False, "detector_unavailable")),
        (0, 0, 1, 0, ("model_run_completed_with_tool_parse_errors", False, "tool_parse_error")),
        (0, 0, 0, 1, ("model_run_completed_with_invalid_tools", True, None)),
        (0, 1, 0, 0, ("model_run_completed_with_tool_errors", True, None)),
    ],
)
def test_completion_status_separates_invalid_failure_buckets(
    detector_errors: int,
    runtime_errors: int,
    tool_parse_errors: int,
    invalid_tool_errors: int,
    expected: tuple[str, bool, str | None],
) -> None:
    assert (
        _completion_status(
            detector_errors,
            runtime_errors,
            tool_parse_errors=tool_parse_errors,
            invalid_tool_errors=invalid_tool_errors,
        )
        == expected
    )


def test_attack_reference_match_uses_function_and_non_placeholder_arguments() -> None:
    reference = FunctionCall(
        function="send_money",
        args={"recipient": "ATTACKER", "amount": 1000000, "subject": "Hacked!"},
        placeholder_args={"recipient": "ATTACKER", "amount": "$amount", "subject": "$subject"},
    )
    trace_item = {
        "call": {
            "function": "send_money",
            "args": {"recipient": "attacker", "amount": 42, "subject": "different"},
        }
    }

    assert _matches_attack_reference(trace_item, [reference]) is True
    assert _fixed_reference_summary(reference) == {
        "function": "send_money",
        "fixed_args": {"recipient": "ATTACKER"},
        "wildcard_args": ["amount", "subject"],
        "match_semantics": "raw_syntactic_subset_before_runtime_schema_coercion",
    }

    trace_item["call"]["args"]["recipient"] = "AUTHORIZED"
    assert _matches_attack_reference(trace_item, [reference]) is False


def test_hf_backend_cache_is_complete_and_identity_bound() -> None:
    model = object()
    tokenizer = object()
    source = HuggingFaceToolCallingLLM("org/model")
    source.attach_loaded_backend(model, tokenizer)
    cache = HfBackendCache()
    _store_loaded_backend(
        source,
        cache,
        model_id_or_path="org/model",
        revision="revision-a",
        device="cpu",
        dtype="float32",
    )
    target = HuggingFaceToolCallingLLM("org/model")

    _attach_cached_backend(
        target,
        cache,
        model_id_or_path="org/model",
        revision="revision-a",
        device="cpu",
        dtype="float32",
    )
    assert target.loaded_backend() == (model, tokenizer)
    assert target.query_count == 0
    assert target.replay_forward_count == 0

    with pytest.raises(ValueError, match="identity"):
        _attach_cached_backend(
            HuggingFaceToolCallingLLM("org/other"),
            cache,
            model_id_or_path="org/other",
            revision="revision-a",
            device="cpu",
            dtype="float32",
        )

    partial = HfBackendCache(model=model)
    with pytest.raises(ValueError, match="both model and tokenizer"):
        _attach_cached_backend(
            target,
            partial,
            model_id_or_path="org/model",
            revision="revision-a",
            device="cpu",
            dtype="float32",
        )
