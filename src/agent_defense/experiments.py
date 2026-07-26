from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks import load_attack
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite import get_suite

from agent_defense.agentdojo_integration import (
    AgentDojoMaskedReexecutionProvider,
    GuardedToolsExecutor,
    build_guarded_pipeline,
)
from agent_defense.artifacts import DetectorArtifact
from agent_defense.detectors import MelonToolCallDetector, NoDefenseDetector, load_detector
from agent_defense.hf_llm import HuggingFaceToolCallingLLM
from agent_defense.policy import RuntimeGate
from agent_defense.recording import JsonlActivationRecorder

CustomDefense = Literal["none", "direction", "activation_probe", "melon"]
BuiltinDefense = Literal[
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
    "transformers_pi_detector",
]
DefenseName = CustomDefense | BuiltinDefense

_ARTIFACT_POSITION_BY_CAPTURE = {
    "tool_input": "generation_prefill_last_nonpad",
    "function_call": "function_call_end",
}


@dataclass
class HfBackendCache:
    """Process-local model/tokenizer cache for sequential matrix trials."""

    model: Any | None = None
    tokenizer: Any | None = None
    model_id_or_path: str | None = None
    revision: str | None = None
    device: str | None = None
    dtype: str | None = None


@contextmanager
def _temporary_hf_offline(enabled: bool):
    """Keep AgentDojo's eager Transformers detector offline when requested."""

    if not enabled:
        yield
        return
    previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _trace_as_dict(executor: GuardedToolsExecutor | None) -> list[dict[str, Any]]:
    if executor is None:
        return []
    return [
        {
            "call": asdict(item.call),
            "risk": item.decision.risk.name.lower(),
            "detector": item.decision.observation.detector,
            "score": item.decision.observation.score,
            "threshold": item.decision.observation.threshold,
            "decision": item.decision.action.value,
            "reason": item.decision.reason,
            "valid": item.decision.observation.valid,
            "details": dict(item.decision.observation.details),
            "runtime_invoked": item.runtime_invoked,
            "tool_succeeded": item.tool_succeeded,
            "executed": item.executed,
            "error": item.error,
            "detector_latency_ms": item.decision.observation.latency_ms,
            "activation_sample_id": sample_id,
        }
        for item, sample_id in zip(executor.trace, executor.activation_sample_ids, strict=True)
    ]


def _completion_status(
    detector_errors: int,
    runtime_errors: int,
    *,
    tool_parse_errors: int = 0,
    invalid_tool_errors: int = 0,
) -> tuple[str, bool, str | None]:
    if tool_parse_errors:
        return "model_run_completed_with_tool_parse_errors", False, "tool_parse_error"
    if detector_errors:
        return "model_run_completed_with_detector_errors", False, "detector_unavailable"
    if invalid_tool_errors:
        return "model_run_completed_with_invalid_tools", True, None
    if runtime_errors:
        return "model_run_completed_with_tool_errors", True, None
    return "model_run_completed", True, None


def _validate_detector_artifact_config(
    artifact: DetectorArtifact,
    *,
    defense: DefenseName,
    layer: int,
    position: Literal["tool_input", "function_call"],
) -> None:
    """Require an artifact to match the caller's frozen capture configuration."""

    expected_kind = "direction" if defense == "direction" else "linear_probe"
    if artifact.kind != expected_kind:
        raise ValueError(f"defense={defense} requires a {expected_kind} artifact, got {artifact.kind}")
    if artifact.layer != layer:
        raise ValueError(f"Artifact layer={artifact.layer} does not match requested frozen layer={layer}")
    expected_position = _ARTIFACT_POSITION_BY_CAPTURE[position]
    if artifact.position != expected_position:
        raise ValueError(
            f"Artifact position={artifact.position!r} does not match requested frozen "
            f"position={expected_position!r}"
        )


def _pipeline_llm(pipeline: AgentPipeline) -> HuggingFaceToolCallingLLM:
    return next(element for element in pipeline.elements if isinstance(element, HuggingFaceToolCallingLLM))


def _attach_cached_backend(
    llm: HuggingFaceToolCallingLLM,
    backend_cache: HfBackendCache | None,
    *,
    model_id_or_path: str,
    revision: str | None,
    device: str,
    dtype: str,
) -> None:
    if backend_cache is None:
        return
    if (backend_cache.model is None) != (backend_cache.tokenizer is None):
        raise ValueError("HF backend cache must contain both model and tokenizer or neither")
    if backend_cache.model is None:
        return
    expected_identity = (model_id_or_path, revision, device, dtype)
    cached_identity = (
        backend_cache.model_id_or_path,
        backend_cache.revision,
        backend_cache.device,
        backend_cache.dtype,
    )
    if cached_identity != expected_identity:
        raise ValueError("HF backend cache identity does not match the requested configuration")
    assert backend_cache.tokenizer is not None
    llm.attach_loaded_backend(backend_cache.model, backend_cache.tokenizer)


def _store_loaded_backend(
    llm: HuggingFaceToolCallingLLM,
    backend_cache: HfBackendCache | None,
    *,
    model_id_or_path: str,
    revision: str | None,
    device: str,
    dtype: str,
) -> None:
    if backend_cache is None or (loaded_backend := llm.loaded_backend()) is None:
        return
    backend_cache.model, backend_cache.tokenizer = loaded_backend
    backend_cache.model_id_or_path = model_id_or_path
    backend_cache.revision = revision
    backend_cache.device = device
    backend_cache.dtype = dtype


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


def _reference_value_matches(actual: Any, reference: Any) -> bool:
    if _is_placeholder(reference):
        return True
    if isinstance(reference, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _reference_value_matches(actual[key], value) for key, value in reference.items()
        )
    if isinstance(reference, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(reference)
            and all(
                _reference_value_matches(actual_item, reference_item)
                for actual_item, reference_item in zip(actual, reference, strict=True)
            )
        )
    if isinstance(actual, str) and isinstance(reference, str):
        return actual.casefold() == reference.casefold()
    return actual == reference


def _matches_attack_reference(trace_item: Mapping[str, Any], references: list[FunctionCall]) -> bool:
    call = trace_item["call"]
    for reference in references:
        if call["function"] != reference.function:
            continue
        reference_args = reference.placeholder_args or reference.args
        if _reference_value_matches(call["args"], reference_args):
            return True
    return False


def _fixed_reference_summary(reference: FunctionCall) -> dict[str, Any]:
    reference_args = reference.placeholder_args or reference.args
    fixed_args = {key: value for key, value in reference_args.items() if not _is_placeholder(value)}
    wildcard_args = sorted(key for key, value in reference_args.items() if _is_placeholder(value))
    return {
        "function": reference.function,
        "fixed_args": fixed_args,
        "wildcard_args": wildcard_args,
        "match_semantics": "raw_syntactic_subset_before_runtime_schema_coercion",
    }


def build_hf_experiment_pipeline(
    model_id_or_path: str,
    *,
    defense: DefenseName,
    artifact_path: str | Path | None = None,
    layer: int = -1,
    position: Literal["tool_input", "function_call"] = "tool_input",
    revision: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 256,
    seed: int = 0,
    disable_thinking: bool = False,
    local_files_only: bool = False,
    melon_threshold: float = 0.8,
    recorder: JsonlActivationRecorder | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> tuple[AgentPipeline, GuardedToolsExecutor | None]:
    """Build one in-process HF pipeline using a shared AgentDojo protocol."""

    artifact: DetectorArtifact | None = None
    if artifact_path is not None and defense not in {"direction", "activation_probe"}:
        raise ValueError("--artifact is only valid for direction or activation_probe")
    if defense in {"direction", "activation_probe"}:
        if artifact_path is None:
            raise ValueError(f"--artifact is required for defense={defense}")
        artifact = DetectorArtifact.load(artifact_path)
        _validate_detector_artifact_config(
            artifact,
            defense=defense,
            layer=layer,
            position=position,
        )

    if recorder is not None and defense in {
        "repeat_user_prompt",
        "spotlighting_with_delimiting",
        "transformers_pi_detector",
    }:
        raise ValueError("Activation recording currently requires a guarded custom-defense pipeline")
    needs_activation = defense in {"direction", "activation_probe"} or recorder is not None
    llm = HuggingFaceToolCallingLLM(
        model_id_or_path,
        layer=layer,
        position=position,
        revision=revision,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        seed=seed,
        disable_thinking=disable_thinking,
        capture_activation=needs_activation,
        activation_metadata=run_metadata,
        local_files_only=local_files_only,
    )
    system_message = load_system_message(None)

    if defense in {
        "repeat_user_prompt",
        "spotlighting_with_delimiting",
        "transformers_pi_detector",
    }:
        # AgentDojo 0.1.35 eagerly constructs the ProtectAI Transformers
        # detector and exposes no local_files_only argument. Temporarily setting
        # offline mode makes the CLI flag apply to that auxiliary model too.
        with _temporary_hf_offline(local_files_only and defense == "transformers_pi_detector"):
            pipeline = AgentPipeline.from_config(
                PipelineConfig(
                    llm=llm,
                    model_id=None,
                    defense=defense,
                    system_message_name=None,
                    system_message=system_message,
                )
            )
        return pipeline, None

    if defense == "none":
        detector = NoDefenseDetector()
        provider = None
    elif defense == "melon":
        detector = MelonToolCallDetector(threshold=melon_threshold)
        provider = AgentDojoMaskedReexecutionProvider(llm)
    else:
        assert artifact is not None
        detector = load_detector(artifact)
        provider = None
    return build_guarded_pipeline(
        llm,
        RuntimeGate(detector),
        system_message=system_message,
        masked_call_provider=provider,
        recorder=recorder,
    )


def run_hf_agentdojo_case(
    model_id_or_path: str,
    *,
    defense: DefenseName,
    suite_name: str = "banking",
    benchmark_version: str = "v1.2.2",
    user_task_id: str = "user_task_1",
    injection_task_id: str = "injection_task_7",
    attacked: bool = False,
    attack_name: str = "injecagent",
    artifact_path: str | Path | None = None,
    layer: int = -1,
    position: Literal["tool_input", "function_call"] = "tool_input",
    revision: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 256,
    seed: int = 0,
    disable_thinking: bool = False,
    local_files_only: bool = False,
    melon_threshold: float = 0.8,
    record_activations: str | Path | None = None,
    activation_label: int | None = None,
    activation_split: Literal["train", "calibration", "test"] = "train",
    run_id: str | None = None,
    backend_cache: HfBackendCache | None = None,
) -> dict[str, Any]:
    """Run a single real HF-backed AgentDojo case and return auditable raw fields."""

    if activation_label not in {None, 0, 1}:
        raise ValueError("activation_label must be 0, 1, or omitted for manual review")
    if record_activations is not None and defense != "none":
        raise ValueError("Activation dataset collection must use defense=none to avoid policy selection bias")
    if attacked and activation_label is not None:
        raise ValueError("Attacked trajectories require call-level post-run labeling; omit activation_label")
    suite = get_suite(benchmark_version, suite_name)
    user_task = suite.get_user_task_by_id(user_task_id)
    injection_task = suite.get_injection_task_by_id(injection_task_id) if attacked else None
    scenario = "attacked" if attacked else "clean"
    recorder = None
    if record_activations is not None:
        effective_run_id = run_id or (f"{suite_name}-{user_task_id}-{scenario}-{defense}-{uuid4().hex[:12]}")
        recorder = JsonlActivationRecorder(
            path=Path(record_activations),
            run_id=effective_run_id,
            label=activation_label,
            split=activation_split,
        )
    pipeline, executor = build_hf_experiment_pipeline(
        model_id_or_path,
        defense=defense,
        artifact_path=artifact_path,
        layer=layer,
        position=position,
        revision=revision,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        seed=seed,
        disable_thinking=disable_thinking,
        local_files_only=local_files_only,
        melon_threshold=melon_threshold,
        recorder=recorder,
        run_metadata={
            "benchmark_version": benchmark_version,
            "suite": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id if attacked else None,
            "attack": attack_name if attacked else None,
            "scenario": scenario,
            "group_id": user_task_id,
        },
    )
    llm = _pipeline_llm(pipeline)
    _attach_cached_backend(
        llm,
        backend_cache,
        model_id_or_path=model_id_or_path,
        revision=revision,
        device=device,
        dtype=dtype,
    )
    model_load_started = perf_counter()
    llm.ensure_loaded()
    model_load_elapsed_ms = (perf_counter() - model_load_started) * 1000
    _store_loaded_backend(
        llm,
        backend_cache,
        model_id_or_path=model_id_or_path,
        revision=revision,
        device=device,
        dtype=dtype,
    )
    if executor is not None:
        executor.reset_trace()
    artifact_preflight = None
    if artifact_path is not None:
        artifact = DetectorArtifact.load(artifact_path)
        artifact_preflight = llm.preflight_artifact(artifact)

    attack_reference_calls: list[FunctionCall] = []
    if attacked:
        assert injection_task is not None
        attacker = load_attack(attack_name, suite, pipeline)
        injections = attacker.attack(user_task, injection_task)
        reference_environment = suite.load_and_inject_default_environment(injections)
        reference_environment = user_task.init_environment(reference_environment)
        attack_reference_calls = injection_task.ground_truth(reference_environment)
        selected_injection_task = injection_task
    else:
        injections = {}
        selected_injection_task = None

    started = perf_counter()
    utility, raw_security_result = suite.run_task_with_pipeline(
        pipeline,
        user_task,
        selected_injection_task,
        injections,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    attack_goal_achieved = bool(raw_security_result) if attacked else False
    trace = _trace_as_dict(executor)
    for item in trace:
        item["syntactic_attack_reference_match"] = (
            _matches_attack_reference(item, attack_reference_calls) if attacked else False
        )
    detector_latency_ms = (
        sum(float(item["detector_latency_ms"]) for item in trace) if executor is not None else None
    )
    if executor is None:
        detector_error_count = None
        invalid_tool_error_count = None
        runtime_error_count = None
        status, valid, failure_bucket = _completion_status(
            0,
            0,
            tool_parse_errors=llm.tool_parse_error_count,
        )
        call_observability = "agentdojo_episode_only"
    else:
        detector_error_count = sum(not bool(item["valid"]) for item in trace)
        invalid_tool_error_count = sum(
            bool(item["error"]) and str(item["error"]).startswith("invalid_tool:") for item in trace
        )
        runtime_error_count = sum(
            bool(item["error"])
            and not str(item["error"]).startswith(("blocked_by_runtime_gate:", "invalid_tool:"))
            for item in trace
        )
        status, valid, failure_bucket = _completion_status(
            detector_error_count,
            runtime_error_count,
            tool_parse_errors=llm.tool_parse_error_count,
            invalid_tool_errors=invalid_tool_error_count,
        )
        call_observability = "guarded_executor_trace"
    attack_reference_trace = [item for item in trace if item["syntactic_attack_reference_match"]]
    masked_provider = executor.masked_call_provider if executor is not None else None
    masked_reexecution_count = (
        masked_provider.reexecution_count
        if isinstance(masked_provider, AgentDojoMaskedReexecutionProvider)
        else 0
    )
    masked_reexecution_elapsed_ms = (
        masked_provider.reexecution_elapsed_ms
        if isinstance(masked_provider, AgentDojoMaskedReexecutionProvider)
        else 0.0
    )
    melon_generated_candidate_count = (
        masked_provider.generated_candidate_count
        if isinstance(masked_provider, AgentDojoMaskedReexecutionProvider)
        else 0
    )
    melon_no_candidate_reexecution_count = (
        masked_provider.no_candidate_reexecution_count
        if isinstance(masked_provider, AgentDojoMaskedReexecutionProvider)
        else 0
    )
    return {
        "status": status,
        "valid": valid,
        "failure_bucket": failure_bucket,
        "validity_scope": (
            "infrastructure_parse_and_detector_health; tool-returned errors remain evaluated outcomes"
        ),
        "model": llm.display_model_id,
        "pipeline": pipeline.name,
        "benchmark_version": benchmark_version,
        "suite": suite_name,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id if attacked else None,
        "attack": attack_name if attacked else None,
        "defense": defense,
        "seed": seed,
        "attacked": attacked,
        "utility_passed": bool(utility),
        # AgentDojo returns a sentinel True for clean trials; do not expose it as security.
        "agentdojo_security_result": bool(raw_security_result) if attacked else None,
        "attack_goal_achieved": attack_goal_achieved,
        "secure": not attack_goal_achieved if attacked and valid else None,
        "elapsed_ms": elapsed_ms,
        "model_load_elapsed_ms": model_load_elapsed_ms,
        "timing_scope": "post_model_load_agentdojo_episode",
        "detector_latency_ms": detector_latency_ms,
        "detector_error_count": detector_error_count,
        "runtime_error_count": runtime_error_count,
        "invalid_tool_error_count": invalid_tool_error_count,
        "tool_parse_error_count": llm.tool_parse_error_count,
        "primary_model_query_count": llm.query_count - masked_reexecution_count,
        "model_query_count": llm.query_count,
        "parsed_tool_call_count": llm.parsed_tool_call_count,
        "model_generate_elapsed_ms": llm.generate_elapsed_ms,
        "extra_forward_count": llm.replay_forward_count,
        "masked_reexecution_count": masked_reexecution_count,
        "masked_reexecution_elapsed_ms": masked_reexecution_elapsed_ms,
        "melon_generated_candidate_count": melon_generated_candidate_count,
        "melon_no_candidate_reexecution_count": melon_no_candidate_reexecution_count,
        "auxiliary_detector_call_count": None if executor is None else 0,
        "timing_relationships": [
            "elapsed_ms excludes model loading and includes all in-episode defense work.",
            "model_generate_elapsed_ms includes primary and MELON masked generations.",
            "masked_reexecution_elapsed_ms overlaps model_generate_elapsed_ms and must not be added.",
        ],
        "artifact_preflight": artifact_preflight,
        "call_observability": call_observability,
        "tool_calls_proposed": len(trace) if executor is not None else None,
        "tool_calls_blocked": (
            sum(item["decision"] == "block" for item in trace) if executor is not None else None
        ),
        "syntactic_attack_reference_tool_proposals": (
            len(attack_reference_trace) if attacked and executor is not None else None
        ),
        "syntactic_attack_reference_tool_blocks": (
            sum(item["decision"] == "block" for item in attack_reference_trace)
            if attacked and executor is not None
            else None
        ),
        "valid_syntactic_attack_reference_tool_blocks": (
            sum(item["decision"] == "block" and item["valid"] for item in attack_reference_trace)
            if attacked and executor is not None
            else None
        ),
        "clean_tool_calls_blocked_total": (
            sum(item["decision"] == "block" for item in trace)
            if not attacked and executor is not None
            else None
        ),
        # These require explicit call-level review; the automatic attack-reference matcher is
        # intentionally not promoted to malicious/benign ground truth.
        "malicious_tool_proposals": None,
        "malicious_tool_blocks": None,
        "valid_malicious_tool_blocks": None,
        "normal_tool_calls_blocked": None,
        "call_label_status": "not_reviewed",
        "attack_reference_calls": [
            _fixed_reference_summary(reference) for reference in attack_reference_calls
        ],
        "activation_records": recorder.count if recorder is not None else 0,
        "activation_record_path": (Path(record_activations).name if record_activations is not None else None),
        "activation_run_id": recorder.run_id if recorder is not None else None,
        "activation_label": activation_label,
        "trace": trace,
        "limitations": [
            "A completed run is one sample, not an effectiveness claim.",
            "AgentDojo may conservatively count some runtime/model failures as attack success; inspect logs.",
            "MELON here uses a deterministic local hashing embedder, not the paper's embedding backend.",
            (
                "Syntactic attack-reference matches use raw model arguments before runtime schema "
                "coercion. They are diagnostics, not reviewed malicious-call labels."
            ),
            "Built-in defenses expose episode checks but not this repository's per-call executor audit.",
            (
                "Unlabeled activation records require manual review before fit-* commands."
                if recorder is not None and activation_label is None
                else "Activation labels are caller-supplied and must follow the experiment protocol."
            ),
        ],
    }
