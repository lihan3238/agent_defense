from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import click
import numpy as np

from agent_defense.artifacts import DetectorArtifact


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _emit_json(value: Any) -> None:
    click.echo(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_json_object_without_duplicates,
                    parse_constant=_reject_nonfinite_json,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise click.ClickException(f"Invalid JSONL at line {line_number}") from error
            if not isinstance(value, dict):
                raise click.ClickException(f"JSONL line {line_number} must contain an object")
            rows.append(value)
    return rows


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        value = json.loads(
            data,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise click.ClickException(f"Invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise click.ClickException(f"{label} JSON must contain an object")
    return value, data


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(dict(value)),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _write_jsonl_new(path: Path, rows: list[Mapping[str, Any]]) -> None:
    serialized = "".join(
        json.dumps(
            _jsonable(dict(row)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as error:
        raise click.ClickException("Output already exists; choose a new file") from error


def _matrix_failure_bucket(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    message = str(error).casefold()
    if "artifact" in message and any(
        marker in message for marker in ("does not match", "missing", "requires", "incompatible")
    ):
        return "artifact_mismatch"
    if isinstance(error, (ImportError, OSError)):
        return "model_unavailable"
    return "runner_error"


def _matrix_backend_identity(model_config: Any, model_override: str | None) -> dict[str, Any]:
    """Return stable model/config identity without retaining a local absolute path."""

    from agent_defense.hf_llm import _display_model_id, _local_checkpoint_content_id

    effective_model = model_override or model_config.model_id_or_path
    device = str(model_config.device)
    device_class = "cuda" if device.casefold().startswith("cuda") else device
    identity = {
        "schema_version": 1,
        "declared_model_id": _display_model_id(model_config.model_id_or_path),
        "effective_model_id": _display_model_id(effective_model),
        "model_override_used": model_override is not None,
        "revision": model_config.revision,
        "layer": model_config.layer,
        "position": model_config.position,
        "device_class": device_class,
        "dtype": model_config.dtype,
        "max_new_tokens": model_config.max_new_tokens,
        "disable_thinking": model_config.disable_thinking,
        "local_files_only": model_config.local_files_only,
    }
    if checkpoint_content_id := _local_checkpoint_content_id(effective_model):
        identity["checkpoint_content_id"] = checkpoint_content_id
    return identity


def _matrix_run_fingerprint(
    *,
    manifest_sha256: str,
    artifact_sha256s: Mapping[str, str],
    backend_identity: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "artifact_sha256s": dict(sorted(artifact_sha256s.items())),
        "backend_identity": dict(backend_identity),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_matrix_rows(
    rows: list[dict[str, Any]],
    *,
    planned_by_id: Mapping[str, Any],
    manifest_sha256: str,
    run_fingerprint: str,
    require_complete: bool,
) -> set[str]:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise click.ClickException(f"Output row {index} has no valid trial_id")
        if trial_id in seen:
            raise click.ClickException("Output contains duplicate trial_id values")
        spec = planned_by_id.get(trial_id)
        if spec is None:
            raise click.ClickException("Output contains trial IDs that are not in this manifest")
        if row.get("manifest_sha256") != manifest_sha256:
            raise click.ClickException(f"Output row {index} does not match the current manifest SHA256")
        if row.get("run_fingerprint") != run_fingerprint:
            raise click.ClickException(f"Output row {index} does not match the current run fingerprint")
        expected_identity = {
            "case_id": spec.case_id,
            "scenario": spec.scenario,
            "seed": spec.seed,
            "defense": spec.defense.name,
            "attacked": spec.scenario == "attacked",
        }
        for key, expected in expected_identity.items():
            actual = row.get(key)
            if type(actual) is not type(expected) or actual != expected:
                raise click.ClickException(
                    f"Output row {index} trial identity disagrees with trial_id for field {key}"
                )
        seen.add(trial_id)

    if require_complete and seen != set(planned_by_id):
        raise click.ClickException("Output does not contain the complete planned trial set")
    return seen


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Runtime prompt-injection defense at the Agent tool-call boundary."""


@main.command()
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
def doctor(json_output: bool) -> None:
    """Check the pinned core environment without downloading a model."""

    packages = {
        name: _package_version(name)
        for name in ("agentdojo", "click", "numpy", "scikit-learn", "torch", "transformers")
    }
    result = {
        "python": platform.python_version(),
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
        "core_ready": all(packages[name] for name in ("agentdojo", "click", "numpy", "scikit-learn")),
        "hf_extra_ready": all(packages[name] for name in ("torch", "transformers")),
        "packages": packages,
        "project_files": {
            "pyproject": Path("pyproject.toml").is_file(),
            "agentdojo_paper": Path("papers/core/2406.13352_AgentDojo_NeurIPS2024.pdf").is_file(),
            "melon_paper": Path("papers/core/2502.05174_MELON_ICML2025.pdf").is_file(),
        },
    }
    if json_output:
        _emit_json(result)
        return
    click.echo(f"Python {result['python']} (supported={result['python_supported']})")
    click.echo(f"Core environment ready: {result['core_ready']}")
    click.echo(f"HF hidden-state extra ready: {result['hf_extra_ready']}")
    for name, version in packages.items():
        click.echo(f"  {name}: {version or 'not installed'}")


@main.command()
@click.option(
    "--defense",
    type=click.Choice(["none", "direction", "activation_probe", "melon"]),
    default="direction",
    show_default=True,
)
@click.option(
    "--scenario",
    type=click.Choice(["clean", "attacked"]),
    default="attacked",
    show_default=True,
)
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
def demo(defense: str, scenario: str, json_output: bool) -> None:
    """Run the two-minute deterministic AgentDojo Banking teaching demo."""

    from agent_defense.demo import run_demo_trial

    record, trace = run_demo_trial(defense, attacked=scenario == "attacked")  # type: ignore[arg-type]
    payload = {
        "evidence_level": "synthetic_activation_fixture",
        "warning": "This validates software control flow, not model-level defense effectiveness.",
        "record": record,
        "trace": trace,
    }
    if json_output:
        _emit_json(payload)
        return
    click.echo("Synthetic activation fixture — not a model effectiveness result.")
    click.echo(
        f"defense={defense} scenario={scenario} utility={record.utility_passed} "
        f"attack_goal_achieved={record.attack_succeeded}"
    )
    for item in trace:
        args = json.dumps(item["args"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        click.echo(
            f"tool={item['tool']} args={args} risk={item['risk']} score={_jsonable(item['score'])} "
            f"threshold={_jsonable(item['threshold'])} triggered={item['triggered']} "
            f"decision={item['decision']} "
            f"executed={item['executed']}"
        )


@main.command("interview-demo")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
def interview_demo(json_output: bool) -> None:
    """Run one-process no-defense/probe before-after cases for a live interview."""

    from agent_defense.demo import run_interview_sequence

    results = run_interview_sequence()
    payload = {
        "evidence_level": "synthetic_activation_fixture",
        "warning": "This demonstrates the runtime boundary, not real-model detector accuracy.",
        "cases": results,
    }
    if json_output:
        _emit_json(payload)
        return
    click.echo("Synthetic activation fixture — runtime-control-flow evidence only.")
    for result in results:
        record = result["record"]
        click.echo(
            f"\n[{result['case']}] utility={record.utility_passed} "
            f"attack_goal_achieved={record.attack_succeeded}"
        )
        for item in result["trace"]:
            args = json.dumps(item["args"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            click.echo(
                f"  tool={item['tool']} args={args} decision={item['decision']} "
                f"executed={item['executed']} reason={item['reason']}"
            )


@main.command("eval-demo")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
def eval_demo(json_output: bool) -> None:
    """Compare all teaching defenses on one clean and one attacked task."""

    from agent_defense.demo import run_demo_matrix

    records, summaries = run_demo_matrix()
    payload = {
        "evidence_level": "synthetic_activation_fixture",
        "warning": "These numbers are contract tests, not paper reproduction results.",
        "records": records,
        "summaries": summaries,
    }
    if json_output:
        _emit_json(payload)
        return
    click.echo("Synthetic activation fixture — not a paper reproduction result.")
    for defense, summary in summaries.items():
        click.echo(
            f"{defense:16} utility={summary['utility_rate']} ASR={summary['attack_success_rate']} "
            f"interception={summary['interception_rate']} false_block={summary['false_block_rate']}"
        )


@main.command("validate-boundary")
@click.option(
    "--defense",
    type=click.Choice(["none", "activation_probe"]),
    default="activation_probe",
    show_default=True,
)
@click.option(
    "--scenario",
    type=click.Choice(["clean", "attacked"]),
    default="attacked",
    show_default=True,
)
def validate_boundary(defense: str, scenario: str) -> None:
    """Audit whether a candidate crossed AgentDojo runtime.run_function."""

    from agent_defense.agentdojo_runner import run_banking_validation

    result = run_banking_validation(defense, attacked=scenario == "attacked")  # type: ignore[arg-type]
    _emit_json(
        {
            "evidence_level": "real_agentdojo_contract_with_synthetic_activation",
            "warning": (
                "The suite/tools/checks are real; model outputs and activations are deterministic fixtures."
            ),
            "result": result,
        }
    )


def _fit(dataset: Path, output: Path, kind: str, fpr: float, score_mode: str = "cosine") -> None:
    from agent_defense.training import fit_artifact_from_samples, load_activation_samples

    samples = load_activation_samples(dataset)
    artifact = fit_artifact_from_samples(
        samples,
        kind=kind,  # type: ignore[arg-type]
        max_false_positive_rate=fpr,
        direction_score_mode=score_mode,  # type: ignore[arg-type]
    )
    artifact.save(output)
    _emit_json({"artifact": output, "kind": artifact.kind, "threshold": artifact.threshold})


@main.command("fit-direction")
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--fpr", type=click.FloatRange(0.0, 1.0, max_open=True), default=0.05, show_default=True)
@click.option(
    "--score-mode",
    type=click.Choice(["cosine", "projection"]),
    default="cosine",
    show_default=True,
)
def fit_direction(dataset: Path, output: Path, fpr: float, score_mode: str) -> None:
    """Fit a danger/refusal direction from train and benign calibration splits."""

    _fit(dataset, output, "direction", fpr, score_mode)


@main.command("fit-probe")
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--fpr", type=click.FloatRange(0.0, 1.0, max_open=True), default=0.05, show_default=True)
def fit_probe(dataset: Path, output: Path, fpr: float) -> None:
    """Fit a train-only-standardized logistic activation probe."""

    _fit(dataset, output, "linear_probe", fpr)


@main.command("evaluate-artifact")
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("artifact", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def evaluate_artifact_command(dataset: Path, artifact: Path) -> None:
    """Evaluate a frozen detector once on the JSONL test split."""

    from agent_defense.training import evaluate_artifact, load_activation_samples

    samples = load_activation_samples(dataset)
    detector_artifact = DetectorArtifact.load(artifact)
    _emit_json(evaluate_artifact(detector_artifact, samples))


@main.command("apply-labels")
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--allow-unlabeled", is_flag=True, help="Keep missing labels as pending review.")
def apply_labels_command(
    dataset: Path,
    manifest: Path,
    output: Path,
    allow_unlabeled: bool,
) -> None:
    """Apply a reviewed sample_id-to-label JSON manifest to recorded activations."""

    from agent_defense.training import apply_label_manifest

    try:
        result = apply_label_manifest(
            dataset,
            manifest,
            output,
            require_all=not allow_unlabeled,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    _emit_json({"output": output, **result})


@main.command("hf-smoke")
@click.option("--model", "model_id", required=True, help="HF model id or local model directory.")
@click.option("--layer", type=int, default=-1, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--dtype", default="auto", show_default=True)
@click.option("--revision", default=None)
@click.option("--local-files-only/--allow-download", default=True, show_default=True)
@click.option("--text", default="Classify this short prompt before taking any action.", show_default=True)
def hf_smoke(
    model_id: str,
    layer: int,
    device: str,
    dtype: str,
    revision: str | None,
    local_files_only: bool,
    text: str,
) -> None:
    """Verify in-process residual-stream access; this is not an effectiveness test."""

    from agent_defense.hf_llm import HuggingFaceToolCallingLLM

    llm = HuggingFaceToolCallingLLM(
        model_id,
        layer=layer,
        device=device,
        dtype=dtype,
        revision=revision,
        local_files_only=local_files_only,
    )
    activation, activation_metadata = llm.capture_text_activation(text)
    _emit_json(
        {
            "evidence_level": "hidden_state_access_smoke",
            "warning": "Standalone text forward only; the Agent runtime path captures generation prefill.",
            "activation_shape": list(activation.shape),
            "activation_dtype": str(activation.dtype),
            "finite": bool(np.all(np.isfinite(activation))),
            "metadata": activation_metadata,
        }
    )


@main.command("hf-tool-call-smoke")
@click.option("--model", "model_id", required=True, help="HF model id or local model directory.")
@click.option("--layer", type=int, default=-1, show_default=True)
@click.option("--position", type=click.Choice(["tool_input", "function_call"]), default="tool_input")
@click.option("--device", default="cpu", show_default=True)
@click.option("--dtype", default="auto", show_default=True)
@click.option("--revision", default=None)
@click.option("--max-new-tokens", type=click.IntRange(1), default=64, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--disable-thinking", is_flag=True, help="Pass enable_thinking=false to compatible templates.")
@click.option("--local-files-only/--allow-download", default=True, show_default=True)
def hf_tool_call_smoke(
    model_id: str,
    layer: int,
    position: str,
    device: str,
    dtype: str,
    revision: str | None,
    max_new_tokens: int,
    seed: int,
    disable_thinking: bool,
    local_files_only: bool,
) -> None:
    """Generate one harmless tool call and capture its runtime activation."""

    from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
    from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime, make_function

    from agent_defense.hf_llm import HuggingFaceToolCallingLLM

    def add(x: int, y: int) -> int:
        """Add two integers.

        :param x: First integer.
        :param y: Second integer.
        """

        return x + y

    runtime = FunctionsRuntime([make_function(add)])
    llm = HuggingFaceToolCallingLLM(
        model_id,
        layer=layer,
        position=position,  # type: ignore[arg-type]
        device=device,
        dtype=dtype,
        revision=revision,
        max_new_tokens=max_new_tokens,
        seed=seed,
        disable_thinking=disable_thinking,
        local_files_only=local_files_only,
    )
    pipeline = AgentPipeline(
        [
            SystemMessage(
                "You are a tool-calling assistant. When explicitly asked, call the available tool."
            ),
            InitQuery(),
            llm,
        ]
    )
    try:
        _, _, _, messages, state = pipeline.query(
            "Call the add tool with x=2 and y=3. Emit the tool call and no prose.",
            runtime,
            EmptyEnv(),
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    calls = messages[-1].get("tool_calls") or []
    activation = state.get("agent_defense.activation")
    if not calls:
        raise click.ClickException("Model did not emit a parseable tool call in the smoke prompt")
    if activation is None:
        capture_error = state.get("agent_defense.activation_error")
        raise click.ClickException(f"Tool call was parsed but activation capture failed: {capture_error}")
    _emit_json(
        {
            "evidence_level": "real_local_model_tool_call_activation_smoke",
            "warning": "This validates the white-box plumbing, not prompt-injection detection quality.",
            "tool_calls": [call.model_dump() for call in calls],
            "activation_shape": list(activation.shape),
            "activation_dtype": str(activation.dtype),
            "finite": bool(np.all(np.isfinite(activation))),
            "metadata": state["agent_defense.activation_metadata"],
        }
    )


@main.command("matrix-plan")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def matrix_plan(manifest: Path) -> None:
    """Validate and print a privacy-safe held-out trial plan without loading a model."""

    from agent_defense.hf_llm import _display_model_id
    from agent_defense.matrix import expand_trials, load_manifest

    try:
        parsed = load_manifest(manifest)
        trials = expand_trials(parsed)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    _emit_json(
        {
            "schema_version": parsed.schema_version,
            "model_id": _display_model_id(parsed.model.model_id_or_path),
            "revision": parsed.model.revision,
            "benchmark": {
                "suite": parsed.benchmark.suite_name,
                "version": parsed.benchmark.benchmark_version,
                "attack": parsed.benchmark.attack_name,
            },
            "defenses": [defense.name for defense in parsed.defenses],
            "cases": [
                {
                    "case_id": case.case_id,
                    "user_task_id": case.user_task_id,
                    "injection_task_id": case.injection_task_id,
                    "seeds": case.seeds,
                }
                for case in parsed.cases
            ],
            "trial_count": len(trials),
            "trial_ids": [trial.trial_id for trial in trials],
        }
    )


@main.command("matrix-run")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--model",
    "model_override",
    default=None,
    help="Override only the local model location; frozen model identity still comes from the artifact.",
)
@click.option("--resume", is_flag=True, help="Skip trial IDs already present in the output JSONL.")
@click.option("--skip-warmup", is_flag=True, help="Do not discard one clean none trial as a warm-up.")
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Record an invalid trial and continue; otherwise stop after persisting the first failure.",
)
def matrix_run(
    manifest: Path,
    output: Path,
    model_override: str | None,
    resume: bool,
    skip_warmup: bool,
    continue_on_error: bool,
) -> None:
    """Run a frozen HF/AgentDojo matrix with one shared in-process model backend."""

    from agent_defense.experiments import (
        HfBackendCache,
        _validate_detector_artifact_config,
        run_hf_agentdojo_case,
    )
    from agent_defense.matrix import (
        BUILTIN_DEFENSES,
        aggregate_results,
        expand_trials,
        load_manifest,
        run_sequential,
    )

    try:
        parsed = load_manifest(manifest)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if output.exists() and not resume:
        raise click.ClickException("Output already exists; pass --resume or choose a new file")

    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    try:
        backend_identity = _matrix_backend_identity(parsed.model, model_override)
    except OSError as error:
        raise click.ClickException(
            f"Could not establish frozen model identity: unreadable identity files ({type(error).__name__})"
        ) from error
    except ValueError as error:
        raise click.ClickException(f"Could not establish frozen model identity: {error}") from error
    checkpoint_content_id = backend_identity.get("checkpoint_content_id")
    artifact_sha256s: dict[str, str] = {}
    for defense in parsed.defenses:
        if defense.artifact_path is None:
            continue
        if not defense.artifact_path.is_file():
            raise click.ClickException(f"Missing frozen artifact for defense={defense.name}")
        try:
            artifact = DetectorArtifact.load(defense.artifact_path)
            _validate_detector_artifact_config(
                artifact,
                defense=defense.name,  # type: ignore[arg-type]
                layer=parsed.model.layer,
                position=parsed.model.position,
            )
            if artifact.model_id != backend_identity["effective_model_id"]:
                raise ValueError(
                    f"Artifact model_id={artifact.model_id!r} does not match effective model "
                    f"{backend_identity['effective_model_id']!r}"
                )
            compatibility = artifact.metadata.get("activation_compatibility", {})
            if not isinstance(compatibility, Mapping):
                raise ValueError("Artifact activation_compatibility must be a mapping")
            artifact_checkpoint_content_id = compatibility.get("checkpoint_content_id")
            if checkpoint_content_id is not None:
                if artifact_checkpoint_content_id is None:
                    raise ValueError("Artifact is missing checkpoint_content_id required by the local model")
                if artifact_checkpoint_content_id != checkpoint_content_id:
                    raise ValueError("Artifact checkpoint_content_id does not match the local model")
            elif artifact_checkpoint_content_id is not None:
                raise ValueError("Artifact checkpoint_content_id requires a matching local model override")
            artifact_sha256s[defense.name] = hashlib.sha256(defense.artifact_path.read_bytes()).hexdigest()
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise click.ClickException(
                f"Invalid frozen artifact for defense={defense.name}: {error}"
            ) from error

    run_fingerprint = _matrix_run_fingerprint(
        manifest_sha256=manifest_sha256,
        artifact_sha256s=artifact_sha256s,
        backend_identity=backend_identity,
    )

    trials = expand_trials(parsed)
    planned_by_id = {trial.trial_id: trial for trial in trials}
    existing: list[dict[str, Any]] = []
    if output.exists():
        existing = _read_jsonl(output)
    existing_ids = _validate_matrix_rows(
        existing,
        planned_by_id=planned_by_id,
        manifest_sha256=manifest_sha256,
        run_fingerprint=run_fingerprint,
        require_complete=False,
    )
    pending = [trial for trial in trials if trial.trial_id not in existing_ids]
    backend_cache = HfBackendCache()

    def runner(**kwargs: Any) -> Mapping[str, Any]:
        if model_override is not None:
            kwargs["model_id_or_path"] = model_override
        return run_hf_agentdojo_case(**kwargs, backend_cache=backend_cache)

    if pending and not skip_warmup:
        warmup = next(
            (trial for trial in trials if trial.scenario == "clean" and trial.defense.name == "none"),
            None,
        )
        if warmup is None:
            raise click.ClickException("Manifest has no clean none trial for warm-up")
        try:
            run_sequential([warmup], runner)
        except Exception as error:
            bucket = _matrix_failure_bucket(error)
            raise click.ClickException(
                f"Warm-up failed: bucket={bucket}, error_type={type(error).__name__}"
            ) from error

    completed = 0
    failures = 0
    for spec in pending:
        started = perf_counter()
        try:
            result = run_sequential([spec], runner)[0]
        except Exception as error:
            failures += 1
            result = {
                "trial_id": spec.trial_id,
                "case_id": spec.case_id,
                "scenario": spec.scenario,
                "defense": spec.defense.name,
                "seed": spec.seed,
                "attacked": spec.scenario == "attacked",
                "status": "runner_exception",
                "valid": False,
                "failure_bucket": _matrix_failure_bucket(error),
                "error_type": type(error).__name__,
                "utility_passed": False,
                "attack_goal_achieved": False,
                "elapsed_ms": (perf_counter() - started) * 1000,
                "detector_latency_ms": None,
                "model_query_count": None,
                "extra_forward_count": None,
                "trace": [],
                "tool_calls_proposed": (None if spec.defense.name in BUILTIN_DEFENSES else 0),
                "tool_calls_blocked": (None if spec.defense.name in BUILTIN_DEFENSES else 0),
            }
        result["manifest_sha256"] = manifest_sha256
        result["run_fingerprint"] = run_fingerprint
        _append_jsonl(output, result)
        completed += 1
        if not result["valid"] and not continue_on_error:
            raise click.ClickException(
                f"Trial {spec.trial_id} failed with bucket={result['failure_bucket']}; "
                "the invalid row was persisted and can be inspected before --resume"
            )

    all_results = _read_jsonl(output) if output.exists() else existing
    _validate_matrix_rows(
        all_results,
        planned_by_id=planned_by_id,
        manifest_sha256=manifest_sha256,
        run_fingerprint=run_fingerprint,
        require_complete=True,
    )
    try:
        summary = aggregate_results(all_results)
    except ValueError as error:
        raise click.ClickException(f"Results could not be aggregated: {error}") from error
    _emit_json(
        {
            "output": output.name,
            "manifest_sha256": manifest_sha256,
            "run_fingerprint": run_fingerprint,
            "scheduled_trials": len(trials),
            "previously_completed": len(existing),
            "completed_this_run": completed,
            "failures_this_run": failures,
            "summary": summary,
        }
    )


@main.command("matrix-apply-reviews")
@click.argument("raw_jsonl", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("review_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_jsonl", type=click.Path(dir_okay=False, path_type=Path))
def matrix_apply_reviews(raw_jsonl: Path, review_json: Path, output_jsonl: Path) -> None:
    """Apply complete human call-level counts to a new matrix JSONL file."""

    from agent_defense.matrix import apply_call_reviews

    resolved_output = output_jsonl.resolve()
    if resolved_output in {raw_jsonl.resolve(), review_json.resolve()}:
        raise click.ClickException("Output must be a new file distinct from both inputs")
    if output_jsonl.exists():
        raise click.ClickException("Output already exists; choose a new file")

    raw_rows = _read_jsonl(raw_jsonl)
    review_manifest, review_bytes = _read_json_object(review_json, label="review manifest")
    review_manifest_sha256 = hashlib.sha256(review_bytes).hexdigest()
    try:
        reviewed_rows = apply_call_reviews(
            raw_rows,
            review_manifest,
            review_manifest_sha256=review_manifest_sha256,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    _write_jsonl_new(output_jsonl, reviewed_rows)
    _emit_json(
        {
            "output": output_jsonl.name,
            "trials": len(reviewed_rows),
            "reviewed_custom_trials": sum(
                row.get("review_manifest_sha256") == review_manifest_sha256 for row in reviewed_rows
            ),
            "review_manifest_sha256": review_manifest_sha256,
        }
    )


@main.command("matrix-summarize")
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def matrix_summarize(results: Path) -> None:
    """Aggregate a matrix JSONL into BU, UA, ASR, coverage, calls, and overhead."""

    from agent_defense.matrix import aggregate_results

    rows = _read_jsonl(results)
    try:
        summary = aggregate_results(rows)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    _emit_json(summary)


@main.command("agentdojo-run")
@click.option("--model", "model_id", required=True, help="HF model id or local model directory.")
@click.option(
    "--defense",
    type=click.Choice(
        [
            "none",
            "direction",
            "activation_probe",
            "melon",
            "repeat_user_prompt",
            "spotlighting_with_delimiting",
            "transformers_pi_detector",
        ]
    ),
    default="none",
    show_default=True,
)
@click.option("--scenario", type=click.Choice(["clean", "attacked"]), default="clean", show_default=True)
@click.option("--artifact", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--suite", "suite_name", default="banking", show_default=True)
@click.option("--benchmark-version", default="v1.2.2", show_default=True)
@click.option("--user-task", "user_task_id", default="user_task_1", show_default=True)
@click.option("--injection-task", "injection_task_id", default="injection_task_7", show_default=True)
@click.option("--attack", "attack_name", default="injecagent", show_default=True)
@click.option("--layer", type=int, default=-1, show_default=True)
@click.option("--position", type=click.Choice(["tool_input", "function_call"]), default="tool_input")
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="auto", show_default=True)
@click.option("--revision", default=None)
@click.option("--max-new-tokens", type=click.IntRange(1), default=256, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--disable-thinking", is_flag=True, help="Pass enable_thinking=false to compatible templates.")
@click.option(
    "--local-files-only/--allow-download",
    default=False,
    show_default=True,
    help="Keep the main LLM and any auxiliary Transformers detector offline; all weights must be cached.",
)
@click.option("--melon-threshold", type=click.FloatRange(-1.0, 1.0), default=0.8, show_default=True)
@click.option(
    "--record-activations",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Append minimal activation records to a Git-ignored JSONL file.",
)
@click.option("--activation-label", type=click.IntRange(0, 1), default=None)
@click.option(
    "--activation-split",
    type=click.Choice(["train", "calibration", "test"]),
    default="train",
    show_default=True,
)
@click.option("--run-id", default=None, help="Stable, non-sensitive identifier for recorded calls.")
def agentdojo_run(
    model_id: str,
    defense: str,
    scenario: str,
    artifact: Path | None,
    suite_name: str,
    benchmark_version: str,
    user_task_id: str,
    injection_task_id: str,
    attack_name: str,
    layer: int,
    position: str,
    device: str,
    dtype: str,
    revision: str | None,
    max_new_tokens: int,
    seed: int,
    disable_thinking: bool,
    local_files_only: bool,
    melon_threshold: float,
    record_activations: Path | None,
    activation_label: int | None,
    activation_split: str,
    run_id: str | None,
) -> None:
    """Run one real in-process-HF AgentDojo case; inspect before aggregating."""

    from agent_defense.experiments import run_hf_agentdojo_case

    if activation_label is not None and record_activations is None:
        raise click.ClickException("--activation-label requires --record-activations")
    try:
        result = run_hf_agentdojo_case(
            model_id,
            defense=defense,  # type: ignore[arg-type]
            suite_name=suite_name,
            benchmark_version=benchmark_version,
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            attacked=scenario == "attacked",
            attack_name=attack_name,
            artifact_path=artifact,
            layer=layer,
            position=position,  # type: ignore[arg-type]
            revision=revision,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            seed=seed,
            disable_thinking=disable_thinking,
            local_files_only=local_files_only,
            melon_threshold=melon_threshold,
            record_activations=record_activations,
            activation_label=activation_label,
            activation_split=activation_split,  # type: ignore[arg-type]
            run_id=run_id,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _emit_json(result)


if __name__ == "__main__":
    main()
