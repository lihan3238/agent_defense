from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_defense.artifacts import DetectorArtifact
from agent_defense.cli import main
from agent_defense.hf_llm import _local_checkpoint_content_id


def test_doctor_reports_pinned_core_environment() -> None:
    result = CliRunner().invoke(main, ["doctor", "--json-output"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["python_supported"] is True
    assert payload["packages"]["agentdojo"] == "0.1.35"
    assert payload["packages"]["click"] == "8.1.8"


def test_fit_and_evaluate_probe_cli(tmp_path) -> None:
    runner = CliRunner()
    artifact = tmp_path / "probe.json"

    fitted = runner.invoke(
        main,
        ["fit-probe", "examples/synthetic_activations.jsonl", str(artifact)],
    )
    evaluated = runner.invoke(
        main,
        ["evaluate-artifact", "examples/synthetic_activations.jsonl", str(artifact)],
    )

    assert fitted.exit_code == 0, fitted.output
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.output)["roc_auc"] == 1.0


def test_apply_labels_cli(tmp_path) -> None:
    dataset = tmp_path / "pending.jsonl"
    manifest = tmp_path / "labels.json"
    output = tmp_path / "labeled.jsonl"
    dataset.write_text(json.dumps({"sample_id": "run:0", "label": None}) + "\n")
    manifest.write_text(json.dumps({"run:0": 0}))

    result = CliRunner().invoke(
        main,
        ["apply-labels", str(dataset), str(manifest), str(output)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["label"] == 0


def test_matrix_plan_expands_frozen_heldout_example() -> None:
    result = CliRunner().invoke(
        main,
        ["matrix-plan", "examples/qwen3-heldout-matrix.example.json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["trial_count"] == 30
    assert {case["user_task_id"] for case in payload["cases"]} == {
        "user_task_1",
        "user_task_11",
        "user_task_13",
    }


def test_matrix_plan_supports_schema_v2_explicit_scenarios(tmp_path: Path) -> None:
    manifest = tmp_path / "scenario-matrix.json"
    payload = {
        "schema_version": 2,
        "model": {
            "model_id_or_path": "org/model",
            "revision": "revision-a",
            "layer": 1,
            "position": "tool_input",
            "device": "cpu",
            "dtype": "float32",
            "max_new_tokens": 16,
            "disable_thinking": True,
            "local_files_only": True,
        },
        "benchmark": {
            "suite_name": "banking",
            "benchmark_version": "v1.1.2",
            "attack_name": "important_instructions",
        },
        "defenses": [{"name": "none"}],
        "cases": [
            {
                "case_id": "clean-user-0",
                "user_task_id": "user_task_0",
                "injection_task_id": None,
                "seeds": [0],
                "scenarios": ["clean"],
            },
            {
                "case_id": "attacked-user-0-injection-0",
                "user_task_id": "user_task_0",
                "injection_task_id": "injection_task_0",
                "seeds": [0],
                "scenarios": ["attacked"],
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(main, ["matrix-plan", str(manifest)])

    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["schema_version"] == 2
    assert plan["trial_count"] == 2
    assert plan["cases"][0]["scenarios"] == ["clean"]


def _write_minimal_matrix(path: Path, *, defenses: list[dict] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {
                    "model_id_or_path": "org/model",
                    "revision": "revision-a",
                    "layer": 1,
                    "position": "tool_input",
                    "device": "cpu",
                    "dtype": "float32",
                    "max_new_tokens": 16,
                    "disable_thinking": True,
                    "local_files_only": True,
                },
                "benchmark": {
                    "suite_name": "banking",
                    "benchmark_version": "v1.2.2",
                    "attack_name": "injecagent",
                },
                "defenses": defenses or [{"name": "none"}],
                "cases": [
                    {
                        "case_id": "case-a",
                        "user_task_id": "user_task_1",
                        "injection_task_id": "injection_task_5",
                        "seeds": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _fake_matrix_case(model_id_or_path, *, defense, attacked, seed, **kwargs):
    del model_id_or_path, kwargs
    return {
        "defense": defense,
        "seed": seed,
        "attacked": attacked,
        "valid": True,
        "failure_bucket": None,
        "utility_passed": True,
        "attack_goal_achieved": attacked,
        "elapsed_ms": 10.0,
        "detector_latency_ms": 0.0,
        "model_query_count": 1,
        "extra_forward_count": 0,
        "trace": [],
        "tool_calls_proposed": 0,
        "tool_calls_blocked": 0,
    }


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_probe_artifact(
    path: Path,
    *,
    layer: int = 1,
    position: str = "generation_prefill_last_nonpad",
    weight: float = 1.0,
    model_id: str = "org/model",
) -> None:
    DetectorArtifact(
        kind="linear_probe",
        weights=(weight,),
        bias=0.0,
        threshold=0.5,
        model_id=model_id,
        layer=layer,
        position=position,
    ).save(path)


def test_matrix_run_persists_raw_trials_and_emits_summary(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    _write_minimal_matrix(manifest)

    monkeypatch.setattr("agent_defense.experiments.run_hf_agentdojo_case", _fake_matrix_case)
    result = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--skip-warmup"],
    )

    assert result.exit_code == 0, result.output
    rows = _read_rows(output)
    assert len(rows) == 2
    assert all(row["manifest_sha256"] for row in rows)
    assert all(row["run_fingerprint"] for row in rows)
    summary = json.loads(result.output)["summary"]["by_defense"]["none"]
    assert summary["bu"] == 1.0
    assert summary["ua"] == 1.0
    assert summary["targeted_asr"] == 1.0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("run_fingerprint", None, "fingerprint"),
        ("run_fingerprint", "0" * 64, "fingerprint"),
        ("manifest_sha256", None, "manifest"),
    ],
)
def test_matrix_resume_rejects_missing_or_wrong_provenance(
    tmp_path,
    monkeypatch,
    field: str,
    replacement: str | None,
    message: str,
) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    _write_minimal_matrix(manifest)
    monkeypatch.setattr("agent_defense.experiments.run_hf_agentdojo_case", _fake_matrix_case)
    first = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--skip-warmup"],
    )
    assert first.exit_code == 0, first.output

    rows = _read_rows(output)
    if replacement is None:
        rows[0].pop(field)
    else:
        rows[0][field] = replacement
    _write_rows(output, rows)
    monkeypatch.setattr(
        "agent_defense.experiments.run_hf_agentdojo_case",
        lambda *args, **kwargs: pytest.fail("resume validation must precede the runner"),
    )

    resumed = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--resume", "--skip-warmup"],
    )

    assert resumed.exit_code != 0
    assert message in resumed.output.casefold()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("case_id", "foreign-case"),
        ("scenario", "attacked"),
        ("suite", "foreign-suite"),
        ("benchmark_version", "v0.0.0"),
        ("user_task_id", "user_task_999"),
        ("injection_task_id", "injection_task_999"),
        ("attack", "foreign-attack"),
        ("seed", 99),
        ("seed", False),
        ("defense", "activation_probe"),
        ("attacked", True),
    ],
)
def test_matrix_resume_rejects_trial_identity_mismatch(
    tmp_path,
    monkeypatch,
    field: str,
    replacement,
) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    _write_minimal_matrix(manifest)
    monkeypatch.setattr("agent_defense.experiments.run_hf_agentdojo_case", _fake_matrix_case)
    first = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--skip-warmup"],
    )
    assert first.exit_code == 0, first.output

    rows = _read_rows(output)
    rows[0][field] = replacement
    _write_rows(output, rows)

    resumed = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--resume", "--skip-warmup"],
    )

    assert resumed.exit_code != 0
    assert "identity" in resumed.output.casefold()


def test_matrix_resume_fingerprint_binds_effective_backend_without_leaking_path(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    model_a = tmp_path / "private-a" / "model-a"
    model_b = tmp_path / "private-b" / "model-b"
    _write_minimal_matrix(manifest)
    monkeypatch.setattr("agent_defense.experiments.run_hf_agentdojo_case", _fake_matrix_case)
    first = CliRunner().invoke(
        main,
        [
            "matrix-run",
            str(manifest),
            str(output),
            "--model",
            str(model_a),
            "--skip-warmup",
        ],
    )
    assert first.exit_code == 0, first.output
    assert str(model_a) not in output.read_text(encoding="utf-8")

    resumed = CliRunner().invoke(
        main,
        [
            "matrix-run",
            str(manifest),
            str(output),
            "--model",
            str(model_b),
            "--resume",
            "--skip-warmup",
        ],
    )

    assert resumed.exit_code != 0
    assert "fingerprint" in resumed.output.casefold()
    assert str(model_b) not in resumed.output


def test_matrix_resume_fingerprint_distinguishes_same_named_local_checkpoints(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    model_a = tmp_path / "private-a" / "same-model"
    model_b = tmp_path / "private-b" / "same-model"
    for model, marker in ((model_a, "a"), (model_b, "b")):
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({"marker": marker}), encoding="utf-8")
        (model / "model.safetensors").write_bytes(marker.encode() * (128 * 1024))
    _write_minimal_matrix(manifest)
    monkeypatch.setattr("agent_defense.experiments.run_hf_agentdojo_case", _fake_matrix_case)

    first = CliRunner().invoke(
        main,
        [
            "matrix-run",
            str(manifest),
            str(output),
            "--model",
            str(model_a),
            "--skip-warmup",
        ],
    )
    assert first.exit_code == 0, first.output
    assert str(model_a) not in output.read_text(encoding="utf-8")

    resumed = CliRunner().invoke(
        main,
        [
            "matrix-run",
            str(manifest),
            str(output),
            "--model",
            str(model_b),
            "--resume",
            "--skip-warmup",
        ],
    )

    assert resumed.exit_code != 0
    assert "fingerprint" in resumed.output.casefold()
    assert str(model_b) not in resumed.output


def test_matrix_identity_read_error_does_not_leak_local_path(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    private_path = "/private/checkpoints/model.safetensors"
    _write_minimal_matrix(manifest)
    monkeypatch.setattr(
        "agent_defense.cli._matrix_backend_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError(private_path)),
    )

    result = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--skip-warmup"],
    )

    assert result.exit_code != 0
    assert "PermissionError" in result.output
    assert private_path not in result.output
    assert not output.exists()


def test_matrix_rejects_local_artifact_content_mismatch_before_any_trial(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    artifact_path = tmp_path / "probe.json"
    model_a = tmp_path / "private-a" / "same-model"
    model_b = tmp_path / "private-b" / "same-model"
    for model, marker in ((model_a, "a"), (model_b, "b")):
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({"marker": marker}), encoding="utf-8")
        (model / "model.safetensors").write_bytes(marker.encode() * (128 * 1024))
    DetectorArtifact(
        kind="linear_probe",
        weights=(1.0,),
        bias=0.0,
        threshold=0.5,
        model_id="same-model",
        layer=1,
        position="generation_prefill_last_nonpad",
        metadata={
            "activation_compatibility": {"checkpoint_content_id": _local_checkpoint_content_id(str(model_a))}
        },
    ).save(artifact_path)
    _write_minimal_matrix(
        manifest,
        defenses=[
            {"name": "none"},
            {"name": "activation_probe", "artifact_path": artifact_path.name},
        ],
    )
    monkeypatch.setattr(
        "agent_defense.experiments.run_hf_agentdojo_case",
        lambda *args, **kwargs: pytest.fail("artifact identity must fail before the first trial"),
    )

    result = CliRunner().invoke(
        main,
        [
            "matrix-run",
            str(manifest),
            str(output),
            "--model",
            str(model_b),
            "--skip-warmup",
        ],
    )

    assert result.exit_code != 0
    assert "checkpoint_content_id" in result.output
    assert not output.exists()
    assert str(model_b) not in result.output


def test_matrix_resume_fingerprint_binds_artifact_content(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    artifact = tmp_path / "probe.json"
    _write_probe_artifact(artifact, weight=1.0)
    _write_minimal_matrix(
        manifest,
        defenses=[
            {"name": "none"},
            {"name": "activation_probe", "artifact_path": artifact.name},
        ],
    )
    monkeypatch.setattr("agent_defense.experiments.run_hf_agentdojo_case", _fake_matrix_case)
    first = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--skip-warmup"],
    )
    assert first.exit_code == 0, first.output

    _write_probe_artifact(artifact, weight=2.0)
    resumed = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--resume", "--skip-warmup"],
    )

    assert resumed.exit_code != 0
    assert "fingerprint" in resumed.output.casefold()


@pytest.mark.parametrize(
    ("artifact_layer", "artifact_position", "artifact_model_id", "message"),
    [
        (7, "generation_prefill_last_nonpad", "org/model", "layer"),
        (1, "function_call_end", "org/model", "position"),
        (1, "generation_prefill_last_nonpad", "org/other", "model_id"),
    ],
)
def test_matrix_rejects_artifact_capture_mismatch_before_runner(
    tmp_path,
    monkeypatch,
    artifact_layer: int,
    artifact_position: str,
    artifact_model_id: str,
    message: str,
) -> None:
    manifest = tmp_path / "matrix.json"
    output = tmp_path / "results.jsonl"
    artifact = tmp_path / "probe.json"
    _write_probe_artifact(
        artifact,
        layer=artifact_layer,
        position=artifact_position,
        model_id=artifact_model_id,
    )
    _write_minimal_matrix(
        manifest,
        defenses=[
            {"name": "none"},
            {"name": "activation_probe", "artifact_path": artifact.name},
        ],
    )
    monkeypatch.setattr(
        "agent_defense.experiments.run_hf_agentdojo_case",
        lambda *args, **kwargs: pytest.fail("artifact validation must precede the runner"),
    )

    result = CliRunner().invoke(
        main,
        ["matrix-run", str(manifest), str(output), "--skip-warmup"],
    )

    assert result.exit_code != 0
    assert "artifact" in result.output.casefold()
    assert message in result.output.casefold()
    assert not output.exists()


def _pending_cli_review_row(
    *,
    trial_id: str,
    defense: str,
    attacked: bool,
    decisions: tuple[tuple[str, bool], ...],
) -> dict:
    row = _fake_matrix_case(
        "model",
        defense=defense,
        attacked=attacked,
        seed=0,
    )
    trace = [{"decision": decision, "valid": valid} for decision, valid in decisions]
    row.update(
        {
            "trial_id": trial_id,
            "case_id": "review-case",
            "scenario": "attacked" if attacked else "clean",
            "trace": trace,
            "tool_calls_proposed": None if defense == "repeat_user_prompt" else len(trace),
            "tool_calls_blocked": (
                None
                if defense == "repeat_user_prompt"
                else sum(decision == "block" for decision, _ in decisions)
            ),
            "malicious_tool_proposals": None,
            "malicious_tool_blocks": None,
            "valid_malicious_tool_blocks": None,
            "normal_tool_calls_blocked": None,
            "call_label_status": "not_reviewed",
            "manifest_sha256": "manifest-sha",
            "run_fingerprint": "run-fingerprint",
        }
    )
    return row


def _call_review_counts() -> dict[str, int]:
    return {
        "malicious_tool_proposals": 1,
        "malicious_tool_blocks": 1,
        "valid_malicious_tool_blocks": 1,
        "normal_tool_calls_blocked": 0,
    }


def test_matrix_apply_reviews_writes_new_reviewed_jsonl_and_preserves_provenance(tmp_path) -> None:
    raw = tmp_path / "raw.jsonl"
    review = tmp_path / "reviews.json"
    output = tmp_path / "reviewed.jsonl"
    custom = _pending_cli_review_row(
        trial_id="custom-attacked",
        defense="activation_probe",
        attacked=True,
        decisions=(("block", True),),
    )
    builtin = _pending_cli_review_row(
        trial_id="builtin-clean",
        defense="repeat_user_prompt",
        attacked=False,
        decisions=(),
    )
    _write_rows(raw, [custom, builtin])
    review.write_text(
        json.dumps({"custom-attacked": _call_review_counts()}, sort_keys=True),
        encoding="utf-8",
    )
    expected_review_sha = hashlib.sha256(review.read_bytes()).hexdigest()

    result = CliRunner().invoke(
        main,
        ["matrix-apply-reviews", str(raw), str(review), str(output)],
    )

    assert result.exit_code == 0, result.output
    rows = _read_rows(output)
    assert rows[0]["call_label_status"] == "reviewed"
    assert rows[0]["review_manifest_sha256"] == expected_review_sha
    assert rows[0]["manifest_sha256"] == custom["manifest_sha256"]
    assert rows[0]["run_fingerprint"] == custom["run_fingerprint"]
    assert rows[1] == _read_rows(raw)[1]
    payload = json.loads(result.output)
    assert payload["reviewed_custom_trials"] == 1
    summarized = CliRunner().invoke(main, ["matrix-summarize", str(output)])
    assert summarized.exit_code == 0, summarized.output
    probe = json.loads(summarized.output)["by_defense"]["activation_probe"]
    assert probe["call_interception_rate"] == 1.0


def test_matrix_apply_reviews_rejects_duplicate_review_trial_id(tmp_path) -> None:
    raw = tmp_path / "raw.jsonl"
    review = tmp_path / "reviews.json"
    output = tmp_path / "reviewed.jsonl"
    custom = _pending_cli_review_row(
        trial_id="custom-attacked",
        defense="activation_probe",
        attacked=True,
        decisions=(("block", True),),
    )
    _write_rows(raw, [custom])
    counts = json.dumps(_call_review_counts(), sort_keys=True)
    review.write_text(
        f'{{"custom-attacked":{counts},"custom-attacked":{counts}}}',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["matrix-apply-reviews", str(raw), str(review), str(output)],
    )

    assert result.exit_code != 0
    assert "invalid review manifest json" in result.output.casefold()
    assert not output.exists()


def test_matrix_apply_reviews_rejects_in_place_or_existing_output(tmp_path) -> None:
    raw = tmp_path / "raw.jsonl"
    review = tmp_path / "reviews.json"
    existing = tmp_path / "existing.jsonl"
    custom = _pending_cli_review_row(
        trial_id="custom-attacked",
        defense="activation_probe",
        attacked=True,
        decisions=(("block", True),),
    )
    _write_rows(raw, [custom])
    review.write_text(json.dumps({"custom-attacked": _call_review_counts()}), encoding="utf-8")

    for output in (raw, review):
        result = CliRunner().invoke(
            main,
            ["matrix-apply-reviews", str(raw), str(review), str(output)],
        )
        assert result.exit_code != 0
        assert "new file" in result.output.casefold()

    existing.write_text("sentinel", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        ["matrix-apply-reviews", str(raw), str(review), str(existing)],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output.casefold()
    assert existing.read_text(encoding="utf-8") == "sentinel"


def test_matrix_summarize_rejects_nonfinite_jsonl(tmp_path) -> None:
    results = tmp_path / "bad.jsonl"
    results.write_text('{"elapsed_ms": NaN}\n', encoding="utf-8")

    result = CliRunner().invoke(main, ["matrix-summarize", str(results)])

    assert result.exit_code != 0
    assert "Invalid JSONL" in result.output
