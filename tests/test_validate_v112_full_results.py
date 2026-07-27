from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agent_defense.matrix import expand_trials, load_manifest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_v112_full_results.py"
_SPEC = importlib.util.spec_from_file_location("validate_v112_full_results", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VALIDATOR
_SPEC.loader.exec_module(_VALIDATOR)

EXPECTED_MANIFEST_NAMES = _VALIDATOR.EXPECTED_MANIFEST_NAMES
ValidationError = _VALIDATOR.ValidationError
validate_full_results = _VALIDATOR.validate_full_results
write_outputs = _VALIDATOR.write_outputs

SUITE_LAYOUT = {
    "banking": (16, 9),
    "slack": (21, 5),
    "travel": (20, 7),
    "workspace": (40, 6),
}


def _cases(suite: str, user_count: int, injection_count: int) -> list[dict]:
    cases = []
    for user_index in range(1, user_count + 1):
        cases.append(
            {
                "case_id": f"{suite}-clean-u{user_index}",
                "user_task_id": f"user_task_{user_index}",
                "injection_task_id": None,
                "seeds": [0],
                "scenarios": ["clean"],
            }
        )
    for user_index in range(1, user_count + 1):
        for injection_index in range(1, injection_count + 1):
            cases.append(
                {
                    "case_id": f"{suite}-attacked-u{user_index}-i{injection_index}",
                    "user_task_id": f"user_task_{user_index}",
                    "injection_task_id": f"injection_task_{injection_index}",
                    "seeds": [0],
                    "scenarios": ["attacked"],
                }
            )
    return cases


def _manifest(suite: str, cases: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "model": {
            "model_id_or_path": "Qwen/Qwen3-8B",
            "revision": "local-snapshot",
            "layer": 22,
            "position": "tool_input",
            "device": "auto",
            "dtype": "bfloat16",
            "max_new_tokens": 256,
            "disable_thinking": True,
            "local_files_only": True,
        },
        "benchmark": {
            "suite_name": suite,
            "benchmark_version": "v1.1.2",
            "attack_name": "important_instructions",
        },
        "defenses": [
            {"name": "none"},
            {
                "name": "melon_paper",
                "melon_threshold": 0.8,
                "melon_embedding_backend": "hf",
                "melon_embedding_model": "models/all-MiniLM-L6-v2",
                "melon_embedding_device": "cpu",
            },
        ],
        "cases": cases,
    }


def _result_row(spec, *, manifest_sha256: str, run_fingerprint: str) -> dict:
    attacked = spec.scenario == "attacked"
    return {
        "trial_id": spec.trial_id,
        "case_id": spec.case_id,
        "scenario": spec.scenario,
        "suite": spec.benchmark.suite_name,
        "benchmark_version": spec.benchmark.benchmark_version,
        "user_task_id": spec.user_task_id,
        "injection_task_id": spec.injection_task_id if attacked else None,
        "attack": spec.benchmark.attack_name if attacked else None,
        "defense": spec.defense.name,
        "seed": spec.seed,
        "attacked": attacked,
        "status": "ok",
        "valid": True,
        "utility_passed": True,
        "attack_goal_achieved": False,
        "elapsed_ms": 1.0,
        "detector_latency_ms": 0.0,
        "model_query_count": 1,
        "extra_forward_count": 1 if spec.defense.name == "melon_paper" else 0,
        "trace": [],
        "tool_calls_proposed": 0,
        "tool_calls_blocked": 0,
        "manifest_sha256": manifest_sha256,
        "run_fingerprint": run_fingerprint,
    }


def _write_raw_for_manifest(manifest_path: Path, raw_path: Path) -> list[dict]:
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    run_fingerprint = hashlib.sha256(f"run:{manifest_path.name}".encode()).hexdigest()
    rows = [
        _result_row(
            spec,
            manifest_sha256=manifest_sha256,
            run_fingerprint=run_fingerprint,
        )
        for spec in expand_trials(load_manifest(manifest_path))
    ]
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _full_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifests_dir = tmp_path / "manifests"
    raw_dir = tmp_path / "raw"
    manifests_dir.mkdir()
    raw_dir.mkdir()
    for suite, (user_count, injection_count) in SUITE_LAYOUT.items():
        cases = _cases(suite, user_count, injection_count)
        for shard in range(4):
            manifest_path = manifests_dir / f"{suite}-shard-{shard}.json"
            manifest_path.write_text(
                json.dumps(_manifest(suite, cases[shard::4]), indent=2) + "\n",
                encoding="utf-8",
            )
            raw_path = raw_dir / f"{manifest_path.stem}.raw.jsonl"
            _write_raw_for_manifest(manifest_path, raw_path)
    assert {path.name for path in manifests_dir.glob("*.json")} == set(EXPECTED_MANIFEST_NAMES)
    return manifests_dir, raw_dir


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validates_fixed_full_matrix_and_writes_deterministic_outputs(tmp_path: Path) -> None:
    manifests_dir, raw_dir = _full_fixture(tmp_path)

    rows, summary = validate_full_results(manifests_dir, raw_dir)

    assert len(rows) == 1452
    assert summary["status"] == "validated"
    assert summary["manifest_count"] == 16
    assert summary["semantic_identity_count"] == 1452
    assert summary["by_defense"] == {
        "none": {"clean": 97, "attacked": 629, "total": 726},
        "melon_paper": {"clean": 97, "attacked": 629, "total": 726},
    }
    merged = tmp_path / "out" / "merged.jsonl"
    summary_path = tmp_path / "out" / "summary.json"
    write_outputs(rows, summary, merged_jsonl=merged, summary_json=summary_path)
    assert len(_read_rows(merged)) == 1452
    assert json.loads(summary_path.read_text(encoding="utf-8"))["trial_count"] == 1452


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "incomplete"),
        ("manifest_sha256", "manifest SHA256"),
        ("run_fingerprint", "exactly one run_fingerprint"),
        ("identity", "identity field"),
    ],
)
def test_rejects_incomplete_mixed_or_misidentified_shard(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    manifests_dir, raw_dir = _full_fixture(tmp_path)
    raw_path = raw_dir / "banking-shard-0.raw.jsonl"
    rows = _read_rows(raw_path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "manifest_sha256":
        rows[0]["manifest_sha256"] = "0" * 64
    elif mutation == "run_fingerprint":
        rows[0]["run_fingerprint"] = "f" * 64
    else:
        rows[0]["seed"] = False
    _write_rows(raw_path, rows)

    with pytest.raises(ValidationError, match=message):
        validate_full_results(manifests_dir, raw_dir)


def test_rejects_duplicate_semantic_identity_across_shards(tmp_path: Path) -> None:
    manifests_dir, raw_dir = _full_fixture(tmp_path)
    first_manifest = manifests_dir / "banking-shard-0.json"
    duplicate_manifest = manifests_dir / "banking-shard-1.json"
    first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    duplicate_payload = json.loads(duplicate_manifest.read_text(encoding="utf-8"))
    first_clean = next(case for case in first_payload["cases"] if case["scenarios"] == ["clean"])
    duplicate_clean = next(case for case in duplicate_payload["cases"] if case["scenarios"] == ["clean"])
    duplicate_clean["user_task_id"] = first_clean["user_task_id"]
    duplicate_manifest.write_text(json.dumps(duplicate_payload, indent=2) + "\n", encoding="utf-8")
    _write_raw_for_manifest(
        duplicate_manifest,
        raw_dir / "banking-shard-1.raw.jsonl",
    )

    with pytest.raises(ValidationError, match="Duplicate semantic trial identity"):
        validate_full_results(manifests_dir, raw_dir)
