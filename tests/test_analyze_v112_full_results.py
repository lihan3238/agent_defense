from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analyze_v112_full_results.py"
_SPEC = importlib.util.spec_from_file_location("analyze_v112_full_results", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ANALYZER
_SPEC.loader.exec_module(_ANALYZER)

AnalysisError = _ANALYZER.AnalysisError
analyze_rows = _ANALYZER.analyze_rows
main = _ANALYZER.main


def _trace(
    *,
    decision: str,
    exact: bool = False,
    valid: bool = True,
    detail_error: str | None = None,
) -> dict:
    return {
        "decision": decision,
        "valid": valid,
        "syntactic_attack_reference_match": exact,
        "details": {} if detail_error is None else {"error": detail_error},
    }


def _row(
    pair: str,
    defense: str,
    scenario: str,
    *,
    suite: str,
    valid: bool,
    utility: bool,
    attack_goal: bool = False,
    injection_task_id: str | None = None,
    failure_bucket: str | None = None,
    trace: list[dict] | None = None,
    elapsed_ms: float = 10.0,
    model_query_count: int | None = 1,
    masked_reexecution_count: int | None = 0,
) -> dict:
    trace = [] if trace is None else trace
    attacked = scenario == "attacked"
    exact_trace = [item for item in trace if item["syntactic_attack_reference_match"]]
    return {
        "trial_id": f"{pair}-{defense}",
        "case_id": pair,
        "scenario": scenario,
        "suite": suite,
        "benchmark_version": "v1.1.2",
        "user_task_id": f"user_task_{pair[-1]}",
        "injection_task_id": injection_task_id if attacked else None,
        "attack": "important_instructions" if attacked else None,
        "defense": defense,
        "seed": 0,
        "attacked": attacked,
        "valid": valid,
        "failure_bucket": failure_bucket,
        "utility_passed": utility,
        "attack_goal_achieved": attack_goal,
        "elapsed_ms": elapsed_ms,
        "detector_latency_ms": 2.0 if defense == "melon_paper" else 0.0,
        "model_generate_elapsed_ms": elapsed_ms - 1.0,
        "model_query_count": model_query_count,
        "primary_model_query_count": 1 if model_query_count is not None else None,
        "extra_forward_count": 0,
        "auxiliary_detector_call_count": 1 if defense == "melon_paper" else 0,
        "masked_reexecution_count": masked_reexecution_count,
        "masked_reexecution_elapsed_ms": (3.0 if masked_reexecution_count else 0.0),
        "melon_generated_candidate_count": 1 if masked_reexecution_count else 0,
        "melon_no_candidate_reexecution_count": 0,
        "trace": trace,
        "tool_calls_proposed": len(trace),
        "tool_calls_blocked": sum(item["decision"] == "block" for item in trace),
        "syntactic_attack_reference_tool_proposals": len(exact_trace) if attacked else None,
        "syntactic_attack_reference_tool_blocks": (
            sum(item["decision"] == "block" for item in exact_trace) if attacked else None
        ),
    }


def _fixture_rows() -> list[dict]:
    return [
        _row(
            "bank-clean-1",
            "none",
            "clean",
            suite="banking",
            valid=True,
            utility=False,
            elapsed_ms=10.0,
        ),
        _row(
            "bank-clean-1",
            "melon_paper",
            "clean",
            suite="banking",
            valid=True,
            utility=True,
            trace=[_trace(decision="block")],
            elapsed_ms=20.0,
            model_query_count=2,
            masked_reexecution_count=1,
        ),
        _row(
            "bank-attack-2",
            "none",
            "attacked",
            suite="banking",
            valid=True,
            utility=True,
            attack_goal=True,
            injection_task_id="injection_task_1",
            trace=[_trace(decision="allow", exact=True)],
            elapsed_ms=12.0,
        ),
        _row(
            "bank-attack-2",
            "melon_paper",
            "attacked",
            suite="banking",
            valid=True,
            utility=False,
            injection_task_id="injection_task_1",
            trace=[
                _trace(decision="block", exact=True),
                _trace(decision="block", exact=False),
            ],
            elapsed_ms=22.0,
            model_query_count=2,
            masked_reexecution_count=1,
        ),
        _row(
            "travel-attack-3",
            "none",
            "attacked",
            suite="travel",
            valid=True,
            utility=False,
            injection_task_id="injection_task_6",
            trace=[_trace(decision="allow")],
            elapsed_ms=15.0,
        ),
        _row(
            "travel-attack-3",
            "melon_paper",
            "attacked",
            suite="travel",
            valid=False,
            utility=False,
            injection_task_id="injection_task_6",
            failure_bucket="tool_parse_error",
            trace=[
                _trace(
                    decision="block",
                    valid=False,
                    detail_error="masked_tool_parse_error",
                )
            ],
            elapsed_ms=25.0,
            model_query_count=2,
            masked_reexecution_count=1,
        ),
        _row(
            "travel-clean-4",
            "none",
            "clean",
            suite="travel",
            valid=False,
            utility=False,
            failure_bucket="tool_parse_error",
            elapsed_ms=14.0,
            model_query_count=None,
            masked_reexecution_count=None,
        ),
        _row(
            "travel-clean-4",
            "melon_paper",
            "clean",
            suite="travel",
            valid=True,
            utility=True,
            elapsed_ms=24.0,
            model_query_count=2,
            masked_reexecution_count=1,
        ),
    ]


def test_analyzes_coverage_outcomes_pairs_calls_failures_and_travel_slice() -> None:
    result = analyze_rows(_fixture_rows())

    overall = result["groups"]["overall"]
    assert overall["coverage"]["all"] == {
        "planned_trials": 8,
        "valid_trials": 6,
        "invalid_trials": 2,
        "valid_coverage": 0.75,
    }
    assert result["groups"]["by_defense"]["none"]["outcomes"]["bu"] == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    assert result["groups"]["by_defense"]["none"]["outcomes"]["ua"]["rate"] == 0.5
    assert result["groups"]["by_defense"]["none"]["outcomes"]["targeted_asr"]["rate"] == 0.5

    paired = result["paired_common_valid"]["overall"]
    assert paired["planned_pairs"] == 4
    assert paired["validity_transition"] == {
        "both_valid": 2,
        "baseline_only_valid": 1,
        "candidate_only_valid": 1,
        "neither_valid": 0,
    }
    assert paired["utility_transition"]["all_common_valid"]["matrix"] == {
        "baseline_false": {"candidate_false": 0, "candidate_true": 1},
        "baseline_true": {"candidate_false": 1, "candidate_true": 0},
    }
    assert paired["targeted_asr_transition"]["matrix"]["baseline_true"] == {
        "candidate_false": 1,
        "candidate_true": 0,
    }
    assert paired["candidate_minus_baseline"]["elapsed_ms"]["median"] == 10.0

    melon_calls = result["groups"]["by_defense"]["melon_paper"]["automatic_call_statistics"]
    assert melon_calls["all_trials"]["automatic_exact_syntactic_attack_reference"] == {
        "proposals": 1,
        "blocks": 1,
        "valid_detector_blocks": 1,
        "invalid_detector_blocks": 0,
        "block_rate": 1.0,
    }
    assert melon_calls["all_trials"]["clean"]["tool_blocks"] == 1
    assert melon_calls["all_trials"]["attacked_non_reference"]["tool_blocks"] == 2
    assert melon_calls["valid_trials"]["attacked_non_reference"]["tool_blocks"] == 1

    failures = overall["failures"]
    assert failures["declared_failure_buckets"] == {"tool_parse_error": 2}
    assert failures["resolved_failure_buckets"] == {
        "masked_tool_parse_error": 1,
        "other_tool_parse_error": 1,
    }
    assert failures["parse_split"]["masked_tool_parse_error"] == 1
    assert failures["parse_split"]["other_tool_parse_error"] == 1

    performance = result["groups"]["by_defense"]["melon_paper"]["performance"]["valid_trials"]
    assert performance["model_query_count"]["observed"] == 3
    assert performance["masked_reexecution_count"]["sum"] == 3.0
    assert result["travel_injection_task_6"]["summary"]["coverage"]["all"]["planned_trials"] == 2
    assert result["travel_injection_task_6"]["paired_common_valid"]["both_valid_pairs"] == 0
    assert "human call labeling" in result["semantics"]["automatic_attack_reference_match"]


def test_cli_writes_deterministic_json_without_overwriting(tmp_path: Path) -> None:
    merged = tmp_path / "merged.jsonl"
    merged.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _fixture_rows()),
        encoding="utf-8",
    )
    output = tmp_path / "analysis.json"

    assert main([str(merged), "--output-json", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["input"]["trial_count"] == 8
    first_content = output.read_text(encoding="utf-8")
    assert main([str(merged), "--output-json", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == first_content


def test_rejects_incomplete_pair() -> None:
    rows = _fixture_rows()
    rows.pop()

    with pytest.raises(AnalysisError, match="incomplete defense pairs"):
        analyze_rows(rows)
