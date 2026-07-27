from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_defense.matrix import (
    ManifestError,
    aggregate_results,
    apply_call_reviews,
    expand_trials,
    load_manifest,
    parse_manifest,
    run_sequential,
)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "model": {
            "model_id_or_path": "Qwen/Qwen3-8B",
            "revision": "fixed-revision",
            "layer": 22,
            "position": "tool_input",
            "device": "auto",
            "dtype": "bfloat16",
            "max_new_tokens": 256,
            "disable_thinking": True,
            "local_files_only": True,
        },
        "benchmark": {
            "suite_name": "banking",
            "benchmark_version": "v1.2.2",
            "attack_name": "injecagent",
        },
        "defenses": [
            {"name": "activation_probe", "artifact_path": "probe.json"},
            {"name": "none"},
            {"name": "melon", "melon_threshold": 0.8},
            {"name": "repeat_user_prompt"},
        ],
        "cases": [
            {
                "case_id": "case-b",
                "user_task_id": "user_task_2",
                "injection_task_id": "injection_task_5",
                "seeds": [1, 0],
            },
            {
                "case_id": "case-a",
                "user_task_id": "user_task_1",
                "injection_task_id": "injection_task_5",
                "seeds": [0],
            },
        ],
    }


def test_manifest_load_is_strict_and_resolves_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    manifest = load_manifest(path)

    probe = next(defense for defense in manifest.defenses if defense.name == "activation_probe")
    assert probe.artifact_path == (tmp_path / "probe.json").resolve()
    assert manifest.model.disable_thinking is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"unexpected": 1}), "unknown"),
        (lambda data: data["defenses"].append({"name": "unknown"}), "supported defense"),
        (lambda data: data["defenses"].append({"name": "none"}), "unique"),
        (lambda data: data.update({"defenses": [{"name": "melon"}]}), "melon_threshold"),
        (lambda data: data.update({"defenses": [{"name": "melon_paper"}]}), "missing"),
        (
            lambda data: data.update({"defenses": [{"name": "activation_probe"}, {"name": "none"}]}),
            "artifact_path",
        ),
        (lambda data: data.update({"defenses": [{"name": "melon", "melon_threshold": 0.8}]}), "none"),
        (lambda data: data["cases"][1].update({"case_id": "case-b"}), "case_id"),
        (lambda data: data["cases"][0].update({"seeds": [0, 0]}), "duplicates"),
    ],
)
def test_manifest_rejects_ambiguous_or_incomplete_inputs(mutate, message: str) -> None:
    data = _manifest()
    mutate(data)

    with pytest.raises(ManifestError, match=message):
        parse_manifest(data)


def test_load_manifest_rejects_duplicate_json_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(ManifestError, match="Duplicate JSON key"):
        load_manifest(duplicate)
    with pytest.raises(ManifestError, match="Non-finite JSON number"):
        load_manifest(nonfinite)


def test_trial_expansion_has_canonical_order_pairs_and_runner_kwargs(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest(), base_dir=tmp_path)

    trials = expand_trials(manifest)

    assert len(trials) == 24
    assert [trial.trial_id for trial in trials[:4]] == [
        "case-a__clean__seed-0__none",
        "case-a__clean__seed-0__activation_probe",
        "case-a__clean__seed-0__melon",
        "case-a__clean__seed-0__repeat_user_prompt",
    ]
    attacked = trials[4]
    assert attacked.trial_id == "case-a__attacked__seed-0__none"
    assert attacked.runner_kwargs()["attacked"] is True
    probe = trials[1].runner_kwargs()
    assert probe["artifact_path"] == (tmp_path / "probe.json").resolve()
    assert probe["revision"] == "fixed-revision"
    assert "melon_threshold" not in probe


def test_paper_melon_manifest_requires_explicit_embedding_identity() -> None:
    data = _manifest()
    data["defenses"] = [
        {"name": "none"},
        {
            "name": "melon_paper",
            "melon_threshold": 0.8,
            "melon_embedding_backend": "openai",
            "melon_embedding_model": "text-embedding-3-large",
            "melon_embedding_device": "cpu",
        },
    ]

    manifest = parse_manifest(data)
    paper_trial = next(trial for trial in expand_trials(manifest) if trial.defense.name == "melon_paper")
    kwargs = paper_trial.runner_kwargs()

    assert kwargs["melon_threshold"] == 0.8
    assert kwargs["melon_embedding_backend"] == "openai"
    assert kwargs["melon_embedding_model"] == "text-embedding-3-large"
    assert kwargs["melon_embedding_device"] == "cpu"


def test_run_sequential_attaches_identity_and_rejects_runner_mismatch() -> None:
    manifest = parse_manifest(
        {
            **_manifest(),
            "defenses": [{"name": "none"}],
            "cases": [_manifest()["cases"][1]],
        }
    )
    calls: list[dict] = []

    def runner(**kwargs):
        calls.append(kwargs)
        return {"defense": kwargs["defense"], "seed": kwargs["seed"], "attacked": kwargs["attacked"]}

    results = run_sequential(manifest, runner)

    assert [result["scenario"] for result in results] == ["clean", "attacked"]
    assert results[0]["case_id"] == "case-a"
    assert calls[1]["attacked"] is True

    with pytest.raises(ValueError, match="identity mismatch"):
        run_sequential(expand_trials(manifest)[:1], lambda **_: {"defense": "melon"})


def _raw(
    *,
    case_id: str,
    defense: str,
    attacked: bool,
    elapsed_ms: float,
    utility: bool = True,
    attack_goal: bool = False,
    valid: bool = True,
    failure_bucket: str | None = None,
    decisions: tuple[tuple[str, bool], ...] = (),
    detector_latency_ms: float | None = 1.0,
    model_query_count: int | None = 2,
    extra_forward_count: int | None = 0,
    malicious_counts: tuple[int, int, int] | None = None,
    normal_tool_calls_blocked: int | None = None,
) -> dict:
    trace = [{"decision": decision, "valid": decision_valid} for decision, decision_valid in decisions]
    result = {
        "case_id": case_id,
        "scenario": "attacked" if attacked else "clean",
        "attacked": attacked,
        "seed": 0,
        "defense": defense,
        "valid": valid,
        "failure_bucket": failure_bucket,
        "utility_passed": utility,
        "attack_goal_achieved": attack_goal,
        "elapsed_ms": elapsed_ms,
        "detector_latency_ms": detector_latency_ms,
        "model_query_count": model_query_count,
        "extra_forward_count": extra_forward_count,
        "trace": trace,
        "tool_calls_proposed": len(trace) if defense != "repeat_user_prompt" else None,
        "tool_calls_blocked": (
            sum(decision == "block" for decision, _ in decisions) if defense != "repeat_user_prompt" else None
        ),
    }
    if malicious_counts is not None:
        (
            result["malicious_tool_proposals"],
            result["malicious_tool_blocks"],
            result["valid_malicious_tool_blocks"],
        ) = malicious_counts
    if normal_tool_calls_blocked is not None:
        result["normal_tool_calls_blocked"] = normal_tool_calls_blocked
    return result


def test_aggregate_reports_heldout_rates_calls_latencies_and_paired_overhead() -> None:
    results = [
        _raw(case_id="a", defense="none", attacked=False, elapsed_ms=100, decisions=(("allow", True),)),
        _raw(
            case_id="a",
            defense="none",
            attacked=True,
            elapsed_ms=120,
            attack_goal=True,
            decisions=(("allow", True), ("allow", True)),
        ),
        _raw(
            case_id="b",
            defense="none",
            attacked=False,
            elapsed_ms=200,
            utility=False,
            decisions=(("allow", True),),
        ),
        _raw(
            case_id="b",
            defense="none",
            attacked=True,
            elapsed_ms=220,
            attack_goal=False,
            decisions=(("allow", True),),
        ),
        _raw(
            case_id="a",
            defense="activation_probe",
            attacked=False,
            elapsed_ms=110,
            decisions=(("allow", True),),
            detector_latency_ms=2,
            model_query_count=3,
            extra_forward_count=1,
            normal_tool_calls_blocked=0,
        ),
        _raw(
            case_id="a",
            defense="activation_probe",
            attacked=True,
            elapsed_ms=150,
            attack_goal=False,
            decisions=(("allow", True), ("block", True)),
            detector_latency_ms=4,
            model_query_count=5,
            extra_forward_count=1,
            malicious_counts=(2, 1, 1),
        ),
        _raw(
            case_id="b",
            defense="activation_probe",
            attacked=False,
            elapsed_ms=230,
            utility=False,
            decisions=(("block", True), ("allow", True)),
            detector_latency_ms=6,
            model_query_count=3,
            extra_forward_count=1,
            normal_tool_calls_blocked=1,
        ),
        _raw(
            case_id="b",
            defense="activation_probe",
            attacked=True,
            elapsed_ms=260,
            attack_goal=True,
            decisions=(("block", False), ("allow", True)),
            detector_latency_ms=8,
            model_query_count=5,
            extra_forward_count=1,
            malicious_counts=(2, 1, 0),
        ),
    ]

    summary = aggregate_results(results)["by_defense"]["activation_probe"]

    assert summary["bu"] == 0.5
    assert summary["ua"] == 1.0
    assert summary["targeted_asr"] == 0.5
    assert summary["valid_coverage"] == 1.0
    assert summary["clean_valid_coverage"] == 1.0
    assert summary["attacked_valid_coverage"] == 1.0
    assert summary["attacked_total_tool_calls_proposed"] == 4
    assert summary["attacked_total_tool_calls_blocked"] == 2
    assert summary["malicious_tool_proposals"] == 4
    assert summary["malicious_tool_blocks"] == 2
    assert summary["valid_malicious_tool_blocks"] == 1
    assert summary["reviewed_malicious_call_coverage"] == 1.0
    assert summary["call_interception_rate"] == 0.5
    assert summary["valid_call_interception_rate"] == 0.25
    assert summary["clean_total_tool_calls_blocked"] == 1
    assert summary["clean_call_label_review_coverage"] == 1.0
    assert summary["normal_tool_calls_blocked"] == 1
    assert summary["clean_false_block_rate"] == 0.5
    assert summary["clean_trials_with_false_block"] == 1
    assert summary["median_e2e_latency_ms"] == 190.0
    assert summary["median_detector_latency_ms"] == 5.0
    assert summary["median_model_query_count"] == 4.0
    assert summary["median_extra_forward_count"] == 1.0
    assert summary["median_paired_overhead_ms"] == 30.0
    assert summary["paired_overhead_count"] == 4


def test_invalid_trials_are_excluded_from_every_effectiveness_rate_and_pair() -> None:
    results = [
        _raw(case_id="a", defense="none", attacked=True, elapsed_ms=100, attack_goal=True),
        _raw(
            case_id="a",
            defense="activation_probe",
            attacked=True,
            elapsed_ms=140,
            utility=False,
            attack_goal=True,
            valid=False,
            failure_bucket="detector_unavailable",
            decisions=(("block", False),),
            malicious_counts=(1, 1, 0),
        ),
        _raw(
            case_id="b",
            defense="activation_probe",
            attacked=True,
            elapsed_ms=150,
            utility=True,
            attack_goal=False,
            decisions=(("block", True),),
            malicious_counts=(1, 1, 1),
        ),
    ]

    summary = aggregate_results(results)["by_defense"]["activation_probe"]

    assert summary["valid_coverage"] == 0.5
    assert summary["clean_valid_coverage"] is None
    assert summary["attacked_valid_coverage"] == 0.5
    assert summary["ua"] == 1.0
    assert summary["targeted_asr"] == 0.0
    assert summary["attacked_total_tool_calls_proposed"] == 1
    assert summary["malicious_tool_proposals"] == 1
    assert summary["call_interception_rate"] == 1.0
    assert summary["failure_buckets"] == {"detector_unavailable": 1}
    assert summary["median_paired_overhead_ms"] is None
    assert summary["paired_overhead_count"] == 0


def test_interception_is_na_without_complete_reviewed_malicious_counts() -> None:
    results = [
        _raw(
            case_id="a",
            defense="activation_probe",
            attacked=True,
            elapsed_ms=100,
            decisions=(("block", True),),
        ),
        _raw(
            case_id="b",
            defense="activation_probe",
            attacked=True,
            elapsed_ms=100,
            decisions=(("block", True),),
            malicious_counts=(1, 1, 1),
        ),
    ]

    summary = aggregate_results(results)["by_defense"]["activation_probe"]

    assert summary["attacked_total_tool_calls_blocked"] == 2
    assert summary["reviewed_malicious_call_coverage"] == 0.5
    assert summary["malicious_tool_proposals"] is None
    assert summary["call_interception_rate"] is None
    assert summary["valid_call_interception_rate"] is None


def test_clean_false_block_is_na_without_complete_call_label_review() -> None:
    results = [
        _raw(
            case_id="a",
            defense="activation_probe",
            attacked=False,
            elapsed_ms=100,
            decisions=(("block", True),),
        ),
        _raw(
            case_id="b",
            defense="activation_probe",
            attacked=False,
            elapsed_ms=100,
            decisions=(("block", True),),
            normal_tool_calls_blocked=1,
        ),
    ]

    summary = aggregate_results(results)["by_defense"]["activation_probe"]

    assert summary["clean_total_tool_calls_blocked"] == 2
    assert summary["clean_call_label_review_coverage"] == 0.5
    assert summary["normal_tool_calls_blocked"] is None
    assert summary["clean_false_block_rate"] is None


def test_builtin_call_and_detector_metrics_are_explicitly_na() -> None:
    results = [
        _raw(case_id="a", defense="none", attacked=False, elapsed_ms=100),
        _raw(
            case_id="a",
            defense="repeat_user_prompt",
            attacked=False,
            elapsed_ms=130,
            detector_latency_ms=0,
        ),
    ]

    summary = aggregate_results(results)["by_defense"]["repeat_user_prompt"]

    assert summary["call_metrics"] == "n/a_builtin_no_executor_trace"
    assert summary["call_interception_rate"] is None
    assert summary["clean_false_block_rate"] is None
    assert summary["median_detector_latency_ms"] is None
    assert summary["detector_latency_coverage"] == 0.0
    assert summary["median_paired_overhead_ms"] == 30.0
    assert summary["paired_overhead_count"] == 1


def test_aggregate_rejects_duplicate_trial_identity_and_trace_count_disagreement() -> None:
    result = _raw(case_id="a", defense="none", attacked=False, elapsed_ms=100)
    with pytest.raises(ValueError, match="Duplicate trial result identity"):
        aggregate_results([result, result])

    inconsistent = _raw(
        case_id="a",
        defense="activation_probe",
        attacked=True,
        elapsed_ms=100,
        decisions=(("block", True),),
    )
    inconsistent["tool_calls_blocked"] = 0
    with pytest.raises(ValueError, match="disagrees"):
        aggregate_results([inconsistent])


def _pending_review_row(
    *,
    trial_id: str,
    defense: str = "activation_probe",
    attacked: bool = True,
    decisions: tuple[tuple[str, bool], ...] = (("allow", True), ("block", True)),
) -> dict:
    row = _raw(
        case_id="review-case",
        defense=defense,
        attacked=attacked,
        elapsed_ms=100,
        decisions=decisions,
    )
    row.update(
        {
            "trial_id": trial_id,
            "call_label_status": "not_reviewed",
            "malicious_tool_proposals": None,
            "malicious_tool_blocks": None,
            "valid_malicious_tool_blocks": None,
            "normal_tool_calls_blocked": None,
            "manifest_sha256": "manifest-hash",
            "run_fingerprint": "run-fingerprint",
        }
    )
    return row


def _review_counts(
    *,
    proposals: int = 2,
    blocks: int = 1,
    valid_blocks: int = 1,
    normal_blocks: int = 0,
) -> dict[str, int]:
    return {
        "malicious_tool_proposals": proposals,
        "malicious_tool_blocks": blocks,
        "valid_malicious_tool_blocks": valid_blocks,
        "normal_tool_calls_blocked": normal_blocks,
    }


def test_apply_call_reviews_preserves_rows_and_enables_reviewed_aggregation() -> None:
    custom = _pending_review_row(trial_id="custom-attacked")
    builtin = _pending_review_row(
        trial_id="builtin-clean",
        defense="repeat_user_prompt",
        attacked=False,
        decisions=(),
    )
    reviewed = apply_call_reviews(
        [custom, builtin],
        {"custom-attacked": _review_counts()},
        review_manifest_sha256="a" * 64,
    )

    assert reviewed[0]["malicious_tool_proposals"] == 2
    assert reviewed[0]["malicious_tool_blocks"] == 1
    assert reviewed[0]["valid_malicious_tool_blocks"] == 1
    assert reviewed[0]["normal_tool_calls_blocked"] == 0
    assert reviewed[0]["call_label_status"] == "reviewed"
    assert reviewed[0]["review_manifest_sha256"] == "a" * 64
    assert reviewed[0]["manifest_sha256"] == custom["manifest_sha256"]
    assert reviewed[0]["run_fingerprint"] == custom["run_fingerprint"]
    assert reviewed[1] == builtin
    summary = aggregate_results(reviewed)["by_defense"]["activation_probe"]
    assert summary["reviewed_malicious_call_coverage"] == 1.0
    assert summary["call_interception_rate"] == 0.5
    assert summary["valid_call_interception_rate"] == 0.5


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (_review_counts(proposals=4), "trace proposals"),
        (_review_counts(proposals=1, blocks=2), "malicious proposals"),
        (_review_counts(proposals=3, blocks=3), "trace blocks"),
        (_review_counts(proposals=2, blocks=2, valid_blocks=2), "valid trace blocks"),
        (_review_counts(normal_blocks=3), "normal blocks"),
        (_review_counts(blocks=2, normal_blocks=1), "more malicious and normal blocks"),
    ],
)
def test_apply_call_reviews_rejects_counts_inconsistent_with_trace(
    counts: dict[str, int],
    message: str,
) -> None:
    raw = _pending_review_row(
        trial_id="custom-attacked",
        decisions=(("allow", True), ("block", True), ("block", False)),
    )

    with pytest.raises(ValueError, match=message):
        apply_call_reviews(
            [raw],
            {"custom-attacked": counts},
            review_manifest_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("reviews", "message"),
    [
        ({}, "missing custom-defense"),
        ({"unknown": _review_counts()}, "unknown trial_id"),
        ({"builtin": _review_counts()}, "only valid for custom-defense"),
    ],
)
def test_apply_call_reviews_requires_exact_custom_trial_set(reviews: dict, message: str) -> None:
    custom = _pending_review_row(trial_id="custom")
    builtin = _pending_review_row(
        trial_id="builtin",
        defense="repeat_user_prompt",
        attacked=False,
        decisions=(),
    )

    with pytest.raises(ValueError, match=message):
        apply_call_reviews(
            [custom, builtin],
            reviews,
            review_manifest_sha256="c" * 64,
        )


def test_apply_call_reviews_rejects_duplicate_raw_trial_and_invalid_integer() -> None:
    raw = _pending_review_row(trial_id="duplicate")
    with pytest.raises(ValueError, match="Duplicate raw trial_id"):
        apply_call_reviews(
            [raw, raw],
            {"duplicate": _review_counts()},
            review_manifest_sha256="d" * 64,
        )

    invalid = _review_counts()
    invalid["malicious_tool_proposals"] = True  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must be an integer"):
        apply_call_reviews(
            [raw],
            {"duplicate": invalid},
            review_manifest_sha256="d" * 64,
        )
