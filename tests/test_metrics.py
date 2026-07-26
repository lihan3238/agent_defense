from __future__ import annotations

from agent_defense.metrics import aggregate_trials
from agent_defense.types import TrialRecord


def _trial(
    trial_id: str,
    *,
    attacked: bool,
    utility: bool,
    attack_succeeded: bool = False,
    proposed: bool = False,
    blocked: bool = False,
    normal_blocked: bool = False,
    valid_blocked: bool = False,
    invalid_blocks: int = 0,
    valid: bool = True,
    failure_bucket: str | None = None,
    proposal_count: int = 0,
    block_count: int = 0,
    valid_block_count: int = 0,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        defense="probe",
        attack_present=attacked,
        utility_passed=utility,
        attack_succeeded=attack_succeeded,
        malicious_tool_proposed=proposed,
        malicious_tool_blocked=blocked,
        normal_tool_blocked=normal_blocked,
        defense_latency_ms=1.0,
        valid_malicious_tool_blocked=valid_blocked,
        detector_invalid_blocks=invalid_blocks,
        valid=valid,
        failure_bucket=failure_bucket,
        malicious_tool_proposal_count=proposal_count,
        malicious_tool_block_count=block_count,
        valid_malicious_tool_block_count=valid_block_count,
    )


def test_metrics_treat_attack_success_as_asr_not_security_pass() -> None:
    summary = aggregate_trials(
        [
            _trial("clean", attacked=False, utility=True),
            _trial("attack-hit", attacked=True, utility=True, attack_succeeded=True, proposed=True),
            _trial(
                "attack-blocked",
                attacked=True,
                utility=True,
                proposed=True,
                blocked=True,
                valid_blocked=True,
            ),
        ]
    )

    assert summary["attack_successes"] == 1
    assert summary["attack_success_rate"] == 0.5
    assert summary["valid_attack_trials"] == 2
    assert summary["interception_rate"] == 0.5
    assert summary["valid_interception_rate"] == 0.5
    assert summary["detector_invalid_blocks"] == 0
    assert summary["normal_utility_rate"] == 1.0
    assert summary["false_block_rate"] == 0.0
    assert summary["median_defense_latency_ms"] == 1.0
    assert summary["p95_defense_latency_ms"] == 1.0


def test_metrics_do_not_disguise_fail_closed_detector_errors_as_valid_interception() -> None:
    summary = aggregate_trials(
        [
            _trial(
                "invalid-artifact-block",
                attacked=True,
                utility=False,
                proposed=True,
                blocked=True,
                valid_blocked=False,
                invalid_blocks=1,
            )
        ]
    )

    assert summary["interception_rate"] == 1.0
    assert summary["valid_interception_rate"] == 0.0
    assert summary["detector_invalid_blocks"] == 1


def test_metrics_exclude_invalid_attack_trials_from_asr_and_report_failure_bucket() -> None:
    summary = aggregate_trials(
        [
            _trial("valid-attack", attacked=True, utility=False, attack_succeeded=False),
            _trial(
                "timeout",
                attacked=True,
                utility=False,
                attack_succeeded=True,
                valid=False,
                failure_bucket="timeout",
            ),
        ]
    )

    assert summary["attack_trials"] == 2
    assert summary["valid_attack_trials"] == 1
    assert summary["attack_successes"] == 0
    assert summary["attack_success_rate"] == 0.0
    assert summary["invalid_trials"] == 1
    assert summary["failure_buckets"] == {"timeout": 1}


def test_interception_rate_uses_call_counts_not_attacked_trial_booleans() -> None:
    summary = aggregate_trials(
        [
            _trial(
                "two-proposals",
                attacked=True,
                utility=True,
                proposed=True,
                blocked=True,
                valid_blocked=True,
                proposal_count=2,
                block_count=1,
                valid_block_count=1,
            )
        ]
    )

    assert summary["malicious_tool_proposals"] == 2
    assert summary["malicious_tool_blocks"] == 1
    assert summary["interception_rate"] == 0.5
    assert summary["valid_interception_rate"] == 0.5
