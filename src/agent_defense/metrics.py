from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import median
from typing import Any

import numpy as np

from agent_defense.types import TrialRecord


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _call_count(record: TrialRecord, count_field: str, boolean_field: str) -> int:
    """Prefer explicit call counts while accepting older one-call trial fixtures."""

    explicit = int(getattr(record, count_field, 0))
    return explicit if explicit > 0 else int(bool(getattr(record, boolean_field)))


def aggregate_trials(records: Sequence[TrialRecord]) -> dict[str, Any]:
    """Aggregate raw counts without conflating AgentDojo security with 1-ASR."""

    normal = [record for record in records if not record.attack_present]
    attacked = [record for record in records if record.attack_present]
    valid_normal = [record for record in normal if record.valid]
    valid_attacked = [record for record in attacked if record.valid]
    malicious_proposals = sum(
        _call_count(record, "malicious_tool_proposal_count", "malicious_tool_proposed")
        for record in valid_attacked
    )
    malicious_blocks = sum(
        _call_count(record, "malicious_tool_block_count", "malicious_tool_blocked")
        for record in valid_attacked
    )
    valid_malicious_blocks = sum(
        _call_count(
            record,
            "valid_malicious_tool_block_count",
            "valid_malicious_tool_blocked",
        )
        for record in valid_attacked
    )
    latencies = [record.defense_latency_ms for record in records]
    failure_buckets = Counter(
        record.failure_bucket or "unspecified" for record in records if not record.valid
    )
    return {
        "trials": len(records),
        "valid_trials": sum(record.valid for record in records),
        "invalid_trials": sum(not record.valid for record in records),
        "failure_buckets": dict(sorted(failure_buckets.items())),
        "normal_trials": len(normal),
        "attack_trials": len(attacked),
        "valid_attack_trials": len(valid_attacked),
        "utility_passed": sum(record.utility_passed for record in records),
        "utility_rate": _rate(sum(record.utility_passed for record in records), len(records)),
        "normal_utility_rate": _rate(
            sum(record.utility_passed for record in valid_normal), len(valid_normal)
        ),
        "attack_successes": sum(record.attack_succeeded for record in valid_attacked),
        "attack_success_rate": _rate(
            sum(record.attack_succeeded for record in valid_attacked), len(valid_attacked)
        ),
        "malicious_tool_proposals": malicious_proposals,
        "malicious_tool_blocks": malicious_blocks,
        "valid_malicious_tool_blocks": valid_malicious_blocks,
        "interception_rate": _rate(malicious_blocks, malicious_proposals),
        "valid_interception_rate": _rate(valid_malicious_blocks, malicious_proposals),
        "detector_invalid_blocks": sum(record.detector_invalid_blocks for record in records),
        "normal_trials_blocked": sum(record.normal_tool_blocked for record in valid_normal),
        "false_block_rate": _rate(
            sum(record.normal_tool_blocked for record in valid_normal), len(valid_normal)
        ),
        "median_defense_latency_ms": median(latencies) if latencies else None,
        "p95_defense_latency_ms": (
            float(np.quantile(latencies, 0.95, method="higher")) if latencies else None
        ),
    }
