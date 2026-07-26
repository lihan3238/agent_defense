from __future__ import annotations

import math

from agent_defense.policy import RuntimeGate
from agent_defense.types import (
    CandidateToolCall,
    DecisionAction,
    DetectionContext,
    ProbeObservation,
)


class FixedDetector:
    name = "fixed"

    def __init__(self, *, triggered: bool, valid: bool = True) -> None:
        self.triggered = triggered
        self.valid = valid

    def inspect(self, context: DetectionContext) -> ProbeObservation:
        del context
        return ProbeObservation(
            detector=self.name,
            score=0.9 if self.triggered else 0.1,
            threshold=0.5,
            triggered=self.triggered,
            valid=self.valid,
            latency_ms=0.0,
            details={} if self.valid else {"error": "missing activation"},
        )


def _context(function: str) -> DetectionContext:
    return DetectionContext(candidate=CandidateToolCall(function, {}))


def test_gate_blocks_triggered_high_impact_call() -> None:
    decision = RuntimeGate(FixedDetector(triggered=True)).decide(_context("send_money"))

    assert decision.action == DecisionAction.BLOCK
    assert decision.reason == "score_above_threshold"


def test_gate_does_not_turn_detector_into_a_blanket_read_blocker() -> None:
    decision = RuntimeGate(FixedDetector(triggered=True)).decide(_context("get_balance"))

    assert decision.action == DecisionAction.ALLOW
    assert decision.reason == "risk_below_enforcement_threshold"


def test_invalid_detector_fails_closed_only_at_enforced_risk_levels() -> None:
    gate = RuntimeGate(FixedDetector(triggered=False, valid=False), fail_closed=True)

    high = gate.decide(_context("update_password"))
    low = gate.decide(_context("list_transactions"))

    assert high.action == DecisionAction.BLOCK
    assert high.reason == "detector_unavailable"
    assert math.isnan(high.observation.score) is False
    assert low.action == DecisionAction.ALLOW
    assert low.reason == "detector_unavailable_fail_open"
