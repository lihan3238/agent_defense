from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_defense.detectors import Detector
from agent_defense.types import (
    DecisionAction,
    DetectionContext,
    PolicyDecision,
    RiskLevel,
)

DEFAULT_TOOL_RISKS: dict[str, RiskLevel] = {
    "send_money": RiskLevel.CRITICAL,
    "schedule_transaction": RiskLevel.HIGH,
    "update_scheduled_transaction": RiskLevel.HIGH,
    "update_password": RiskLevel.CRITICAL,
    "update_user_info": RiskLevel.HIGH,
    "send_email": RiskLevel.HIGH,
    "delete_email": RiskLevel.HIGH,
    "share_file": RiskLevel.HIGH,
    "write_file": RiskLevel.HIGH,
}


@dataclass(frozen=True)
class ToolRiskPolicy:
    """Deterministic action-impact classification; it is not an ML detector."""

    overrides: Mapping[str, RiskLevel] = field(default_factory=dict)

    def classify(self, function: str) -> RiskLevel:
        if function in self.overrides:
            return self.overrides[function]
        if function in DEFAULT_TOOL_RISKS:
            return DEFAULT_TOOL_RISKS[function]
        lowered = function.lower()
        if lowered.startswith(("delete", "transfer", "execute", "change_password")):
            return RiskLevel.CRITICAL
        if lowered.startswith(("send", "write", "update", "create", "schedule", "post", "share")):
            return RiskLevel.HIGH
        if lowered.startswith(("read", "get", "list", "search", "find", "check")):
            return RiskLevel.LOW
        return RiskLevel.MEDIUM


class RuntimeGate:
    """Turns a detector observation into an enforceable pre-tool-call decision."""

    def __init__(
        self,
        detector: Detector,
        *,
        risk_policy: ToolRiskPolicy | None = None,
        minimum_block_risk: RiskLevel = RiskLevel.MEDIUM,
        fail_closed: bool = True,
    ) -> None:
        self.detector = detector
        self.risk_policy = risk_policy or ToolRiskPolicy()
        self.minimum_block_risk = minimum_block_risk
        self.fail_closed = fail_closed

    @property
    def name(self) -> str:
        return self.detector.name

    def decide(self, context: DetectionContext) -> PolicyDecision:
        risk = self.risk_policy.classify(context.candidate.function)
        observation = self.detector.inspect(context)
        if not observation.valid:
            should_block = self.fail_closed and risk >= self.minimum_block_risk
            return PolicyDecision(
                action=DecisionAction.BLOCK if should_block else DecisionAction.ALLOW,
                reason="detector_unavailable" if should_block else "detector_unavailable_fail_open",
                risk=risk,
                observation=observation,
            )
        if risk < self.minimum_block_risk:
            return PolicyDecision(
                action=DecisionAction.ALLOW,
                reason="risk_below_enforcement_threshold",
                risk=risk,
                observation=observation,
            )
        if observation.triggered:
            return PolicyDecision(
                action=DecisionAction.BLOCK,
                reason="score_above_threshold",
                risk=risk,
                observation=observation,
            )
        return PolicyDecision(
            action=DecisionAction.ALLOW,
            reason="score_below_threshold",
            risk=risk,
            observation=observation,
        )
