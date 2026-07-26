from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RiskLevel(IntEnum):
    """Impact tier of a candidate tool call."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class DecisionAction(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True)
class CandidateToolCall:
    """A model-proposed action that has not been executed yet."""

    function: str
    args: Mapping[str, Any]
    call_id: str | None = None

    def canonical_text(self) -> str:
        payload = json.dumps(dict(self.args), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"{self.function}({payload})"


@dataclass(frozen=True)
class DetectionContext:
    """Minimal detector input at the pre-execution boundary."""

    candidate: CandidateToolCall
    activation: NDArray[np.floating[Any]] | None = None
    masked_candidates: tuple[CandidateToolCall, ...] = ()
    messages: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeObservation:
    detector: str
    score: float
    threshold: float
    triggered: bool
    valid: bool
    latency_ms: float
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    action: DecisionAction
    reason: str
    risk: RiskLevel
    observation: ProbeObservation


@dataclass(frozen=True)
class DecisionTrace:
    call: CandidateToolCall
    decision: PolicyDecision
    runtime_invoked: bool
    tool_succeeded: bool
    error: str | None = None

    @property
    def executed(self) -> bool:
        """Backward-compatible alias for whether the call crossed into the runtime."""

        return self.runtime_invoked


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    defense: str
    attack_present: bool
    utility_passed: bool
    attack_succeeded: bool
    malicious_tool_proposed: bool
    malicious_tool_blocked: bool
    normal_tool_blocked: bool
    defense_latency_ms: float
    valid_malicious_tool_blocked: bool = False
    detector_invalid_blocks: int = 0
    valid: bool = True
    failure_bucket: str | None = None
    malicious_tool_proposal_count: int = 0
    malicious_tool_block_count: int = 0
    valid_malicious_tool_block_count: int = 0
