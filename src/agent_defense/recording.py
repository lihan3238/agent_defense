from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from agent_defense.types import DetectionContext, PolicyDecision

_SAFE_METADATA_KEYS = {
    "model_id",
    "checkpoint_content_id",
    "revision",
    "model_dtype",
    "quantization_config_hash",
    "tokenizer_class",
    "layer",
    "module_path",
    "state_kind",
    "position",
    "token_id",
    "token_text",
    "chat_template_hash",
    "tool_schema_hash",
    "system_message_hash",
    "completion_hash",
    "render_mode",
    "generation_seed",
    "extra_forward_count",
    "benchmark_version",
    "suite",
    "user_task_id",
    "injection_task_id",
    "attack",
    "scenario",
    "group_id",
    "trajectory_id",
    "step_index",
    "injection_present",
    "episode_security_violation",
    "candidate_call_dangerous",
}


@dataclass
class JsonlActivationRecorder:
    """Write minimal activation samples outside Git; never records prompt text."""

    path: Path
    run_id: str
    label: int | None
    split: Literal["train", "calibration", "test"]
    _counter: int = 0

    def __post_init__(self) -> None:
        if not self.run_id or "\n" in self.run_id or "\r" in self.run_id:
            raise ValueError("run_id must be a non-empty single-line identifier")
        if not self.path.exists():
            return
        prefix = f"{self.run_id}:"
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    sample_id = str(json.loads(line)["sample_id"])
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Cannot safely append to malformed activation JSONL at line {line_number}"
                    ) from error
                if sample_id.startswith(prefix):
                    raise ValueError(
                        f"run_id {self.run_id!r} already exists in {self.path}; choose a new run_id"
                    )

    @property
    def count(self) -> int:
        return self._counter

    def record(self, context: DetectionContext, decision: PolicyDecision) -> str | None:
        if context.activation is None:
            return None
        activation = np.asarray(context.activation, dtype=np.float64)
        if activation.ndim != 1 or not np.all(np.isfinite(activation)):
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(context.metadata)
        required = ("model_id", "layer", "position")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"Activation metadata is missing required recorder keys: {missing}")
        safe_metadata = {key: metadata[key] for key in _SAFE_METADATA_KEYS if key in metadata}
        safe_metadata.update(
            {
                "tool": context.candidate.function,
                "risk": decision.risk.name.lower(),
                "decision": decision.action.value,
                "label_requires_review": self.label is None,
            }
        )
        safe_metadata.setdefault("trajectory_id", self.run_id)
        safe_metadata.setdefault("step_index", self._counter)
        sample_id = f"{self.run_id}:{self._counter}"
        payload = {
            "sample_id": sample_id,
            "label": self.label,
            "split": self.split,
            "activation": activation.tolist(),
            "model_id": metadata["model_id"],
            "layer": metadata["layer"],
            "position": metadata["position"],
            "metadata": safe_metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self._counter += 1
        return sample_id
