from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DetectorArtifact:
    """Portable, non-pickle representation of a direction or linear probe."""

    kind: Literal["direction", "linear_probe"]
    weights: tuple[float, ...]
    bias: float
    threshold: float
    model_id: str
    layer: int
    position: str
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported artifact schema: {self.schema_version}")
        if not self.weights:
            raise ValueError("Detector weights must not be empty")
        if self.kind == "direction" and abs(sum(value * value for value in self.weights) - 1.0) > 1e-4:
            raise ValueError("Direction weights must be unit normalized")

    @property
    def dimension(self) -> int:
        return len(self.weights)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> DetectorArtifact:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["weights"] = tuple(float(value) for value in data["weights"])
        return cls(**data)
