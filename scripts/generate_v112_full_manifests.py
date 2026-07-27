from __future__ import annotations

import argparse
import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

from agentdojo.task_suite import get_suite

SUITES = ("banking", "slack", "travel", "workspace")
EXPECTED_COUNTS = {
    "banking": (16, 144),
    "slack": (21, 105),
    "travel": (20, 140),
    "workspace": (40, 240),
}
SHARD_COUNT = 4
AGENTDOJO_VERSION = "0.1.35"


def _numeric_suffix(identifier: str) -> int:
    match = re.search(r"(\d+)$", identifier)
    if match is None:
        raise ValueError(f"Identifier has no numeric suffix: {identifier!r}")
    return int(match.group(1))


def _base_manifest(suite_name: str) -> dict[str, Any]:
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
            "suite_name": suite_name,
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
    }


def _suite_cases(suite_name: str) -> list[dict[str, Any]]:
    suite = get_suite("v1.1.2", suite_name)
    user_task_ids = sorted(suite.user_tasks, key=_numeric_suffix)
    injection_task_ids = sorted(suite.injection_tasks, key=_numeric_suffix)
    cases: list[dict[str, Any]] = []
    for user_task_id in user_task_ids:
        user_index = _numeric_suffix(user_task_id)
        cases.append(
            {
                "case_id": f"{suite_name}-{len(cases):03d}-clean-u{user_index}",
                "user_task_id": user_task_id,
                "injection_task_id": None,
                "seeds": [0],
                "scenarios": ["clean"],
            }
        )
    for user_task_id in user_task_ids:
        user_index = _numeric_suffix(user_task_id)
        for injection_task_id in injection_task_ids:
            injection_index = _numeric_suffix(injection_task_id)
            cases.append(
                {
                    "case_id": (f"{suite_name}-{len(cases):03d}-attacked-u{user_index}-i{injection_index}"),
                    "user_task_id": user_task_id,
                    "injection_task_id": injection_task_id,
                    "seeds": [0],
                    "scenarios": ["attacked"],
                }
            )
    clean_count = sum(case["scenarios"] == ["clean"] for case in cases)
    attacked_count = sum(case["scenarios"] == ["attacked"] for case in cases)
    if (clean_count, attacked_count) != EXPECTED_COUNTS[suite_name]:
        raise RuntimeError(f"Unexpected {suite_name} catalog: clean={clean_count}, attacked={attacked_count}")
    return cases


def generate(output_dir: Path) -> dict[str, Any]:
    installed_version = version("agentdojo")
    if installed_version != AGENTDOJO_VERSION:
        raise RuntimeError(f"Expected agentdojo=={AGENTDOJO_VERSION}, found agentdojo=={installed_version}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_version": "v1.1.2",
        "attack": "important_instructions",
        "defenses": ["none", "melon_paper"],
        "shard_count": SHARD_COUNT,
        "manifests": [],
    }
    total_base_cases = 0
    shard_base_counts = [0] * SHARD_COUNT
    for suite_name in SUITES:
        cases = _suite_cases(suite_name)
        total_base_cases += len(cases)
        for shard_index in range(SHARD_COUNT):
            shard_cases = cases[shard_index::SHARD_COUNT]
            shard_base_counts[shard_index] += len(shard_cases)
            payload = {**_base_manifest(suite_name), "cases": shard_cases}
            filename = f"{suite_name}-shard-{shard_index}.json"
            (output_dir / filename).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            index["manifests"].append(
                {
                    "file": filename,
                    "suite": suite_name,
                    "shard_index": shard_index,
                    "base_case_count": len(shard_cases),
                    "trial_count": len(shard_cases) * 2,
                }
            )
    index["base_case_count"] = total_base_cases
    index["trial_count"] = total_base_cases * 2
    index["shard_base_case_counts"] = shard_base_counts
    index["shard_trial_counts"] = [count * 2 for count in shard_base_counts]
    if total_base_cases != 726 or index["trial_count"] != 1452:
        raise RuntimeError(f"Unexpected full-matrix size: {index}")
    if shard_base_counts != [182, 182, 181, 181]:
        raise RuntimeError(f"Unexpected shard balance: {shard_base_counts}")
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
