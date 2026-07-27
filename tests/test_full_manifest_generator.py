from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

from agent_defense.matrix import expand_trials, load_manifest


def _load_generator_module():
    script = Path(__file__).parents[1] / "scripts" / "generate_v112_full_manifests.py"
    spec = importlib.util.spec_from_file_location("generate_v112_full_manifests", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_v112_generator_has_exact_unique_denominators(tmp_path: Path) -> None:
    module = _load_generator_module()

    index = module.generate(tmp_path)
    trials = [
        trial
        for path in sorted(tmp_path.glob("*-shard-*.json"))
        for trial in expand_trials(load_manifest(path))
    ]
    counts = Counter((trial.defense.name, trial.scenario) for trial in trials)
    semantic_identities = {
        (
            trial.benchmark.suite_name,
            trial.user_task_id,
            trial.injection_task_id if trial.scenario == "attacked" else None,
            trial.scenario,
            trial.seed,
            trial.defense.name,
        )
        for trial in trials
    }

    assert index["shard_trial_counts"] == [364, 364, 362, 362]
    assert counts == Counter(
        {
            ("none", "clean"): 97,
            ("none", "attacked"): 629,
            ("melon_paper", "clean"): 97,
            ("melon_paper", "attacked"): 629,
        }
    )
    assert len(trials) == len(semantic_identities) == 1452
