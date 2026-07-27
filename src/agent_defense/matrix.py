from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

Scenario = Literal["clean", "attacked"]

CUSTOM_DEFENSES = frozenset({"none", "direction", "activation_probe", "melon", "melon_paper"})
BUILTIN_DEFENSES = frozenset(
    {
        "repeat_user_prompt",
        "spotlighting_with_delimiting",
        "transformers_pi_detector",
    }
)
VALID_DEFENSES = CUSTOM_DEFENSES | BUILTIN_DEFENSES

_CALL_REVIEW_FIELDS = (
    "malicious_tool_proposals",
    "malicious_tool_blocks",
    "valid_malicious_tool_blocks",
    "normal_tool_calls_blocked",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SCENARIOS: tuple[Scenario, ...] = ("clean", "attacked")


class ManifestError(ValueError):
    """A matrix manifest is malformed or ambiguous."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id_or_path: str
    revision: str
    layer: int
    position: Literal["tool_input", "function_call"]
    device: str
    dtype: str
    max_new_tokens: int
    disable_thinking: bool
    local_files_only: bool


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    suite_name: str
    benchmark_version: str
    attack_name: str


@dataclass(frozen=True, slots=True)
class DefenseConfig:
    name: str
    artifact_path: Path | None = None
    melon_threshold: float | None = None
    melon_embedding_backend: Literal["hf", "openai"] | None = None
    melon_embedding_model: str | None = None
    melon_embedding_device: str | None = None


@dataclass(frozen=True, slots=True)
class CaseConfig:
    case_id: str
    user_task_id: str
    injection_task_id: str
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MatrixManifest:
    schema_version: int
    model: ModelConfig
    benchmark: BenchmarkConfig
    defenses: tuple[DefenseConfig, ...]
    cases: tuple[CaseConfig, ...]


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One immutable call to ``run_hf_agentdojo_case`` plus matrix identity."""

    trial_id: str
    case_id: str
    scenario: Scenario
    seed: int
    defense: DefenseConfig
    model: ModelConfig
    benchmark: BenchmarkConfig
    user_task_id: str
    injection_task_id: str

    def runner_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model_id_or_path": self.model.model_id_or_path,
            "defense": self.defense.name,
            "suite_name": self.benchmark.suite_name,
            "benchmark_version": self.benchmark.benchmark_version,
            "user_task_id": self.user_task_id,
            "injection_task_id": self.injection_task_id,
            "attacked": self.scenario == "attacked",
            "attack_name": self.benchmark.attack_name,
            "layer": self.model.layer,
            "position": self.model.position,
            "revision": self.model.revision,
            "device": self.model.device,
            "dtype": self.model.dtype,
            "max_new_tokens": self.model.max_new_tokens,
            "seed": self.seed,
            "disable_thinking": self.model.disable_thinking,
            "local_files_only": self.model.local_files_only,
        }
        if self.defense.artifact_path is not None:
            kwargs["artifact_path"] = self.defense.artifact_path
        if self.defense.melon_threshold is not None:
            kwargs["melon_threshold"] = self.defense.melon_threshold
        if self.defense.melon_embedding_backend is not None:
            kwargs["melon_embedding_backend"] = self.defense.melon_embedding_backend
        if self.defense.melon_embedding_model is not None:
            kwargs["melon_embedding_model"] = self.defense.melon_embedding_model
        if self.defense.melon_embedding_device is not None:
            kwargs["melon_embedding_device"] = self.defense.melon_embedding_device
        return kwargs


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ManifestError(f"Non-finite JSON number is not allowed: {value}")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{location} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        raise ManifestError(f"{location} has invalid keys: {', '.join(parts)}")


def _string(value: Any, location: str, *, safe_id: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise ManifestError(f"{location} must not have surrounding whitespace")
    if safe_id and _SAFE_ID.fullmatch(value) is None:
        raise ManifestError(f"{location} must match {_SAFE_ID.pattern}")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ManifestError(f"{location} must be >= {minimum}")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{location} must be a boolean")
    return value


def _finite_number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{location} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ManifestError(f"{location} must be finite")
    if minimum is not None and number < minimum:
        raise ManifestError(f"{location} must be >= {minimum}")
    return number


def _parse_model(value: Any) -> ModelConfig:
    raw = _mapping(value, "model")
    _exact_keys(
        raw,
        {
            "model_id_or_path",
            "revision",
            "layer",
            "position",
            "device",
            "dtype",
            "max_new_tokens",
            "disable_thinking",
            "local_files_only",
        },
        "model",
    )
    position = _string(raw["position"], "model.position")
    if position not in {"tool_input", "function_call"}:
        raise ManifestError("model.position must be 'tool_input' or 'function_call'")
    return ModelConfig(
        model_id_or_path=_string(raw["model_id_or_path"], "model.model_id_or_path"),
        revision=_string(raw["revision"], "model.revision"),
        layer=_integer(raw["layer"], "model.layer"),
        position=position,  # type: ignore[arg-type]
        device=_string(raw["device"], "model.device"),
        dtype=_string(raw["dtype"], "model.dtype"),
        max_new_tokens=_integer(raw["max_new_tokens"], "model.max_new_tokens", minimum=1),
        disable_thinking=_boolean(raw["disable_thinking"], "model.disable_thinking"),
        local_files_only=_boolean(raw["local_files_only"], "model.local_files_only"),
    )


def _parse_benchmark(value: Any) -> BenchmarkConfig:
    raw = _mapping(value, "benchmark")
    _exact_keys(raw, {"suite_name", "benchmark_version", "attack_name"}, "benchmark")
    return BenchmarkConfig(
        suite_name=_string(raw["suite_name"], "benchmark.suite_name"),
        benchmark_version=_string(raw["benchmark_version"], "benchmark.benchmark_version"),
        attack_name=_string(raw["attack_name"], "benchmark.attack_name"),
    )


def _resolve_artifact(value: Any, location: str, base_dir: Path) -> Path:
    raw_path = Path(_string(value, location))
    return (base_dir / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()


def _parse_defense(value: Any, index: int, base_dir: Path) -> DefenseConfig:
    location = f"defenses[{index}]"
    raw = _mapping(value, location)
    name = _string(raw.get("name"), f"{location}.name")
    if name not in VALID_DEFENSES:
        raise ManifestError(f"{location}.name is not a supported defense: {name!r}")
    if name in {"direction", "activation_probe"}:
        _exact_keys(raw, {"name", "artifact_path"}, location)
        return DefenseConfig(
            name=name,
            artifact_path=_resolve_artifact(raw["artifact_path"], f"{location}.artifact_path", base_dir),
        )
    if name == "melon":
        _exact_keys(raw, {"name", "melon_threshold"}, location)
        threshold = _finite_number(raw["melon_threshold"], f"{location}.melon_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ManifestError(f"{location}.melon_threshold must be between 0 and 1")
        return DefenseConfig(name=name, melon_threshold=threshold)
    if name == "melon_paper":
        expected = {
            "name",
            "melon_threshold",
            "melon_embedding_backend",
            "melon_embedding_model",
            "melon_embedding_device",
        }
        _exact_keys(raw, expected, location)
        threshold = _finite_number(raw["melon_threshold"], f"{location}.melon_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ManifestError(f"{location}.melon_threshold must be between 0 and 1")
        backend = _string(raw["melon_embedding_backend"], f"{location}.melon_embedding_backend")
        if backend not in {"hf", "openai"}:
            raise ManifestError(f"{location}.melon_embedding_backend must be 'hf' or 'openai'")
        return DefenseConfig(
            name=name,
            melon_threshold=threshold,
            melon_embedding_backend=backend,  # type: ignore[arg-type]
            melon_embedding_model=_string(raw["melon_embedding_model"], f"{location}.melon_embedding_model"),
            melon_embedding_device=_string(
                raw["melon_embedding_device"], f"{location}.melon_embedding_device"
            ),
        )
    _exact_keys(raw, {"name"}, location)
    return DefenseConfig(name=name)


def _parse_case(value: Any, index: int) -> CaseConfig:
    location = f"cases[{index}]"
    raw = _mapping(value, location)
    _exact_keys(raw, {"case_id", "user_task_id", "injection_task_id", "seeds"}, location)
    seeds_raw = raw["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ManifestError(f"{location}.seeds must be a non-empty array")
    seeds = tuple(
        _integer(seed, f"{location}.seeds[{seed_index}]", minimum=0)
        for seed_index, seed in enumerate(seeds_raw)
    )
    if len(set(seeds)) != len(seeds):
        raise ManifestError(f"{location}.seeds must not contain duplicates")
    return CaseConfig(
        case_id=_string(raw["case_id"], f"{location}.case_id", safe_id=True),
        user_task_id=_string(raw["user_task_id"], f"{location}.user_task_id"),
        injection_task_id=_string(raw["injection_task_id"], f"{location}.injection_task_id"),
        seeds=seeds,
    )


def parse_manifest(data: Any, *, base_dir: str | Path | None = None) -> MatrixManifest:
    """Validate an already-decoded manifest without accepting implicit defaults."""

    raw = _mapping(data, "manifest")
    _exact_keys(raw, {"schema_version", "model", "benchmark", "defenses", "cases"}, "manifest")
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != 1:
        raise ManifestError(f"Unsupported schema_version: {schema_version}")
    resolved_base = Path.cwd() if base_dir is None else Path(base_dir)

    defenses_raw = raw["defenses"]
    if not isinstance(defenses_raw, list) or not defenses_raw:
        raise ManifestError("defenses must be a non-empty array")
    defenses = tuple(_parse_defense(value, index, resolved_base) for index, value in enumerate(defenses_raw))
    defense_names = [defense.name for defense in defenses]
    if len(set(defense_names)) != len(defense_names):
        raise ManifestError("defense names must be unique")
    if "none" not in defense_names:
        raise ManifestError("defenses must include 'none' as the paired overhead baseline")

    cases_raw = raw["cases"]
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ManifestError("cases must be a non-empty array")
    cases = tuple(_parse_case(value, index) for index, value in enumerate(cases_raw))
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ManifestError("case_id values must be unique")

    return MatrixManifest(
        schema_version=schema_version,
        model=_parse_model(raw["model"]),
        benchmark=_parse_benchmark(raw["benchmark"]),
        defenses=defenses,
        cases=cases,
    )


def load_manifest(path: str | Path) -> MatrixManifest:
    """Load strict JSON, rejecting duplicate keys, NaN/Infinity, and unknown fields."""

    manifest_path = Path(path)
    try:
        data = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ManifestError(f"Invalid JSON in {manifest_path}: {error}") from error
    return parse_manifest(data, base_dir=manifest_path.resolve().parent)


def expand_trials(manifest: MatrixManifest) -> tuple[TrialSpec, ...]:
    """Expand canonical case/seed/scenario/defense order into stable trial IDs."""

    defenses = sorted(manifest.defenses, key=lambda item: (item.name != "none", item.name))
    trials: list[TrialSpec] = []
    for case in sorted(manifest.cases, key=lambda item: item.case_id):
        for seed in sorted(case.seeds):
            for scenario in _SCENARIOS:
                for defense in defenses:
                    trial_id = f"{case.case_id}__{scenario}__seed-{seed}__{defense.name}"
                    trials.append(
                        TrialSpec(
                            trial_id=trial_id,
                            case_id=case.case_id,
                            scenario=scenario,
                            seed=seed,
                            defense=defense,
                            model=manifest.model,
                            benchmark=manifest.benchmark,
                            user_task_id=case.user_task_id,
                            injection_task_id=case.injection_task_id,
                        )
                    )
    return tuple(trials)


def run_sequential(
    manifest_or_trials: MatrixManifest | Sequence[TrialSpec],
    runner: Callable[..., Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run trial specs in order and attach identities omitted by the single-case runner.

    Exceptions deliberately propagate: callers can persist an explicit invalid result with
    their own failure bucket instead of this layer silently inventing failure semantics.
    """

    trials = (
        expand_trials(manifest_or_trials)
        if isinstance(manifest_or_trials, MatrixManifest)
        else tuple(manifest_or_trials)
    )
    results: list[dict[str, Any]] = []
    for spec in trials:
        raw = runner(**spec.runner_kwargs())
        if not isinstance(raw, Mapping):
            raise TypeError(f"Runner returned {type(raw).__name__} for {spec.trial_id}, expected mapping")
        result = dict(raw)
        expected = {
            "defense": spec.defense.name,
            "seed": spec.seed,
            "attacked": spec.scenario == "attacked",
        }
        for key, expected_value in expected.items():
            if key in result and result[key] != expected_value:
                raise ValueError(
                    f"Runner identity mismatch for {spec.trial_id}: "
                    f"{key}={result[key]!r}, expected {expected_value!r}"
                )
        result.update(
            {
                "trial_id": spec.trial_id,
                "case_id": spec.case_id,
                "scenario": spec.scenario,
                **expected,
            }
        )
        results.append(result)
    return results


@dataclass(frozen=True, slots=True)
class _TraceCounts:
    proposed: int
    blocked: int
    valid_blocked: int


@dataclass(frozen=True, slots=True)
class _Result:
    case_id: str
    scenario: Scenario
    seed: int
    defense: str
    valid: bool
    utility_passed: bool
    attack_goal_achieved: bool
    elapsed_ms: float
    detector_latency_ms: float | None
    model_query_count: int | None
    extra_forward_count: int | None
    failure_bucket: str | None
    trace_counts: _TraceCounts | None
    malicious_counts: _TraceCounts | None
    normal_tool_calls_blocked: int | None


def _optional_count(raw: Mapping[str, Any], key: str, location: str) -> int | None:
    value = raw.get(key)
    return None if value is None else _integer(value, f"{location}.{key}", minimum=0)


def _optional_latency(raw: Mapping[str, Any], key: str, location: str) -> float | None:
    value = raw.get(key)
    return None if value is None else _finite_number(value, f"{location}.{key}", minimum=0.0)


def _trace_counts(raw: Mapping[str, Any], defense: str, location: str) -> _TraceCounts | None:
    if defense in BUILTIN_DEFENSES:
        return None
    trace = raw.get("trace")
    if not isinstance(trace, list):
        raise ValueError(f"{location}.trace must be an array for custom defense {defense!r}")
    blocked = 0
    valid_blocked = 0
    for index, item in enumerate(trace):
        trace_location = f"{location}.trace[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{trace_location} must be an object")
        decision = item.get("decision")
        if decision not in {"allow", "block"}:
            raise ValueError(f"{trace_location}.decision must be 'allow' or 'block'")
        trace_valid = item.get("valid")
        if not isinstance(trace_valid, bool):
            raise ValueError(f"{trace_location}.valid must be a boolean")
        if decision == "block":
            blocked += 1
            valid_blocked += int(trace_valid)
    proposed = len(trace)
    declared_proposed = _optional_count(raw, "tool_calls_proposed", location)
    declared_blocked = _optional_count(raw, "tool_calls_blocked", location)
    if declared_proposed is not None and declared_proposed != proposed:
        raise ValueError(f"{location}.tool_calls_proposed disagrees with trace length")
    if declared_blocked is not None and declared_blocked != blocked:
        raise ValueError(f"{location}.tool_calls_blocked disagrees with trace decisions")
    return _TraceCounts(proposed=proposed, blocked=blocked, valid_blocked=valid_blocked)


def _reviewed_malicious_counts(raw: Mapping[str, Any], defense: str, location: str) -> _TraceCounts | None:
    """Read only explicit reviewed labels; executor trace is not attack ground truth."""

    field_names = (
        "malicious_tool_proposals",
        "malicious_tool_blocks",
        "valid_malicious_tool_blocks",
    )
    values = [raw.get(name) for name in field_names]
    if all(value is None for value in values) or defense in BUILTIN_DEFENSES:
        return None
    if any(value is None for value in values):
        raise ValueError(f"{location} must provide all reviewed malicious-tool count fields or none")
    proposed, blocked, valid_blocked = (
        _integer(value, f"{location}.{name}", minimum=0)
        for name, value in zip(field_names, values, strict=True)
    )
    if blocked > proposed:
        raise ValueError(f"{location}.malicious_tool_blocks exceeds malicious_tool_proposals")
    if valid_blocked > blocked:
        raise ValueError(f"{location}.valid_malicious_tool_blocks exceeds malicious_tool_blocks")
    return _TraceCounts(proposed=proposed, blocked=blocked, valid_blocked=valid_blocked)


def _normalize_result(raw_value: Mapping[str, Any], index: int) -> _Result:
    location = f"results[{index}]"
    raw = _mapping(raw_value, location)
    case_id = _string(raw.get("case_id"), f"{location}.case_id", safe_id=True)
    defense = _string(raw.get("defense"), f"{location}.defense")
    if defense not in VALID_DEFENSES:
        raise ValueError(f"{location}.defense is unsupported: {defense!r}")

    attacked = raw.get("attacked")
    if attacked is not None and not isinstance(attacked, bool):
        raise ValueError(f"{location}.attacked must be a boolean")
    scenario_value = raw.get("scenario")
    if scenario_value is None:
        if attacked is None:
            raise ValueError(f"{location} requires scenario or attacked")
        scenario_value = "attacked" if attacked else "clean"
    if scenario_value not in _SCENARIOS:
        raise ValueError(f"{location}.scenario must be 'clean' or 'attacked'")
    scenario: Scenario = scenario_value
    if attacked is not None and attacked != (scenario == "attacked"):
        raise ValueError(f"{location}.scenario disagrees with attacked")

    valid = raw.get("valid")
    utility_passed = raw.get("utility_passed")
    if not isinstance(valid, bool):
        raise ValueError(f"{location}.valid must be a boolean")
    if not isinstance(utility_passed, bool):
        raise ValueError(f"{location}.utility_passed must be a boolean")
    attack_goal = raw.get("attack_goal_achieved")
    if scenario == "attacked":
        if not isinstance(attack_goal, bool):
            raise ValueError(f"{location}.attack_goal_achieved must be a boolean for attacked trials")
    elif attack_goal not in {None, False}:
        raise ValueError(f"{location}.attack_goal_achieved must be false or null for clean trials")

    failure_bucket = raw.get("failure_bucket")
    if failure_bucket is not None:
        failure_bucket = _string(failure_bucket, f"{location}.failure_bucket")
    trace_counts = _trace_counts(raw, defense, location)
    malicious_counts = _reviewed_malicious_counts(raw, defense, location)
    if trace_counts is not None and malicious_counts is not None:
        if malicious_counts.proposed > trace_counts.proposed:
            raise ValueError(f"{location}.malicious_tool_proposals exceeds total trace proposals")
        if malicious_counts.blocked > trace_counts.blocked:
            raise ValueError(f"{location}.malicious_tool_blocks exceeds total trace blocks")
        if malicious_counts.valid_blocked > trace_counts.valid_blocked:
            raise ValueError(f"{location}.valid_malicious_tool_blocks exceeds valid total trace blocks")
    normal_tool_calls_blocked = _optional_count(raw, "normal_tool_calls_blocked", location)
    if defense in BUILTIN_DEFENSES:
        normal_tool_calls_blocked = None
    elif normal_tool_calls_blocked is not None and trace_counts is not None:
        if normal_tool_calls_blocked > trace_counts.blocked:
            raise ValueError(f"{location}.normal_tool_calls_blocked exceeds total trace blocks")

    return _Result(
        case_id=case_id,
        scenario=scenario,
        seed=_integer(raw.get("seed"), f"{location}.seed", minimum=0),
        defense=defense,
        valid=valid,
        utility_passed=utility_passed,
        attack_goal_achieved=bool(attack_goal),
        elapsed_ms=_finite_number(raw.get("elapsed_ms"), f"{location}.elapsed_ms", minimum=0.0),
        detector_latency_ms=(
            None if defense in BUILTIN_DEFENSES else _optional_latency(raw, "detector_latency_ms", location)
        ),
        model_query_count=_optional_count(raw, "model_query_count", location),
        extra_forward_count=_optional_count(raw, "extra_forward_count", location),
        failure_bucket=failure_bucket,
        trace_counts=trace_counts,
        malicious_counts=malicious_counts,
        normal_tool_calls_blocked=normal_tool_calls_blocked,
    )


def _parse_call_review(value: Any, trial_id: str) -> dict[str, int]:
    location = f"reviews[{trial_id!r}]"
    raw = _mapping(value, location)
    _exact_keys(raw, set(_CALL_REVIEW_FIELDS), location)
    return {field: _integer(raw[field], f"{location}.{field}", minimum=0) for field in _CALL_REVIEW_FIELDS}


def apply_call_reviews(
    results: Sequence[Mapping[str, Any]],
    review_manifest: Mapping[str, Any],
    *,
    review_manifest_sha256: str,
) -> list[dict[str, Any]]:
    """Apply complete human call-count reviews without mutating the raw rows."""

    if re.fullmatch(r"[0-9a-f]{64}", review_manifest_sha256) is None:
        raise ValueError("review_manifest_sha256 must be a lowercase SHA256 digest")
    if not results:
        raise ValueError("Raw matrix results must not be empty")

    rows_by_id: dict[str, tuple[Mapping[str, Any], _Result, int]] = {}
    ordered_ids: list[str] = []
    for index, raw_value in enumerate(results):
        location = f"results[{index}]"
        raw = _mapping(raw_value, location)
        trial_id = _string(raw.get("trial_id"), f"{location}.trial_id")
        if trial_id in rows_by_id:
            raise ValueError(f"Duplicate raw trial_id: {trial_id!r}")
        normalized = _normalize_result(raw, index)
        rows_by_id[trial_id] = (raw, normalized, index)
        ordered_ids.append(trial_id)

    reviews_raw = _mapping(review_manifest, "reviews")
    parsed_reviews: dict[str, dict[str, int]] = {}
    for raw_trial_id, value in reviews_raw.items():
        trial_id = _string(raw_trial_id, "reviews trial_id")
        parsed_reviews[trial_id] = _parse_call_review(value, trial_id)

    all_ids = set(rows_by_id)
    custom_ids = {
        trial_id
        for trial_id, (_, normalized, _) in rows_by_id.items()
        if normalized.defense in CUSTOM_DEFENSES
    }
    review_ids = set(parsed_reviews)
    unknown_ids = sorted(review_ids - all_ids)
    if unknown_ids:
        raise ValueError(f"Review contains unknown trial_id values: {unknown_ids}")
    builtin_ids = sorted(review_ids - custom_ids)
    if builtin_ids:
        raise ValueError(f"Review is only valid for custom-defense trials, got: {builtin_ids}")
    missing_ids = sorted(custom_ids - review_ids)
    if missing_ids:
        raise ValueError(f"Review is missing custom-defense trial_id values: {missing_ids}")

    reviewed_rows: list[dict[str, Any]] = []
    for trial_id in ordered_ids:
        raw, normalized, index = rows_by_id[trial_id]
        if normalized.defense in BUILTIN_DEFENSES:
            reviewed_rows.append(dict(raw))
            continue

        location = f"results[{index}]"
        if raw.get("call_label_status") not in {None, "not_reviewed"}:
            raise ValueError(f"{location} already has a non-pending call_label_status")
        if raw.get("review_manifest_sha256") is not None or any(
            raw.get(field) is not None for field in _CALL_REVIEW_FIELDS
        ):
            raise ValueError(f"{location} already contains reviewed call-count fields")

        trace_counts = normalized.trace_counts
        assert trace_counts is not None
        counts = parsed_reviews[trial_id]
        malicious_proposals = counts["malicious_tool_proposals"]
        malicious_blocks = counts["malicious_tool_blocks"]
        valid_malicious_blocks = counts["valid_malicious_tool_blocks"]
        normal_blocks = counts["normal_tool_calls_blocked"]
        if malicious_proposals > trace_counts.proposed:
            raise ValueError(f"Review for {trial_id!r} has malicious proposals above trace proposals")
        if malicious_blocks > malicious_proposals:
            raise ValueError(f"Review for {trial_id!r} has malicious blocks above malicious proposals")
        if malicious_blocks > trace_counts.blocked:
            raise ValueError(f"Review for {trial_id!r} has malicious blocks above trace blocks")
        if valid_malicious_blocks > malicious_blocks:
            raise ValueError(f"Review for {trial_id!r} has valid malicious blocks above malicious blocks")
        if valid_malicious_blocks > trace_counts.valid_blocked:
            raise ValueError(f"Review for {trial_id!r} has valid malicious blocks above valid trace blocks")
        if normal_blocks > trace_counts.blocked:
            raise ValueError(f"Review for {trial_id!r} has normal blocks above trace blocks")
        if malicious_blocks + normal_blocks > trace_counts.blocked:
            raise ValueError(
                f"Review for {trial_id!r} assigns more malicious and normal blocks than trace blocks"
            )

        reviewed = dict(raw)
        reviewed.update(counts)
        reviewed["call_label_status"] = "reviewed"
        reviewed["review_manifest_sha256"] = review_manifest_sha256
        reviewed_rows.append(reviewed)

    return reviewed_rows


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _median(values: Sequence[int | float]) -> float | None:
    return None if not values else float(median(values))


def _summary_for_defense(
    defense: str,
    records: Sequence[_Result],
    baseline: Mapping[tuple[str, Scenario, int], _Result],
) -> dict[str, Any]:
    valid = [record for record in records if record.valid]
    clean = [record for record in records if record.scenario == "clean"]
    attacked = [record for record in records if record.scenario == "attacked"]
    valid_clean = [record for record in clean if record.valid]
    valid_attacked = [record for record in attacked if record.valid]
    failure_buckets = Counter(
        record.failure_bucket or "unspecified" for record in records if not record.valid
    )

    paired_overheads = []
    for record in valid:
        pair = baseline.get((record.case_id, record.scenario, record.seed))
        if pair is not None and pair.valid:
            paired_overheads.append(record.elapsed_ms - pair.elapsed_ms)

    summary: dict[str, Any] = {
        "trials": len(records),
        "valid_trials": len(valid),
        "invalid_trials": len(records) - len(valid),
        "valid_coverage": _rate(len(valid), len(records)),
        "clean_trials": len(clean),
        "valid_clean_trials": len(valid_clean),
        "clean_valid_coverage": _rate(len(valid_clean), len(clean)),
        "attacked_trials": len(attacked),
        "valid_attacked_trials": len(valid_attacked),
        "attacked_valid_coverage": _rate(len(valid_attacked), len(attacked)),
        "clean_utility_passes": sum(record.utility_passed for record in valid_clean),
        "attacked_utility_passes": sum(record.utility_passed for record in valid_attacked),
        "attack_successes": sum(record.attack_goal_achieved for record in valid_attacked),
        "bu": _rate(sum(record.utility_passed for record in valid_clean), len(valid_clean)),
        "ua": _rate(sum(record.utility_passed for record in valid_attacked), len(valid_attacked)),
        "targeted_asr": _rate(
            sum(record.attack_goal_achieved for record in valid_attacked), len(valid_attacked)
        ),
        "median_e2e_latency_ms": _median([record.elapsed_ms for record in valid]),
        "median_detector_latency_ms": _median(
            [record.detector_latency_ms for record in valid if record.detector_latency_ms is not None]
        ),
        "detector_latency_coverage": _rate(
            sum(record.detector_latency_ms is not None for record in valid), len(valid)
        ),
        "median_model_query_count": _median(
            [record.model_query_count for record in valid if record.model_query_count is not None]
        ),
        "model_query_coverage": _rate(
            sum(record.model_query_count is not None for record in valid), len(valid)
        ),
        "median_extra_forward_count": _median(
            [record.extra_forward_count for record in valid if record.extra_forward_count is not None]
        ),
        "extra_forward_coverage": _rate(
            sum(record.extra_forward_count is not None for record in valid), len(valid)
        ),
        "median_paired_overhead_ms": _median(paired_overheads),
        "paired_overhead_count": len(paired_overheads),
        "failure_buckets": dict(sorted(failure_buckets.items())),
    }

    if defense in BUILTIN_DEFENSES:
        summary.update(
            {
                "call_metrics": "n/a_builtin_no_executor_trace",
                "attacked_total_tool_calls_proposed": None,
                "attacked_total_tool_calls_blocked": None,
                "malicious_tool_proposals": None,
                "malicious_tool_blocks": None,
                "valid_malicious_tool_blocks": None,
                "reviewed_malicious_call_coverage": None,
                "call_interception_rate": None,
                "valid_call_interception_rate": None,
                "clean_total_tool_calls_proposed": None,
                "clean_total_tool_calls_blocked": None,
                "clean_call_label_review_coverage": None,
                "normal_tool_calls_blocked": None,
                "clean_false_block_rate": None,
                "clean_trials_with_false_block": None,
            }
        )
        return summary

    attacked_trace_counts = [record.trace_counts for record in valid_attacked]
    clean_trace_counts = [record.trace_counts for record in valid_clean]
    attacked_total_proposed = sum(count.proposed for count in attacked_trace_counts if count is not None)
    attacked_total_blocked = sum(count.blocked for count in attacked_trace_counts if count is not None)
    clean_total_proposed = sum(count.proposed for count in clean_trace_counts if count is not None)
    clean_total_blocked = sum(count.blocked for count in clean_trace_counts if count is not None)

    reviewed_counts = [record.malicious_counts for record in valid_attacked]
    reviewed_coverage = _rate(sum(count is not None for count in reviewed_counts), len(valid_attacked))
    reviewed_complete = bool(valid_attacked) and all(count is not None for count in reviewed_counts)
    malicious_proposed = (
        sum(count.proposed for count in reviewed_counts if count is not None) if reviewed_complete else None
    )
    malicious_blocked = (
        sum(count.blocked for count in reviewed_counts if count is not None) if reviewed_complete else None
    )
    valid_malicious_blocked = (
        sum(count.valid_blocked for count in reviewed_counts if count is not None)
        if reviewed_complete
        else None
    )
    normal_block_reviews = [record.normal_tool_calls_blocked for record in valid_clean]
    clean_review_coverage = _rate(sum(value is not None for value in normal_block_reviews), len(valid_clean))
    clean_review_complete = bool(valid_clean) and all(value is not None for value in normal_block_reviews)
    reviewed_normal_blocks = (
        sum(value for value in normal_block_reviews if value is not None) if clean_review_complete else None
    )
    clean_trials_with_false_block = (
        sum(bool(value) for value in normal_block_reviews if value is not None)
        if clean_review_complete
        else None
    )
    summary.update(
        {
            "call_metrics": (
                "reviewed_malicious_labels_available"
                if reviewed_complete
                else "n/a_missing_reviewed_malicious_labels"
            ),
            "attacked_total_tool_calls_proposed": attacked_total_proposed,
            "attacked_total_tool_calls_blocked": attacked_total_blocked,
            "malicious_tool_proposals": malicious_proposed,
            "malicious_tool_blocks": malicious_blocked,
            "valid_malicious_tool_blocks": valid_malicious_blocked,
            "reviewed_malicious_call_coverage": reviewed_coverage,
            "call_interception_rate": (
                None
                if malicious_proposed is None or malicious_blocked is None
                else _rate(malicious_blocked, malicious_proposed)
            ),
            "valid_call_interception_rate": (
                None
                if malicious_proposed is None or valid_malicious_blocked is None
                else _rate(valid_malicious_blocked, malicious_proposed)
            ),
            "clean_total_tool_calls_proposed": clean_total_proposed,
            "clean_total_tool_calls_blocked": clean_total_blocked,
            "clean_call_label_review_coverage": clean_review_coverage,
            "normal_tool_calls_blocked": reviewed_normal_blocks,
            "clean_false_block_rate": (
                None
                if clean_trials_with_false_block is None
                else _rate(clean_trials_with_false_block, len(valid_clean))
            ),
            "clean_trials_with_false_block": clean_trials_with_false_block,
        }
    )
    return summary


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate held-out raw results without putting invalid trials in rate denominators."""

    normalized = [_normalize_result(raw, index) for index, raw in enumerate(results)]
    identities: set[tuple[str, Scenario, int, str]] = set()
    for record in normalized:
        identity = (record.case_id, record.scenario, record.seed, record.defense)
        if identity in identities:
            raise ValueError(f"Duplicate trial result identity: {identity}")
        identities.add(identity)

    baseline = {
        (record.case_id, record.scenario, record.seed): record
        for record in normalized
        if record.defense == "none"
    }
    defenses = sorted({record.defense for record in normalized}, key=lambda name: (name != "none", name))
    return {
        "schema_version": 1,
        "trials": len(normalized),
        "by_defense": {
            defense: _summary_for_defense(
                defense,
                [record for record in normalized if record.defense == defense],
                baseline,
            )
            for defense in defenses
        },
    }
