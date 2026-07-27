from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_defense.matrix import MatrixManifest, TrialSpec, aggregate_results, expand_trials, load_manifest

SUITE_COUNTS = {
    "banking": {"clean": 16, "attacked": 144},
    "slack": {"clean": 21, "attacked": 105},
    "travel": {"clean": 20, "attacked": 140},
    "workspace": {"clean": 40, "attacked": 240},
}
EXPECTED_DEFENSES = ("none", "melon_paper")
EXPECTED_PER_DEFENSE = {"clean": 97, "attacked": 629, "total": 726}
EXPECTED_TRIALS = 1452
EXPECTED_MANIFEST_NAMES = tuple(f"{suite}-shard-{shard}.json" for suite in SUITE_COUNTS for shard in range(4))
EXPECTED_RAW_NAMES = tuple(f"{Path(name).stem}.raw.jsonl" for name in EXPECTED_MANIFEST_NAMES)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(ValueError):
    """The full result set is incomplete, mixed, or inconsistent with its manifests."""


@dataclass(frozen=True, slots=True)
class ValidatedShard:
    manifest_name: str
    raw_name: str
    manifest_sha256: str
    run_fingerprint: str
    manifest: MatrixManifest
    trials: tuple[TrialSpec, ...]
    rows: tuple[dict[str, Any], ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ValidationError(
            f"Could not read raw result file {path.name}: {type(error).__name__}"
        ) from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonfinite,
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                raise ValidationError(
                    f"Invalid JSONL in {path.name} at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValidationError(f"{path.name} line {line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _expected_identity(spec: TrialSpec) -> dict[str, Any]:
    return {
        "case_id": spec.case_id,
        "scenario": spec.scenario,
        "suite": spec.benchmark.suite_name,
        "benchmark_version": spec.benchmark.benchmark_version,
        "user_task_id": spec.user_task_id,
        "injection_task_id": spec.injection_task_id if spec.scenario == "attacked" else None,
        "attack": spec.benchmark.attack_name if spec.scenario == "attacked" else None,
        "seed": spec.seed,
        "defense": spec.defense.name,
        "attacked": spec.scenario == "attacked",
    }


def _short_ids(values: Sequence[str]) -> str:
    preview = ", ".join(repr(value) for value in values[:3])
    suffix = " ..." if len(values) > 3 else ""
    return f"{preview}{suffix}"


def _load_manifest(path: Path) -> MatrixManifest:
    try:
        return load_manifest(path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        message = str(error).replace(str(path), path.name)
        raise ValidationError(f"Invalid manifest {path.name}: {message}") from error


def _validate_shard(manifest_path: Path, raw_path: Path) -> ValidatedShard:
    manifest = _load_manifest(manifest_path)
    declared_suite = manifest_path.name.split("-shard-", maxsplit=1)[0]
    if manifest.benchmark.suite_name != declared_suite:
        raise ValidationError(
            f"{manifest_path.name} declares suite {manifest.benchmark.suite_name!r}, "
            f"expected {declared_suite!r}"
        )
    if manifest.benchmark.benchmark_version != "v1.1.2":
        raise ValidationError(f"{manifest_path.name} must use benchmark version v1.1.2")
    if manifest.benchmark.attack_name != "important_instructions":
        raise ValidationError(f"{manifest_path.name} must use attack important_instructions")
    defense_names = tuple(defense.name for defense in manifest.defenses)
    if set(defense_names) != set(EXPECTED_DEFENSES) or len(defense_names) != len(EXPECTED_DEFENSES):
        raise ValidationError(
            f"{manifest_path.name} must contain exactly defenses {list(EXPECTED_DEFENSES)!r}"
        )

    trials = expand_trials(manifest)
    planned_by_id = {trial.trial_id: trial for trial in trials}
    if len(planned_by_id) != len(trials):
        raise ValidationError(f"{manifest_path.name} expands to duplicate trial_id values")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = _read_jsonl(raw_path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()

    for line_number, row in enumerate(rows, start=1):
        location = f"{raw_path.name} row {line_number}"
        trial_id = row.get("trial_id")
        if type(trial_id) is not str or not trial_id:
            raise ValidationError(f"{location} has no valid trial_id")
        if trial_id in rows_by_id:
            raise ValidationError(f"{raw_path.name} contains duplicate trial_id {trial_id!r}")
        spec = planned_by_id.get(trial_id)
        if spec is None:
            raise ValidationError(f"{location} has a trial_id outside {manifest_path.name}")
        if row.get("manifest_sha256") != manifest_sha256:
            raise ValidationError(f"{location} does not match the manifest SHA256")

        fingerprint = row.get("run_fingerprint")
        if type(fingerprint) is not str or SHA256_RE.fullmatch(fingerprint) is None:
            raise ValidationError(f"{location} has no valid lowercase SHA256 run_fingerprint")
        fingerprints.add(fingerprint)

        for field, expected in _expected_identity(spec).items():
            if field not in row:
                raise ValidationError(f"{location} is missing identity field {field!r}")
            actual = row[field]
            if type(actual) is not type(expected) or actual != expected:
                raise ValidationError(
                    f"{location} identity field {field!r} disagrees with trial_id {trial_id!r}"
                )
        rows_by_id[trial_id] = row

    if len(fingerprints) != 1:
        raise ValidationError(
            f"{raw_path.name} must contain exactly one run_fingerprint; found {len(fingerprints)}"
        )
    missing = sorted(set(planned_by_id) - set(rows_by_id))
    if missing:
        raise ValidationError(
            f"{raw_path.name} is incomplete: missing {len(missing)} trial_id values ({_short_ids(missing)})"
        )

    canonical_rows = tuple(rows_by_id[trial.trial_id] for trial in trials)
    return ValidatedShard(
        manifest_name=manifest_path.name,
        raw_name=raw_path.name,
        manifest_sha256=manifest_sha256,
        run_fingerprint=next(iter(fingerprints)),
        manifest=manifest,
        trials=trials,
        rows=canonical_rows,
    )


def _require_exact_files(directory: Path, *, expected: set[str], pattern: str, label: str) -> None:
    if not directory.is_dir():
        raise ValidationError(f"{label} directory is missing or is not a directory")
    actual = {path.name for path in directory.glob(pattern) if path.is_file()}
    if label == "manifest":
        actual.discard("index.json")
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValidationError(f"{label} file set is not the expected 16-file matrix: {'; '.join(details)}")


def _semantic_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["suite"],
        row["benchmark_version"],
        row["scenario"],
        row["user_task_id"],
        row["injection_task_id"],
        row["attack"],
        row["seed"],
        row["defense"],
    )


def _count_payload(counter: Counter[tuple[str, str]], defense: str) -> dict[str, int]:
    clean = counter[(defense, "clean")]
    attacked = counter[(defense, "attacked")]
    return {"clean": clean, "attacked": attacked, "total": clean + attacked}


def validate_full_results(
    manifests_dir: str | Path,
    raw_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests_path = Path(manifests_dir)
    raw_path = Path(raw_dir)
    _require_exact_files(
        manifests_path,
        expected=set(EXPECTED_MANIFEST_NAMES),
        pattern="*.json",
        label="manifest",
    )
    _require_exact_files(
        raw_path,
        expected=set(EXPECTED_RAW_NAMES),
        pattern="*.jsonl",
        label="raw result",
    )

    shards = []
    for manifest_name in EXPECTED_MANIFEST_NAMES:
        manifest_file = manifests_path / manifest_name
        raw_file = raw_path / f"{manifest_file.stem}.raw.jsonl"
        shards.append(_validate_shard(manifest_file, raw_file))

    reference_model = shards[0].manifest.model
    reference_defenses = shards[0].manifest.defenses
    for shard in shards[1:]:
        if shard.manifest.model != reference_model:
            raise ValidationError(f"{shard.manifest_name} has a different model configuration")
        if shard.manifest.defenses != reference_defenses:
            raise ValidationError(f"{shard.manifest_name} has a different defense configuration")

    merged_rows: list[dict[str, Any]] = []
    seen_trial_ids: set[str] = set()
    seen_semantic: dict[tuple[Any, ...], tuple[str, str]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    suite_counts: Counter[tuple[str, str, str]] = Counter()
    for shard in shards:
        for row in shard.rows:
            trial_id = row["trial_id"]
            if trial_id in seen_trial_ids:
                raise ValidationError(f"Duplicate trial_id across raw files: {trial_id!r}")
            seen_trial_ids.add(trial_id)
            semantic = _semantic_identity(row)
            previous = seen_semantic.get(semantic)
            if previous is not None:
                raise ValidationError(
                    "Duplicate semantic trial identity across raw files: "
                    f"{trial_id!r} in {shard.raw_name} duplicates {previous[0]!r} in {previous[1]}"
                )
            seen_semantic[semantic] = (trial_id, shard.raw_name)
            counts[(row["defense"], row["scenario"])] += 1
            suite_counts[(row["suite"], row["defense"], row["scenario"])] += 1
            merged_rows.append(row)

    if len(merged_rows) != EXPECTED_TRIALS:
        raise ValidationError(f"Full matrix must contain {EXPECTED_TRIALS} trials; found {len(merged_rows)}")
    by_defense = {defense: _count_payload(counts, defense) for defense in EXPECTED_DEFENSES}
    for defense, actual in by_defense.items():
        if actual != EXPECTED_PER_DEFENSE:
            raise ValidationError(
                f"Defense {defense!r} has wrong full-matrix counts: "
                f"expected {EXPECTED_PER_DEFENSE}, found {actual}"
            )

    by_suite: dict[str, dict[str, dict[str, int]]] = {}
    for suite, expected in SUITE_COUNTS.items():
        by_suite[suite] = {}
        for defense in EXPECTED_DEFENSES:
            actual = {
                "clean": suite_counts[(suite, defense, "clean")],
                "attacked": suite_counts[(suite, defense, "attacked")],
            }
            actual["total"] = actual["clean"] + actual["attacked"]
            expected_with_total = {**expected, "total": expected["clean"] + expected["attacked"]}
            if actual != expected_with_total:
                raise ValidationError(
                    f"Suite {suite!r}, defense {defense!r} has wrong counts: "
                    f"expected {expected_with_total}, found {actual}"
                )
            by_suite[suite][defense] = actual

    try:
        metrics = aggregate_results(merged_rows)
    except ValueError as error:
        raise ValidationError(f"Merged rows fail result-schema aggregation: {error}") from error
    valid_trials = sum(row.get("valid") is True for row in merged_rows)
    summary = {
        "schema_version": 1,
        "status": "validated",
        "benchmark_version": "v1.1.2",
        "attack": "important_instructions",
        "manifest_count": len(shards),
        "raw_file_count": len(shards),
        "trial_count": len(merged_rows),
        "semantic_identity_count": len(seen_semantic),
        "valid_trials": valid_trials,
        "invalid_trials": len(merged_rows) - valid_trials,
        "by_defense": by_defense,
        "by_suite": by_suite,
        "per_manifest": [
            {
                "manifest": shard.manifest_name,
                "raw": shard.raw_name,
                "manifest_sha256": shard.manifest_sha256,
                "run_fingerprint": shard.run_fingerprint,
                "trial_count": len(shard.rows),
            }
            for shard in shards
        ],
        "metrics": metrics,
    }
    return merged_rows, summary


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise ValidationError(f"Output already exists: {path.name}") from error


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    merged_jsonl: Path | None,
    summary_json: Path | None,
) -> None:
    requested = [path.resolve() for path in (merged_jsonl, summary_json) if path is not None]
    if len(set(requested)) != len(requested):
        raise ValidationError("Merged JSONL and summary JSON must use different output files")
    existing = [path.name for path in (merged_jsonl, summary_json) if path is not None and path.exists()]
    if existing:
        raise ValidationError(f"Output already exists: {', '.join(existing)}")
    if merged_jsonl is not None:
        _write_new(merged_jsonl, _serialize_jsonl(rows))
    if summary_json is not None:
        _write_new(
            summary_json,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and optionally merge the fixed AgentDojo v1.1.2 full matrix."
    )
    parser.add_argument("manifests_dir", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--merged-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rows, summary = validate_full_results(args.manifests_dir, args.raw_dir)
        write_outputs(
            rows,
            summary,
            merged_jsonl=args.merged_jsonl,
            summary_json=args.summary_json,
        )
    except ValidationError as error:
        print(f"validation failed: {error}")
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
