#!/usr/bin/env python3
"""Resolve the newest protocol-compatible result for every trial coordinate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


METRICS = (
    "r2", "rmse", "nrmse_range", "mae", "fit_seconds",
    "prediction_median_seconds", "prediction_microseconds_per_sample",
    "expression_extraction_seconds", "analysis_seconds", "expression_node_count",
    "expression_depth", "simplified_node_count", "simplified_depth",
    "simplified_to_ground_truth_size_ratio", "ground_truth_variable_recall",
    "training_scale_simplified_node_count", "training_scale_simplified_depth",
)
SCALINGS = ("raw", "domain_minmax")
SUITE_PATTERN = re.compile(r"benchmark_suite_v(\d+)$")


def suite_version(name: str) -> int:
    match = SUITE_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unsupported benchmark-suite name: {name}")
    return int(match.group(1))


def containing_suite(path: Path) -> str:
    for parent in path.parents:
        if SUITE_PATTERN.fullmatch(parent.name):
            return parent.name
    raise ValueError(f"Result is not contained in a benchmark suite: {path}")


def effective_algorithms(configuration: dict[str, object]) -> dict[str, object]:
    algorithms = configuration["algorithms"]
    assert isinstance(algorithms, dict)
    return {
        name: details.get("parameters", {})
        for name, details in algorithms.items()
    }


def compatible_with_reference(
    candidate: dict[str, object], reference: dict[str, object], problem: str
) -> bool:
    if problem not in candidate.get("problems", []):
        return False
    fixed_keys = (
        "configuration_schema_version", "repetition_protocol", "execution_controls",
        "seeds", "input_scalings",
    )
    if any(candidate.get(key) != reference.get(key) for key in fixed_keys):
        return False
    if effective_algorithms(candidate) != effective_algorithms(reference):
        return False
    candidate_generation = candidate.get("dataset_generation", {}).get(problem)
    reference_generation = reference.get("dataset_generation", {}).get(problem)
    return candidate_generation == reference_generation


def load_analysis(result_path: Path, result: dict[str, object]) -> None:
    sidecar = result_path.with_name(f"{result_path.stem}.analysis.json")
    result["analysis_available"] = False
    if not sidecar.exists():
        return
    analysis = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if analysis.get("source_result_sha256") != expected_hash:
        raise ValueError(f"Stale analysis sidecar: {sidecar}")
    result.update(analysis)
    result["analysis_available"] = True


def resolve_trials(
    project_dir: Path, reference: dict[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    configurations: dict[str, dict[str, object]] = {}
    for path in (project_dir / "configs").glob("benchmark_suite_v*.json"):
        configuration = json.loads(path.read_text(encoding="utf-8"))
        configurations[str(configuration["name"])] = configuration

    algorithms = tuple(effective_algorithms(reference))
    seeds = tuple(int(seed) for seed in reference["seeds"])
    resolved: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for problem in reference["problems"]:
        problem_root = project_dir / "results" / str(problem)
        suites = sorted(
            (
                path for path in problem_root.glob("benchmark_suite_v*")
                if path.name in configurations
                and compatible_with_reference(configurations[path.name], reference, str(problem))
            ),
            key=lambda path: suite_version(path.name),
        )
        for scaling in SCALINGS:
            for algorithm in algorithms:
                for seed in seeds:
                    candidates: list[tuple[int, int, Path, str]] = []
                    for suite in suites:
                        success = suite / scaling / algorithm / f"seed-{seed}.json"
                        failure = suite / scaling / "failures" / algorithm / f"seed-{seed}.json"
                        if success.exists():
                            candidates.append((suite_version(suite.name), success.stat().st_mtime_ns,
                                               success, "success"))
                        if failure.exists():
                            candidates.append((suite_version(suite.name), failure.stat().st_mtime_ns,
                                               failure, "failed"))
                    coordinate = {
                        "problem": problem, "input_scaling": scaling,
                        "algorithm": algorithm, "seed": seed,
                    }
                    if not candidates:
                        manifest.append({**coordinate, "status": "missing", "suite": "", "path": ""})
                        continue
                    _, _, selected_path, status = max(candidates, key=lambda item: (item[0], item[1]))
                    record = json.loads(selected_path.read_text(encoding="utf-8"))
                    record.update(coordinate)
                    record["resolved_suite"] = containing_suite(selected_path)
                    record["resolved_path"] = str(selected_path.relative_to(project_dir))
                    if status == "success":
                        load_analysis(selected_path, record)
                    resolved.append(record)
                    manifest.append({
                        **coordinate, "status": status, "suite": record["resolved_suite"],
                        "path": record["resolved_path"],
                        "analysis_available": record.get("analysis_available", False),
                    })
    return resolved, manifest


def aggregate(
    records: list[dict[str, object]], reference: dict[str, object]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected = len(reference["seeds"])
    uncontrolled = set(reference.get("repetition_protocol", {}).get(
        "uncontrolled_repetition_algorithms", []
    ))
    for problem in reference["problems"]:
        for scaling in SCALINGS:
            for algorithm in effective_algorithms(reference):
                selected = [
                    record for record in records
                    if record["problem"] == problem
                    and record["input_scaling"] == scaling
                    and record["algorithm"] == algorithm
                ]
                trials = [record for record in selected if record.get("status") == "success"]
                failures = [record for record in selected if record.get("status") == "failed"]
                parsed = sum(record.get("expression_parse_success") is True for record in trials)
                complexity_valid = sum(record.get("complexity_valid") is True for record in trials)
                simplified = sum(record.get("complexity_source") == "simplified" for record in trials)
                fallback = sum(
                    record.get("complexity_source") == "unsimplified_fallback" for record in trials
                )
                symbolic_trials = [
                    record for record in trials if record.get("symbolic_exact_match") is not None
                ]
                row: dict[str, object] = {
                    "problem": problem,
                    "algorithm": algorithm,
                    "input_scaling": scaling,
                    "repetition_type": "uncontrolled" if algorithm in uncontrolled else "seed-controlled",
                    "expected_trials": expected,
                    "successful_trials": len(trials),
                    "failed_trials": len(failures),
                    "missing_trials": expected - len(trials) - len(failures),
                    "analysis_trials": sum(record.get("analysis_available") is True for record in trials),
                    "parsed_expression_trials": parsed,
                    "complexity_valid_trials": complexity_valid,
                    "simplified_complexity_trials": simplified,
                    "unsimplified_fallback_trials": fallback,
                    "symbolic_exact_matches": sum(
                        record.get("symbolic_exact_match") is True for record in symbolic_trials
                    ),
                    "source_suites": ";".join(sorted(
                        {str(record["resolved_suite"]) for record in selected},
                        key=suite_version,
                    )),
                }
                for metric in METRICS:
                    values = [
                        float(record[metric]) for record in trials
                        if record.get(metric) is not None
                    ]
                    row[f"{metric}_mean"] = float(np.mean(values)) if values else None
                    row[f"{metric}_std"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    )
                    row[f"{metric}_median"] = float(np.median(values)) if values else None
                    row[f"{metric}_min"] = float(np.min(values)) if values else None
                    row[f"{metric}_max"] = float(np.max(values)) if values else None
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latest_summary(
    project_dir: Path, config_path: Path, output_dir: Path
) -> dict[str, int]:
    reference = json.loads(config_path.read_text(encoding="utf-8"))
    records, manifest = resolve_trials(project_dir, reference)
    rows = aggregate(records, reference)
    write_csv(output_dir / "summary.csv", rows)
    write_csv(output_dir / "manifest.csv", manifest)
    statuses = {
        status: sum(row["status"] == status for row in manifest)
        for status in ("success", "failed", "missing")
    }
    statuses["summary_rows"] = len(rows)
    statuses["coordinates"] = len(manifest)
    statuses["analysis_missing"] = sum(
        row["status"] == "success" and not row.get("analysis_available", False)
        for row in manifest
    )
    return statuses


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=project_dir / "configs" / "benchmark_suite_v10.json",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results" / "latest")
    args = parser.parse_args()
    statuses = write_latest_summary(project_dir, args.config, args.output_dir)
    print(
        f"Wrote {statuses['summary_rows']} summary rows and "
        f"{statuses['coordinates']} resolved coordinates"
    )
    print("Coverage: " + ", ".join(f"{key}={value}" for key, value in statuses.items()))


if __name__ == "__main__":
    main()
