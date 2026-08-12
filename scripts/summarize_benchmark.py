#!/usr/bin/env python3
"""Collate repeated surrogate benchmark JSON files into a compact CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "r2",
    "rmse",
    "nrmse_range",
    "mae",
    "fit_seconds",
    "prediction_median_seconds",
    "prediction_microseconds_per_sample",
    "analysis_seconds",
    "expression_node_count",
    "expression_depth",
    "simplified_node_count",
    "simplified_depth",
    "simplified_to_ground_truth_size_ratio",
    "ground_truth_variable_recall",
    "training_scale_simplified_node_count",
    "training_scale_simplified_depth",
)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="cantilever")
    parser.add_argument("--input-scaling", default="raw")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / "configs/benchmark_suite_v3.json",
    )
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = project_dir / "results" / args.problem / str(configuration["name"])
    if "input_scalings" in configuration:
        result_root = result_root / args.input_scaling
    legacy_root = project_dir / "results" / args.problem / "benchmark"
    if not result_root.exists() and configuration["name"] in {
        "benchmark_pilot_v1",
        "benchmark_suite_v1",
    }:
        result_root = legacy_root
    expected_seeds = [int(seed) for seed in configuration["seeds"]]
    expected_algorithms = list(configuration["algorithms"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for path in sorted(result_root.glob("*/seed-*.json")):
        if path.name.endswith(".analysis.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if int(result["seed"]) not in expected_seeds:
            continue
        analysis_path = path.with_name(f"{path.stem}.analysis.json")
        result["analysis_available"] = analysis_path.exists()
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if analysis.get("source_result_sha256") != expected_hash:
                raise ValueError(f"Stale analysis sidecar for {path}")
            result.update(analysis)
        else:
            # Do not reuse analysis embedded by the older, coupled evaluator.
            for metric in METRICS:
                if metric not in {"r2", "rmse", "nrmse_range", "mae", "fit_seconds",
                                  "prediction_median_seconds",
                                  "prediction_microseconds_per_sample"}:
                    result[metric] = None
            for key in ("expression_parse_success", "expression_simplify_success",
                        "symbolic_exact_match", "complexity_valid", "complexity_source"):
                result[key] = None
        grouped.setdefault(str(result["algorithm"]), []).append(result)
    failures: dict[str, list[dict[str, object]]] = {}
    for path in sorted((result_root / "failures").glob("*/seed-*.json")):
        failure = json.loads(path.read_text(encoding="utf-8"))
        if int(failure["seed"]) not in expected_seeds:
            continue
        failures.setdefault(str(failure["algorithm"]), []).append(failure)

    repetition_protocol = configuration.get("repetition_protocol", {})
    uncontrolled = set(
        repetition_protocol.get("uncontrolled_repetition_algorithms", [])
    )

    rows: list[dict[str, object]] = []
    for algorithm in expected_algorithms:
        trials = grouped.get(algorithm, [])
        failed_seeds = sorted(
            int(failure["seed"]) for failure in failures.get(algorithm, [])
        )
        trials = [
            trial for trial in trials if int(trial["seed"]) not in set(failed_seeds)
        ]
        completed_seeds = sorted(int(trial["seed"]) for trial in trials)
        missing_seeds = sorted(
            set(expected_seeds) - set(completed_seeds) - set(failed_seeds)
        )
        symbolic_trials = [
            trial for trial in trials if trial.get("symbolic_exact_match") is not None
        ]
        symbolic_matches = sum(
            trial.get("symbolic_exact_match") is True for trial in symbolic_trials
        )
        parsed_trials = sum(
            trial.get("expression_parse_success") is True for trial in trials
        )
        analysis_trials = sum(trial.get("analysis_available") is True for trial in trials)
        complexity_valid_trials = sum(
            trial.get("complexity_valid") is True for trial in trials
        )
        simplified_complexity_trials = sum(
            trial.get("complexity_source") == "simplified" for trial in trials
        )
        fallback_trials = sum(
            trial.get("complexity_source") == "unsimplified_fallback" for trial in trials
        )
        row: dict[str, object] = {
            "algorithm": algorithm,
            "input_scaling": args.input_scaling,
            "repetition_type": (
                "uncontrolled" if algorithm in uncontrolled else "seed-controlled"
            ),
            "expected_trials": len(expected_seeds),
            "successful_trials": len(trials),
            "failed_trials": len(failed_seeds),
            "missing_trials": len(missing_seeds),
            "failed_or_missing_trials": len(failed_seeds) + len(missing_seeds),
            "seeds": ";".join(str(seed) for seed in completed_seeds),
            "failed_seeds": ";".join(str(seed) for seed in failed_seeds),
            "missing_seeds": ";".join(str(seed) for seed in missing_seeds),
            "analysis_trials": analysis_trials,
            "analysis_coverage": analysis_trials / len(trials) if trials else None,
            "parsed_expression_trials": parsed_trials,
            "expression_parse_success_rate": (
                parsed_trials / len(trials) if trials else None
            ),
            "complexity_valid_trials": complexity_valid_trials,
            "complexity_coverage": (
                complexity_valid_trials / len(trials) if trials else None
            ),
            "simplified_complexity_trials": simplified_complexity_trials,
            "simplified_complexity_coverage": (
                simplified_complexity_trials / len(trials) if trials else None
            ),
            "unsimplified_fallback_trials": fallback_trials,
            "unsimplified_fallback_rate": (
                fallback_trials / len(trials) if trials else None
            ),
            "symbolic_exact_matches": symbolic_matches,
            "symbolic_exact_match_rate": (
                symbolic_matches / len(symbolic_trials) if symbolic_trials else None
            ),
        }
        for metric in METRICS:
            values = [trial[metric] for trial in trials if trial.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row[f"{metric}_median"] = float(np.median(values)) if values else None
            row[f"{metric}_min"] = float(np.min(values)) if values else None
            row[f"{metric}_max"] = float(np.max(values)) if values else None
        rows.append(row)

    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / "summary.csv"
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} algorithms")


if __name__ == "__main__":
    main()
