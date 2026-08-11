#!/usr/bin/env python3
"""Collate repeated surrogate benchmark JSON files into a compact CSV."""

from __future__ import annotations

import argparse
import csv
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
    "expression_analysis_seconds",
    "simplified_node_count",
    "simplified_depth",
    "simplified_to_ground_truth_size_ratio",
    "ground_truth_variable_recall",
)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="cantilever")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / "configs/benchmark_suite_v2.json",
    )
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = (
        project_dir / "results" / args.problem / str(configuration["name"])
    )
    legacy_root = project_dir / "results" / args.problem / "benchmark"
    if not result_root.exists() and configuration["name"] != "benchmark_suite_v2":
        result_root = legacy_root
    expected_seeds = [int(seed) for seed in configuration["seeds"]]
    expected_algorithms = list(configuration["algorithms"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for path in sorted(result_root.glob("*/seed-*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(str(result["algorithm"]), []).append(result)

    rows: list[dict[str, object]] = []
    for algorithm in expected_algorithms:
        trials = grouped.get(algorithm, [])
        completed_seeds = sorted(int(trial["seed"]) for trial in trials)
        missing_seeds = sorted(set(expected_seeds) - set(completed_seeds))
        symbolic_trials = [
            trial for trial in trials if trial.get("symbolic_exact_match") is not None
        ]
        symbolic_matches = sum(
            trial.get("symbolic_exact_match") is True for trial in symbolic_trials
        )
        parsed_trials = sum(
            trial.get("expression_parse_success") is True for trial in trials
        )
        row: dict[str, object] = {
            "algorithm": algorithm,
            "expected_trials": len(expected_seeds),
            "successful_trials": len(trials),
            "failed_or_missing_trials": len(missing_seeds),
            "seeds": ";".join(str(seed) for seed in completed_seeds),
            "missing_seeds": ";".join(str(seed) for seed in missing_seeds),
            "parsed_expression_trials": parsed_trials,
            "expression_parse_success_rate": (
                parsed_trials / len(trials) if trials else None
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
