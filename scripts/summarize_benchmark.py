#!/usr/bin/env python3
"""Collate repeated surrogate benchmark JSON files into a compact CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


RESULT_ROOT = Path("results/cantilever/benchmark")
CONFIG_PATH = Path("configs/benchmark_pilot_v1.json")
METRICS = ("r2", "rmse", "nrmse_range", "mae", "fit_seconds", "model_size")


def main() -> None:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_seeds = [int(seed) for seed in configuration["seeds"]]
    expected_algorithms = list(configuration["algorithms"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for path in sorted(RESULT_ROOT.glob("*/seed-*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(str(result["algorithm"]), []).append(result)

    rows: list[dict[str, object]] = []
    for algorithm in expected_algorithms:
        trials = grouped.get(algorithm, [])
        completed_seeds = sorted(int(trial["seed"]) for trial in trials)
        missing_seeds = sorted(set(expected_seeds) - set(completed_seeds))
        row: dict[str, object] = {
            "algorithm": algorithm,
            "expected_trials": len(expected_seeds),
            "successful_trials": len(trials),
            "failed_or_missing_trials": len(missing_seeds),
            "seeds": ";".join(str(seed) for seed in completed_seeds),
            "missing_seeds": ";".join(str(seed) for seed in missing_seeds),
        }
        for metric in METRICS:
            values = [trial[metric] for trial in trials if trial.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / "summary.csv"
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} algorithms")


if __name__ == "__main__":
    main()
