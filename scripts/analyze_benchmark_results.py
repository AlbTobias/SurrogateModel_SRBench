#!/usr/bin/env python3
"""Post-process stored benchmark expressions into analysis sidecars."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--benchmark", default="benchmark_suite_v3")
    parser.add_argument(
        "--config", type=Path, default=project_dir / "configs/benchmark_suite_v3.json"
    )
    parser.add_argument("--input-scaling", choices=("raw", "domain_minmax"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    expected_seeds = {int(seed) for seed in configuration["seeds"]}
    expected_algorithms = set(configuration["algorithms"])

    root = project_dir / "results" / args.problem / args.benchmark
    roots = [root / args.input_scaling] if args.input_scaling else [root / "raw", root / "domain_minmax"]
    failures: list[Path] = []
    tasks: list[tuple[Path, Path]] = []
    skipped = 0
    for scaling_root in roots:
        for result_path in sorted(scaling_root.glob("*/seed-*.json")):
            if result_path.name.endswith(".analysis.json"):
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (int(result["seed"]) not in expected_seeds
                    or str(result["algorithm"]) not in expected_algorithms):
                continue
            output = result_path.with_name(f"{result_path.stem}.analysis.json")
            if output.exists() and not args.force:
                skipped += 1
                continue
            tasks.append((result_path, output))

    def run(task: tuple[Path, Path]) -> tuple[Path, int]:
        result_path, output = task
        process = subprocess.run(
            [sys.executable, str(project_dir / "scripts/analyze_expression_result.py"),
             str(result_path), "--output", str(output), "--timeout", str(args.timeout)],
            check=False,
        )
        return result_path, process.returncode

    completed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            result_path, returncode = future.result()
            if returncode:
                failures.append(result_path)
            else:
                completed += 1
    print(f"Analysis complete: written={completed} skipped={skipped} failed={len(failures)}")
    if failures:
        print("Failed: " + ", ".join(str(path) for path in failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
