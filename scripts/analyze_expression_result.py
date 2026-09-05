#!/usr/bin/env python3
"""Analyze one stored benchmark expression without rerunning its estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment.metrics.expression_analysis import analyze_expression, time_limit
from experiment.metrics.input_scaling import (
    SCALING_DOMAIN_MINMAX,
    expression_to_raw_scale,
)


ANALYSIS_VERSION = "expression_analysis_v2"


def analyze_result(
    result: dict[str, object], timeout_seconds: int, simplify: bool = True
) -> dict[str, object]:
    feature_names = list(result.get("feature_names") or result["input_scaling_parameters"])
    expression = result.get("symbolic_model")
    started = time.perf_counter()
    training_analysis = analyze_expression(
        expression,
        feature_names,
        "training_scale_without_ground_truth",
        timeout_seconds=timeout_seconds,
        simplify=simplify,
    )

    back_transform_error = None
    if expression and result["input_scaling"] == SCALING_DOMAIN_MINMAX:
        try:
            with time_limit(timeout_seconds):
                raw_expression = expression_to_raw_scale(
                    str(expression), feature_names, result["input_scaling_parameters"]
                )
        except Exception as error:
            raw_expression = None
            back_transform_error = f"{type(error).__name__}: {error}"
    else:
        raw_expression = expression

    physical_analysis = analyze_expression(
        str(raw_expression) if raw_expression is not None else None,
        feature_names,
        str(result["problem"]),
        timeout_seconds=timeout_seconds,
        simplify=simplify,
    )
    parse_success = physical_analysis["expression_parse_success"] is True
    simplify_success = physical_analysis["expression_simplify_success"] is True
    return {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_status": "complete",
        "problem": result["problem"],
        "algorithm": result["algorithm"],
        "seed": result["seed"],
        "input_scaling": result["input_scaling"],
        "expression_timeout_seconds_per_stage": timeout_seconds,
        "simplification_requested": simplify,
        "analysis_seconds": time.perf_counter() - started,
        "analysis_python": platform.python_version(),
        "analysis_platform": platform.platform(),
        "raw_scale_symbolic_model": raw_expression,
        "raw_scale_back_transform_error": back_transform_error,
        "complexity_valid": parse_success,
        "complexity_source": (
            "simplified" if simplify_success else "unsimplified_fallback" if parse_success else None
        ),
        **{f"training_scale_{key}": value for key, value in training_analysis.items()},
        **physical_analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-simplification", action="store_true")
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    analysis = analyze_result(result, args.timeout, simplify=not args.skip_simplification)
    analysis["source_result"] = args.result.name
    analysis["source_result_sha256"] = hashlib.sha256(args.result.read_bytes()).hexdigest()
    output = args.output or args.result.with_name(
        f"{args.result.stem}.analysis.json"
    )
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
