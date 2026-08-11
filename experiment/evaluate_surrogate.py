#!/usr/bin/env python3
"""Evaluate an SRBench adapter on predefined surrogate train/test datasets."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, sep="\t", compression="infer")
    if "target" not in frame:
        raise ValueError(f'{path} must contain a "target" column')
    return frame.drop(columns="target"), frame["target"].to_numpy()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="gplearn")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--profile", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    algorithm = importlib.import_module(f"experiment.methods.{args.algorithm}.regressor")
    estimator = clone(algorithm.est)
    supported = estimator.get_params(deep=True)
    benchmark_name = None
    configured_parameters: dict[str, object] = {}
    if args.config:
        configuration = json.loads(args.config.read_text(encoding="utf-8"))
        benchmark_name = configuration.get("name")
        configured_parameters = configuration.get("algorithms", {}).get(args.algorithm, {})
        if args.profile == "benchmark" and not configured_parameters:
            raise ValueError(f"No benchmark parameters configured for {args.algorithm}")
    overrides: dict[str, object] = {}
    for key in ("random_state", "seed"):
        if key in supported:
            overrides[key] = args.seed
    if args.profile == "smoke":
        overrides.update(getattr(algorithm, "eval_kwargs", {}).get("test_params", {}))
        for key in ("population_size", "npop"):
            if key in supported:
                overrides[key] = args.population_size
        for key in (
            "generations",
            "ngens",
            "n_iter",
            "niterations",
            "number_of_generations",
        ):
            if key in supported:
                overrides[key] = args.iterations
        for key in ("max_time", "timeout_in_seconds", "timer_limit"):
            if key in supported:
                overrides[key] = args.time_limit
        if "n_trials" in supported:
            overrides["n_trials"] = 2
    elif args.profile == "benchmark":
        unknown = sorted(set(configured_parameters) - set(supported))
        if unknown:
            raise ValueError(
                f"Unsupported configured parameters for {args.algorithm}: {unknown}"
            )
        overrides.update(configured_parameters)
    estimator.set_params(**{key: value for key, value in overrides.items() if key in supported})

    x_train, y_train = load_dataset(args.train)
    x_test, y_test = load_dataset(args.test)
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Train and test feature columns differ")
    if "feature_names" in supported:
        estimator.set_params(feature_names=list(x_train.columns))

    use_dataframe = getattr(algorithm, "eval_kwargs", {}).get("use_dataframe", True)
    x_train_fit = x_train if use_dataframe else x_train.to_numpy()
    x_test_fit = x_test if use_dataframe else x_test.to_numpy()

    fit_started = time.perf_counter()
    estimator.fit(x_train_fit, y_train)
    fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    prediction = np.asarray(estimator.predict(x_test_fit)).reshape(-1)
    prediction_seconds = time.perf_counter() - prediction_started

    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    target_range = float(np.max(y_test) - np.min(y_test))
    model_function = getattr(algorithm, "model", None)
    if model_function is None:
        expression = None
    elif len(inspect.signature(model_function).parameters) >= 2:
        expression = model_function(estimator, x_train)
    else:
        expression = model_function(estimator)
    complexity_function = getattr(algorithm, "complexity", None)
    model_size = complexity_function(estimator) if complexity_function else None

    result = {
        "problem": "cantilever_tip_deflection",
        "algorithm": args.algorithm,
        "seed": args.seed,
        "seed_parameter": next(
            (key for key in ("random_state", "seed") if key in supported),
            None,
        ),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "requested_population_size": (
            args.population_size if args.profile == "smoke" else None
        ),
        "requested_iterations": args.iterations if args.profile == "smoke" else None,
        "time_limit": args.time_limit if args.profile == "smoke" else None,
        "profile": args.profile,
        "benchmark_name": benchmark_name,
        "config_file": str(args.config) if args.config else None,
        "use_dataframe": use_dataframe,
        "applied_parameters": {
            key: value for key, value in overrides.items() if key in supported
        },
        "rmse": rmse,
        "nrmse_range": rmse / target_range,
        "mae": float(mean_absolute_error(y_test, prediction)),
        "r2": float(r2_score(y_test, prediction)),
        "max_absolute_error": float(np.max(np.abs(y_test - prediction))),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "model_size": model_size,
        "symbolic_model": expression,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    json.dump(result, sys.stdout, indent=2, default=json_default)
    print()


if __name__ == "__main__":
    main()
