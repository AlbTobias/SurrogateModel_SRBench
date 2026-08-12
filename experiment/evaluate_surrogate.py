#!/usr/bin/env python3
"""Evaluate an SRBench adapter on predefined surrogate train/test datasets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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

from experiment.metrics.input_scaling import (
    SCALING_DOMAIN_MINMAX,
    SUPPORTED_SCALINGS,
    scale_frame_to_unit_interval,
    scaling_metadata,
)


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, sep="\t", compression="infer")
    if "target" not in frame:
        raise ValueError(f'{path} must contain a "target" column')
    return frame.drop(columns="target"), frame["target"].to_numpy()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_dataset(path: Path) -> str:
    """Hash uncompressed table content so gzip timestamps do not affect identity."""

    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="gplearn")
    parser.add_argument("--problem", default="cantilever")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--profile", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prediction-repeats", type=int)
    parser.add_argument("--input-scaling", choices=SUPPORTED_SCALINGS, default="raw")
    args = parser.parse_args()

    algorithm = importlib.import_module(f"experiment.methods.{args.algorithm}.regressor")
    estimator = clone(algorithm.est)
    supported = estimator.get_params(deep=True)
    benchmark_name = None
    configuration: dict[str, object] = {}
    configured_parameters: dict[str, object] = {}
    configuration_role = None
    configuration_rationale = None
    configuration_sha256 = None
    if args.config:
        configuration = json.loads(args.config.read_text(encoding="utf-8"))
        configuration_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
        benchmark_name = configuration.get("name")
        algorithm_configuration = configuration.get("algorithms", {}).get(
            args.algorithm, {}
        )
        configured_parameters = algorithm_configuration.get(
            "parameters", algorithm_configuration
        )
        configuration_role = algorithm_configuration.get("configuration_role")
        configuration_rationale = algorithm_configuration.get("rationale")
        if args.profile == "benchmark" and not configured_parameters:
            raise ValueError(f"No benchmark parameters configured for {args.algorithm}")
    prediction_repeats = args.prediction_repeats or int(
        configuration.get("execution_controls", {}).get(
            "prediction_repeats", configuration.get("prediction_repeats", 1)
        )
    )
    if prediction_repeats < 1:
        raise ValueError("prediction repeats must be at least one")
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
    feature_names = list(x_train.columns)
    domain_metadata = scaling_metadata(args.problem, feature_names)
    if args.input_scaling == SCALING_DOMAIN_MINMAX:
        x_train_model = scale_frame_to_unit_interval(x_train, domain_metadata)
        x_test_model = scale_frame_to_unit_interval(x_test, domain_metadata)
    else:
        x_train_model = x_train
        x_test_model = x_test
    if "feature_names" in supported:
        estimator.set_params(feature_names=feature_names)

    use_dataframe = getattr(algorithm, "eval_kwargs", {}).get("use_dataframe", True)
    x_train_fit = x_train_model if use_dataframe else x_train_model.to_numpy()
    x_test_fit = x_test_model if use_dataframe else x_test_model.to_numpy()

    evaluation_started = time.perf_counter()
    fit_started = time.perf_counter()
    estimator.fit(x_train_fit, y_train)
    fit_seconds = time.perf_counter() - fit_started

    prediction_timings: list[float] = []
    prediction = None
    for _ in range(prediction_repeats):
        prediction_started = time.perf_counter()
        current_prediction = np.asarray(estimator.predict(x_test_fit)).reshape(-1)
        prediction_timings.append(time.perf_counter() - prediction_started)
        if prediction is None:
            prediction = current_prediction
    assert prediction is not None

    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    target_range = float(np.max(y_test) - np.min(y_test))
    extraction_started = time.perf_counter()
    model_function = getattr(algorithm, "model", None)
    if model_function is None:
        expression = None
    elif len(inspect.signature(model_function).parameters) >= 2:
        expression = model_function(estimator, x_train)
    else:
        expression = model_function(estimator)
    complexity_function = getattr(algorithm, "complexity", None)
    model_size = complexity_function(estimator) if complexity_function else None
    expression_extraction_seconds = time.perf_counter() - extraction_started

    evaluation_seconds = time.perf_counter() - evaluation_started

    result = {
        "status": "success",
        "problem": args.problem,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "seed_parameter": next(
            (key for key in ("random_state", "seed") if key in supported),
            None,
        ),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "feature_names": feature_names,
        "dataset_policy": "fixed across algorithm trials",
        "dataset_hash_scope": "SHA-256 of uncompressed table bytes",
        "train_dataset_sha256": sha256_dataset(args.train),
        "test_dataset_sha256": sha256_dataset(args.test),
        "requested_population_size": (
            args.population_size if args.profile == "smoke" else None
        ),
        "requested_iterations": args.iterations if args.profile == "smoke" else None,
        "time_limit": args.time_limit if args.profile == "smoke" else None,
        "profile": args.profile,
        "benchmark_name": benchmark_name,
        "config_file": str(args.config) if args.config else None,
        "configuration_schema_version": configuration.get(
            "configuration_schema_version"
        ),
        "configuration_sha256": configuration_sha256,
        "use_dataframe": use_dataframe,
        "input_scaling": args.input_scaling,
        "input_scaling_applied": args.input_scaling == SCALING_DOMAIN_MINMAX,
        "input_scaling_source": "fixed published problem domain",
        "input_scaling_formula": (
            "z = 2 * (x - lower) / (upper - lower) - 1"
            if args.input_scaling == SCALING_DOMAIN_MINMAX
            else "x unchanged"
        ),
        "input_scaling_parameters": domain_metadata,
        "target_scaling": "none; target remains in original units",
        "algorithm_configuration_role": configuration_role,
        "algorithm_configuration_rationale": configuration_rationale,
        "configured_parameters": configured_parameters,
        "applied_parameters": {
            key: value for key, value in overrides.items() if key in supported
        },
        "rmse": rmse,
        "nrmse_range": rmse / target_range,
        "mae": float(mean_absolute_error(y_test, prediction)),
        "r2": float(r2_score(y_test, prediction)),
        "max_absolute_error": float(np.max(np.abs(y_test - prediction))),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_timings[0],
        "prediction_repeats": prediction_repeats,
        "prediction_mean_seconds": float(np.mean(prediction_timings)),
        "prediction_median_seconds": float(np.median(prediction_timings)),
        "prediction_std_seconds": (
            float(np.std(prediction_timings, ddof=1))
            if len(prediction_timings) > 1
            else None
        ),
        "prediction_microseconds_per_sample": (
            1e6 * float(np.median(prediction_timings)) / len(y_test)
        ),
        "expression_extraction_seconds": expression_extraction_seconds,
        "evaluation_seconds": evaluation_seconds,
        "timing_scope": (
            "fit/predict/expression extraction inside the running container; "
            "symbolic analysis, "
            "image pull, container startup, data generation, and metric serialization excluded"
        ),
        "model_size": model_size,
        "symbolic_model": expression,
        "symbolic_model_coordinate_system": args.input_scaling,
        "symbolic_analysis": "separate post-processing sidecar",
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
