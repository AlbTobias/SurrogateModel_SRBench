#!/usr/bin/env python3
"""Evaluate an SRBench adapter on predefined surrogate train/test datasets."""

from __future__ import annotations

import argparse
import importlib
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="gplearn")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=30)
    args = parser.parse_args()

    algorithm = importlib.import_module(f"experiment.methods.{args.algorithm}.regressor")
    estimator = clone(algorithm.est)
    supported = estimator.get_params(deep=True)
    overrides = {
        "random_state": args.seed,
        "population_size": args.population_size,
        "generations": args.generations,
    }
    estimator.set_params(**{key: value for key, value in overrides.items() if key in supported})

    x_train, y_train = load_dataset(args.train)
    x_test, y_test = load_dataset(args.test)
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Train and test feature columns differ")
    if "feature_names" in supported:
        estimator.set_params(feature_names=list(x_train.columns))

    fit_started = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    prediction = np.asarray(estimator.predict(x_test)).reshape(-1)
    prediction_seconds = time.perf_counter() - prediction_started

    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    target_range = float(np.max(y_test) - np.min(y_test))
    expression = algorithm.model(estimator, x_train) if algorithm.model else None
    model_size = algorithm.complexity(estimator) if getattr(algorithm, "complexity", None) else None

    result = {
        "problem": "cantilever_tip_deflection",
        "algorithm": args.algorithm,
        "seed": args.seed,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "population_size": args.population_size,
        "generations": args.generations,
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
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
