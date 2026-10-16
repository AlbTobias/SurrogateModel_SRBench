#!/usr/bin/env python3
"""Generate fixed train/test data for the OTL Circuit surrogate problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


COLUMNS = (
    "base_resistance_1",
    "base_resistance_2",
    "feedback_resistance",
    "collector_resistance_1",
    "collector_resistance_2",
    "current_gain",
)
LOWER = np.array([50.0, 25.0, 0.5, 1.2, 0.25, 50.0])
UPPER = np.array([150.0, 70.0, 3.0, 2.5, 1.2, 300.0])


def otl_midpoint_voltage(x: np.ndarray) -> np.ndarray:
    """Return the midpoint voltage of the output-transformerless circuit."""

    rb1, rb2, rf, rc1, rc2, beta = x.T
    vb1 = 12.0 * rb2 / (rb1 + rb2)
    denominator = beta * (rc2 + 9.0) + rf
    term1 = (vb1 + 0.74) * beta * (rc2 + 9.0) / denominator
    term2 = 11.35 * rf / denominator
    term3 = 0.74 * rf * beta * (rc2 + 9.0) / (denominator * rc1)
    return term1 + term2 + term3


def sample_inputs(seed: int, samples: int) -> np.ndarray:
    unit_sample = qmc.LatinHypercube(
        d=len(COLUMNS), scramble=True, seed=seed
    ).random(samples)
    return qmc.scale(unit_sample, LOWER, UPPER)


def write_dataset(path: Path, x: np.ndarray) -> None:
    frame = pd.DataFrame(x, columns=COLUMNS)
    frame["target"] = otl_midpoint_voltage(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/otl_circuit"))
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--train-seed", type=int, default=20260903)
    parser.add_argument("--test-seed", type=int, default=20260904)
    args = parser.parse_args()

    write_dataset(
        args.output_dir / "train.tsv.gz",
        sample_inputs(args.train_seed, args.train_size),
    )
    write_dataset(
        args.output_dir / "test.tsv.gz",
        sample_inputs(args.test_seed, args.test_size),
    )


if __name__ == "__main__":
    main()
