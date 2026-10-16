#!/usr/bin/env python3
"""Generate fixed train/test data for the Robot Arm surrogate problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


COLUMNS = (
    "angle_1",
    "angle_2",
    "angle_3",
    "angle_4",
    "length_1",
    "length_2",
    "length_3",
    "length_4",
)
LOWER = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
UPPER = np.array(
    [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi, 1.0, 1.0, 1.0, 1.0]
)


def robot_arm_distance(x: np.ndarray) -> np.ndarray:
    """Return the planar distance from the arm endpoint to the origin."""

    angles = np.cumsum(x[:, :4], axis=1)
    lengths = x[:, 4:]
    horizontal = np.sum(lengths * np.cos(angles), axis=1)
    vertical = np.sum(lengths * np.sin(angles), axis=1)
    return np.sqrt(horizontal**2 + vertical**2)


def sample_inputs(seed: int, samples: int) -> np.ndarray:
    unit_sample = qmc.LatinHypercube(
        d=len(COLUMNS), scramble=True, seed=seed
    ).random(samples)
    return qmc.scale(unit_sample, LOWER, UPPER)


def write_dataset(path: Path, x: np.ndarray) -> None:
    frame = pd.DataFrame(x, columns=COLUMNS)
    frame["target"] = robot_arm_distance(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/robot_arm"))
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--train-seed", type=int, default=20260905)
    parser.add_argument("--test-seed", type=int, default=20260906)
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
