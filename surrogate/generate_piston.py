#!/usr/bin/env python3
"""Generate fixed train/test data for the Piston surrogate problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


COLUMNS = ("piston_mass", "surface_area", "initial_volume", "spring_coefficient",
           "atmospheric_pressure", "ambient_temperature", "gas_temperature")
LOWER = np.array([30.0, 0.005, 0.002, 1000.0, 90000.0, 290.0, 340.0])
UPPER = np.array([60.0, 0.020, 0.010, 5000.0, 110000.0, 296.0, 360.0])


def piston_cycle_time(x: np.ndarray) -> np.ndarray:
    """Cycle time of a piston moving in a cylinder, in seconds."""
    mass, area, volume_0, spring, pressure_0, temp_a, temp_0 = x.T
    a_term = pressure_0 * area + 19.62 * mass - spring * volume_0 / area
    volume = area / (2.0 * spring) * (
        np.sqrt(a_term**2 + 4.0 * spring * pressure_0 * volume_0 * temp_a / temp_0)
        - a_term
    )
    restoring_term = spring + area**2 * pressure_0 * volume_0 * temp_a / (
        temp_0 * volume**2
    )
    return 2.0 * np.pi * np.sqrt(mass / restoring_term)


def sample_inputs(seed: int, samples: int) -> np.ndarray:
    unit_sample = qmc.LatinHypercube(d=len(COLUMNS), seed=seed).random(samples)
    return qmc.scale(unit_sample, LOWER, UPPER)


def write_dataset(path: Path, x: np.ndarray) -> None:
    frame = pd.DataFrame(x, columns=COLUMNS)
    frame["target"] = piston_cycle_time(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/piston"))
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    write_dataset(
        args.output_dir / "train.tsv.gz", sample_inputs(args.seed, args.train_size)
    )
    write_dataset(
        args.output_dir / "test.tsv.gz", sample_inputs(args.seed + 1, args.test_size)
    )


if __name__ == "__main__":
    main()
