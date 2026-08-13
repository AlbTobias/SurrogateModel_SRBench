#!/usr/bin/env python3
"""Generate fixed train/test data for the Wing Weight surrogate problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


COLUMNS = (
    "wing_area",
    "fuel_weight",
    "aspect_ratio",
    "sweep_angle_degrees",
    "dynamic_pressure",
    "taper_ratio",
    "thickness_chord_ratio",
    "ultimate_load_factor",
    "design_gross_weight",
    "paint_weight",
)
LOWER = np.array([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025])
UPPER = np.array([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08])


def wing_weight(x: np.ndarray) -> np.ndarray:
    """Estimate light-aircraft wing weight in pounds."""

    (area, fuel, aspect, sweep_degrees, pressure, taper, thickness,
     load, gross_weight, paint) = x.T
    sweep_radians = np.deg2rad(sweep_degrees)
    cosine = np.cos(sweep_radians)
    structural_weight = (
        0.036
        * area**0.758
        * fuel**0.0035
        * (aspect / cosine**2) ** 0.6
        * pressure**0.006
        * taper**0.04
        * (100.0 * thickness / cosine) ** -0.3
        * (load * gross_weight) ** 0.49
    )
    return structural_weight + area * paint


def sample_inputs(seed: int, samples: int) -> np.ndarray:
    unit_sample = qmc.LatinHypercube(d=len(COLUMNS), seed=seed).random(samples)
    return qmc.scale(unit_sample, LOWER, UPPER)


def write_dataset(path: Path, x: np.ndarray) -> None:
    frame = pd.DataFrame(x, columns=COLUMNS)
    frame["target"] = wing_weight(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/wing_weight"))
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    write_dataset(
        args.output_dir / "train.tsv.gz", sample_inputs(args.seed, args.train_size)
    )
    write_dataset(
        args.output_dir / "test.tsv.gz", sample_inputs(args.seed + 1, args.test_size)
    )


if __name__ == "__main__":
    main()
