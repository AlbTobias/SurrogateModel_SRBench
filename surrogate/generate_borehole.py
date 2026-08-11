#!/usr/bin/env python3
"""Generate fixed train/test data for the Borehole surrogate problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


COLUMNS = ("borehole_radius", "influence_radius", "upper_transmissivity",
           "upper_head", "lower_transmissivity", "lower_head",
           "borehole_length", "hydraulic_conductivity")
LOWER = np.array([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0])
UPPER = np.array([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0])


def borehole_flow(x: np.ndarray) -> np.ndarray:
    """Water flow rate through a borehole in cubic metres per year."""
    rw, radius, tu, hu, tl, hl, length, kw = x.T
    log_ratio = np.log(radius / rw)
    numerator = 2.0 * np.pi * tu * (hu - hl)
    denominator = log_ratio * (
        1.0 + 2.0 * length * tu / (log_ratio * rw**2 * kw) + tu / tl
    )
    return numerator / denominator


def sample_inputs(seed: int, samples: int) -> np.ndarray:
    unit_sample = qmc.LatinHypercube(d=len(COLUMNS), seed=seed).random(samples)
    return qmc.scale(unit_sample, LOWER, UPPER)


def write_dataset(path: Path, x: np.ndarray) -> None:
    frame = pd.DataFrame(x, columns=COLUMNS)
    frame["target"] = borehole_flow(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/borehole"))
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    write_dataset(
        args.output_dir / "train.tsv.gz", sample_inputs(args.seed, args.train_size)
    )
    write_dataset(
        args.output_dir / "test.tsv.gz", sample_inputs(args.seed + 1, args.test_size)
    )


if __name__ == "__main__":
    main()
