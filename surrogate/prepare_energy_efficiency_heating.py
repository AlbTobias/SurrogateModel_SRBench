#!/usr/bin/env python3
"""Prepare the UCI Energy Efficiency heating-load surrogate split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


SOURCE_URL = "https://archive.ics.uci.edu/static/public/242/data.csv"
SOURCE_SHA256 = "db44dbe453acd464b5cf65be2fb01a28aa9c5b2630300e65fbe28cde35f5d96f"
SOURCE_COLUMNS = ("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2")
FEATURE_COLUMNS = (
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        with urlopen(SOURCE_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    actual_hash = sha256(path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected Energy Efficiency source SHA-256 for {path}: "
            f"{actual_hash}; expected {SOURCE_SHA256}"
        )


def load_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    if tuple(source.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected Energy Efficiency columns {tuple(source.columns)}; "
            f"expected {SOURCE_COLUMNS}"
        )
    if source.shape != (768, 10):
        raise ValueError(
            f"Unexpected Energy Efficiency shape {source.shape}; expected (768, 10)"
        )
    if source.isna().any().any():
        raise ValueError("Energy Efficiency source unexpectedly contains missing values")
    selected = source.loc[:, [*SOURCE_COLUMNS[:8], "Y1"]].copy()
    selected.columns = (*FEATURE_COLUMNS, "target")
    if selected.duplicated(list(FEATURE_COLUMNS)).any():
        raise ValueError("Energy Efficiency source contains repeated input designs")
    return selected


def split_dataset(
    source: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic, disjoint building designs without replacement."""

    if train_size + test_size > len(source):
        raise ValueError("Requested train and test sizes exceed available designs")
    indices = np.random.default_rng(seed).permutation(len(source))
    train = source.iloc[indices[:train_size]].reset_index(drop=True)
    test = source.iloc[indices[train_size : train_size + test_size]].reset_index(drop=True)
    return train, test


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/energy_efficiency_heating")
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=368)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    source_path = args.source or args.output_dir / "source.csv"
    if args.source is None:
        download_source(source_path)
    elif sha256(source_path) != SOURCE_SHA256:
        raise ValueError(
            f"Supplied source does not match the pinned Energy Efficiency dataset: {source_path}"
        )
    source = load_source(source_path)
    train, test = split_dataset(source, args.seed, args.train_size, args.test_size)
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
