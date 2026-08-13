#!/usr/bin/env python3
"""Prepare a leakage-resistant UCI concrete-strength surrogate split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


SOURCE_URL = "https://archive.ics.uci.edu/static/public/165/data.csv"
SOURCE_SHA256 = "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af"
SOURCE_COLUMNS = (
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
    "Concrete compressive strength",
)
FEATURE_COLUMNS = (
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
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
            f"Unexpected concrete source SHA-256 for {path}: {actual_hash}; "
            f"expected {SOURCE_SHA256}"
        )


def load_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    if tuple(source.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected concrete columns {tuple(source.columns)}; "
            f"expected {SOURCE_COLUMNS}"
        )
    if source.shape != (1030, 9):
        raise ValueError(f"Unexpected concrete shape {source.shape}; expected (1030, 9)")
    if source.isna().any().any():
        raise ValueError("Concrete source unexpectedly contains missing values")
    source.columns = (*FEATURE_COLUMNS, "target")
    return source


def aggregate_replicates(source: pd.DataFrame) -> pd.DataFrame:
    """Average strength replicates so each material design occurs exactly once."""

    designs = (
        source.groupby(list(FEATURE_COLUMNS), as_index=False, sort=False)["target"]
        .mean()
        .loc[:, [*FEATURE_COLUMNS, "target"]]
    )
    if designs.duplicated(list(FEATURE_COLUMNS)).any():
        raise ValueError("Concrete aggregation left repeated input designs")
    return designs


def split_dataset(
    designs: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic, disjoint material designs without replacement."""

    if train_size + test_size > len(designs):
        raise ValueError("Requested train and test sizes exceed unique designs")
    indices = np.random.default_rng(seed).permutation(len(designs))
    train = designs.iloc[indices[:train_size]].reset_index(drop=True)
    test = designs.iloc[indices[train_size : train_size + test_size]].reset_index(drop=True)
    return train, test


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/concrete_strength"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=592)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    source_path = args.source or args.output_dir / "source.csv"
    if args.source is None:
        download_source(source_path)
    elif sha256(source_path) != SOURCE_SHA256:
        raise ValueError(
            f"Supplied source does not match the pinned concrete dataset: {source_path}"
        )
    designs = aggregate_replicates(load_source(source_path))
    if designs.shape != (992, 9):
        raise ValueError(
            f"Unexpected unique-design shape {designs.shape}; expected (992, 9)"
        )
    train, test = split_dataset(
        designs,
        args.seed,
        args.train_size,
        args.test_size,
    )
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
