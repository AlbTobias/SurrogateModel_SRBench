#!/usr/bin/env python3
"""Prepare the UCI Electrical Grid Stability regression problem."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/471/"
    "electrical+grid+stability+simulated+data.zip"
)
ARCHIVE_SHA256 = "32eeb20cf1877127aade7c9a76a6ab6f2ff1fdd05270bc591d549f26c7402cad"
DATA_MEMBER = "Data_for_UCI_named.csv"
DATA_MEMBER_SHA256 = "7afc3154fed9cd65258b3403b60381fe6b0d832bf817a1ea4866395d05a84a83"
SOURCE_COLUMNS = (
    "tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4", "stab", "stabf",
)
FEATURE_COLUMNS = (
    "reaction_time_producer",
    "reaction_time_consumer_1",
    "reaction_time_consumer_2",
    "reaction_time_consumer_3",
    "nominal_power_consumer_1",
    "nominal_power_consumer_2",
    "nominal_power_consumer_3",
    "price_elasticity_producer",
    "price_elasticity_consumer_1",
    "price_elasticity_consumer_2",
    "price_elasticity_consumer_3",
)
SELECTED_SOURCE_COLUMNS = (
    "tau1", "tau2", "tau3", "tau4", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        with urlopen(SOURCE_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    actual_hash = sha256(path)
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError(
            f"Unexpected Electrical Grid archive SHA-256 for {path}: {actual_hash}; "
            f"expected {ARCHIVE_SHA256}"
        )


def extract_source(archive_path: Path, source_path: Path) -> None:
    with ZipFile(archive_path) as archive:
        if DATA_MEMBER not in archive.namelist():
            raise ValueError(f"Electrical Grid archive does not contain {DATA_MEMBER}")
        contents = archive.read(DATA_MEMBER)
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != DATA_MEMBER_SHA256:
        raise ValueError(
            f"Unexpected Electrical Grid data-member SHA-256: {actual_hash}; "
            f"expected {DATA_MEMBER_SHA256}"
        )
    source_path.write_bytes(contents)


def load_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    if tuple(source.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected Electrical Grid columns {tuple(source.columns)}; "
            f"expected {SOURCE_COLUMNS}"
        )
    if source.shape != (10000, 14):
        raise ValueError(
            f"Unexpected Electrical Grid shape {source.shape}; expected (10000, 14)"
        )
    if source.isna().any().any():
        raise ValueError("Electrical Grid source unexpectedly contains missing values")
    if not np.allclose(source["p1"], -(source["p2"] + source["p3"] + source["p4"])):
        raise ValueError("Electrical Grid p1 does not match its documented dependency")
    selected = source.loc[:, [*SELECTED_SOURCE_COLUMNS, "stab"]].copy()
    selected.columns = (*FEATURE_COLUMNS, "target")
    if selected.duplicated(list(FEATURE_COLUMNS)).any():
        raise ValueError("Electrical Grid source contains repeated input designs")
    return selected


def split_dataset(
    source: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_size + test_size > len(source):
        raise ValueError("Requested train and test sizes exceed available simulations")
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
        "--output-dir", type=Path, default=Path("data/electrical_grid_stability")
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    archive_path = args.archive or args.output_dir / "electrical_grid_stability.zip"
    if args.archive is None:
        download_archive(archive_path)
    elif sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError(
            f"Supplied archive is not the pinned Electrical Grid dataset: {archive_path}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / DATA_MEMBER
    extract_source(archive_path, source_path)
    source = load_source(source_path)
    train, test = split_dataset(source, args.seed, args.train_size, args.test_size)
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
