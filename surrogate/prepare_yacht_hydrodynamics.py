#!/usr/bin/env python3
"""Prepare a hull-disjoint split of the UCI Yacht Hydrodynamics dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCE_URL = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
ARCHIVE_SHA256 = "aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac"
DATA_MEMBER = "yacht_hydrodynamics.data"
DATA_MEMBER_SHA256 = "00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1"
HULL_COLUMNS = (
    "longitudinal_buoyancy_position",
    "prismatic_coefficient",
    "length_displacement_ratio",
    "beam_draught_ratio",
    "length_beam_ratio",
)
FEATURE_COLUMNS = (*HULL_COLUMNS, "froude_number")


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
            f"Unexpected Yacht archive SHA-256 for {path}: {actual_hash}; "
            f"expected {ARCHIVE_SHA256}"
        )


def extract_source(archive_path: Path, source_path: Path) -> None:
    with ZipFile(archive_path) as archive:
        if DATA_MEMBER not in archive.namelist():
            raise ValueError(f"Yacht archive does not contain {DATA_MEMBER}")
        contents = archive.read(DATA_MEMBER)
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != DATA_MEMBER_SHA256:
        raise ValueError(
            f"Unexpected Yacht data-member SHA-256: {actual_hash}; "
            f"expected {DATA_MEMBER_SHA256}"
        )
    source_path.write_bytes(contents)


def load_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=[*FEATURE_COLUMNS, "target"],
    )
    if source.shape != (308, 7):
        raise ValueError(f"Unexpected Yacht shape {source.shape}; expected (308, 7)")
    if source.isna().any().any():
        raise ValueError("Yacht source unexpectedly contains missing values")
    hull_sizes = source.groupby(list(HULL_COLUMNS), sort=False).size()
    if len(hull_sizes) != 22 or set(hull_sizes) != {14}:
        raise ValueError("Yacht source does not contain 22 hulls with 14 runs each")
    if source.duplicated(list(FEATURE_COLUMNS)).any():
        raise ValueError("Yacht source contains repeated complete input designs")
    return source


def split_dataset(
    source: pd.DataFrame, seed: int, train_hulls: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every operating point for a hull wholly in train or test."""

    hulls = source.loc[:, HULL_COLUMNS].drop_duplicates().reset_index(drop=True)
    if not 0 < train_hulls < len(hulls):
        raise ValueError("train_hulls must leave at least one hull in each split")
    order = np.random.default_rng(seed).permutation(len(hulls))
    training_keys = {
        tuple(row) for row in hulls.iloc[order[:train_hulls]].to_numpy()
    }
    is_training = source.loc[:, HULL_COLUMNS].apply(tuple, axis=1).isin(training_keys)
    train = source.loc[is_training].reset_index(drop=True)
    test = source.loc[~is_training].reset_index(drop=True)
    return train, test


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/yacht_hydrodynamics")
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--train-hulls", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    archive_path = args.archive or args.output_dir / "yacht_hydrodynamics.zip"
    if args.archive is None:
        download_archive(archive_path)
    elif sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError(f"Supplied archive is not the pinned Yacht dataset: {archive_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / DATA_MEMBER
    extract_source(archive_path, source_path)
    train, test = split_dataset(load_source(source_path), args.seed, args.train_hulls)
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
