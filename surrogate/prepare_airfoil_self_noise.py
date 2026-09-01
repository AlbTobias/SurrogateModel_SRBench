#!/usr/bin/env python3
"""Prepare the UCI Airfoil Self-Noise surrogate split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/291/airfoil%2Bself%2Bnoise.zip"
)
ARCHIVE_SHA256 = "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
DATA_MEMBER = "airfoil_self_noise.dat"
DATA_MEMBER_SHA256 = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
FEATURE_COLUMNS = (
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement_thickness",
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
            f"Unexpected Airfoil Self-Noise archive SHA-256 for {path}: "
            f"{actual_hash}; expected {ARCHIVE_SHA256}"
        )


def extract_source(archive_path: Path, source_path: Path) -> None:
    with ZipFile(archive_path) as archive:
        if DATA_MEMBER not in archive.namelist():
            raise ValueError(f"Airfoil archive does not contain {DATA_MEMBER}")
        contents = archive.read(DATA_MEMBER)
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != DATA_MEMBER_SHA256:
        raise ValueError(
            f"Unexpected Airfoil data-member SHA-256: {actual_hash}; "
            f"expected {DATA_MEMBER_SHA256}"
        )
    source_path.write_bytes(contents)


def load_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[*FEATURE_COLUMNS, "target"],
    )
    if source.shape != (1503, 6):
        raise ValueError(
            f"Unexpected Airfoil Self-Noise shape {source.shape}; expected (1503, 6)"
        )
    if source.isna().any().any():
        raise ValueError("Airfoil Self-Noise source unexpectedly contains missing values")
    if source.duplicated(list(FEATURE_COLUMNS)).any():
        raise ValueError("Airfoil Self-Noise source contains repeated input designs")
    return source


def split_dataset(
    source: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic, disjoint wind-tunnel observations without replacement."""

    if train_size + test_size > len(source):
        raise ValueError("Requested train and test sizes exceed available observations")
    indices = np.random.default_rng(seed).permutation(len(source))
    train = source.iloc[indices[:train_size]].reset_index(drop=True)
    test = source.iloc[indices[train_size : train_size + test_size]].reset_index(drop=True)
    return train, test


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/airfoil_self_noise"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=1103)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    archive_path = args.archive or args.output_dir / "airfoil_self_noise.zip"
    if args.archive is None:
        download_archive(archive_path)
    elif sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError(f"Supplied archive is not the pinned Airfoil dataset: {archive_path}")
    source_path = args.output_dir / DATA_MEMBER
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_source(archive_path, source_path)
    source = load_source(source_path)
    train, test = split_dataset(source, args.seed, args.train_size, args.test_size)
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
