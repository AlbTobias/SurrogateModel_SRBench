#!/usr/bin/env python3
"""Prepare the UCI MCNP6 lattice-physics k-inf regression problem."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/1091/"
    "lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results).zip"
)
ARCHIVE_SHA256 = "ee7f7366e82dd9d6e361ff1a1c4bbeda77c0a19ef461c2b7c73aa3b56ce4cd1a"
DATA_MEMBER = "raw.csv"
DATA_MEMBER_SHA256 = "377fad93fefe79483d45f194c57d886f19a9d28ae383a2a838e413a63fdaa253"
FEATURE_COLUMNS = tuple(f"fuel_pin_enrichment_{index:02d}" for index in range(1, 40))


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
        with urlopen(SOURCE_URL, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    actual_hash = sha256(path)
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError(
            f"Unexpected Lattice Physics archive SHA-256 for {path}: {actual_hash}; "
            f"expected {ARCHIVE_SHA256}"
        )


def extract_source(archive_path: Path, source_path: Path) -> None:
    with ZipFile(archive_path) as archive:
        if DATA_MEMBER not in archive.namelist():
            raise ValueError(f"Lattice Physics archive does not contain {DATA_MEMBER}")
        contents = archive.read(DATA_MEMBER)
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != DATA_MEMBER_SHA256:
        raise ValueError(
            f"Unexpected Lattice Physics data-member SHA-256: {actual_hash}; "
            f"expected {DATA_MEMBER_SHA256}"
        )
    source_path.write_bytes(contents)


def load_source(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape != (24000, 41):
        raise ValueError(
            f"Unexpected Lattice Physics shape {raw.shape}; expected (24000, 41)"
        )
    if raw.isna().any().any():
        raise ValueError("Lattice Physics source unexpectedly contains missing values")
    selected = raw.iloc[:, [*range(2, 41), 0]].copy()
    selected.columns = (*FEATURE_COLUMNS, "target")
    if selected.duplicated(list(FEATURE_COLUMNS)).any():
        raise ValueError("Lattice Physics source contains repeated input designs")
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
        "--output-dir", type=Path, default=Path("data/lattice_physics_kinf")
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

    archive_path = args.archive or args.output_dir / "lattice_physics.zip"
    if args.archive is None:
        download_archive(archive_path)
    elif sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError(
            f"Supplied archive is not the pinned Lattice Physics dataset: {archive_path}"
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
