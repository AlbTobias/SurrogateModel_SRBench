#!/usr/bin/env python3
"""Prepare a fixed fuel-flow surrogate split from the UCI naval simulator."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/316/"
    "condition+based+maintenance+of+naval+propulsion+plants.zip"
)
ARCHIVE_SHA256 = "91a3815da80b5ab7e2d5b82ac82f1c2cbf89182c7a65bcdf240db1e014423cb9"
DATA_MEMBER = "UCI CBM Dataset/data.txt"
DATA_SHA256 = "de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6"
SOURCE_COLUMN_COUNT = 18
SELECTED_COLUMNS = {
    1: "ship_speed",
    16: "compressor_decay",
    17: "turbine_decay",
    15: "target",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
            f"Unexpected naval archive SHA-256 for {path}: {actual_hash}; "
            f"expected {ARCHIVE_SHA256}"
        )


def load_source(archive: Path) -> pd.DataFrame:
    with ZipFile(archive) as bundle:
        content = bundle.read(DATA_MEMBER)
    if sha256_bytes(content) != DATA_SHA256:
        raise ValueError("Naval simulator data member does not match its pinned hash")

    source = pd.read_csv(BytesIO(content), sep=r"\s+", header=None)
    if source.shape != (11934, SOURCE_COLUMN_COUNT):
        raise ValueError(
            f"Unexpected naval source shape {source.shape}; expected (11934, 18)"
        )
    if source.isna().any().any():
        raise ValueError("Naval source unexpectedly contains missing values")

    frame = source.loc[:, list(SELECTED_COLUMNS)].copy()
    frame.columns = list(SELECTED_COLUMNS.values())
    if frame.duplicated(["ship_speed", "compressor_decay", "turbine_decay"]).any():
        raise ValueError("Simulator control triples are not unique")
    return frame


def split_dataset(
    frame: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic, disjoint simulator cases without replacement."""

    if train_size + test_size > len(frame):
        raise ValueError("Requested train and test sizes exceed the source dataset")
    indices = np.random.default_rng(seed).permutation(len(frame))
    train = frame.iloc[indices[:train_size]].reset_index(drop=True)
    test = frame.iloc[indices[train_size : train_size + test_size]].reset_index(drop=True)
    return train, test


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/naval_propulsion")
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    archive = args.source or args.output_dir / "source.zip"
    if args.source is None:
        download_archive(archive)
    elif sha256(archive) != ARCHIVE_SHA256:
        raise ValueError(f"Supplied source does not match the pinned archive: {archive}")
    train, test = split_dataset(
        load_source(archive), args.seed, args.train_size, args.test_size
    )
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
