#!/usr/bin/env python3
"""Prepare fixed chronological-pool splits for gas-turbine NOx prediction."""

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
    "https://archive.ics.uci.edu/static/public/551/"
    "gas+turbine+co+and+nox+emission+data+set.zip"
)
ARCHIVE_SHA256 = "55fdb1acc25f05bd9c77aac285c424e9154ffb0dbf1bbadb56b69a1142231295"
SOURCE_COLUMNS = ("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX")
FEATURE_COLUMNS = (
    "ambient_temperature",
    "ambient_pressure",
    "ambient_humidity",
    "air_filter_pressure_difference",
    "gas_turbine_exhaust_pressure",
    "turbine_inlet_temperature",
    "turbine_after_temperature",
    "turbine_energy_yield",
    "compressor_discharge_pressure",
)
MEMBER_HASHES = {
    2011: "d87ceef9aa59533cc7d924d10de241b1b06ecd11f9b26bab59191ea0f8a76b9a",
    2012: "be54b9d0e1a7de40c55d32fa489e75de892b000c066b5a09f09a19124ee29100",
    2013: "13c437bb440ec2045bd12057e6654c41dd4107a661eac16ba2e878e897a08f9e",
    2014: "c2a03c92c9c3207aad0c6be7de8d9b5b4bfa4720ad0efb2c1f21b6cec4d3f3fa",
    2015: "9b08f35fde0d4b138232a605db4093c2b8bf9d6757e6f1fbd9534ad616c13591",
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
            f"Unexpected emissions archive SHA-256 for {path}: {actual_hash}; "
            f"expected {ARCHIVE_SHA256}"
        )


def load_source(archive: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with ZipFile(archive) as bundle:
        for year, expected_hash in MEMBER_HASHES.items():
            member = f"gt_{year}.csv"
            content = bundle.read(member)
            if sha256_bytes(content) != expected_hash:
                raise ValueError(f"{member} does not match its pinned SHA-256")
            frame = pd.read_csv(BytesIO(content))
            if tuple(frame.columns) != SOURCE_COLUMNS:
                raise ValueError(f"Unexpected columns in {member}: {tuple(frame.columns)}")
            frame["year"] = year
            frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    if source.shape != (36733, 12):
        raise ValueError(f"Unexpected source shape {source.shape}; expected (36733, 12)")
    if source.isna().any().any():
        raise ValueError("Emissions source unexpectedly contains missing values")
    return source


def model_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Select nine operating inputs and NOx; exclude year and the CO co-target."""

    frame = source.loc[:, [*SOURCE_COLUMNS[:9], "NOX"]].copy()
    frame.columns = (*FEATURE_COLUMNS, "target")
    return frame


def split_dataset(
    source: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample training from 2011--2013 and testing from 2014--2015."""

    training_pool = source[source["year"] <= 2013]
    testing_pool = source[source["year"] >= 2014]
    if train_size > len(training_pool) or test_size > len(testing_pool):
        raise ValueError("Requested sizes exceed their chronological source pools")
    generator = np.random.default_rng(seed)
    train_indices = generator.choice(training_pool.index, train_size, replace=False)
    test_indices = generator.choice(testing_pool.index, test_size, replace=False)
    return (
        model_frame(training_pool.loc[train_indices]).reset_index(drop=True),
        model_frame(testing_pool.loc[test_indices]).reset_index(drop=True),
    )


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/gas_turbine_nox"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260819)
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
