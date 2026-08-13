"""Tests for deterministic surrogate benchmark simulators and designs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surrogate.generate_borehole import LOWER as BOREHOLE_LOWER
from surrogate.generate_borehole import UPPER as BOREHOLE_UPPER
from surrogate.generate_borehole import borehole_flow, sample_inputs as sample_borehole
from surrogate.generate_piston import LOWER as PISTON_LOWER
from surrogate.generate_piston import UPPER as PISTON_UPPER
from surrogate.generate_piston import piston_cycle_time, sample_inputs as sample_piston
from surrogate.generate_wing_weight import LOWER as WING_LOWER
from surrogate.generate_wing_weight import UPPER as WING_UPPER
from surrogate.generate_wing_weight import sample_inputs as sample_wing
from surrogate.generate_wing_weight import wing_weight
from surrogate.prepare_ccpp import FEATURE_COLUMNS, split_dataset
from surrogate.prepare_naval_propulsion import split_dataset as split_naval
from surrogate.prepare_gas_turbine_nox import (
    FEATURE_COLUMNS as NOX_FEATURE_COLUMNS,
    SOURCE_COLUMNS as NOX_SOURCE_COLUMNS,
    split_dataset as split_nox,
)
from surrogate.prepare_concrete_strength import (
    FEATURE_COLUMNS as CONCRETE_FEATURE_COLUMNS,
    aggregate_replicates as aggregate_concrete,
    split_dataset as split_concrete,
)


def test_borehole_reference_value() -> None:
    midpoint = ((BOREHOLE_LOWER + BOREHOLE_UPPER) / 2.0).reshape(1, -1)
    np.testing.assert_allclose(borehole_flow(midpoint), [70.87291263681897])


def test_piston_reference_value() -> None:
    midpoint = ((PISTON_LOWER + PISTON_UPPER) / 2.0).reshape(1, -1)
    np.testing.assert_allclose(piston_cycle_time(midpoint), [0.4643970224718025])


def test_wing_weight_reference_value() -> None:
    midpoint = ((WING_LOWER + WING_UPPER) / 2.0).reshape(1, -1)
    np.testing.assert_allclose(wing_weight(midpoint), [267.6246925724748])


def test_problem_designs_are_reproducible_bounded_and_distinct() -> None:
    for sampler, lower, upper in (
        (sample_borehole, BOREHOLE_LOWER, BOREHOLE_UPPER),
        (sample_piston, PISTON_LOWER, PISTON_UPPER),
        (sample_wing, WING_LOWER, WING_UPPER),
    ):
        first = sampler(123, 25)
        second = sampler(123, 25)
        other = sampler(124, 25)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)
        assert np.all(first >= lower)
        assert np.all(first <= upper)


def test_ccpp_split_is_reproducible_disjoint_and_value_preserving() -> None:
    frame = pd.DataFrame(
        {
            **{name: np.arange(20, dtype=float) for name in FEATURE_COLUMNS},
            "target": np.arange(20, dtype=float),
        }
    )
    train, test = split_dataset(frame, seed=123, train_size=5, test_size=10)
    repeated_train, repeated_test = split_dataset(
        frame, seed=123, train_size=5, test_size=10
    )
    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    assert set(train["target"]).isdisjoint(test["target"])
    assert set(train["target"]) | set(test["target"]) <= set(frame["target"])


def test_naval_split_is_reproducible_disjoint_and_value_preserving() -> None:
    frame = pd.DataFrame(
        {
            "ship_speed": np.arange(20, dtype=float),
            "compressor_decay": np.linspace(0.95, 1.0, 20),
            "turbine_decay": np.linspace(0.975, 1.0, 20),
            "target": np.arange(20, dtype=float),
        }
    )
    train, test = split_naval(frame, seed=456, train_size=5, test_size=10)
    repeated_train, repeated_test = split_naval(
        frame, seed=456, train_size=5, test_size=10
    )
    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    assert set(train["target"]).isdisjoint(test["target"])


def test_nox_split_respects_chronology_and_excludes_co_target() -> None:
    rows = []
    for year in range(2011, 2016):
        for sample in range(5):
            values = {column: float(year * 100 + sample) for column in NOX_SOURCE_COLUMNS}
            values["NOX"] = float(year * 10 + sample)
            values["CO"] = -float(year * 10 + sample)
            values["year"] = year
            rows.append(values)
    source = pd.DataFrame(rows)
    train, test = split_nox(source, seed=789, train_size=6, test_size=6)
    repeated_train, repeated_test = split_nox(
        source, seed=789, train_size=6, test_size=6
    )

    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    assert list(train.columns) == [*NOX_FEATURE_COLUMNS, "target"]
    assert "year" not in train and "CO" not in train
    assert train["target"].max() < 20140
    assert test["target"].min() >= 20140


def test_concrete_replicates_are_averaged_before_disjoint_split() -> None:
    rows = []
    for design in range(12):
        row = {name: float(design) for name in CONCRETE_FEATURE_COLUMNS}
        rows.append({**row, "target": float(design)})
    rows.extend(
        [
            {**rows[2], "target": 6.0},
            {**rows[2], "target": 10.0},
        ]
    )
    designs = aggregate_concrete(pd.DataFrame(rows))
    train, test = split_concrete(designs, seed=321, train_size=5, test_size=7)
    repeated_train, repeated_test = split_concrete(
        designs, seed=321, train_size=5, test_size=7
    )

    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    assert len(designs) == 12
    assert designs.loc[designs[CONCRETE_FEATURE_COLUMNS[0]] == 2, "target"].item() == 6.0
    train_designs = set(map(tuple, train.loc[:, CONCRETE_FEATURE_COLUMNS].to_numpy()))
    test_designs = set(map(tuple, test.loc[:, CONCRETE_FEATURE_COLUMNS].to_numpy()))
    assert train_designs.isdisjoint(test_designs)
