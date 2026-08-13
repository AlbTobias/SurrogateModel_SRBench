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
