"""Tests for deterministic surrogate benchmark simulators and designs."""

from __future__ import annotations

import numpy as np

from surrogate.generate_borehole import LOWER as BOREHOLE_LOWER
from surrogate.generate_borehole import UPPER as BOREHOLE_UPPER
from surrogate.generate_borehole import borehole_flow, sample_inputs as sample_borehole
from surrogate.generate_piston import LOWER as PISTON_LOWER
from surrogate.generate_piston import UPPER as PISTON_UPPER
from surrogate.generate_piston import piston_cycle_time, sample_inputs as sample_piston


def test_borehole_reference_value() -> None:
    midpoint = ((BOREHOLE_LOWER + BOREHOLE_UPPER) / 2.0).reshape(1, -1)
    np.testing.assert_allclose(borehole_flow(midpoint), [70.87291263681897])


def test_piston_reference_value() -> None:
    midpoint = ((PISTON_LOWER + PISTON_UPPER) / 2.0).reshape(1, -1)
    np.testing.assert_allclose(piston_cycle_time(midpoint), [0.4643970224718025])


def test_problem_designs_are_reproducible_bounded_and_distinct() -> None:
    for sampler, lower, upper in (
        (sample_borehole, BOREHOLE_LOWER, BOREHOLE_UPPER),
        (sample_piston, PISTON_LOWER, PISTON_UPPER),
    ):
        first = sampler(123, 25)
        second = sampler(123, 25)
        other = sampler(124, 25)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)
        assert np.all(first >= lower)
        assert np.all(first <= upper)
