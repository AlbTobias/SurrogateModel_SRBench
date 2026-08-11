"""Tests for fixed-domain scaling and symbolic back-transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp

from experiment.metrics.expression_analysis import parse_expression
from experiment.metrics.input_scaling import (
    expression_to_raw_scale,
    scale_frame_to_unit_interval,
    scaling_metadata,
)


def test_domain_bounds_map_exactly_to_minus_and_plus_one() -> None:
    features = ["force", "length", "modulus", "width", "height"]
    metadata = scaling_metadata("cantilever", features)
    lower = pd.DataFrame([[0.5, 0.8, 0.8, 0.7, 0.7]], columns=features)
    upper = pd.DataFrame([[1.5, 1.2, 1.2, 1.3, 1.3]], columns=features)

    np.testing.assert_allclose(scale_frame_to_unit_interval(lower, metadata), -1.0)
    np.testing.assert_allclose(scale_frame_to_unit_interval(upper, metadata), 1.0)


def test_scaling_uses_registered_domain_not_sample_extrema() -> None:
    features = ["force", "length", "modulus", "width", "height"]
    metadata = scaling_metadata("cantilever", features)
    midpoint = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 1.0]], columns=features)

    np.testing.assert_allclose(scale_frame_to_unit_interval(midpoint, metadata), 0.0)


def test_normalized_expression_is_back_transformed_to_raw_variables() -> None:
    features = ["force", "length", "modulus", "width", "height"]
    metadata = scaling_metadata("cantilever", features)
    raw_text = expression_to_raw_scale("2*force + length", features, metadata)
    raw_expression = parse_expression(raw_text, features)
    symbols = {symbol.name: symbol for symbol in raw_expression.free_symbols}

    assert sp.simplify(
        raw_expression - (4 * symbols["force"] + 5 * symbols["length"] - 9)
    ) == 0


def test_feature_domain_mismatch_is_rejected() -> None:
    try:
        scaling_metadata("cantilever", ["force", "unknown"])
    except ValueError as error:
        assert "feature mismatch" in str(error)
    else:
        raise AssertionError("expected domain mismatch to fail")
