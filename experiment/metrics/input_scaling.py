"""Explicit input scaling and symbolic back-transformation utilities."""

from __future__ import annotations

import pandas as pd
import sympy as sp

from experiment.metrics.expression_analysis import parse_expression
from surrogate.problem_metadata import PROBLEM_DOMAINS


SCALING_RAW = "raw"
SCALING_DOMAIN_MINMAX = "domain_minmax"
SUPPORTED_SCALINGS = (SCALING_RAW, SCALING_DOMAIN_MINMAX)


def scaling_metadata(problem: str, feature_names: list[str]) -> dict[str, object]:
    """Return ordered, JSON-serializable fixed-domain scaling metadata."""

    if problem not in PROBLEM_DOMAINS:
        raise ValueError(f"No fixed input domain is registered for {problem}")
    domain = PROBLEM_DOMAINS[problem]
    missing = sorted(set(feature_names) - set(domain))
    extra = sorted(set(domain) - set(feature_names))
    if missing or extra:
        raise ValueError(
            f"Dataset/domain feature mismatch for {problem}: "
            f"missing bounds={missing}, unexpected bounds={extra}"
        )
    return {
        feature: {"lower": domain[feature][0], "upper": domain[feature][1]}
        for feature in feature_names
    }


def scale_frame_to_unit_interval(
    frame: pd.DataFrame, metadata: dict[str, object]
) -> pd.DataFrame:
    """Map each fixed physical domain to [-1, 1] without fitting to samples."""

    scaled = frame.copy()
    for feature in frame.columns:
        bounds = metadata[feature]
        lower = float(bounds["lower"])
        upper = float(bounds["upper"])
        scaled[feature] = 2.0 * (frame[feature] - lower) / (upper - lower) - 1.0
    return scaled


def expression_to_raw_scale(
    expression: str,
    feature_names: list[str],
    metadata: dict[str, object],
) -> str:
    """Substitute normalized features with their raw-domain affine forms."""

    parsed = parse_expression(expression, feature_names)
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    symbols = {symbol.name: symbol for symbol in parsed.free_symbols}
    for feature in feature_names:
        if feature not in symbols:
            continue
        bounds = metadata[feature]
        lower = sp.Rational(str(bounds["lower"]))
        upper = sp.Rational(str(bounds["upper"]))
        raw_symbol = sp.Symbol(feature, real=True)
        substitutions[symbols[feature]] = 2 * (raw_symbol - lower) / (upper - lower) - 1
    return str(parsed.subs(substitutions, simultaneous=True))
