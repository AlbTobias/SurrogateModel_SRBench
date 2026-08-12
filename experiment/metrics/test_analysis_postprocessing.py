"""Tests for analysis performed after algorithm execution."""

from scripts.analyze_expression_result import ANALYSIS_VERSION, analyze_result


def test_normalized_result_is_analyzed_in_both_coordinate_systems() -> None:
    result = {
        "problem": "cantilever",
        "algorithm": "example",
        "seed": 2,
        "input_scaling": "domain_minmax",
        "feature_names": ["force", "length", "modulus", "width", "height"],
        "input_scaling_parameters": {
            "force": {"lower": 0.5, "upper": 1.5},
            "length": {"lower": 0.8, "upper": 1.2},
            "modulus": {"lower": 0.8, "upper": 1.2},
            "width": {"lower": 0.7, "upper": 1.3},
            "height": {"lower": 0.7, "upper": 1.3},
        },
        "symbolic_model": "2*force + length",
    }

    analysis = analyze_result(result, timeout_seconds=5)

    assert analysis["analysis_version"] == ANALYSIS_VERSION
    assert analysis["training_scale_expression_parse_success"] is True
    assert analysis["expression_parse_success"] is True
    assert analysis["complexity_valid"] is True
    assert analysis["complexity_source"] == "simplified"
    assert analysis["raw_scale_symbolic_model"] == "4*force + 5*length - 9"
