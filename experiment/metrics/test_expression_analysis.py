"""Tests for framework-independent expression analysis."""

from experiment.metrics.expression_analysis import GROUND_TRUTH, analyze_expression


def test_parses_gplearn_prefix_notation_and_detects_exact_match() -> None:
    result = analyze_expression(
        "div(mul(mul(4, force), mul(length, mul(length, length))), "
        "mul(modulus, mul(width, mul(height, mul(height, height)))))",
        ["force", "length", "modulus", "width", "height"],
        "cantilever",
    )

    assert result["expression_parse_success"]
    assert result["symbolic_exact_match"]
    assert result["expression_variable_count"] == 5
    assert result["simplified_node_count"] <= result["expression_node_count"]


def test_records_variables_operators_depth_and_constants() -> None:
    result = analyze_expression(
        "sin(x) + 2*y**2", ["x", "y"], "unknown"
    )

    assert result["expression_parse_success"]
    assert result["expression_variables"] == ["x", "y"]
    assert {"Add", "Mul", "Pow", "sin"}.issubset(result["expression_operators"])
    assert result["expression_depth"] >= 2
    assert result["expression_constant_count"] >= 2
    assert result["symbolic_exact_match"] is None


def test_parses_itea_transform_names_as_sympy_operations() -> None:
    result = analyze_expression(
        "Log(x) + Tanh(y) + PSQRT(x-y)", ["x", "y"], "unknown"
    )

    assert result["expression_parse_success"]
    assert {"log", "tanh", "Abs", "Pow"}.issubset(result["expression_operators"])


def test_reports_missing_or_invalid_expressions_without_raising() -> None:
    missing = analyze_expression(None, ["x"], "unknown")
    invalid = analyze_expression("add(x,", ["x"], "unknown")

    assert not missing["expression_parse_success"]
    assert "no symbolic expression" in missing["expression_parse_error"]
    assert not invalid["expression_parse_success"]
    assert invalid["expression_parse_error"]


def test_equivalent_algebraic_form_matches_borehole_ground_truth() -> None:
    features = [
        "borehole_radius",
        "influence_radius",
        "upper_transmissivity",
        "upper_head",
        "lower_transmissivity",
        "lower_head",
        "borehole_length",
        "hydraulic_conductivity",
    ]
    expression = (
        "2*pi*upper_transmissivity*(upper_head-lower_head) / "
        "(log(influence_radius/borehole_radius) + "
        "2*borehole_length*upper_transmissivity / "
        "(borehole_radius**2*hydraulic_conductivity) + "
        "log(influence_radius/borehole_radius)*"
        "upper_transmissivity/lower_transmissivity)"
    )

    result = analyze_expression(expression, features, "borehole")
    assert result["symbolic_exact_match"]


def test_piston_ground_truth_parses_and_matches_itself() -> None:
    features = [
        "piston_mass",
        "surface_area",
        "initial_volume",
        "spring_coefficient",
        "atmospheric_pressure",
        "ambient_temperature",
        "gas_temperature",
    ]
    result = analyze_expression(GROUND_TRUTH["piston"], features, "piston")

    assert result["expression_parse_success"]
    assert result["expression_simplify_success"]
    assert result["symbolic_exact_match"]
    assert result["ground_truth_variable_recall"] == 1.0


def test_wing_weight_ground_truth_parses_and_matches_itself() -> None:
    features = [
        "wing_area", "fuel_weight", "aspect_ratio", "sweep_angle_degrees",
        "dynamic_pressure", "taper_ratio", "thickness_chord_ratio",
        "ultimate_load_factor", "design_gross_weight", "paint_weight",
    ]
    result = analyze_expression(GROUND_TRUTH["wing_weight"], features, "wing_weight")

    assert result["expression_parse_success"]
    assert result["symbolic_exact_match"]
    assert result["ground_truth_variable_recall"] == 1.0
