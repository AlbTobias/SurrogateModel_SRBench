"""Framework-independent structural analysis of symbolic-regression output."""

from __future__ import annotations

from contextlib import contextmanager
import signal
from typing import Iterator

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)


GROUND_TRUTH = {
    "cantilever": (
        "4*force*length**3/(modulus*width*height**3)"
    ),
    "borehole": (
        "2*pi*upper_transmissivity*(upper_head-lower_head) / "
        "(log(influence_radius/borehole_radius) * "
        "(1 + 2*borehole_length*upper_transmissivity / "
        "(log(influence_radius/borehole_radius)*borehole_radius**2*"
        "hydraulic_conductivity) + upper_transmissivity/lower_transmissivity))"
    ),
    "piston": (
        "2*pi*sqrt(piston_mass / (spring_coefficient + surface_area**2 * "
        "atmospheric_pressure*initial_volume*ambient_temperature / "
        "(gas_temperature * (surface_area/(2*spring_coefficient) * "
        "(sqrt((atmospheric_pressure*surface_area + 19.62*piston_mass - "
        "spring_coefficient*initial_volume/surface_area)**2 + "
        "4*spring_coefficient*atmospheric_pressure*initial_volume*"
        "ambient_temperature/gas_temperature) - "
        "(atmospheric_pressure*surface_area + 19.62*piston_mass - "
        "spring_coefficient*initial_volume/surface_area)))**2)))"
    ),
}


class ExpressionTimeoutError(TimeoutError):
    """Raised when symbolic parsing or simplification exceeds its time limit."""


@contextmanager
def time_limit(seconds: int) -> Iterator[None]:
    """Limit a symbolic operation without leaving a process-wide alarm active."""

    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handle_timeout(signum: int, frame: object) -> None:
        raise ExpressionTimeoutError(f"symbolic analysis exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    previous_alarm = signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_alarm:
            signal.alarm(previous_alarm)


def _local_dictionary(feature_names: list[str]) -> dict[str, object]:
    namespace: dict[str, object] = {
        name: sp.Symbol(name, real=True) for name in feature_names
    }
    namespace.update(
        {
            "add": lambda left, right: left + right,
            "sub": lambda left, right: left - right,
            "mul": lambda left, right: left * right,
            "div": lambda left, right: left / right,
            "Id": lambda value: value,
            "pi": sp.pi,
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "abs": sp.Abs,
            "Abs": sp.Abs,
            "Sin": sp.sin,
            "Cos": sp.cos,
            "Tanh": sp.tanh,
            "Log": sp.log,
            "Exp": sp.exp,
            "SqrtAbs": lambda value: sp.sqrt(sp.Abs(value)),
            "PSQRT": lambda value: sp.sqrt(sp.Abs(value)),
        }
    )
    return namespace


def parse_expression(expression: str, feature_names: list[str]) -> sp.Expr:
    """Parse the prefix and infix forms returned by the included adapters."""

    return parse_expr(
        expression,
        local_dict=_local_dictionary(feature_names),
        transformations=standard_transformations + (convert_xor,),
        evaluate=True,
    )


def expression_depth(expression: sp.Expr) -> int:
    """Return the longest root-to-leaf edge count (an atom has depth zero)."""

    if not expression.args:
        return 0
    return 1 + max(expression_depth(argument) for argument in expression.args)


def structural_metrics(expression: sp.Expr) -> dict[str, object]:
    nodes = list(sp.preorder_traversal(expression))
    operator_nodes = [node for node in nodes if node.args]
    constants = [node for node in nodes if node.is_Atom and node.is_number]
    return {
        "node_count": len(nodes),
        "depth": expression_depth(expression),
        "operator_count": len(operator_nodes),
        "operators": sorted({type(node).__name__ for node in operator_nodes}),
        "constant_count": len(constants),
        "variable_count": len(expression.free_symbols),
        "variables": sorted(str(symbol) for symbol in expression.free_symbols),
    }


def analyze_expression(
    expression: str | None,
    feature_names: list[str],
    problem: str,
    timeout_seconds: int = 5,
) -> dict[str, object]:
    """Parse, simplify, measure, and compare an expression with ground truth."""

    result: dict[str, object] = {
        "expression_parse_success": False,
        "expression_parse_error": None,
        "expression_simplify_success": False,
        "expression_simplify_error": None,
        "symbolic_comparison_error": None,
        "symbolic_exact_match": None,
    }
    if not expression:
        result["expression_parse_error"] = "adapter returned no symbolic expression"
        return result

    try:
        with time_limit(timeout_seconds):
            parsed = parse_expression(expression, feature_names)
        raw_metrics = structural_metrics(parsed)
        result.update({f"expression_{key}": value for key, value in raw_metrics.items()})
        result["expression_parse_success"] = True
    except Exception as error:
        result["expression_parse_error"] = f"{type(error).__name__}: {error}"
        return result

    simplified = None
    try:
        with time_limit(timeout_seconds):
            candidates = (
                parsed,
                sp.cancel(parsed),
                sp.factor(parsed),
                sp.factor(sp.cancel(parsed)),
                sp.simplify(parsed),
            )
            simplified = min(
                candidates,
                key=lambda candidate: (
                    structural_metrics(candidate)["node_count"],
                    len(str(candidate)),
                ),
            )
        simplified_metrics = structural_metrics(simplified)
        result.update(
            {f"simplified_{key}": value for key, value in simplified_metrics.items()}
        )
        result["simplified_expression"] = str(simplified)
        result["expression_simplify_success"] = True
    except Exception as error:
        result["expression_simplify_error"] = f"{type(error).__name__}: {error}"

    ground_truth_text = GROUND_TRUTH.get(problem)
    if ground_truth_text:
        try:
            with time_limit(timeout_seconds):
                ground_truth = parse_expression(ground_truth_text, feature_names)
                residual = sp.cancel(sp.together(parsed - ground_truth))
            truth_metrics = structural_metrics(ground_truth)
            result.update(
                {f"ground_truth_{key}": value for key, value in truth_metrics.items()}
            )
            result["ground_truth_expression"] = str(ground_truth)
            result["symbolic_exact_match"] = residual == 0
            result["ground_truth_variable_recall"] = (
                len(parsed.free_symbols & ground_truth.free_symbols)
                / len(ground_truth.free_symbols)
            )
            if simplified is not None:
                result["simplified_to_ground_truth_size_ratio"] = (
                    structural_metrics(simplified)["node_count"]
                    / truth_metrics["node_count"]
                )
        except Exception as error:
            result["symbolic_comparison_error"] = f"{type(error).__name__}: {error}"
    return result
