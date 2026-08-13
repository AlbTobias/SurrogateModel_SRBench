"""Tests for the final benchmark configuration contract."""

import json
from pathlib import Path


CONFIG = Path(__file__).resolve().parents[2] / "configs/benchmark_suite_v3.json"
CONFIG_V4 = Path(__file__).resolve().parents[2] / "configs/benchmark_suite_v4.json"
CONFIG_V5 = Path(__file__).resolve().parents[2] / "configs/benchmark_suite_v5.json"
CONFIG_V6 = Path(__file__).resolve().parents[2] / "configs/benchmark_suite_v6.json"


def test_algorithm_parameter_provenance_is_complete() -> None:
    configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
    allowed_roles = {
        "project_budget_override",
        "project_budget_and_execution_override",
        "documented_adapter_defaults",
    }

    for algorithm, entry in configuration["algorithms"].items():
        assert entry["configuration_role"] in allowed_roles, algorithm
        assert entry["rationale"], algorithm
        assert entry["parameters"], algorithm


def test_cleanup_preserves_effective_final_parameters() -> None:
    configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
    parameters = {
        algorithm: entry["parameters"]
        for algorithm, entry in configuration["algorithms"].items()
    }

    assert parameters == {
        "gplearn": {"population_size": 500, "generations": 100},
        "operon": {
            "population_size": 1000,
            "generations": 1000,
            "max_evaluations": 1_000_000,
        },
        "pysr": {
            "niterations": 20,
            "ncyclesperiteration": 500,
            "population_size": 100,
            "populations": 3,
            "timeout_in_seconds": 60,
            "deterministic": True,
            "procs": 0,
            "multithreading": False,
        },
        "geneticengine": {
            "population_size": 200,
            "number_of_generations": 100,
            "timer_stop_criteria": True,
            "timer_limit": 60,
        },
        "itea": {"npop": 1000, "ngens": 500},
        "eql": {"n_iter": 10_000},
    }


def test_execution_controls_are_not_algorithm_parameters() -> None:
    configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
    controls = configuration["execution_controls"]

    assert controls["threads"] == 1
    assert controls["prediction_repeats"] == 5
    assert controls["expression_analysis_timeout_seconds_per_stage"] == 60


def test_suite_v4_only_extends_problem_and_dataset_configuration() -> None:
    suite_v3 = json.loads(CONFIG.read_text(encoding="utf-8"))
    suite_v4 = json.loads(CONFIG_V4.read_text(encoding="utf-8"))

    assert suite_v4["problems"] == ["cantilever", "borehole", "piston", "ccpp"]
    assert set(suite_v4["dataset_generation"]) == set(suite_v4["problems"])
    for key in ("seeds", "input_scalings", "execution_controls"):
        assert suite_v4[key] == suite_v3[key]
    assert {
        algorithm: entry["parameters"]
        for algorithm, entry in suite_v4["algorithms"].items()
    } == {
        algorithm: entry["parameters"]
        for algorithm, entry in suite_v3["algorithms"].items()
    }


def test_suite_v5_preserves_v4_protocol_and_adds_naval_problem() -> None:
    suite_v4 = json.loads(CONFIG_V4.read_text(encoding="utf-8"))
    suite_v5 = json.loads(CONFIG_V5.read_text(encoding="utf-8"))

    assert suite_v5["problems"] == [
        "cantilever", "borehole", "piston", "ccpp", "naval_propulsion"
    ]
    assert set(suite_v5["dataset_generation"]) == set(suite_v5["problems"])
    for key in ("seeds", "input_scalings", "execution_controls"):
        assert suite_v5[key] == suite_v4[key]
    assert {
        algorithm: entry["parameters"]
        for algorithm, entry in suite_v5["algorithms"].items()
    } == {
        algorithm: entry["parameters"]
        for algorithm, entry in suite_v4["algorithms"].items()
    }


def test_suite_v6_preserves_v5_protocol_and_adds_wing_weight() -> None:
    suite_v5 = json.loads(CONFIG_V5.read_text(encoding="utf-8"))
    suite_v6 = json.loads(CONFIG_V6.read_text(encoding="utf-8"))

    assert suite_v6["problems"] == [
        "cantilever", "borehole", "piston", "ccpp", "naval_propulsion",
        "wing_weight",
    ]
    assert set(suite_v6["dataset_generation"]) == set(suite_v6["problems"])
    for key in ("seeds", "input_scalings", "execution_controls"):
        assert suite_v6[key] == suite_v5[key]
    assert {
        algorithm: entry["parameters"]
        for algorithm, entry in suite_v6["algorithms"].items()
    } == {
        algorithm: entry["parameters"]
        for algorithm, entry in suite_v5["algorithms"].items()
    }
