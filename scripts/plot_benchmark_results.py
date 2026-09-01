#!/usr/bin/env python3
"""Create adviser-facing figures from the protocol-valid benchmark summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROBLEMS = {
    "cantilever": ("benchmark_suite_v3", "Cantilever"),
    "borehole": ("benchmark_suite_v3", "Borehole"),
    "piston": ("benchmark_suite_v3", "Piston"),
    "ccpp": ("benchmark_suite_v4", "CCPP"),
    "naval_propulsion": ("benchmark_suite_v5", "Naval"),
    "wing_weight": ("benchmark_suite_v6", "Wing Weight"),
    "gas_turbine_nox": ("benchmark_suite_v7", "Gas Turbine NOx"),
    "concrete_strength": ("benchmark_suite_v8", "Concrete Strength"),
    "energy_efficiency_heating": ("benchmark_suite_v9", "Energy Efficiency"),
    "airfoil_self_noise": ("benchmark_suite_v10", "Airfoil Self-Noise"),
}
ALGORITHMS = ("gplearn", "operon", "pysr", "geneticengine", "itea", "eql")
SCALINGS = ("raw", "domain_minmax")


def load_rows(project_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for problem, (suite, label) in PROBLEMS.items():
        for scaling in SCALINGS:
            path = project_dir / "results" / problem / suite / scaling / "summary.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing summary: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row.update(problem=problem, problem_label=label, scaling=scaling)
                    rows.append(row)
    return rows


def number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else np.nan


def save_heatmap(rows: list[dict[str, object]], scaling: str, output: Path) -> None:
    selected = {(str(r["problem"]), str(r["algorithm"])): r for r in rows if r["scaling"] == scaling}
    matrix = np.array([
        [number(selected[(problem, algorithm)], "nrmse_range_mean") for algorithm in ALGORITHMS]
        for problem in PROBLEMS
    ])
    shown = np.log10(np.maximum(matrix, 1e-12))
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    image = ax.imshow(shown, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=30, ha="right")
    ax.set_yticks(range(len(PROBLEMS)), [label for _, label in PROBLEMS.values()])
    ax.set_title(f"Mean range-normalized RMSE ({scaling.replace('_', ' ')})")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "--" if np.isnan(value) else f"{value:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > np.nanmedian(shown) else "black")
    fig.colorbar(image, ax=ax, label=r"$\log_{10}(\mathrm{NRMSE}_{range})$")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_r2_heatmap(rows: list[dict[str, object]], scaling: str, output: Path) -> None:
    selected = {(str(r["problem"]), str(r["algorithm"])): r for r in rows if r["scaling"] == scaling}
    matrix = np.array([
        [number(selected[(problem, algorithm)], "r2_mean") for algorithm in ALGORITHMS]
        for problem in PROBLEMS
    ])
    shown = np.clip(matrix, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    image = ax.imshow(shown, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=30, ha="right")
    ax.set_yticks(range(len(PROBLEMS)), [label for _, label in PROBLEMS.values()])
    ax.set_title(f"Mean coefficient of determination ({scaling.replace('_', ' ')})")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "--" if np.isnan(value) else f"{value:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] < -0.65 else "black")
    fig.colorbar(image, ax=ax, label=r"Mean $R^2$ (color scale clipped to $[-1,1]$)")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_positive_metric_heatmap(
    rows: list[dict[str, object]], scaling: str, metric: str, title: str,
    colorbar_label: str, output: Path
) -> None:
    selected = {(str(r["problem"]), str(r["algorithm"])): r for r in rows if r["scaling"] == scaling}
    matrix = np.array([
        [number(selected[(problem, algorithm)], metric) for algorithm in ALGORITHMS]
        for problem in PROBLEMS
    ])
    shown = np.log10(np.where(matrix > 0, matrix, np.nan))
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    image = ax.imshow(shown, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=30, ha="right")
    ax.set_yticks(range(len(PROBLEMS)), [label for _, label in PROBLEMS.values()])
    ax.set_title(f"{title} ({scaling.replace('_', ' ')})")
    midpoint = float(np.nanmedian(shown))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "--" if np.isnan(value) else f"{value:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] < midpoint else "black")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_framework_bars(
    rows: list[dict[str, object]], metric: str, ylabel: str, title: str, output: Path
) -> None:
    values: dict[str, list[float]] = {scaling: [] for scaling in SCALINGS}
    for scaling in SCALINGS:
        for algorithm in ALGORITHMS:
            observations = [
                number(row, metric)
                for row in rows
                if row["scaling"] == scaling and row["algorithm"] == algorithm
            ]
            values[scaling].append(float(np.nanmedian(observations)))
    positions = np.arange(len(ALGORITHMS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    ax.bar(positions - width / 2, values["raw"], width, label="Raw inputs")
    ax.bar(positions + width / 2, values["domain_minmax"], width,
           label="Domain-normalized inputs")
    ax.set_xticks(positions, ALGORITHMS, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=project_dir / "results" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(project_dir)
    save_heatmap(rows, "raw", args.output_dir / "nrmse_heatmap_raw.png")
    save_heatmap(rows, "domain_minmax", args.output_dir / "nrmse_heatmap_domain_minmax.png")
    save_r2_heatmap(rows, "raw", args.output_dir / "r2_heatmap_raw.png")
    save_r2_heatmap(rows, "domain_minmax", args.output_dir / "r2_heatmap_domain_minmax.png")
    save_framework_bars(
        rows,
        "simplified_node_count_mean",
        "Simplified expression-tree nodes",
        "Expression complexity by framework",
        args.output_dir / "complexity_by_framework.png",
    )
    save_framework_bars(
        rows,
        "fit_seconds_mean",
        "Fitting time (seconds)",
        "Fitting time by framework",
        args.output_dir / "fit_time_by_framework.png",
    )
    save_positive_metric_heatmap(
        rows,
        "raw",
        "simplified_node_count_mean",
        "Mean simplified expression-tree node count",
        r"$\log_{10}(\mathrm{node\ count})$",
        args.output_dir / "complexity_heatmap_raw.png",
    )
    save_positive_metric_heatmap(
        rows,
        "domain_minmax",
        "simplified_node_count_mean",
        "Mean simplified expression-tree node count",
        r"$\log_{10}(\mathrm{node\ count})$",
        args.output_dir / "complexity_heatmap_domain_minmax.png",
    )
    save_positive_metric_heatmap(
        rows,
        "raw",
        "fit_seconds_mean",
        "Mean fitting time",
        r"$\log_{10}(\mathrm{seconds})$",
        args.output_dir / "fit_time_heatmap_raw.png",
    )
    save_positive_metric_heatmap(
        rows,
        "domain_minmax",
        "fit_seconds_mean",
        "Mean fitting time",
        r"$\log_{10}(\mathrm{seconds})$",
        args.output_dir / "fit_time_heatmap_domain_minmax.png",
    )
    print(f"Wrote ten figures to {args.output_dir}")


if __name__ == "__main__":
    main()
