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


def save_scaling_effect(rows: list[dict[str, object]], output: Path) -> None:
    lookup = {(str(r["problem"]), str(r["algorithm"]), str(r["scaling"])): r for r in rows}
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for algorithm in ALGORITHMS:
        raw = np.array([number(lookup[(p, algorithm, "raw")], "nrmse_range_mean") for p in PROBLEMS])
        scaled = np.array([number(lookup[(p, algorithm, "domain_minmax")], "nrmse_range_mean") for p in PROBLEMS])
        ax.scatter(raw, scaled, label=algorithm, s=45, alpha=0.85)
    finite = [number(r, "nrmse_range_mean") for r in rows]
    finite = np.array([v for v in finite if np.isfinite(v) and v > 0])
    lower, upper = finite.min(), finite.max()
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Raw-input mean $\mathrm{NRMSE}_{range}$")
    ax.set_ylabel(r"Normalized-input mean $\mathrm{NRMSE}_{range}$")
    ax.set_title("Effect of domain normalization")
    ax.legend(ncol=2)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_accuracy_complexity(rows: list[dict[str, object]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    markers = {"raw": "o", "domain_minmax": "^"}
    colors = dict(zip(ALGORITHMS, plt.get_cmap("tab10").colors))
    for row in rows:
        error = number(row, "nrmse_range_mean")
        nodes = number(row, "simplified_node_count_mean")
        if not np.isfinite(error) or not np.isfinite(nodes) or error <= 0 or nodes <= 0:
            continue
        ax.scatter(nodes, error, marker=markers[str(row["scaling"])],
                   color=colors[str(row["algorithm"])], alpha=0.72, s=42)
    for algorithm in ALGORITHMS:
        ax.scatter([], [], color=colors[algorithm], label=algorithm)
    ax.scatter([], [], color="black", marker="o", label="raw")
    ax.scatter([], [], color="black", marker="^", label="domain normalized")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean simplified expression-tree node count")
    ax.set_ylabel(r"Mean $\mathrm{NRMSE}_{range}$")
    ax.set_title("Predictive error and symbolic complexity")
    ax.legend(ncol=2, fontsize=8)
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
    save_scaling_effect(rows, args.output_dir / "normalization_effect.png")
    save_accuracy_complexity(rows, args.output_dir / "accuracy_complexity.png")
    print(f"Wrote four figures to {args.output_dir}")


if __name__ == "__main__":
    main()
