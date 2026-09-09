#!/usr/bin/env python3
"""Create adviser-facing figures from the protocol-valid benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize_latest_results import write_latest_summary


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
ALGORITHM_LABELS = {
    "gplearn": "gplearn",
    "operon": "Operon",
    "pysr": "PySR",
    "geneticengine": "GeneticEngine",
    "itea": "ITEA",
    "eql": "EQL",
}


def load_rows(project_dir: Path) -> list[dict[str, object]]:
    summary = project_dir / "results" / "latest" / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(
            f"Missing resolved summary: {summary}. Run scripts/summarize_latest_results.py first."
        )
    rows: list[dict[str, object]] = []
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            problem = str(row["problem"])
            row.update(problem_label=PROBLEMS[problem][1], scaling=row["input_scaling"])
            rows.append(row)
    expected = len(PROBLEMS) * len(SCALINGS) * len(ALGORITHMS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} resolved summary rows, found {len(rows)}")
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
    colorbar_label: str, output: Path, mark_fallbacks: bool = False
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
            if mark_fallbacks and number(selected[(list(PROBLEMS)[i], ALGORITHMS[j])],
                                                 "unsimplified_fallback_trials") > 0:
                label += "*"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] < midpoint else "black")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_normalization_effect(rows: list[dict[str, object]], output: Path) -> None:
    selected = {
        (str(row["problem"]), str(row["algorithm"]), str(row["scaling"])): row
        for row in rows
    }
    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    all_values: list[float] = []
    for index, algorithm in enumerate(ALGORITHMS):
        raw_values = []
        normalized_values = []
        for problem in PROBLEMS:
            raw = number(selected[(problem, algorithm, "raw")], "nrmse_range_mean")
            normalized = number(
                selected[(problem, algorithm, "domain_minmax")], "nrmse_range_mean"
            )
            if np.isfinite(raw) and np.isfinite(normalized) and raw > 0 and normalized > 0:
                raw_values.append(raw)
                normalized_values.append(normalized)
                all_values.extend((raw, normalized))
        ax.scatter(raw_values, normalized_values, s=42, alpha=0.8,
                   color=colors(index), label=ALGORITHM_LABELS[algorithm])
    lower = 10 ** np.floor(np.log10(min(all_values)))
    upper = 10 ** np.ceil(np.log10(max(all_values)))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Mean NRMSE with raw inputs")
    ax.set_ylabel("Mean NRMSE with domain-normalized inputs")
    ax.set_title("Effect of domain normalization")
    ax.grid(which="both", linestyle=":", alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def load_trial_nrmse(project_dir: Path) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], int]]:
    values = {(scaling, algorithm): [] for scaling in SCALINGS for algorithm in ALGORITHMS}
    failures = {(scaling, algorithm): 0 for scaling in SCALINGS for algorithm in ALGORITHMS}
    manifest = project_dir / "results" / "latest" / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing resolved manifest: {manifest}. Run scripts/summarize_latest_results.py first."
        )
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scaling = row["input_scaling"]
            algorithm = row["algorithm"]
            if row["status"] == "failed":
                failures[(scaling, algorithm)] += 1
            elif row["status"] == "success":
                result = json.loads((project_dir / row["path"]).read_text(encoding="utf-8"))
                value = result.get("nrmse_range")
                if value is not None and np.isfinite(float(value)) and float(value) > 0:
                    values[(scaling, algorithm)].append(float(value))
    return values, failures


def save_repetition_distributions(project_dir: Path, output_dir: Path) -> None:
    values, failures = load_trial_nrmse(project_dir)
    rng = np.random.default_rng(20260908)
    filenames = {
        "raw": "repetition_nrmse_raw.png",
        "domain_minmax": "repetition_nrmse_domain_minmax.png",
    }
    for scaling in SCALINGS:
        fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
        groups = [values[(scaling, algorithm)] for algorithm in ALGORITHMS]
        scaling_values = [value for observations in groups for value in observations]
        lower = 10 ** np.floor(np.log10(min(scaling_values)))
        upper = 10 ** np.ceil(np.log10(max(scaling_values)))
        box = ax.boxplot(groups, tick_labels=[ALGORITHM_LABELS[a] for a in ALGORITHMS],
                         showfliers=False, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#8fb9dd")
            patch.set_alpha(0.65)
        for position, (algorithm, observations) in enumerate(zip(ALGORITHMS, groups), start=1):
            jitter = rng.uniform(-0.16, 0.16, len(observations))
            ax.scatter(position + jitter, observations, s=8, alpha=0.28,
                       color="#244a6b", linewidths=0)
            failed = failures[(scaling, algorithm)]
            if failed:
                ax.text(position, 0.97, f"{failed} failed", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="#a51c30")
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        ax.set_title("Raw inputs" if scaling == "raw" else "Domain-normalized inputs")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.35)
        ax.set_ylabel("Repetition-level range-normalized RMSE")
        fig.savefig(output_dir / filenames[scaling], dpi=220)
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
    parser.add_argument(
        "--config", type=Path,
        default=project_dir / "configs" / "benchmark_suite_v10.json",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=project_dir / "results" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses = write_latest_summary(
        project_dir, args.config, project_dir / "results" / "latest"
    )
    if statuses["missing"]:
        raise RuntimeError(
            f"Resolved benchmark contains {statuses['missing']} missing trial coordinates"
        )
    if statuses["analysis_missing"]:
        raise RuntimeError(
            "Resolved benchmark contains "
            f"{statuses['analysis_missing']} successful trials without current analysis sidecars"
        )
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
        mark_fallbacks=True,
    )
    save_positive_metric_heatmap(
        rows,
        "domain_minmax",
        "simplified_node_count_mean",
        "Mean simplified expression-tree node count",
        r"$\log_{10}(\mathrm{node\ count})$",
        args.output_dir / "complexity_heatmap_domain_minmax.png",
        mark_fallbacks=True,
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
    save_normalization_effect(rows, args.output_dir / "normalization_effect.png")
    save_repetition_distributions(project_dir, args.output_dir)
    print(
        f"Wrote thirteen figures to {args.output_dir} from resolved results "
        f"(success={statuses['success']}, failed={statuses['failed']})"
    )


if __name__ == "__main__":
    main()
