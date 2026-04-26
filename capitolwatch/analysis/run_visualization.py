# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Visualization pipeline for all 6 clustering experiments.

Generates static PNG files and saves them to data/figures/.

Plots produced per experiment (6 experiments x 3 plots = 18 files):
    - heatmap_{algo}_{feature_type}.png  : mean feature value per cluster
    - sizes_{algo}_{feature_type}.png    : number of politicians per cluster
    - pca_{algo}_{feature_type}.png      : 2D scatter plot via PCA

Plus one barplot per internal metric:
    - metrics_silhouette.png

Usage:
    python -m capitolwatch.analysis.run_visualization
"""

from pathlib import Path

import pandas as pd

from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.run_evaluation import _build_experiment_configs
from capitolwatch.analysis.visualization import (
    plot_centroid_heatmap,
    plot_cluster_sizes,
    plot_dimensionality_reduction,
    plot_metrics_barplot,
)

OUTPUT_DIR = Path("data/figures")
INTERNAL_METRICS = ["silhouette"]


def run_simple_plots(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Generate heatmaps and cluster-size barplots for all 6 experiments.

    No dimensionality reduction involved — these plots work directly on
    the raw feature values, so they are fast and easy to interpret.

    Args:
        output_dir (Path): Directory where PNG files are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = _build_experiment_configs()

    for cfg in configs:
        algo = cfg["algo"]
        feature_type = cfg["feature_type"]
        label = f"{algo} / {feature_type}"
        print(f"  {label}")

        # Re-fit the algorithm to get cluster labels.
        _, labels = cfg["loader"]()

        # Load raw (unscaled) feature matrix so heatmap values are readable.
        raw_matrix = load_features(feature_type)
        feature_names = list(raw_matrix.columns)
        X_raw = raw_matrix.to_numpy()

        plot_centroid_heatmap(
            X_raw,
            labels,
            feature_names,
            title=f"Mean investment per cluster — {label}",
            output_path=output_dir / f"heatmap_{algo}_{feature_type}.png",
        )
        plot_cluster_sizes(
            labels,
            title=f"Cluster sizes — {label}",
            output_path=output_dir / f"sizes_{algo}_{feature_type}.png",
        )


def run_metrics_barplots(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Generate one comparison barplot per internal metric.

    Reads evaluation_results.csv produced by run_evaluation.py.

    Args:
        output_dir (Path): Directory containing the CSV and where PNGs are
            saved.

    Raises:
        FileNotFoundError: If evaluation_results.csv does not exist yet.
    """
    results_csv = output_dir / "evaluation_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found. Run run_evaluation.py first."
        )

    results_df = pd.read_csv(results_csv)
    # Build a readable experiment label for the X axis.
    results_df["experiment"] = (
        results_df["algo_name"] + "\n" + results_df["feature_type"]
    )

    for metric in INTERNAL_METRICS:
        print(f"  Metrics barplot: {metric}")
        plot_metrics_barplot(
            results_df,
            metric_col=metric,
            title=f"Comparison — {metric.replace('_', ' ').title()}",
            output_path=output_dir / f"metrics_{metric}.png",
        )


def run_pca_plots(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Generate PCA 2D scatter plots for all 6 experiments.

    Each point is one politician. Color = cluster. Grey crosses = outliers.

    Args:
        output_dir (Path): Directory where PNG files are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = _build_experiment_configs()

    for cfg in configs:
        algo = cfg["algo"]
        feature_type = cfg["feature_type"]
        label = f"{algo} / {feature_type}"
        print(f"  (PCA) {label}")

        X, labels = cfg["loader"]()

        plot_dimensionality_reduction(
            X,
            labels,
            title=f"{label} — PCA",
            output_path=output_dir / f"pca_{algo}_{feature_type}.png",
        )


if __name__ == "__main__":
    print("\n=== Simple plots (heatmaps + sizes) ===")
    run_simple_plots()

    print("\n=== Metrics barplots ===")
    run_metrics_barplots()

    print("\n=== PCA scatter plots ===")
    run_pca_plots()

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
