# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Static visualization utilities for clustering results.

All functions accept generic (X, labels) inputs so they can be reused
across K-Means, DBSCAN and SOM without code duplication.

Main functions:
    - plot_dimensionality_reduction() : PCA or t-SNE scatter plot
    - plot_centroid_heatmap()         : mean feature values per cluster
    - plot_metrics_barplot()          : compare one metric across experiments
    - plot_cluster_sizes()  : bar chart of how many politicians per cluster
    - generate_all_plots()            : run all of the above for 6 experiments
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUTPUT_DIR = Path("data/visualizations")

CLUSTER_PALETTE = "tab10"
OUTLIER_COLOR = "#AAAAAA"

# Metrics produced by run_evaluation.py that we want barplots for.
INTERNAL_METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz"]


def _save_or_show(fig: plt.Figure, output_path: Optional[Path]) -> None:
    """
    Save the figure to disk, or display it if no path is given.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        output_path (Optional[Path]): Destination file. None = interactive.
    """
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_dimensionality_reduction(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    title: str = "",
    output_path: Optional[Path] = None,
) -> None:
    """
    Scatter plot of clustering results reduced to 2D.

    Outliers (label == -1) are always shown in grey with a different marker
    so they stand out without a dedicated cluster color.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            -1 = outlier.
        method (str): Reduction method — "pca" or "tsne".
        title (str): Plot title. Auto-generated if empty.
        output_path (Optional[Path]): Save path. Displays
            interactively if None.

    Raises:
        ValueError: If method is not "pca" or "tsne".
    """
    if method not in ("pca", "tsne"):
        raise ValueError(f"method must be 'pca' or 'tsne', got '{method}'")

    # Reduce to 2D. perplexity for t-SNE must be < n_samples.
    if method == "pca":
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        perplexity = min(30, len(X) - 1)
        X_2d = TSNE(
            n_components=2, perplexity=perplexity, random_state=42
        ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))

    unique_labels = sorted(set(labels))
    # Build a discrete color palette with as many colors as there are clusters.
    palette = sns.color_palette(CLUSTER_PALETTE, n_colors=len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        is_outlier = label == -1
        color = OUTLIER_COLOR if is_outlier else palette[i]
        marker = "x" if is_outlier else "o"
        n = mask.sum()
        if is_outlier:
            cluster_name = f"Outliers ({n})"
        else:
            cluster_name = f"Cluster {label} ({n})"

        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[color],
            marker=marker,
            s=60,
            alpha=0.8,
            label=cluster_name,
        )

    ax.set_title(title or f"Clustering ({method.upper()})", fontsize=13)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend(loc="best", fontsize=9)

    _save_or_show(fig, output_path)


def plot_centroid_heatmap(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    title: str = "",
    output_path: Optional[Path] = None,
) -> None:
    """
    Heatmap of mean feature values per cluster.

    Rows are clusters, columns are investment subtypes (or any feature names
    passed in). Outlier points (label == -1) are excluded. This reveals which
    investment types define each cluster at a glance.

    Args:
        feature_matrix (np.ndarray): Shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels, aligned with feature_matrix rows.
        feature_names (list): Column names (length must equal n_features).
        title (str): Plot title.
        output_path (Optional[Path]): Save path. Displays
            interactively if None.
    """
    # Drop engineered numeric meta-features — they are on a different scale
    # and would crush the color range, hiding the subtype signal.
    META_FEATURES = {"total_assets", "diversity", "concentration"}
    subtype_names = [f for f in feature_names if f not in META_FEATURES]

    # Build a DataFrame to leverage groupby easily.
    df = pd.DataFrame(feature_matrix, columns=feature_names)
    df["_label"] = labels

    # Exclude outliers — their "centroid" is not meaningful.
    df = df[df["_label"] != -1]

    # Compute mean of each subtype column per cluster.
    centroid_df = df.groupby("_label")[subtype_names].mean()
    centroid_df.index = [f"Cluster {i}" for i in centroid_df.index]

    # Wider figure when there are many features (e.g. 38 subtypes).
    fig_width = max(14, len(feature_names) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, max(4, len(centroid_df) * 1.2)))

    sns.heatmap(
        centroid_df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        annot=len(feature_names) <= 20,  # annotations only if not too crowded
        fmt=".2f",
    )

    ax.set_title(title or "Centroid heatmap", fontsize=13)
    ax.set_xlabel("Investment subtype")
    ax.set_ylabel("Cluster")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    _save_or_show(fig, output_path)


def plot_metrics_barplot(
    results_df: pd.DataFrame,
    metric_col: str,
    title: str = "",
    output_path: Optional[Path] = None,
) -> None:
    """
    Bar chart comparing one clustering metric across all experiments.

    Each bar is one (algorithm, feature_type) combination. This makes it
    easy to see which combination performs best for a given metric.

    Args:
        results_df (pd.DataFrame): Must contain "experiment" and metric_col.
        metric_col (str): Column to plot (e.g. "silhouette").
        title (str): Plot title.
        output_path (Optional[Path]): Save path. Displays
            interactively if None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=results_df,
        x="experiment",
        y=metric_col,
        hue="experiment",
        palette=CLUSTER_PALETTE,
        legend=False,
        ax=ax,
    )

    ax.set_title(title or metric_col.replace("_", " ").title(), fontsize=13)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    # Rotate long experiment labels so they don't overlap.
    plt.xticks(rotation=30, ha="right", fontsize=9)

    _save_or_show(fig, output_path)


def plot_cluster_sizes(
    labels: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
) -> None:
    """
    Bar chart showing how many politicians are in each cluster.

    Outliers (label == -1) are included and labelled "Outliers".

    Args:
        labels (np.ndarray): Cluster labels of shape (n_samples,).
        title (str): Plot title.
        output_path (Optional[Path]): Save path. Displays
            interactively if None.
    """
    unique, counts = np.unique(labels, return_counts=True)
    bar_labels = [
        "Outliers" if lbl == -1 else f"Cluster {lbl}" for lbl in unique
    ]
    colors = [OUTLIER_COLOR if lbl == -1 else None for lbl in unique]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(bar_labels, counts)

    # Apply grey color to the outlier bar specifically.
    for bar, color in zip(bars, colors):
        if color:
            bar.set_color(color)

    # Show count on top of each bar.
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(title or "Cluster sizes", fontsize=13)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of politicians")

    _save_or_show(fig, output_path)


def generate_all_plots(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Generate and save all static visualizations for the 6 experiments.

    For each (algorithm, feature_type) pair this produces:
      - PCA scatter plot
      - t-SNE scatter plot
      - Centroid heatmap
      - Cluster size bar chart

    Then, for the full comparison table:
      - One barplot per internal metric (silhouette, DB, CH)

    Args:
        output_dir (Path): Root directory where all PNG files are saved.
    """
    # Import here to avoid a circular dependency at module level.
    from capitolwatch.analysis.run_evaluation import _build_experiment_configs
    from capitolwatch.analysis.feature_store import load_features

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the evaluation results CSV and add a combined experiment label.
    results_csv = output_dir / "evaluation_results.csv"
    results_df = pd.read_csv(results_csv)
    results_df["experiment"] = (
        results_df["algo_name"] + " / " + results_df["feature_type"]
    )

    configs = _build_experiment_configs()

    for cfg in configs:
        algo = cfg["algo"]
        feature_type = cfg["feature_type"]
        label = f"{algo} / {feature_type}"
        print(f"  Plotting {label} ...", end=" ", flush=True)

        X, labels = cfg["loader"]()

        # Load raw (unscaled) features to show interpretable values in heatmap.
        raw_matrix = load_features(feature_type)
        feature_names = list(raw_matrix.columns)
        X_raw = raw_matrix.to_numpy()

        plot_dimensionality_reduction(
            X, labels, method="pca",
            title=f"{label} — PCA",
            output_path=output_dir / f"pca_{algo}_{feature_type}.png",
        )
        plot_dimensionality_reduction(
            X, labels, method="tsne",
            title=f"{label} — t-SNE",
            output_path=output_dir / f"tsne_{algo}_{feature_type}.png",
        )
        plot_centroid_heatmap(
            X_raw, labels, feature_names,
            title=f"Centroids — {label}",
            output_path=output_dir / f"heatmap_{algo}_{feature_type}.png",
        )
        plot_cluster_sizes(
            labels,
            title=f"Cluster sizes — {label}",
            output_path=output_dir / f"sizes_{algo}_{feature_type}.png",
        )
        print("done")

    # One barplot per internal metric across all 6 experiments.
    for metric in INTERNAL_METRICS:
        plot_metrics_barplot(
            results_df,
            metric_col=metric,
            title=f"Comparison — {metric.replace('_', ' ').title()}",
            output_path=output_dir / f"metrics_{metric}.png",
        )
        print(f"  Metrics barplot: {metric} ... done")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()
