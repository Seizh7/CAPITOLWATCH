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
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

OUTPUT_DIR = Path("data/figures")

CLUSTER_PALETTE = "tab10"
OUTLIER_COLOR = "#AAAAAA"

# Metrics produced by run_evaluation.py that we want barplots for.
INTERNAL_METRICS = ["silhouette"]


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
    title: str = "",
    output_path: Optional[Path] = None,
) -> None:
    """
    Scatter plot of clustering results reduced to 2D via PCA.

    Outliers (label == -1) are always shown in grey with a different marker
    so they stand out without a dedicated cluster color.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            -1 = outlier.
        title (str): Plot title. Auto-generated if empty.
        output_path (Optional[Path]): Save path. Displays
            interactively if None.
    """
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)

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

    ax.set_title(title or "Clustering (PCA)", fontsize=13)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
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
