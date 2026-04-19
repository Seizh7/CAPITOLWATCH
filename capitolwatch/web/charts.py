# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Plotly chart factory for the Streamlit dashboard.

Every public function in this module returns a go.Figure that can be
displayed directly with st.plotly_chart(). No data loading or heavy
transformation happens here. Callers are responsible for passing clean
arrays and labels.

Main functions:
    scatter_pca_plotly        : PCA 2D scatter with all reduction inside
    heatmap_centroids_plotly  : mean investment value per cluster
    barplot_metrics_plotly    : metric comparison across experiments
    heatmap_confusion_plotly  : cluster x party confusion matrix
    som_umatrix_plotly        : SOM Unified Distance Matrix
    som_map_plotly            : politicians mapped on the SOM grid
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA


# --- Module-level constants (override externally if needed) ---

PARTY_COLOR_MAP: dict[str, str] = {
    "Republican": "#E82026",
    "Democratic": "#3333CC",
    "Independent": "#6E8B3D",
}

OUTLIER_COLOR: str = "#AAAAAA"

# tab10-like discrete color list for cluster labels
CLUSTER_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Feature columns that are on a different scale and should be hidden from
# centroid heatmaps so they don't crush the investment-subtype signal
DEFAULT_META_FEATURES: tuple[str, ...] = (
    "total_assets",
    "diversity",
    "concentration",
)


def _build_scatter_figure(
    X_2d: np.ndarray,
    labels: np.ndarray,
    hover_texts: list[str],
    title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """
    Internal helper that turns pre-reduced 2D coordinates into a figure.

    Outliers (label == -1) are always shown in grey with an X marker.
    Each cluster gets a colour from CLUSTER_COLORS.

    Args:
        X_2d (np.ndarray): 2-D coordinates of shape (n_samples, 2).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            Label -1 is treated as an outlier.
        hover_texts (list[str]): One tooltip string per sample.
        title (str): Figure title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.

    Returns:
        go.Figure: Plotly scatter figure.
    """
    fig = go.Figure()

    for label in sorted(set(labels)):
        mask = labels == label
        is_outlier = label == -1
        color = (
            OUTLIER_COLOR
            if is_outlier
            else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        )
        symbol = "x" if is_outlier else "circle"
        count = mask.sum()
        name = (
            f"Outliers ({count})"
            if is_outlier
            else f"Cluster {label} ({count})"
        )

        fig.add_trace(
            go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode="markers",
                name=name,
                text=[hover_texts[i] for i in np.where(mask)[0]],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    color=color,
                    symbol=symbol,
                    size=9,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="Cluster",
        height=480,
        template="plotly_white",
    )
    return fig


def scatter_pca_plotly(
    X: np.ndarray,
    labels: np.ndarray,
    hover_texts: list[str],
    title: str = "",
) -> go.Figure:
    """
    Reduce X to 2D with PCA and produce an interactive scatter plot.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            Label -1 = outlier.
        hover_texts (list[str]): Tooltip string for each sample.
        title (str): Figure title. Auto-generated if empty.

    Returns:
        go.Figure: 2-D PCA scatter figure.
    """
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    return _build_scatter_figure(
        X_2d,
        labels,
        hover_texts,
        title=title or "PCA 2D — clustering results",
        x_label="PCA Component 1",
        y_label="PCA Component 2",
    )


def heatmap_centroids_plotly(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    title: str = "",
    meta_features: tuple[str, ...] = DEFAULT_META_FEATURES,
) -> go.Figure:
    """
    Interactive heatmap of mean feature values per cluster.

    Meta-features (total_assets, diversity, concentration) are excluded
    because they live on a different scale from the investment subtype
    frequencies and would crush the colour range.

    Outliers (label == -1) are excluded from centroid computation because
    they do not belong to any cluster.

    Args:
        feature_matrix (np.ndarray): Shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels aligned row-for-row with
            feature_matrix.
        feature_names (list[str]): Column names; length must equal n_features.
        title (str): Figure title.
        meta_features (tuple[str, ...]): Feature names to exclude from the
            heatmap columns.

    Returns:
        go.Figure: Plotly heatmap figure.
    """
    subtype_names = [f for f in feature_names if f not in meta_features]

    df = pd.DataFrame(feature_matrix, columns=feature_names)
    df["_label"] = labels
    # exclude noise label so outlier "centroids" don't pollute the heatmap
    df = df[df["_label"] != -1]

    centroid_df = df.groupby("_label")[subtype_names].mean()
    y_labels = [f"Cluster {i}" for i in centroid_df.index]

    fig = go.Figure(
        go.Heatmap(
            z=centroid_df.values,
            x=subtype_names,
            y=y_labels,
            colorscale="YlOrRd",
            hovertemplate=(
                "Subtype: %{x}<br>Cluster: %{y}"
                "<br>Mean: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "Centroid heatmap",
        xaxis_title="Investment subtype",
        yaxis_title="Cluster",
        xaxis_tickangle=-45,
        height=max(300, len(centroid_df) * 100 + 200),
        template="plotly_white",
    )
    return fig


def barplot_metrics_plotly(
    results_df: pd.DataFrame,
    metric_col: str,
    title: str = "",
    experiment_col: str = "experiment",
) -> go.Figure:
    """
    Bar chart comparing one clustering metric across all experiments.

    Args:
        results_df (pd.DataFrame): Must contain experiment_col and metric_col.
        metric_col (str): Column to plot (e.g. "silhouette").
        title (str): Figure title. Auto-generated if empty.
        experiment_col (str): Column used as X-axis group labels.

    Returns:
        go.Figure: Plotly bar figure.
    """
    metric_values = results_df[metric_col].round(4)

    fig = go.Figure(
        go.Bar(
            x=results_df[experiment_col],
            y=metric_values,
            text=metric_values.astype(str),
            textposition="outside",
            marker_color=CLUSTER_COLORS[: len(results_df)],
            hovertemplate=(
                "%{x}<br>"
                + metric_col.replace("_", " ").title()
                + ": %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or metric_col.replace("_", " ").title(),
        xaxis_title="Experiment",
        yaxis_title=metric_col.replace("_", " ").title(),
        xaxis_tickangle=-20,
        height=450,
        template="plotly_white",
    )
    return fig


def heatmap_confusion_plotly(
    conf_matrix: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """
    Interactive heatmap of a cluster x party confusion matrix.

    Args:
        conf_matrix (pd.DataFrame): Rows = cluster ids, columns = party names.
            As returned by
            capitolwatch.analysis.evaluation.build_confusion_matrix().
        title (str): Figure title.

    Returns:
        go.Figure: Plotly annotated heatmap figure.
    """
    z = conf_matrix.values
    x_labels = list(conf_matrix.columns)
    y_labels = [f"Cluster {i}" for i in conf_matrix.index]
    # cell annotations as strings for the texttemplate
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate=(
                "Party: %{x}<br>Cluster: %{y}<br>Count: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "Cluster x Party confusion matrix",
        xaxis_title="Party",
        yaxis_title="Cluster",
        height=max(280, len(y_labels) * 90 + 150),
        template="plotly_white",
    )
    return fig


def som_umatrix_plotly(
    umatrix: np.ndarray,
    title: str = "",
) -> go.Figure:
    """
    Interactive heatmap of the SOM Unified Distance Matrix.

    Dark cells indicate cluster cores (low distance to neighbours).
    Light cells indicate boundaries between clusters.

    Args:
        umatrix (np.ndarray): U-Matrix of shape (m, n), values in [0, 1].
            As returned by SOMClusterer.compute_umatrix().
        title (str): Figure title.

    Returns:
        go.Figure: Plotly heatmap figure.
    """
    fig = go.Figure(
        go.Heatmap(
            z=umatrix,
            colorscale="Blues_r",
            colorbar=dict(title="Distance"),
            hovertemplate=(
                "Row: %{y}<br>Col: %{x}<br>Distance: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "SOM U-Matrix",
        xaxis_title="Neuron column",
        yaxis_title="Neuron row",
        height=480,
        template="plotly_white",
    )
    return fig


def som_map_plotly(
    umatrix: np.ndarray,
    bmu_coords: list[tuple[int, int]],
    hover_texts: list[str],
    party_colors: list[str],
    title: str = "",
    jitter_scale: float = 0.18,
) -> go.Figure:
    """
    Politicians mapped on the SOM grid, overlaid on the U-Matrix background.

    A small random jitter is added to BMU positions so that politicians
    sharing the same Best Matching Unit do not overlap completely.

    Args:
        umatrix (np.ndarray): U-Matrix of shape (m, n).
        bmu_coords (list[tuple[int, int]]): (row, col) BMU for each politician.
        hover_texts (list[str]): Tooltip text per politician.
        party_colors (list[str]): Marker hex colour per politician (usually
            built from PARTY_COLOR_MAP).
        title (str): Figure title.
        jitter_scale (float): Half-width of the uniform jitter applied to
            prevent total overlap of same-BMU politicians (default: 0.18).

    Returns:
        go.Figure: Plotly figure combining the U-Matrix and a scatter overlay.
    """
    fig = go.Figure()

    # --- layer 0: U-Matrix background ---
    fig.add_trace(
        go.Heatmap(
            z=umatrix,
            colorscale="Blues_r",
            showscale=True,
            colorbar=dict(title="Distance", x=1.02),
            hovertemplate=(
                "Row: %{y}, Col: %{x}<br>Distance: %{z:.4f}<extra></extra>"
            ),
        )
    )

    # --- layer 1: politicians as scatter markers ---
    rng = np.random.default_rng(seed=42)
    rows = np.array([c[0] for c in bmu_coords], dtype=float)
    cols = np.array([c[1] for c in bmu_coords], dtype=float)
    rows += rng.uniform(-jitter_scale, jitter_scale, size=len(rows))
    cols += rng.uniform(-jitter_scale, jitter_scale, size=len(cols))

    fig.add_trace(
        go.Scatter(
            x=cols,
            y=rows,
            mode="markers",
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                color=party_colors,
                size=10,
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            showlegend=False,
        )
    )

    # --- legend for party colours (synthetic traces, no data) ---
    for party, color in PARTY_COLOR_MAP.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=party,
                marker=dict(color=color, size=10),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title or "SOM political map",
        xaxis_title="Neuron column",
        yaxis_title="Neuron row",
        legend_title="Party",
        height=520,
        template="plotly_white",
    )
    return fig


if __name__ == "__main__":
    # Quick test with synthetic data — no real dataset required.
    import numpy as np
    from sklearn.datasets import make_blobs

    X_test, y_test = make_blobs(n_samples=60, centers=3, random_state=0)
    hover = [f"Sample {i} (cluster {y_test[i]})" for i in range(len(X_test))]

    fig_pca = scatter_pca_plotly(X_test, y_test, hover, title="Test PCA")
    fig_pca.show()

    results_test = pd.DataFrame(
        {
            "experiment": [
                "KM/fb", "KM/fw", "DB/fb", "DB/fw", "SOM/fb", "SOM/fw"
            ],
            "silhouette": [0.68, 0.76, 0.42, 0.49, 0.28, 0.47],
        }
    )
    fig_bar = barplot_metrics_plotly(
        results_test, "silhouette", title="Test barplot"
    )
    fig_bar.show()
