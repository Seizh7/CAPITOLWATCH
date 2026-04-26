# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
SOM experiment script.

Loads features from the feature store, normalizes them with MinMaxScaler
(required because SOM neuron weights are initialized in [0, 1]), trains
the SOM, extracts clusters via K-Means on neuron weights, then saves the
U-Matrix heatmap and the grid politician map.

Usage:
    python -m capitolwatch.analysis.clustering.run_som
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.preprocessing import normalize_features
from capitolwatch.analysis.clustering.som import SOMClusterer


def _load_and_normalize(feature_type: str) -> tuple:
    """
    Load a feature matrix from the store and apply MinMaxScaler.

    Args:
        feature_type (str): One of "freq_baseline" or "freq_weighted".

    Returns:
        tuple: (X: np.ndarray, politician_labels: pd.DataFrame)
    """
    politician_labels = load_features("politician_labels")
    matrix = load_features(feature_type)
    matrix_scaled, _ = normalize_features(matrix, MinMaxScaler())
    return matrix_scaled.to_numpy(), politician_labels


def _train_som(
    X: np.ndarray,
    feature_type: str,
    m: int = 7,
    n: int = 7,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    n_iterations: int = 1000,
) -> SOMClusterer:
    """
    Initialize, train a SOM, and save the bare U-Matrix heatmap.

    Args:
        X (np.ndarray): Normalized feature matrix.
        feature_type (str): Used to name the output PNG.
        m (int): Number of SOM grid rows.
        n (int): Number of SOM grid columns.
        sigma (float): Initial neighborhood radius.
        learning_rate (float): Initial weight update step size.
        n_iterations (int): Number of training iterations.

    Returns:
        SOMClusterer: Fitted clusterer instance.
    """
    clusterer = SOMClusterer(
        m=m,
        n=n,
        sigma=sigma,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        random_seed=42,
    )
    clusterer.fit(X)

    # save the bare U-Matrix (no overlay) to inspect cluster boundaries
    clusterer.plot_umatrix(
        save_path=f"data/figures/som_umatrix_{feature_type}.png",
    )
    return clusterer


def _extract_and_visualize(
    clusterer: SOMClusterer,
    X: np.ndarray,
    feature_type: str,
    politician_labels,
    n_clusters: int = 3,
) -> None:
    """
    Extract clusters from trained SOM and save both visualizations.

    Args:
        clusterer (SOMClusterer): Fitted SOM (fit() already called).
        X (np.ndarray): Normalized feature matrix.
        feature_type (str): Used to name the output PNGs.
        politician_labels (pd.DataFrame): Politician metadata.
        n_clusters (int): Number of clusters for K-Means on neuron weights.
    """
    clusterer.extract_clusters(n_clusters=n_clusters)

    # U-Matrix with politician overlay color-coded by party
    clusterer.plot_umatrix(
        matrix=X,
        politician_labels=politician_labels,
        save_path=(
            f"data/figures/som_umatrix_overlay_{feature_type}.png"
        ),
    )

    # grid map with abbreviated names and cluster colors
    clusterer.plot_som_map(
        X,
        politician_labels=politician_labels,
        save_path=f"data/figures/som_map_{feature_type}.png",
    )


def run_som_experiment(
    feature_type: str,
    m: int = 7,
    n: int = 7,
    n_clusters: int = 4,
    n_iterations: int = 1000,
) -> dict:
    """
    Run the full SOM experiment for one feature type.

    Steps:
    1. Load and normalize features with MinMaxScaler.
    2. Train the SOM (m x n grid).
    3. Save U-Matrix heatmap.
    4. Extract clusters (K-Means on neuron weights).
    5. Save U-Matrix overlay and grid map.
    6. Return a structured result dict.

    Args:
        feature_type (str): One of "freq_baseline" or "freq_weighted".
        m (int): Number of SOM grid rows.
        n (int): Number of SOM grid columns.
        n_clusters (int): Number of clusters to extract.
        n_iterations (int): Number of SOM training iterations.

    Returns:
        dict: {
            "feature_type": str,
            "grid_size": tuple[int, int],
            "n_clusters": int,
            "n_iterations": int,
            "cluster_sizes": dict,
            "labels": np.ndarray,
            "bmu_coords": list[tuple],
        }
    """
    X, politician_labels = _load_and_normalize(feature_type)

    clusterer = _train_som(
        X,
        feature_type,
        m=m,
        n=n,
        n_iterations=n_iterations,
    )

    _extract_and_visualize(
        clusterer,
        X,
        feature_type,
        politician_labels,
        n_clusters=n_clusters,
    )

    # count members per cluster
    unique, counts = np.unique(clusterer.labels_, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}

    return {
        "feature_type": feature_type,
        "grid_size": (m, n),
        "n_clusters": n_clusters,
        "n_iterations": n_iterations,
        "cluster_sizes": cluster_sizes,
        "labels": clusterer.labels_,
        "bmu_coords": clusterer.bmu_coords_,
    }


def print_results(results: dict) -> None:
    """
    Print SOM experiment results.

    Args:
        results (dict): Output of run_som_experiment().
    """
    print(f"\n--- SOM on {results['feature_type']} ---")
    m, n = results["grid_size"]
    print(f"Grid: {m}x{n} | Iterations: {results['n_iterations']}")
    print(f"Clusters extracted: {results['n_clusters']}")
    print(f"Cluster sizes: {results['cluster_sizes']}")

    # grid utilization: how many of the m*n neurons were actually activated
    unique_bmus = len(set(results["bmu_coords"]))
    total_neurons = m * n
    print(
        f"Grid utilization: {unique_bmus}/{total_neurons} neurons activated "
        f"({100 * unique_bmus / total_neurons:.1f}%)"
    )


if __name__ == "__main__":
    for feature_type in ("freq_baseline", "freq_weighted"):
        results = run_som_experiment(feature_type)
        print_results(results)
