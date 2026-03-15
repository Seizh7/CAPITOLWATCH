# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
DBSCAN experiment script.

Loads features from the feature store, normalizes them, runs a grid search
over (eps, min_samples), fits DBSCAN with the best params, then identifies
outliers and saves the PCA visualization.

Usage:
    python -m capitolwatch.analysis.clustering.run_dbscan
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.preprocessing import normalize_features
from capitolwatch.analysis.clustering.dbscan import DBSCANClusterer


def _load_and_normalize(feature_type: str) -> tuple:
    """
    Load a feature matrix from the store and apply StandardScaler.

    DBSCAN relies on a distance metric (eps is a distance threshold),
    so StandardScaler is the appropriate normalizer here: it centers
    each feature at 0 with unit variance, making distances comparable
    across dimensions.

    Args:
        feature_type (str): One of "freq_baseline" or "freq_weighted".

    Returns:
        tuple: (X: np.ndarray, politician_labels: pd.DataFrame)
    """
    politician_labels = load_features("politician_labels")
    matrix = load_features(feature_type)
    matrix_scaled, _ = normalize_features(matrix, StandardScaler())
    return matrix_scaled.to_numpy(), politician_labels


def _run_grid_search(
    clusterer: DBSCANClusterer,
    X: np.ndarray,
    feature_type: str,
) -> tuple:
    """
    Run grid search over (eps, min_samples) and save the heatmap.

    Args:
        clusterer (DBSCANClusterer): Unfitted clusterer instance.
        X (np.ndarray): Normalized feature matrix.
        feature_type (str): Used to name the output PNG.

    Returns:
        tuple: (grid_results: list[dict], best_params: dict)
    """
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    min_samples_values = [3, 5, 7, 10]

    # run cartesian product over (eps, min_samples) and rank by silhouette
    # cosine distance in [0, 1]: 0 = identical direction, 1 = orthogonal
    grid_results = clusterer.grid_search(X, eps_values, min_samples_values)

    # save the silhouette heatmap for inspection
    save_path = f"data/visualizations/dbscan_grid_{feature_type}.png"
    clusterer.plot_grid_search(
        grid_results,
        eps_values=eps_values,
        min_samples_values=min_samples_values,
        save_path=save_path,
    )

    # pick the best (eps, min_samples) respecting the noise-ratio threshold
    best_params = clusterer.find_best_params(grid_results, n_total=len(X))

    return grid_results, best_params


def _fit_best(
    X: np.ndarray,
    best_params: dict,
    feature_type: str,
    politician_labels,
) -> tuple:
    """
    Fit DBSCAN with best (eps, min_samples), extract outliers, save PCA plot.

    Args:
        X (np.ndarray): Normalized feature matrix.
        best_params (dict): Must contain keys "eps" and "min_samples".
        feature_type (str): Used to name the output PNG.
        politician_labels (pd.DataFrame): Politician metadata.

    Returns:
        tuple: (fitted_clusterer: DBSCANClusterer, outliers: list[dict])
    """
    # instantiate a fresh clusterer with the params selected by grid search
    # use cosine metric: measures overlap between investment profiles
    clusterer = DBSCANClusterer(
        eps=best_params["eps"],
        min_samples=best_params["min_samples"],
        metric="cosine",
    )

    # fit the model: assigns a label to every politician
    clusterer.fit(X)

    # save a 2-D PCA projection with cluster colors and outlier annotations
    clusterer.plot_clusters_pca(
        X,
        politician_labels=politician_labels,
        save_path=f"data/visualizations/dbscan_pca_{feature_type}.png",
    )

    # collect politicians assigned noise label -1
    outliers = clusterer.get_outliers(politician_labels)

    return clusterer, outliers


def run_dbscan_experiment(feature_type: str) -> dict:
    """
    Run the full DBSCAN experiment for one feature type.

    Steps:
    1. Load and normalize features.
    2. Grid search over (eps, min_samples).
    3. Fit DBSCAN with best params.
    4. Extract and return outliers + cluster sizes.

    Args:
        feature_type (str): One of "freq_baseline" or "freq_weighted".

    Returns:
        dict: {
            "feature_type": str,
            "best_params": dict,
            "best_silhouette": float or None,
            "n_clusters": int,
            "n_outliers": int,
            "cluster_sizes": dict,
            "outliers": list[dict],
            "grid_results": list[dict],
        }
    """
    X, politician_labels = _load_and_normalize(feature_type)

    temp_clusterer = DBSCANClusterer(metric="cosine")
    grid_results, best_params = _run_grid_search(
        temp_clusterer, X, feature_type
    )
    clusterer, outliers = _fit_best(
        X, best_params, feature_type, politician_labels
    )

    # count members per cluster, excluding noise label -1
    unique, counts = np.unique(clusterer.labels_, return_counts=True)
    cluster_sizes = {
        int(k): int(v) for k, v in zip(unique, counts) if k != -1
    }

    return {
        "feature_type": feature_type,
        "best_params": best_params,
        "best_silhouette": best_params.get("silhouette"),
        "n_clusters": clusterer.n_clusters_ if clusterer else 0,
        "n_outliers": clusterer.n_outliers_ if clusterer else 0,
        "cluster_sizes": cluster_sizes,
        "outliers": outliers,
        "grid_results": grid_results,
    }


def print_results(results: dict) -> None:
    """
    Print DBSCAN experiment results.

    Args:
        results (dict): Output of run_dbscan_experiment().
    """
    print(f"\n--- DBSCAN on {results['feature_type']} ---")
    print(f"Best params: eps={results['best_params'].get('eps')} | "
          f"min_samples={results['best_params'].get('min_samples')}")
    sil = results["best_silhouette"]
    sil_str = f"{sil:.4f}" if sil is not None else "N/A"
    print(f"Silhouette: {sil_str}")
    print(f"Clusters found: {results['n_clusters']} | "
          f"Outliers: {results['n_outliers']}")
    print(f"Cluster sizes: {results['cluster_sizes']}")

    if results["outliers"]:
        print("Outlier politicians (label = -1):")
        for o in results["outliers"]:
            print(f"  index={o['index']} | "
                  f"{o['first_name']} {o['last_name']} | "
                  f"party={o['party']}")

    print("\nTop 5 grid search results:")
    for r in results["grid_results"][:5]:
        sil = r["silhouette"]
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        print(f"  eps={r['eps']} | min_samples={r['min_samples']} | "
              f"clusters={r['n_clusters']} | outliers={r['n_outliers']} | "
              f"silhouette={sil_str}")


if __name__ == "__main__":
    for feature_type in ("freq_baseline", "freq_weighted"):
        results = run_dbscan_experiment(feature_type)
        print_results(results)
