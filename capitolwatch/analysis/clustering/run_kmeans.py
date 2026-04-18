# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
K-Means experiment script.

Loads features from the feature store, normalizes them, runs K-Means
with elbow + silhouette analysis.

Usage:
    python -m capitolwatch.analysis.clustering.run_kmeans
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.preprocessing import normalize_features
from capitolwatch.analysis.clustering.kmeans import KMeansClusterer


def _load_and_normalize(feature_type: str) -> tuple:
    """
    Load a feature matrix from the store and apply StandardScaler.

    Args:
        feature_type (str): One of "freq_baseline" or "freq_weighted".

    Returns:
        tuple: (X: np.ndarray, politician_labels: pd.DataFrame)
    """
    politician_labels = load_features("politician_labels")
    matrix = load_features(feature_type)
    matrix_scaled, _ = normalize_features(matrix, StandardScaler())
    return matrix_scaled.to_numpy(), politician_labels


def _find_and_plot_optimal_k(
    clusterer: KMeansClusterer,
    X: np.ndarray,
    feature_type: str,
) -> tuple:
    """
    Run elbow + silhouette analysis and save plots to disk.

    Args:
        clusterer (KMeansClusterer): Unfitted clusterer instance.
        X (np.ndarray): Normalized feature matrix.
        feature_type (str): Used to name the output PNG files.

    Returns:
        tuple: (best_k: int, best_silhouette: float)
    """
    k_values, inertias, sil_scores = clusterer.find_optimal_k(X)
    clusterer.plot_elbow(
        k_values, inertias,
        save_path=f"data/visualizations/kmeans_elbow_{feature_type}.png",
    )
    clusterer.plot_silhouette(
        k_values, sil_scores,
        save_path=f"data/visualizations/kmeans_silhouette_{feature_type}.png",
    )
    best_k = k_values[sil_scores.index(max(sil_scores))]
    return best_k, round(max(sil_scores), 4)


def _fit_and_analyze(X: np.ndarray, best_k: int, politician_labels) -> tuple:
    """
    Fit K-Means with best_k and extract cluster sizes and singleton outliers.

    Args:
        X (np.ndarray): Normalized feature matrix.
        best_k (int): Number of clusters to use.
        politician_labels (pd.DataFrame): Labels with first_name, last_name,
            party.

    Returns:
        tuple: (cluster_sizes: dict, outliers: list[dict])
    """
    clusterer = KMeansClusterer(n_clusters=best_k)
    clusterer.fit(X)
    unique, counts = np.unique(clusterer.labels_, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    outliers = []
    for i, lbl in enumerate(clusterer.labels_):
        if counts[lbl] == 1:
            row = politician_labels.iloc[i]
            outliers.append({
                "index": i,
                "first_name": row.get("first_name", ""),
                "last_name": row.get("last_name", ""),
                "party": row.get("party", "N/A"),
            })
    return cluster_sizes, outliers


def _forced_k_investigation(
        X: np.ndarray, k_min: int = 3, k_max: int = 5
) -> list:
    """
    Fit K-Means for each K in [k_min, k_max] and record silhouette + sizes.

    Args:
        X (np.ndarray): Normalized feature matrix.
        k_min (int): First forced K to test.
        k_max (int): Last forced K to test (inclusive).

    Returns:
        list[dict]: [{"k": int, "silhouette": float, "sizes": dict}, ...]
    """
    results = []
    for k in range(k_min, k_max + 1):
        c = KMeansClusterer(n_clusters=k)
        c.fit(X)
        sil = silhouette_score(X, c.labels_)
        u, ct = np.unique(c.labels_, return_counts=True)
        results.append({
            "k": k,
            "silhouette": round(sil, 4),
            "sizes": dict(zip(u.tolist(), ct.tolist())),
        })
    return results


def run_kmeans_experiment(feature_type: str) -> dict:
    """
    Run the full K-Means experiment for one feature type.

    Args:
        feature_type (str): One of "freq_baseline" or "freq_weighted".

    Returns:
        dict: {
            "feature_type": str,
            "best_k": int,
            "best_silhouette": float,
            "cluster_sizes": dict,
            "outliers": list[dict],
            "forced_k_results": list[dict],
        }
    """
    X, politician_labels = _load_and_normalize(feature_type)
    best_k, best_silhouette = _find_and_plot_optimal_k(
        KMeansClusterer(), X, feature_type
    )
    cluster_sizes, outliers = _fit_and_analyze(X, best_k, politician_labels)
    forced_k_results = _forced_k_investigation(X)

    return {
        "feature_type": feature_type,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "cluster_sizes": cluster_sizes,
        "outliers": outliers,
        "forced_k_results": forced_k_results,
    }


def print_results(results: dict) -> None:
    """
    Print experiment result.

    Args:
        results (dict): Output of run_kmeans_experiment().
    """
    print(f"\n--- K-Means on {results['feature_type']} ---")
    print(f"Optimal K (auto): {results['best_k']} "
          f"(silhouette={results['best_silhouette']:.4f})")
    print(f"Cluster sizes: {results['cluster_sizes']}")

    if results["outliers"]:
        print("Singleton clusters (potential outliers):")
        for o in results["outliers"]:
            print(f"  index={o['index']} | "
                  f"{o['first_name']} {o['last_name']} | "
                  f"party={o['party']}")

    print("  Forced K investigation (K=3 to 5):")
    for r in results["forced_k_results"]:
        print(f"  K={r['k']} | silhouette={r['silhouette']:.4f} "
              f"| cluster sizes={r['sizes']}")


if __name__ == "__main__":
    for feature_type in ("freq_baseline", "freq_weighted"):
        results = run_kmeans_experiment(feature_type)
        print_results(results)
