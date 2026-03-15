# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Evaluation pipeline for all 6 clustering experiments.

Loads features from the feature store, re-fits each algorithm with its
optimal parameters (no plots, no grid search side-effects), computes the
three internal metrics via evaluation.py, builds a comparison table, and
exports it to CSV.

The 6 experiments:
    - kmeans x freq_baseline
    - kmeans x freq_weighted
    - dbscan x freq_baseline
    - dbscan x freq_weighted
    - som    x freq_baseline
    - som    x freq_weighted

Usage:
    python -m capitolwatch.analysis.run_evaluation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from capitolwatch.analysis.evaluation import (
    build_comparison_table,
    evaluate_clustering,
    export_results,
)
from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.preprocessing import normalize_features


def _load_standard(feature_type: str) -> np.ndarray:
    """
    Load a feature matrix and apply StandardScaler.

    Args:
        feature_type (str): Key in the feature store (e.g. "freq_baseline").

    Returns:
        np.ndarray: Scaled feature matrix of shape (n_samples, n_features).
    """
    matrix = load_features(feature_type)
    scaled, _ = normalize_features(matrix, StandardScaler())
    return scaled.to_numpy()


def _load_minmax(feature_type: str) -> np.ndarray:
    """
    Load a feature matrix and apply MinMaxScaler.

    Args:
        feature_type (str): Key in the feature store (e.g. "freq_baseline").

    Returns:
        np.ndarray: Scaled feature matrix of shape (n_samples, n_features).
    """
    matrix = load_features(feature_type)
    scaled, _ = normalize_features(matrix, MinMaxScaler())
    return scaled.to_numpy()


def _get_kmeans_labels(feature_type: str, k_range: tuple = (2, 15)) -> tuple:
    """
    Fit K-Means with the silhouette-optimal K and return (X, labels).

    Args:
        feature_type (str): Key in the feature store.
        k_range (tuple): (k_min, k_max) inclusive range to test.

    Returns:
        tuple: (X: np.ndarray, labels: np.ndarray)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = _load_standard(feature_type)
    k_min, k_max = k_range

    best_k = k_min
    best_sil = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # silhouette requires at least 2 clusters with >1 sample each
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)
    return X, labels.astype(int)


def _get_dbscan_labels(
    feature_type: str,
    eps_values: list = None,
    min_samples_values: list = None,
    max_noise_ratio: float = 0.20,
) -> tuple:
    """
    Run cosine-DBSCAN grid search and return (X, labels) for the best params.

    Args:
        feature_type (str): Key in the feature store.
        eps_values (list): eps values to test. Defaults to [0.1..0.6].
        min_samples_values (list): min_samples values to test.
        max_noise_ratio (float): Maximum tolerated noise ratio; fallback to
            best silhouette if no config passes the threshold.

    Returns:
        tuple: (X: np.ndarray, labels: np.ndarray)
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    if eps_values is None:
        eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    if min_samples_values is None:
        min_samples_values = [3, 5, 7, 10]

    X = _load_standard(feature_type)
    n_total = len(X)

    candidates = []
    for eps in eps_values:
        for min_s in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_s, metric="cosine")
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = int(np.sum(labels == -1))

            if n_clusters < 2:
                continue

            # silhouette computed on clustered points only
            mask = labels != -1
            if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
                continue

            sil = silhouette_score(X[mask], labels[mask])
            candidates.append(
                {
                    "eps": eps,
                    "min_samples": min_s,
                    "labels": labels,
                    "n_clusters": n_clusters,
                    "n_outliers": n_outliers,
                    "noise_ratio": n_outliers / n_total,
                    "silhouette": sil,
                }
            )

    if not candidates:
        # degenerate case: every config produces 0 or 1 cluster
        return X, np.full(n_total, -1, dtype=int)

    # prefer configs that pass the noise threshold; fall back to best
    # silhouette
    passing = [c for c in candidates if c["noise_ratio"] <= max_noise_ratio]
    pool = passing if passing else candidates
    best = max(pool, key=lambda c: c["silhouette"])

    return X, best["labels"].astype(int)


def _get_som_labels(
    feature_type: str,
    m: int = 7,
    n: int = 7,
    n_clusters: int = 3,
    n_iterations: int = 1000,
) -> tuple:
    """
    Train SOM and extract cluster labels via K-Means on neuron weights.

    Args:
        feature_type (str): Key in the feature store.
        m (int): SOM grid rows.
        n (int): SOM grid columns.
        n_clusters (int): Number of clusters to extract from neuron weights.
        n_iterations (int): Number of SOM training iterations.

    Returns:
        tuple: (X: np.ndarray, labels: np.ndarray)
    """
    from capitolwatch.analysis.clustering.som import SOMClusterer

    X = _load_minmax(feature_type)
    clusterer = SOMClusterer(
        m=m,
        n=n,
        sigma=1.0,
        learning_rate=0.5,
        n_iterations=n_iterations,
        random_seed=42,
    )
    clusterer.fit(X)
    clusterer.extract_clusters(n_clusters=n_clusters)
    return X, clusterer.labels_.astype(int)


def _build_experiment_configs() -> list:
    """
    Return the list of (algo_name, feature_type, loader_fn) triples.

    Returns:
        list: Each element is a dict with keys "algo", "feature_type",
            "loader" (callable returning (X, labels)).
    """
    return [
        {
            "algo": "kmeans",
            "feature_type": "freq_baseline",
            "loader": lambda: _get_kmeans_labels("freq_baseline"),
        },
        {
            "algo": "kmeans",
            "feature_type": "freq_weighted",
            "loader": lambda: _get_kmeans_labels("freq_weighted"),
        },
        {
            "algo": "dbscan",
            "feature_type": "freq_baseline",
            "loader": lambda: _get_dbscan_labels("freq_baseline"),
        },
        {
            "algo": "dbscan",
            "feature_type": "freq_weighted",
            "loader": lambda: _get_dbscan_labels("freq_weighted"),
        },
        {
            "algo": "som",
            "feature_type": "freq_baseline",
            "loader": lambda: _get_som_labels("freq_baseline"),
        },
        {
            "algo": "som",
            "feature_type": "freq_weighted",
            "loader": lambda: _get_som_labels("freq_weighted"),
        },
    ]


def run_all_evaluations(
    output_path: str = "data/visualizations/evaluation_results.csv",
) -> pd.DataFrame:
    """
    Run all 6 experiments, compute internal metrics, and export results.

    For each (algo, feature_type) pair:
      1. Load and normalize feature matrix.
      2. Fit the algorithm with its optimal parameters.
      3. Compute silhouette, Davies-Bouldin, and Calinski-Harabasz.
      4. Collect results into a sorted comparison table.
      5. Export the table to CSV.

    Args:
        output_path (str): Path for the exported CSV file.

    Returns:
        pd.DataFrame: Sorted comparison table (best silhouette first).
    """
    configs = _build_experiment_configs()
    results = []

    for cfg in configs:
        algo = cfg["algo"]
        feature_type = cfg["feature_type"]
        print(f"  Evaluating {algo} / {feature_type} ...", end=" ", flush=True)

        X, labels = cfg["loader"]()
        result = evaluate_clustering(X, labels, algo, feature_type)
        results.append(result)

        sil = result["silhouette"]
        sil_str = f"{sil:.4f}" if not np.isnan(float(sil)) else "nan"
        print(
            f"clusters={result['n_clusters']} | "
            f"outliers={result['n_outliers']} | "
            f"silhouette={sil_str}"
        )

    df = build_comparison_table(results)
    export_results(df, output_path)
    print(f"\nResults exported to: {output_path}")
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print the comparison table with aligned columns.

    Args:
        df (pd.DataFrame): Output of run_all_evaluations() or
            build_comparison_table().
    """
    print("\n=== Internal Metrics — All Experiments ===\n")
    col_fmt = {
        "algo_name": "Algorithm",
        "feature_type": "Feature type",
        "n_clusters": "Clusters",
        "n_outliers": "Outliers",
        "silhouette": "Silhouette",
        "davies_bouldin": "Davies-Bouldin",
        "calinski_harabasz": "Calinski-Harabasz",
    }
    display = df.rename(columns=col_fmt)

    # format float columns to 4 decimal places
    for col in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]:
        display[col] = display[col].apply(
            lambda v: f"{v:.4f}" if not pd.isna(v) else "nan"
        )

    print(display.to_string(index=False))


if __name__ == "__main__":
    print("Running internal metric evaluation on all 6 experiments\n")
    df = run_all_evaluations()
    print_comparison_table(df)
