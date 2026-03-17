# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Run cluster analysis for all 6 experiments and generate Markdown reports.

Uses the same parameters as the evaluation step, so results are consistent.
Saves one report per experiment to data/visualizations/cluster_profiles/.

Usage:
    python -m capitolwatch.analysis.run_cluster_analysis
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from capitolwatch.analysis.cluster_analysis import run_analysis
from capitolwatch.analysis.data_loader import (
    load_assets_with_products,
    load_politicians,
)
from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.preprocessing import normalize_features


def _load_standard(feature_type: str) -> np.ndarray:
    """
    Load and scale features with StandardScaler (for K-Means and DBSCAN).

    Args:
        feature_type (str): Which features to load.

    Returns:
        np.ndarray: Scaled feature matrix.
    """
    matrix = load_features(feature_type)
    scaled, _ = normalize_features(matrix, StandardScaler())
    return scaled.to_numpy()


def _load_minmax(feature_type: str) -> np.ndarray:
    """
    Load and scale features to [0, 1] (for SOM).

    Args:
        feature_type (str): Which features to load.

    Returns:
        np.ndarray: Scaled feature matrix.
    """
    matrix = load_features(feature_type)
    scaled, _ = normalize_features(matrix, MinMaxScaler())
    return scaled.to_numpy()


def _get_kmeans_labels(feature_type: str, k_range: tuple = (2, 15)) -> tuple:
    """
    Run K-Means and find the best K using silhouette score.

    Args:
        feature_type (str): Which features to use.
        k_range (tuple): Range of K values to test (min, max).

    Returns:
        tuple: (feature_matrix, cluster_labels)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = _load_standard(feature_type)
    k_min, k_max = k_range

    best_k = k_min
    best_sil = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X)
        if len(np.unique(labs)) < 2:
            continue
        sil = silhouette_score(X, labs)
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
    Run DBSCAN with grid search to find good parameters.

    Tests different eps and min_samples values, keeping results with
    reasonable noise levels.

    Args:
        feature_type (str): Which features to use.
        eps_values (list): Values of eps to test (default: 0.1 to 0.6).
        min_samples_values (list): Values of min_samples to test.
        max_noise_ratio (float): Maximum fraction of outliers to accept (20%).

    Returns:
        tuple: (feature_matrix, cluster_labels)
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
            labs = db.fit_predict(X)
            n_clusters = len(set(labs)) - (1 if -1 in labs else 0)
            n_outliers = int(np.sum(labs == -1))

            if n_clusters < 2:
                continue

            mask = labs != -1
            if mask.sum() < 2 or len(np.unique(labs[mask])) < 2:
                continue

            sil = silhouette_score(X[mask], labs[mask])
            candidates.append({
                "eps": eps,
                "min_samples": min_s,
                "labels": labs,
                "noise_ratio": n_outliers / n_total,
                "silhouette": sil,
            })

    if not candidates:
        return X, np.full(n_total, -1, dtype=int)

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
    Train a Self-Organizing Map and extract cluster labels.

    Args:
        feature_type (str): Which features to use.
        m (int): Grid height (default: 7).
        n (int): Grid width (default: 7).
        n_clusters (int): How many clusters to create (default: 3).
        n_iterations (int): Training iterations (default: 1000).

    Returns:
        tuple: (feature_matrix, cluster_labels)
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


def run_all_analyses(
    output_dir: str = "data/visualizations/cluster_profiles",
) -> dict:
    """
    Analyze clusters for all 6 experiments.

    Loads data once, then runs K-Means, DBSCAN, and SOM (each with two
    feature types), generating a report for each.

    Args:
        output_dir (str): Where to save the reports (default:
            data/visualizations/cluster_profiles/).

    Returns:
        dict: Results keyed by experiment name, e.g. "kmeans/freq_baseline".
    """
    from pathlib import Path

    out_path = Path(output_dir)

    # Load the raw data once
    print("Loading raw data from database")
    politicians_df = load_politicians()
    assets_df = load_assets_with_products()
    print(f"  Politicians: {len(politicians_df)} | Assets: {len(assets_df)}")

    experiments = [
        ("kmeans", "freq_baseline", _get_kmeans_labels),
        ("kmeans", "freq_weighted", _get_kmeans_labels),
        ("dbscan", "freq_baseline", _get_dbscan_labels),
        ("dbscan", "freq_weighted", _get_dbscan_labels),
        ("som", "freq_baseline", _get_som_labels),
        ("som", "freq_weighted", _get_som_labels),
    ]

    all_profiles = {}

    for algo_name, feature_type, loader_fn in experiments:
        _, labels = loader_fn(feature_type)

        profiles = run_analysis(
            labels=labels,
            politicians_df=politicians_df,
            assets_df=assets_df,
            algo_name=algo_name,
            feature_type=feature_type,
            output_dir=out_path,
        )
        all_profiles[f"{algo_name}/{feature_type}"] = profiles

    print(f"\nAll reports saved to: {out_path.resolve()}")
    return all_profiles


if __name__ == "__main__":
    run_all_analyses()
