# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)


def calculate_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the mean Silhouette Coefficient for a clustering result.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            Label -1 is treated as noise and excluded.

    Returns:
        float: Mean silhouette score in [-1, 1], or np.nan if fewer than
            2 valid clusters remain after filtering.
    """
    # Filter out outliers (label == -1) using a boolean mask
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    # Guard — return np.nan if fewer than 2 distinct clusters remain
    if len(np.unique(labels_filtered)) < 2:
        return np.nan

    # Call sklearn silhouette_score and return the result
    return silhouette_score(X_filtered, labels_filtered)


def calculate_davies_bouldin_index(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the Davies-Bouldin Index for a clustering result.

    Lower is better (0 = perfect separation). Outlier points (label == -1)
    are excluded before computation.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            Label -1 is treated as noise and excluded.

    Returns:
        float: Davies-Bouldin index >= 0, or np.nan if fewer than
            2 valid clusters remain after filtering.
    """
    # Filter out outliers using the same mask pattern as above
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    # Guard — return np.nan if fewer than 2 distinct clusters remain
    if len(np.unique(labels_filtered)) < 2:
        return np.nan

    # Call sklearn davies_bouldin_score and return the result
    return davies_bouldin_score(X_filtered, labels_filtered)


def calculate_calinski_harabasz_index(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the Calinski-Harabasz Index for a clustering result.

    Higher is better (no upper bound). Outlier points (label == -1)
    are excluded before computation.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
            Label -1 is treated as noise and excluded.

    Returns:
        float: Calinski-Harabasz index > 0, or np.nan if fewer than
            2 valid clusters remain after filtering.
    """
    # Filter out outliers
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    # Guard — return np.nan if fewer than 2 distinct clusters remain
    if len(np.unique(labels_filtered)) < 2:
        return np.nan

    # Call sklearn calinski_harabasz_score and return the result
    return calinski_harabasz_score(X_filtered, labels_filtered)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    algo_name: str,
    feature_type: str,
) -> dict:
    """
    Compute all three internal metrics for one clustering experiment.

    Outliers (label == -1, produced by DBSCAN) are excluded from metric
    calculations but their count is reported in the result dict so that
    the caller can assess the coverage of the clustering.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
        algo_name (str): Name of the algorithm (e.g., "kmeans", "dbscan",
            "som").
        feature_type (str): Name of the feature set used (e.g.,
            "freq_baseline", "freq_weighted").

    Returns:
        dict: Keys — algo_name, feature_type, n_clusters, n_outliers,
            silhouette, davies_bouldin, calinski_harabasz.
    """
    # Count the number of outliers (labels == -1)
    n_outliers = int(np.sum(labels == -1))

    # Count valid clusters (unique labels excluding -1)
    n_clusters = int(len(np.unique(labels[labels != -1])))

    # Call the three metric functions defined above
    silhouette = calculate_silhouette_score(X, labels)
    davies_bouldin = calculate_davies_bouldin_index(X, labels)
    calinski_harabasz = calculate_calinski_harabasz_index(X, labels)

    return {
        "algo_name": algo_name,
        "feature_type": feature_type,
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }


def build_comparison_table(results: list) -> pd.DataFrame:
    """
    Assemble a list of evaluation dicts into a sorted comparison DataFrame.

    The table is sorted by silhouette score (descending) so the best
    experiment appears first.

    Args:
        results (list): List of dicts as returned by evaluate_clustering().

    Returns:
        pd.DataFrame: One row per experiment, columns matching the keys
            of the input dicts.
    """
    # Create a DataFrame from the list of dicts
    df = pd.DataFrame(results)

    # Sort by silhouette descending, reset the index
    df = df.sort_values(
        by="silhouette", ascending=False
    ).reset_index(drop=True)

    return df


def export_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the comparison DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Comparison table as returned by
            build_comparison_table().
        output_path (str): Absolute or relative path for the CSV file.

    Returns:
        None
    """
    # Export df to CSV without the DataFrame index
    df.to_csv(output_path, index=False)


def calculate_ari(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    """
    Compute the Adjusted Rand Index between true and predicted labels.

    The ARI is corrected for chance: 0.0 means random assignment,
    1.0 means perfect agreement.

    Args:
        labels_true (np.ndarray): Ground-truth class labels (e.g., party
            encoded as integers). Shape (n_samples,).
        labels_pred (np.ndarray): Cluster labels produced by the algorithm.
            Shape (n_samples,). Label -1 (noise) is excluded before scoring.

    Returns:
        float: ARI score in [-1, 1], or np.nan if fewer than 2 unique
            predicted labels remain after filtering.
    """
    # Create a boolean mask that keeps only non-outlier points
    mask = labels_pred != -1

    # Apply the mask to both arrays
    lt_filtered = labels_true[mask]
    lp_filtered = labels_pred[mask]

    # Guard — return np.nan if fewer than 2 unique labels in lp_filtered
    if len(np.unique(lp_filtered)) < 2:
        return np.nan

    return adjusted_rand_score(lt_filtered, lp_filtered)


def calculate_nmi(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    """
    Compute the Normalized Mutual Information between true and predicted
    labels.

    Args:
        labels_true (np.ndarray): Ground-truth labels. Shape (n_samples,).
        labels_pred (np.ndarray): Cluster labels. Shape (n_samples,).
            Label -1 (noise) is excluded before scoring.

    Returns:
        float: NMI score in [0, 1], or np.nan if fewer than 2 unique
            predicted labels remain after filtering.
    """
    # Same mask + guard pattern as calculate_ari
    mask = labels_pred != -1
    lt_filtered = labels_true[mask]
    lp_filtered = labels_pred[mask]

    if len(np.unique(lp_filtered)) < 2:
        return np.nan

    # average_method="arithmetic" is the sklearn default but explicit is better
    return normalized_mutual_info_score(
        lt_filtered, lp_filtered, average_method="arithmetic"
    )


def calculate_v_measure(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> dict:
    """
    Compute homogeneity, completeness, and V-Measure.

    Args:
        labels_true (np.ndarray): Ground-truth labels. Shape (n_samples,).
        labels_pred (np.ndarray): Cluster labels. Shape (n_samples,).
            Label -1 (noise) is excluded before scoring.

    Returns:
        dict: Keys — homogeneity, completeness, v_measure.
            Values float in [0, 1], or np.nan if guard triggers.
    """
    # Same mask + guard pattern as calculate_ari
    mask = labels_pred != -1
    lt_filtered = labels_true[mask]
    lp_filtered = labels_pred[mask]

    if len(np.unique(lp_filtered)) < 2:
        return {
            "homogeneity": np.nan, "completeness": np.nan, "v_measure": np.nan
        }

    # homogeneity_completeness_v_measure returns a tuple (h, c, v)
    h, c, v = homogeneity_completeness_v_measure(lt_filtered, lp_filtered)
    return {"homogeneity": h, "completeness": c, "v_measure": v}


def evaluate_clustering_external(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    algo_name: str,
    feature_type: str,
) -> dict:
    """
    Compute all external metrics for one clustering experiment.

    Args:
        labels_true (np.ndarray): Integer-encoded party labels (R=0,D=1,I=2).
        labels_pred (np.ndarray): Cluster labels from the algorithm.
        algo_name (str): Algorithm name (e.g., "kmeans").
        feature_type (str): Feature set name (e.g., "freq_baseline").

    Returns:
        dict: Keys — algo_name, feature_type, ari, nmi, homogeneity,
            completeness, v_measure.
    """
    ari = calculate_ari(labels_true, labels_pred)
    nmi = calculate_nmi(labels_true, labels_pred)

    # Call calculate_v_measure and unpack the result dict into local variables
    vm = calculate_v_measure(labels_true, labels_pred)
    # Build and return the result dict with all 7 keys
    return {
        "algo_name": algo_name,
        "feature_type": feature_type,
        "ari": ari,
        "nmi": nmi,
        "homogeneity": vm["homogeneity"],
        "completeness": vm["completeness"],
        "v_measure": vm["v_measure"],
    }


def build_confusion_matrix(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    party_names: list,
) -> pd.DataFrame:
    """
    Build a cluster x party confusion matrix.

    Noise points (labels_pred == -1) are excluded.

    Args:
        labels_true (np.ndarray): Integer-encoded party labels.
        labels_pred (np.ndarray): Cluster labels.
        party_names (list): Human-readable party names ordered by integer
            encoding (e.g., ["Republican", "Democratic", "Independent"]).

    Returns:
        pd.DataFrame: Rows = cluster ids, columns = party names.
    """
    # Apply mask to remove outliers from both arrays
    mask = labels_pred != -1
    labels_true_filtered = labels_true[mask]
    labels_pred_filtered = labels_pred[mask]

    # pd.crosstab counts co-occurrences between two arrays
    matrix = pd.crosstab(labels_pred_filtered, labels_true_filtered)

    # Map each integer code that actually appears to its party name
    party_map = {i: name for i, name in enumerate(party_names)}
    matrix.columns = [party_map[col] for col in matrix.columns]

    return matrix


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans, DBSCAN

    # Generate synthetic data
    X, _ = make_blobs(
        n_samples=300, centers=4, cluster_std=0.60, random_state=0
    )

    # Run KMeans
    kmeans_labels = KMeans(n_clusters=4).fit_predict(X)
    kmeans_results = evaluate_clustering(
        X, kmeans_labels, "kmeans", "synthetic"
    )

    # Run DBSCAN
    dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    dbscan_results = evaluate_clustering(
        X, dbscan_labels, "dbscan", "synthetic"
    )

    # Run SOM (placeholder - replace with actual SOM implementation)
    som_labels = np.random.randint(0, 4, size=X.shape[0])
    som_results = evaluate_clustering(X, som_labels, "som", "synthetic")

    # Build comparison table
    comparison_df = build_comparison_table([
        kmeans_results, dbscan_results, som_results
    ])
    print(comparison_df)
