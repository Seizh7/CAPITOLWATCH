# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from capitolwatch.analysis.evaluation import (
    build_comparison_table,
    calculate_calinski_harabasz_index,
    calculate_davies_bouldin_index,
    calculate_silhouette_score,
    evaluate_clustering,
    export_results,
)


def make_clean_blobs(n_samples: int = 100, n_clusters: int = 3) -> tuple:
    """
    Generate well-separated blobs with no outliers.

    Returns:
        tuple: (X np.ndarray, labels np.ndarray)
    """
    X, labels = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42,
    )
    return X, labels.astype(int)


def make_blobs_with_outliers(
    n_samples: int = 90,
    n_clusters: int = 3,
    n_outliers: int = 10,
) -> tuple:
    """
    Generate blobs where some labels are replaced by -1 to simulate DBSCAN
    noise points.

    Returns:
        tuple: (X np.ndarray, labels np.ndarray with some -1)
    """
    X, labels = make_blobs(
        n_samples=n_samples + n_outliers,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=0,
    )
    labels = labels.astype(int)
    # mark the last n_outliers samples as noise
    labels[-n_outliers:] = -1
    return X, labels


class TestCalculateSilhouetteScore:

    def test_valid_clusters_returns_float(self):
        """Normal case: 3 clusters, no outliers → float in [-1, 1]."""
        X, labels = make_clean_blobs(n_clusters=3)
        score = calculate_silhouette_score(X, labels)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_well_separated_clusters_high_score(self):
        """Tight, well-separated blobs should produce silhouette > 0.7."""
        X, labels = make_clean_blobs(n_clusters=3)
        score = calculate_silhouette_score(X, labels)
        assert score > 0.7

    def test_outliers_are_excluded(self):
        """Score with outliers must equal score computed on filtered data."""
        X, labels = make_blobs_with_outliers(n_outliers=10)
        score_with_outliers = calculate_silhouette_score(X, labels)

        mask = labels != -1
        score_filtered = calculate_silhouette_score(X[mask], labels[mask])
        assert abs(score_with_outliers - score_filtered) < 1e-9

    def test_single_cluster_returns_nan(self):
        """A single cluster after filtering cannot produce a silhouette."""
        X, _ = make_clean_blobs(n_clusters=3)
        # all points in cluster 0, rest are outliers
        labels = np.zeros(len(X), dtype=int)
        labels[-5:] = -1
        # only label 0 remains → 1 unique cluster
        result = calculate_silhouette_score(X, labels)
        assert np.isnan(result)

    def test_all_outliers_returns_nan(self):
        """If all points are noise, no cluster exists."""
        X, _ = make_clean_blobs()
        labels = np.full(len(X), -1, dtype=int)
        result = calculate_silhouette_score(X, labels)
        assert np.isnan(result)

    def test_no_outliers_match_sklearn(self):
        """Result must match sklearn silhouette_score directly."""
        from sklearn.metrics import silhouette_score

        X, labels = make_clean_blobs(n_clusters=3)
        expected = silhouette_score(X, labels)
        assert abs(calculate_silhouette_score(X, labels) - expected) < 1e-9


class TestCalculateDaviesBouldinIndex:

    def test_valid_clusters_returns_non_negative_float(self):
        """DBI must be >= 0 for valid clustering."""
        X, labels = make_clean_blobs(n_clusters=3)
        score = calculate_davies_bouldin_index(X, labels)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_well_separated_clusters_low_score(self):
        """Tight, well-separated blobs should produce DBI < 0.5."""
        X, labels = make_clean_blobs(n_clusters=3)
        score = calculate_davies_bouldin_index(X, labels)
        assert score < 0.5

    def test_outliers_excluded(self):
        """DBI with outliers must equal DBI on filtered data."""
        X, labels = make_blobs_with_outliers(n_outliers=10)
        score_with_outliers = calculate_davies_bouldin_index(X, labels)

        mask = labels != -1
        score_filtered = calculate_davies_bouldin_index(
            X[mask], labels[mask]
        )
        assert abs(score_with_outliers - score_filtered) < 1e-9

    def test_single_cluster_returns_nan(self):
        """Single cluster after filtering → nan."""
        X, _ = make_clean_blobs()
        labels = np.zeros(len(X), dtype=int)
        result = calculate_davies_bouldin_index(X, labels)
        assert np.isnan(result)

    def test_all_outliers_returns_nan(self):
        """All noise → nan."""
        X, _ = make_clean_blobs()
        labels = np.full(len(X), -1, dtype=int)
        result = calculate_davies_bouldin_index(X, labels)
        assert np.isnan(result)


class TestCalculateCalinskiHarabaszIndex:

    def test_valid_clusters_returns_positive_float(self):
        """CHI must be > 0 for valid clustering."""
        X, labels = make_clean_blobs(n_clusters=3)
        score = calculate_calinski_harabasz_index(X, labels)
        assert isinstance(score, float)
        assert score > 0.0

    def test_outliers_excluded(self):
        """CHI with outliers must equal CHI on filtered data."""
        X, labels = make_blobs_with_outliers(n_outliers=10)
        score_with_outliers = calculate_calinski_harabasz_index(X, labels)

        mask = labels != -1
        score_filtered = calculate_calinski_harabasz_index(
            X[mask], labels[mask]
        )
        assert abs(score_with_outliers - score_filtered) < 1e-9

    def test_single_cluster_returns_nan(self):
        """Single cluster after filtering → nan."""
        X, _ = make_clean_blobs()
        labels = np.zeros(len(X), dtype=int)
        result = calculate_calinski_harabasz_index(X, labels)
        assert np.isnan(result)

    def test_all_outliers_returns_nan(self):
        """All noise → nan."""
        X, _ = make_clean_blobs()
        labels = np.full(len(X), -1, dtype=int)
        result = calculate_calinski_harabasz_index(X, labels)
        assert np.isnan(result)


class TestEvaluateClustering:

    def test_returns_dict_with_required_keys(self):
        """evaluate_clustering must return all 7 expected keys."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "kmeans", "freq_baseline")
        expected_keys = {
            "algo_name",
            "feature_type",
            "n_clusters",
            "n_outliers",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        }
        assert set(result.keys()) == expected_keys

    def test_metadata_stored_correctly(self):
        """algo_name and feature_type are stored as provided."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "dbscan", "freq_weighted")
        assert result["algo_name"] == "dbscan"
        assert result["feature_type"] == "freq_weighted"

    def test_no_outliers_counted_for_kmeans(self):
        """K-Means produces no label -1 → n_outliers must be 0."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "kmeans", "freq_baseline")
        assert result["n_outliers"] == 0

    def test_outliers_counted_correctly(self):
        """n_outliers must match the number of -1 labels."""
        X, labels = make_blobs_with_outliers(n_outliers=10)
        result = evaluate_clustering(X, labels, "dbscan", "freq_weighted")
        assert result["n_outliers"] == 10

    def test_n_clusters_excludes_outliers(self):
        """n_clusters must not count label -1 as a cluster."""
        X, labels = make_blobs_with_outliers(n_clusters=3, n_outliers=10)
        result = evaluate_clustering(X, labels, "dbscan", "freq_baseline")
        assert result["n_clusters"] == 3

    def test_n_outliers_is_python_int(self):
        """n_outliers must be a Python int, not numpy.int64."""
        X, labels = make_blobs_with_outliers(n_outliers=5)
        result = evaluate_clustering(X, labels, "dbscan", "freq_baseline")
        assert isinstance(result["n_outliers"], int)

    def test_n_clusters_is_python_int(self):
        """n_clusters must be a Python int, not numpy.int64."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "kmeans", "freq_baseline")
        assert isinstance(result["n_clusters"], int)

    def test_all_outliers_produces_nan_metrics(self):
        """When all labels are -1, all three metrics must be nan."""
        X, _ = make_clean_blobs()
        labels = np.full(len(X), -1, dtype=int)
        result = evaluate_clustering(X, labels, "dbscan", "freq_baseline")
        assert np.isnan(result["silhouette"])
        assert np.isnan(result["davies_bouldin"])
        assert np.isnan(result["calinski_harabasz"])

    def test_som_labels_accepted(self):
        # SOM produces integer labels >= 0, must be evaluated like any algo.
        X, labels = make_clean_blobs(n_clusters=4)
        result = evaluate_clustering(X, labels, "som", "freq_baseline")
        assert result["algo_name"] == "som"
        assert result["n_outliers"] == 0
        assert not np.isnan(result["silhouette"])

    def test_metrics_have_expected_sign(self):
        """Silhouette in [-1,1], DBI >= 0, CHI > 0 for valid clustering."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "kmeans", "freq_baseline")
        assert -1.0 <= result["silhouette"] <= 1.0
        assert result["davies_bouldin"] >= 0.0
        assert result["calinski_harabasz"] > 0.0


class TestBuildComparisonTable:

    def _make_results(self) -> list:
        """Produce 3 fake evaluation dicts with distinct silhouette scores."""
        return [
            {
                "algo_name": "kmeans",
                "feature_type": "freq_baseline",
                "n_clusters": 3,
                "n_outliers": 0,
                "silhouette": 0.45,
                "davies_bouldin": 0.80,
                "calinski_harabasz": 120.0,
            },
            {
                "algo_name": "dbscan",
                "feature_type": "freq_weighted",
                "n_clusters": 3,
                "n_outliers": 15,
                "silhouette": 0.62,
                "davies_bouldin": 0.55,
                "calinski_harabasz": 200.0,
            },
            {
                "algo_name": "som",
                "feature_type": "freq_baseline",
                "n_clusters": 3,
                "n_outliers": 0,
                "silhouette": 0.51,
                "davies_bouldin": 0.70,
                "calinski_harabasz": 150.0,
            },
        ]

    def test_returns_dataframe(self):
        """build_comparison_table must return a DataFrame."""
        df = build_comparison_table(self._make_results())
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self):
        """One row per experiment."""
        results = self._make_results()
        df = build_comparison_table(results)
        assert len(df) == len(results)

    def test_sorted_by_silhouette_descending(self):
        """First row must have the highest silhouette score."""
        df = build_comparison_table(self._make_results())
        scores = df["silhouette"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_index_is_reset(self):
        """Index must start at 0 after sort."""
        df = build_comparison_table(self._make_results())
        assert list(df.index) == list(range(len(df)))

    def test_all_columns_present(self):
        """DataFrame must contain all 7 expected columns."""
        df = build_comparison_table(self._make_results())
        expected_cols = {
            "algo_name",
            "feature_type",
            "n_clusters",
            "n_outliers",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_nan_silhouette_sorted_last(self):
        # Experiments with nan silhouette (all-outlier case) sort to bottom.
        results = self._make_results()
        results.append(
            {
                "algo_name": "dbscan",
                "feature_type": "freq_baseline",
                "n_clusters": 0,
                "n_outliers": 100,
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
            }
        )
        df = build_comparison_table(results)
        # NaN sorts last with na_position default in pandas
        assert np.isnan(df.iloc[-1]["silhouette"])


class TestExportResults:

    def test_creates_csv_file(self):
        """export_results must create a file at the given path."""
        df = pd.DataFrame(
            [{"algo_name": "kmeans", "silhouette": 0.5}]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            export_results(df, path)
            assert os.path.exists(path)

    def test_csv_has_correct_columns(self):
        """CSV headers must match DataFrame columns."""
        df = pd.DataFrame(
            [{"algo_name": "kmeans", "silhouette": 0.5, "n_clusters": 3}]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            export_results(df, path)
            loaded = pd.read_csv(path)
            assert list(loaded.columns) == list(df.columns)

    def test_csv_row_count_matches_dataframe(self):
        """CSV must contain the same number of rows as the DataFrame."""
        df = pd.DataFrame(
            [
                {"algo_name": "kmeans", "silhouette": 0.5},
                {"algo_name": "dbscan", "silhouette": 0.6},
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            export_results(df, path)
            loaded = pd.read_csv(path)
            assert len(loaded) == len(df)

    def test_no_index_column_in_csv(self):
        """The DataFrame index must not appear as a column in the CSV."""
        df = pd.DataFrame([{"algo_name": "som", "silhouette": 0.55}])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            export_results(df, path)
            loaded = pd.read_csv(path)
            assert "Unnamed: 0" not in loaded.columns
