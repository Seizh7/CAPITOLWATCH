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
    build_confusion_matrix,
    calculate_ari,
    calculate_nmi,
    calculate_silhouette_score,
    calculate_v_measure,
    evaluate_clustering,
    evaluate_clustering_external,
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


class TestEvaluateClustering:

    def test_returns_dict_with_required_keys(self):
        """evaluate_clustering must return all 5 expected keys."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "kmeans", "freq_baseline")
        expected_keys = {
            "algo_name",
            "feature_type",
            "n_clusters",
            "n_outliers",
            "silhouette",
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

    def test_all_outliers_produces_nan_silhouette(self):
        """When all labels are -1, silhouette must be nan."""
        X, _ = make_clean_blobs()
        labels = np.full(len(X), -1, dtype=int)
        result = evaluate_clustering(X, labels, "dbscan", "freq_baseline")
        assert np.isnan(result["silhouette"])

    def test_som_labels_accepted(self):
        # SOM produces integer labels >= 0, must be evaluated like any algo.
        X, labels = make_clean_blobs(n_clusters=4)
        result = evaluate_clustering(X, labels, "som", "freq_baseline")
        assert result["algo_name"] == "som"
        assert result["n_outliers"] == 0
        assert not np.isnan(result["silhouette"])

    def test_silhouette_in_valid_range(self):
        """Silhouette must be in [-1, 1] for a valid clustering."""
        X, labels = make_clean_blobs(n_clusters=3)
        result = evaluate_clustering(X, labels, "kmeans", "freq_baseline")
        assert -1.0 <= result["silhouette"] <= 1.0


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
            },
            {
                "algo_name": "dbscan",
                "feature_type": "freq_weighted",
                "n_clusters": 3,
                "n_outliers": 15,
                "silhouette": 0.62,
            },
            {
                "algo_name": "som",
                "feature_type": "freq_baseline",
                "n_clusters": 3,
                "n_outliers": 0,
                "silhouette": 0.51,
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
        """DataFrame must contain all 5 expected columns."""
        df = build_comparison_table(self._make_results())
        expected_cols = {
            "algo_name",
            "feature_type",
            "n_clusters",
            "n_outliers",
            "silhouette",
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


def make_party_labels(n: int = 79) -> np.ndarray:
    """
    Generate synthetic party labels (0=R, 1=D, 2=I) matching dataset size.

    Returns:
        np.ndarray: Integer array of shape (n,).
    """
    # ~55% R, ~43% D, ~2% I — approximate real distribution
    rng = np.random.default_rng(0)
    labels = rng.choice([0, 1, 2], size=n, p=[0.55, 0.43, 0.02])
    return labels.astype(int)


class TestCalculateAri:

    def test_perfect_agreement_returns_one(self):
        """When pred == true, ARI must be 1.0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        score = calculate_ari(labels, labels.copy())
        assert abs(score - 1.0) < 1e-9

    def test_random_labels_near_zero(self):
        """Two independent random labelings should give ARI near 0."""
        rng = np.random.default_rng(42)
        lt = rng.integers(0, 3, size=200)
        lp = rng.integers(0, 3, size=200)
        score = calculate_ari(lt, lp)
        assert -0.1 < score < 0.2

    def test_returns_float(self):
        """ARI must be a float."""
        lt = np.array([0, 0, 1, 1])
        lp = np.array([0, 0, 1, 1])
        assert isinstance(calculate_ari(lt, lp), float)

    def test_outliers_excluded(self):
        """ARI computed with outliers must equal ARI on filtered data."""
        lt = np.array([0, 1, 0, 1, 0, 1])
        lp = np.array([0, 1, 0, -1, 0, -1])
        mask = lp != -1
        expected = calculate_ari(lt[mask], lp[mask])
        assert abs(calculate_ari(lt, lp) - expected) < 1e-9

    def test_all_outliers_returns_nan(self):
        """All pred == -1 → np.nan."""
        lt = np.array([0, 1, 0])
        lp = np.full(3, -1)
        assert np.isnan(calculate_ari(lt, lp))

    def test_single_cluster_after_filter_returns_nan(self):
        """Only one unique pred label after filtering → np.nan."""
        lt = np.array([0, 1, 0, 1])
        lp = np.array([0, 0, 0, -1])
        assert np.isnan(calculate_ari(lt, lp))

    def test_range_is_valid(self):
        """Score must lie in [-1, 1] for any valid input."""
        lt = make_party_labels()
        lp = np.random.default_rng(7).integers(0, 3, size=79)
        score = calculate_ari(lt, lp)
        assert -1.0 <= score <= 1.0


class TestCalculateNmi:

    def test_perfect_agreement_returns_one(self):
        """When pred == true, NMI must be 1.0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        score = calculate_nmi(labels, labels.copy())
        assert abs(score - 1.0) < 1e-9

    def test_returns_non_negative_float(self):
        """NMI must be >= 0."""
        lt = make_party_labels()
        lp = np.random.default_rng(3).integers(0, 3, size=79)
        score = calculate_nmi(lt, lp)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_outliers_excluded(self):
        """NMI with outliers must equal NMI on filtered data."""
        lt = np.array([0, 1, 0, 1, 0, 1])
        lp = np.array([0, 1, 0, -1, 0, -1])
        mask = lp != -1
        expected = calculate_nmi(lt[mask], lp[mask])
        assert abs(calculate_nmi(lt, lp) - expected) < 1e-9

    def test_all_outliers_returns_nan(self):
        """All pred == -1 → np.nan."""
        lt = np.array([0, 1, 0])
        lp = np.full(3, -1)
        assert np.isnan(calculate_nmi(lt, lp))

    def test_range_is_valid(self):
        """NMI must lie in [0, 1]."""
        lt = make_party_labels()
        lp = np.random.default_rng(9).integers(0, 3, size=79)
        score = calculate_nmi(lt, lp)
        assert 0.0 <= score <= 1.0


class TestCalculateVMeasure:

    def test_returns_dict_with_three_keys(self):
        # calculate_v_measure must return homogeneity, completeness, v_measure.
        labels = np.array([0, 0, 1, 1])
        result = calculate_v_measure(labels, labels.copy())
        assert set(result.keys()) == {
            "homogeneity", "completeness", "v_measure"
        }

    def test_perfect_agreement_all_ones(self):
        """All three values must be 1.0 when pred == true."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = calculate_v_measure(labels, labels.copy())
        assert abs(result["homogeneity"] - 1.0) < 1e-9
        assert abs(result["completeness"] - 1.0) < 1e-9
        assert abs(result["v_measure"] - 1.0) < 1e-9

    def test_all_outliers_returns_nan_dict(self):
        """All pred == -1 → all three values must be np.nan."""
        lt = np.array([0, 1, 0])
        lp = np.full(3, -1)
        result = calculate_v_measure(lt, lp)
        assert np.isnan(result["homogeneity"])
        assert np.isnan(result["completeness"])
        assert np.isnan(result["v_measure"])

    def test_outliers_excluded(self):
        """Result with outliers must match result on filtered data."""
        lt = np.array([0, 1, 0, 1, 0, 1])
        lp = np.array([0, 1, 0, -1, 0, -1])
        mask = lp != -1
        expected = calculate_v_measure(lt[mask], lp[mask])
        result = calculate_v_measure(lt, lp)
        assert abs(result["v_measure"] - expected["v_measure"]) < 1e-9

    def test_values_in_range(self):
        """All three values must be in [0, 1]."""
        lt = make_party_labels()
        lp = np.random.default_rng(5).integers(0, 3, size=79)
        result = calculate_v_measure(lt, lp)
        for key in ("homogeneity", "completeness", "v_measure"):
            assert 0.0 <= result[key] <= 1.0


class TestEvaluateClusteringExternal:

    def _make_inputs(self) -> tuple:
        lt = make_party_labels(79)
        rng = np.random.default_rng(1)
        lp = rng.integers(0, 3, size=79)
        return lt, lp

    def test_returns_dict_with_required_keys(self):
        """evaluate_clustering_external must return all 7 expected keys."""
        lt, lp = self._make_inputs()
        result = evaluate_clustering_external(
            lt, lp, "kmeans", "freq_baseline"
        )
        expected_keys = {
            "algo_name",
            "feature_type",
            "ari",
            "nmi",
            "homogeneity",
            "completeness",
            "v_measure",
        }
        assert set(result.keys()) == expected_keys

    def test_metadata_stored_correctly(self):
        """algo_name and feature_type are stored as provided."""
        lt, lp = self._make_inputs()
        result = evaluate_clustering_external(
            lt, lp, "dbscan", "freq_weighted"
        )
        assert result["algo_name"] == "dbscan"
        assert result["feature_type"] == "freq_weighted"

    def test_scores_are_floats(self):
        """All 5 metric values must be floats."""
        lt, lp = self._make_inputs()
        result = evaluate_clustering_external(lt, lp, "som", "freq_baseline")
        for key in ("ari", "nmi", "homogeneity", "completeness", "v_measure"):
            assert isinstance(result[key], float), f"{key} is not float"

    def test_all_outliers_produces_nan_metrics(self):
        """All pred == -1 → all five metrics must be nan."""
        lt = make_party_labels(79)
        lp = np.full(79, -1, dtype=int)
        result = evaluate_clustering_external(
            lt, lp, "dbscan", "freq_baseline"
        )
        for key in ("ari", "nmi", "homogeneity", "completeness", "v_measure"):
            assert np.isnan(result[key]), f"{key} should be nan"

    def test_perfect_clustering_returns_one(self):
        """When pred == true, all metrics must return 1.0."""
        labels = np.array([0] * 40 + [1] * 30 + [2] * 9)
        result = evaluate_clustering_external(
            labels, labels.copy(), "kmeans", "freq_baseline"
        )
        assert abs(result["ari"] - 1.0) < 1e-9
        assert abs(result["nmi"] - 1.0) < 1e-9
        assert abs(result["v_measure"] - 1.0) < 1e-9


class TestBuildConfusionMatrix:

    def _make_inputs(self) -> tuple:
        """Return (labels_true, labels_pred, party_names)."""
        lt = np.array([0, 0, 1, 1, 0, 1, 2])
        lp = np.array([0, 0, 1, 1, 1, 0, 2])
        names = ["Republican", "Democratic", "Independent"]
        return lt, lp, names

    def test_returns_dataframe(self):
        """build_confusion_matrix must return a DataFrame."""
        lt, lp, names = self._make_inputs()
        matrix = build_confusion_matrix(lt, lp, names)
        assert isinstance(matrix, pd.DataFrame)

    def test_columns_are_party_names(self):
        """Column names must be human-readable party names."""
        lt, lp, names = self._make_inputs()
        matrix = build_confusion_matrix(lt, lp, names)
        for col in matrix.columns:
            assert col in names

    def test_outliers_excluded_from_matrix(self):
        """Points with labels_pred == -1 must not appear in the matrix."""
        lt = np.array([0, 1, 0, 1, 0])
        lp = np.array([0, 1, -1, 0, 1])
        names = ["Republican", "Democratic", "Independent"]
        matrix = build_confusion_matrix(lt, lp, names)
        # total count in matrix must equal number of non-outlier points
        assert matrix.values.sum() == (lp != -1).sum()

    def test_counts_are_correct(self):
        """Cell values must match manual counts."""
        lt = np.array([0, 0, 1, 1])
        lp = np.array([0, 0, 1, 1])
        names = ["Republican", "Democratic", "Independent"]
        matrix = build_confusion_matrix(lt, lp, names)
        assert matrix.loc[0, "Republican"] == 2
        assert matrix.loc[1, "Democratic"] == 2

    def test_missing_party_does_not_crash(self):
        # If a party is absent from the filtered data, no crash should occur.
        # No Independent (2) in the data after filtering
        lt = np.array([0, 0, 1, 1])
        lp = np.array([0, 1, 0, 1])
        names = ["Republican", "Democratic", "Independent"]
        matrix = build_confusion_matrix(lt, lp, names)
        # Independent column must be absent — only present parties are columns
        assert "Independent" not in matrix.columns
        assert "Republican" in matrix.columns
        assert "Democratic" in matrix.columns
