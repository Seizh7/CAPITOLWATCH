# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from capitolwatch.analysis.clustering.kmeans import KMeansClusterer


def make_matrix(n_samples=60, n_clusters=3, random_state=0):
    """Generate a well-separated blob dataset for deterministic tests."""
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=random_state,
    )
    return X


class TestKMeansClustererInit:

    def test_default_params(self):
        """Default n_clusters=3, random_state=42."""
        c = KMeansClusterer()
        assert c.n_clusters == 3
        assert c.random_state == 42

    def test_custom_params(self):
        """Constructor stores provided hyperparameters."""
        c = KMeansClusterer(n_clusters=5, random_state=7)
        assert c.n_clusters == 5
        assert c.random_state == 7

    def test_not_fitted_initially(self):
        """Model, labels and inertia must be None before fit."""
        c = KMeansClusterer()
        assert c._model is None
        assert c.labels_ is None
        assert c.inertia_ is None


class TestKMeansClustererFit:

    def test_fit_returns_self(self):
        """fit() must return self for method chaining."""
        X = make_matrix()
        c = KMeansClusterer(n_clusters=3)
        result = c.fit(X)
        assert result is c

    def test_labels_shape_after_fit(self):
        """labels_ must have one entry per sample."""
        X = make_matrix(n_samples=60)
        c = KMeansClusterer(n_clusters=3)
        c.fit(X)
        assert c.labels_.shape == (60,)

    def test_labels_values_in_range(self):
        """All labels must be integers in [0, n_clusters - 1]."""
        X = make_matrix()
        c = KMeansClusterer(n_clusters=3)
        c.fit(X)
        assert set(c.labels_).issubset(set(range(3)))

    def test_inertia_positive_after_fit(self):
        """Inertia must be a positive float after fit."""
        X = make_matrix()
        c = KMeansClusterer(n_clusters=3)
        c.fit(X)
        assert isinstance(c.inertia_, float)
        assert c.inertia_ > 0


class TestKMeansClustererPredict:

    def test_predict_returns_array(self):
        """predict() must return a numpy array."""
        X = make_matrix()
        c = KMeansClusterer(n_clusters=3)
        c.fit(X)
        labels = c.predict(X)
        assert isinstance(labels, np.ndarray)

    def test_predict_shape(self):
        """predict() output shape must match number of input samples."""
        X = make_matrix(n_samples=60)
        c = KMeansClusterer(n_clusters=3)
        c.fit(X)
        assert c.predict(X).shape == (60,)

    def test_predict_raises_if_not_fitted(self):
        """predict() must raise RuntimeError before fit."""
        X = make_matrix()
        c = KMeansClusterer()
        with pytest.raises(RuntimeError):
            c.predict(X)


class TestKMeansClustererGetParams:

    def test_get_params_keys(self):
        """get_params() must return n_clusters and random_state keys."""
        c = KMeansClusterer(n_clusters=4, random_state=1)
        params = c.get_params()
        assert "n_clusters" in params
        assert "random_state" in params

    def test_get_params_values(self):
        """get_params() must reflect constructor arguments."""
        c = KMeansClusterer(n_clusters=4, random_state=1)
        params = c.get_params()
        assert params["n_clusters"] == 4
        assert params["random_state"] == 1


class TestFindOptimalK:

    def test_returns_three_lists(self):
        """find_optimal_k() must return a tuple of 3 lists."""
        X = make_matrix()
        c = KMeansClusterer()
        result = c.find_optimal_k(X, k_min=2, k_max=6)
        assert len(result) == 3

    def test_k_values_range(self):
        """k_values must cover [k_min, k_max] inclusive."""
        X = make_matrix()
        c = KMeansClusterer()
        k_values, _, _ = c.find_optimal_k(X, k_min=2, k_max=6)
        assert k_values == list(range(2, 7))

    def test_lists_same_length(self):
        """inertias and silhouette_scores must have same length as k_values."""
        X = make_matrix()
        c = KMeansClusterer()
        k_values, inertias, sil_scores = c.find_optimal_k(X, k_min=2, k_max=6)
        assert len(inertias) == len(k_values)
        assert len(sil_scores) == len(k_values)

    def test_inertias_decreasing(self):
        """Inertia must decrease (or stay equal) as K increases."""
        X = make_matrix()
        c = KMeansClusterer()
        _, inertias, _ = c.find_optimal_k(X, k_min=2, k_max=8)
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_silhouette_scores_in_valid_range(self):
        """Silhouette scores must be in [-1, 1]."""
        X = make_matrix()
        c = KMeansClusterer()
        _, _, sil_scores = c.find_optimal_k(X, k_min=2, k_max=6)
        for score in sil_scores:
            assert -1.0 <= score <= 1.0

    def test_optimal_k_detects_true_clusters(self):
        """On 3-cluster blobs, best silhouette K should be 3."""
        X = make_matrix(n_clusters=3, random_state=42)
        c = KMeansClusterer()
        k_values, _, sil_scores = c.find_optimal_k(X, k_min=2, k_max=8)
        best_k = k_values[sil_scores.index(max(sil_scores))]
        assert best_k == 3
