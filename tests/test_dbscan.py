# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from capitolwatch.analysis.clustering.dbscan import DBSCANClusterer


def make_matrix_with_outlier(n_samples=60, n_clusters=3, random_state=0):
    """
    Generate a well-separated blob dataset with one explicit outlier.

    The outlier is placed far from the blobs to mimic Rick Scott's position
    in the real feature space.
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=random_state,
    )
    outlier = np.array([[20.0, 20.0]])
    return np.vstack([X, outlier])


def make_politician_labels(n=61):
    """Build a minimal politician_labels DataFrame."""
    rows = [
        {
            "first_name": f"Pol{i}",
            "last_name": f"Name{i}",
            "party": "R" if i % 2 == 0 else "D",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestDBSCANClustererInit:

    def test_default_params(self):
        """Default eps=0.5, min_samples=5."""
        c = DBSCANClusterer()
        assert c.eps == 0.5
        assert c.min_samples == 5

    def test_custom_params(self):
        """Constructor stores provided hyperparameters."""
        c = DBSCANClusterer(eps=1.0, min_samples=3)
        assert c.eps == 1.0
        assert c.min_samples == 3

    def test_not_fitted_initially(self):
        """Model, labels, n_clusters and n_outliers must be None before fit."""
        c = DBSCANClusterer()
        assert c._model is None
        assert c.labels_ is None
        assert c.n_clusters_ is None
        assert c.n_outliers_ is None


class TestDBSCANClustererFit:

    def test_fit_returns_self(self):
        """fit() must return self for method chaining."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        result = c.fit(X)
        assert result is c

    def test_labels_shape_after_fit(self):
        """labels_ must have one entry per sample."""
        X = make_matrix_with_outlier(n_samples=60)
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        # 60 blob points + 1 explicit outlier = 61 total
        assert c.labels_.shape == (61,)

    def test_outlier_gets_minus_one(self):
        """The explicit far outlier (last row) must receive label -1."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        # The last row is the explicit outlier placed at (20, 20)
        assert c.labels_[-1] == -1

    def test_n_outliers_at_least_one(self):
        """n_outliers_ must be >= 1 when explicit outlier is present."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        assert c.n_outliers_ >= 1

    def test_n_clusters_positive(self):
        """n_clusters_ must be >= 1 with well-separated blobs."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        assert c.n_clusters_ >= 1

    def test_labels_contain_minus_one(self):
        """labels_ array must contain at least one -1 (noise label)."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        assert -1 in c.labels_


class TestDBSCANClustererGetParams:

    def test_get_params_returns_dict(self):
        """get_params() must return a dict with eps and min_samples."""
        c = DBSCANClusterer(eps=1.5, min_samples=7)
        params = c.get_params()
        assert isinstance(params, dict)
        assert params["eps"] == 1.5
        assert params["min_samples"] == 7


class TestDBSCANGridSearch:

    def test_grid_search_returns_list(self):
        """grid_search() must return a list."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        results = c.grid_search(
            X, eps_values=[0.5, 1.0], min_samples_values=[3, 5]
        )
        assert isinstance(results, list)

    def test_grid_search_length(self):
        """Result list length must equal len(eps) * len(min_samples)."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        eps_v = [0.5, 1.0]
        ms_v = [3, 5]
        results = c.grid_search(
            X, eps_values=eps_v, min_samples_values=ms_v
        )
        assert len(results) == len(eps_v) * len(ms_v)

    def test_grid_search_result_keys(self):
        """Each result dict must have the expected keys."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        results = c.grid_search(
            X, eps_values=[0.7], min_samples_values=[3]
        )
        expected_keys = {
            "eps", "min_samples", "n_clusters", "n_outliers", "silhouette"
        }
        assert expected_keys.issubset(set(results[0].keys()))

    def test_grid_search_sorted_by_silhouette(self):
        """Valid silhouette results must appear before None entries."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        results = c.grid_search(X)
        valid = [r for r in results if r["silhouette"] is not None]
        none_entries = [r for r in results if r["silhouette"] is None]
        # All valid entries must come before None entries
        assert results.index(valid[0]) < results.index(none_entries[0]) if (
            valid and none_entries
        ) else True

    def test_grid_search_silhouette_range(self):
        """Silhouette scores (when not None) must be in [-1, 1]."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        results = c.grid_search(
            X, eps_values=[0.7, 1.0], min_samples_values=[3]
        )
        for r in results:
            if r["silhouette"] is not None:
                assert -1.0 <= r["silhouette"] <= 1.0


class TestDBSCANGetOutliers:

    def test_get_outliers_returns_list(self):
        """get_outliers() must return a list."""
        X = make_matrix_with_outlier()
        labels = make_politician_labels(n=61)
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        outliers = c.get_outliers(labels)
        assert isinstance(outliers, list)

    def test_get_outliers_content(self):
        """Outlier dict must contain index, first_name, last_name, party."""
        X = make_matrix_with_outlier()
        labels = make_politician_labels(n=61)
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        outliers = c.get_outliers(labels)
        if outliers:
            expected_keys = {"index", "first_name", "last_name", "party"}
            assert expected_keys.issubset(set(outliers[0].keys()))

    def test_get_outliers_last_point_detected(self):
        """The explicit outlier (last row, index 60) must be detected."""
        X = make_matrix_with_outlier()
        labels = make_politician_labels(n=61)
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        outliers = c.get_outliers(labels)
        # Index 60 is the explicit far outlier added by
        # make_matrix_with_outlier
        outlier_indices = [o["index"] for o in outliers]
        assert 60 in outlier_indices

    def test_get_outliers_raises_if_not_fitted(self):
        """get_outliers() must raise RuntimeError before fit()."""
        c = DBSCANClusterer()
        labels = make_politician_labels()
        with pytest.raises(RuntimeError, match="not fitted"):
            c.get_outliers(labels)


class TestDBSCANFindBestParams:

    def test_find_best_params_returns_dict(self):
        """find_best_params() must return a dict."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        results = c.grid_search(
            X, eps_values=[0.5, 0.7, 1.0], min_samples_values=[3, 5]
        )
        best = c.find_best_params(results, n_total=len(X))
        assert isinstance(best, dict)

    def test_find_best_params_respects_noise_ratio(self):
        """Result must have noise_ratio <= max_noise_ratio when possible."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        results = c.grid_search(
            X, eps_values=[0.5, 0.7, 1.0], min_samples_values=[3, 5]
        )
        n_total = len(X)
        best = c.find_best_params(
            results, n_total=n_total, max_noise_ratio=0.20
        )
        valid = [
            r for r in results
            if r["silhouette"] is not None
            and r["n_outliers"] / n_total <= 0.20
        ]
        if valid:
            assert best["n_outliers"] / n_total <= 0.20
            assert best["silhouette"] == max(
                r["silhouette"] for r in valid
            )

    def test_find_best_params_fallback_when_all_noisy(self):
        """Fallback to grid_results[0] when no result passes noise filter."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        # eps=0.3 with min_samples=10 will produce many outliers
        results = c.grid_search(
            X, eps_values=[0.3], min_samples_values=[10]
        )
        # Use n_total=1 to force all results to fail the noise ratio check
        best = c.find_best_params(results, n_total=1, max_noise_ratio=0.0)
        assert best is results[0]


class TestDBSCANPlotGridSearch:

    def test_plot_grid_search_saves_file(self, tmp_path):
        """plot_grid_search() must create the PNG file."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        eps_v = [0.5, 1.0]
        ms_v = [3, 5]
        results = c.grid_search(
            X, eps_values=eps_v, min_samples_values=ms_v
        )
        out = str(tmp_path / "grid.png")
        c.plot_grid_search(results, eps_v, ms_v, save_path=out)
        import os
        assert os.path.exists(out)

    def test_plot_grid_search_file_nonempty(self, tmp_path):
        """The saved PNG must not be an empty file."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        eps_v = [0.5, 1.0]
        ms_v = [3, 5]
        results = c.grid_search(
            X, eps_values=eps_v, min_samples_values=ms_v
        )
        out = str(tmp_path / "grid.png")
        c.plot_grid_search(results, eps_v, ms_v, save_path=out)
        import os
        assert os.path.getsize(out) > 0


class TestDBSCANPlotClustersPCA:

    def test_plot_clusters_pca_saves_file(self, tmp_path):
        """plot_clusters_pca() must create the PNG file."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        out = str(tmp_path / "pca.png")
        c.plot_clusters_pca(X, save_path=out)
        import os
        assert os.path.exists(out)

    def test_plot_clusters_pca_file_nonempty(self, tmp_path):
        """The saved PNG must not be an empty file."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        out = str(tmp_path / "pca.png")
        c.plot_clusters_pca(X, save_path=out)
        import os
        assert os.path.getsize(out) > 0

    def test_plot_clusters_pca_with_labels(self, tmp_path):
        """plot_clusters_pca() must not raise when politician_labels given."""
        X = make_matrix_with_outlier()
        labels = make_politician_labels(n=61)
        c = DBSCANClusterer(eps=0.7, min_samples=3)
        c.fit(X)
        out = str(tmp_path / "pca_labels.png")
        c.plot_clusters_pca(X, politician_labels=labels, save_path=out)
        import os
        assert os.path.exists(out)

    def test_plot_clusters_pca_raises_if_not_fitted(self, tmp_path):
        """plot_clusters_pca() must raise RuntimeError before fit()."""
        X = make_matrix_with_outlier()
        c = DBSCANClusterer()
        with pytest.raises(RuntimeError, match="not fitted"):
            c.plot_clusters_pca(X, save_path=str(tmp_path / "out.png"))
