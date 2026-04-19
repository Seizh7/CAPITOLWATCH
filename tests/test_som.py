# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from capitolwatch.analysis.clustering.som import SOMClusterer


def make_normalized_matrix(
    n_samples: int = 60,
    n_centers: int = 3,
    random_state: int = 0
) -> np.ndarray:
    """
    Generate a blob dataset normalized to [0, 1] — required for SOM.

    Args:
        n_samples (int): Number of samples.
        n_centers (int): Number of blob centers.
        random_state (int): Seed for reproducibility.

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, 2) in [0, 1].
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=0.5,
        random_state=random_state,
    )
    return MinMaxScaler().fit_transform(X)


def make_politician_labels(n: int = 60) -> pd.DataFrame:
    """
    Build a minimal politician_labels DataFrame.

    Args:
        n (int): Number of rows.

    Returns:
        pd.DataFrame: DataFrame with first_name, last_name, party columns.
    """
    rows = [
        {
            "first_name": f"Pol{i}",
            "last_name": f"Name{i}",
            "party": "Republican" if i % 2 == 0 else "Democratic",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestSOMClustererInit:

    def test_default_params(self):
        """Default grid is 10x10 with standard minisom defaults."""
        c = SOMClusterer()
        assert c.m == 10
        assert c.n == 10
        assert c.sigma == 1.0
        assert c.learning_rate == 0.5
        assert c.n_iterations == 1000
        assert c.random_seed == 42

    def test_custom_params(self):
        """Constructor stores all provided hyperparameters."""
        c = SOMClusterer(m=5, n=5, sigma=0.5, learning_rate=0.3,
                         n_iterations=200, random_seed=0)
        assert c.m == 5
        assert c.n == 5
        assert c.sigma == 0.5
        assert c.learning_rate == 0.3
        assert c.n_iterations == 200
        assert c.random_seed == 0

    def test_not_fitted_initially(self):
        """All fitted attributes must be None before fit()."""
        c = SOMClusterer()
        assert c._som is None
        assert c.labels_ is None
        assert c.bmu_coords_ is None
        assert c.n_clusters is None


class TestSOMClustererFit:

    def test_fit_returns_self(self):
        """fit() must return self for method chaining."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        result = c.fit(X)
        assert result is c

    def test_som_initialized_after_fit(self):
        """_som must be a MiniSom instance after fit()."""
        from minisom import MiniSom
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        assert isinstance(c._som, MiniSom)

    def test_bmu_coords_set_after_fit(self):
        """bmu_coords_ must have one entry per sample after fit()."""
        X = make_normalized_matrix(n_samples=60)
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        assert c.bmu_coords_ is not None
        assert len(c.bmu_coords_) == 60

    def test_bmu_coords_within_grid(self):
        """Every BMU coordinate must be a valid (row, col) within the grid."""
        X = make_normalized_matrix(n_samples=30)
        c = SOMClusterer(m=4, n=6, n_iterations=50)
        c.fit(X)
        for row, col in c.bmu_coords_:
            assert 0 <= row < 4
            assert 0 <= col < 6

    def test_labels_none_before_extract_clusters(self):
        """labels_ must remain None until extract_clusters() is called."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        assert c.labels_ is None


class TestSOMClustererGetBmuCoords:

    def test_returns_list_of_tuples(self):
        """get_bmu_coords must return a list of (row, col) tuples."""
        X = make_normalized_matrix(n_samples=20)
        c = SOMClusterer(m=4, n=4, n_iterations=50)
        c.fit(X)
        coords = c.get_bmu_coords(X)
        assert isinstance(coords, list)
        assert len(coords) == 20
        for item in coords:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_raises_before_fit(self):
        """get_bmu_coords must raise RuntimeError if model not fitted."""
        X = make_normalized_matrix(n_samples=10)
        c = SOMClusterer()
        with pytest.raises(RuntimeError):
            c.get_bmu_coords(X)


class TestSOMClustererComputeUmatrix:

    def test_umatrix_shape(self):
        """U-Matrix shape must match the SOM grid dimensions (m, n)."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=6, n_iterations=100)
        c.fit(X)
        umatrix = c.compute_umatrix()
        assert umatrix.shape == (5, 6)

    def test_umatrix_values_range(self):
        """U-Matrix values must be non-negative."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        umatrix = c.compute_umatrix()
        assert np.all(umatrix >= 0)

    def test_raises_before_fit(self):
        """compute_umatrix must raise RuntimeError if model not fitted."""
        c = SOMClusterer()
        with pytest.raises(RuntimeError):
            c.compute_umatrix()


class TestSOMClustererExtractClusters:

    def test_labels_shape(self):
        """extract_clusters must produce one label per sample."""
        X = make_normalized_matrix(n_samples=60)
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        labels = c.extract_clusters(n_clusters=3)
        assert labels.shape == (60,)

    def test_labels_stored(self):
        """extract_clusters must store labels in self.labels_."""
        X = make_normalized_matrix(n_samples=60)
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        labels = c.extract_clusters(n_clusters=3)
        assert c.labels_ is not None
        np.testing.assert_array_equal(c.labels_, labels)

    def test_n_clusters_stored(self):
        """n_clusters attribute must be updated by extract_clusters."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        c.extract_clusters(n_clusters=4)
        assert c.n_clusters == 4

    def test_labels_in_valid_range(self):
        """All labels must be in [0, n_clusters - 1]."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        labels = c.extract_clusters(n_clusters=3)
        assert set(labels).issubset({0, 1, 2})

    def test_raises_before_fit(self):
        """extract_clusters must raise RuntimeError if model not fitted."""
        c = SOMClusterer()
        with pytest.raises(RuntimeError):
            c.extract_clusters()


class TestSOMClustererGetParams:

    def test_returns_dict_with_expected_keys(self):
        """get_params must return all six hyperparameter keys."""
        c = SOMClusterer(m=7, n=8, sigma=0.8, learning_rate=0.4,
                         n_iterations=500, random_seed=1)
        params = c.get_params()
        assert params == {
            "m": 7,
            "n": 8,
            "sigma": 0.8,
            "learning_rate": 0.4,
            "n_iterations": 500,
            "random_seed": 1,
        }


class TestSOMClustererPlots:

    def test_plot_umatrix_saves_file(self, tmp_path):
        """plot_umatrix must create a PNG file at the given path."""
        X = make_normalized_matrix()
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        save_path = str(tmp_path / "umatrix.png")
        c.plot_umatrix(save_path=save_path)
        assert os.path.isfile(save_path)

    def test_plot_umatrix_with_overlay_saves_file(self, tmp_path):
        """plot_umatrix with matrix+labels overlay must create a PNG."""
        X = make_normalized_matrix(n_samples=30)
        labels_df = make_politician_labels(30)
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        save_path = str(tmp_path / "umatrix_overlay.png")
        c.plot_umatrix(matrix=X, politician_labels=labels_df,
                       save_path=save_path)
        assert os.path.isfile(save_path)

    def test_plot_som_map_saves_file(self, tmp_path):
        """plot_som_map must create a PNG after extract_clusters."""
        X = make_normalized_matrix(n_samples=30)
        labels_df = make_politician_labels(30)
        c = SOMClusterer(m=5, n=5, n_iterations=100)
        c.fit(X)
        c.extract_clusters(n_clusters=3)
        save_path = str(tmp_path / "som_map.png")
        c.plot_som_map(X, politician_labels=labels_df, save_path=save_path)
        assert os.path.isfile(save_path)

    def test_plot_som_map_raises_before_extract_clusters(self, tmp_path):
        """plot_som_map must raise RuntimeError if labels not set."""
        X = make_normalized_matrix(n_samples=20)
        c = SOMClusterer(m=4, n=4, n_iterations=50)
        c.fit(X)
        save_path = str(tmp_path / "som_map.png")
        with pytest.raises(RuntimeError):
            c.plot_som_map(X, save_path=save_path)

    def test_plot_umatrix_raises_before_fit(self, tmp_path):
        """plot_umatrix must raise RuntimeError if model not fitted."""
        c = SOMClusterer()
        save_path = str(tmp_path / "umatrix.png")
        with pytest.raises(RuntimeError):
            c.plot_umatrix(save_path=save_path)

