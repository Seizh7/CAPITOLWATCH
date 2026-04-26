# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from capitolwatch.analysis.clustering.base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """
    K-Means clustering class around sklearn.cluster.KMeans.

    Provides elbow and silhouette analysis to find the optimal number
    of clusters K, then exposes fit/predict.

    Attributes:
        n_clusters (int): Number of clusters to form.
        random_state (int): Seed for reproducibility.
        labels_ (np.ndarray or None): Cluster labels after fit.
        inertia_ (float or None): Sum of squared distances to centroid
            after fit.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, matrix: np.ndarray) -> "KMeansClusterer":
        """
        Train K-Means on the feature matrix.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).

        Returns:
            self
        """
        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,  # Run 10 times and keep the best result
        )
        self._model.fit(matrix)
        self.labels_ = self._model.labels_
        self.inertia_ = self._model.inertia_
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to samples.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._model.predict(matrix)

    def get_params(self) -> dict:
        """
        Return the hyperparameters of this clusterer.

        Returns:
            dict: {"n_clusters": int, "random_state": int}
        """
        return {
            "n_clusters": self.n_clusters, "random_state": self.random_state
        }

    def find_optimal_k(
        self,
        matrix: np.ndarray,
        k_min: int = 2,
        k_max: int = 15,
    ) -> tuple:
        """
        Run K-Means for each K in [k_min, k_max] and collect inertia and
        silhouette scores.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).
            k_min (int): Minimum number of clusters to test.
            k_max (int): Maximum number of clusters to test.

        Returns:
            tuple:
                - k_values (list[int]): Tested values of K.
                - inertias (list[float]): Inertia for each K.
                - silhouette_scores (list[float]): Silhouette score for each K.
        """
        k_values = list(range(k_min, k_max + 1))
        inertias = []
        silhouette_scores_list = []

        # Loop over k_values, fit KMeans, append inertia and silhouette score
        for k in k_values:
            temp_model = KMeans(
                # Run 10 times and keep the best result
                n_clusters=k, random_state=self.random_state, n_init=10
            )
            temp_model.fit(matrix)
            inertias.append(temp_model.inertia_)
            score = silhouette_score(matrix, temp_model.labels_)
            silhouette_scores_list.append(score)

        return k_values, inertias, silhouette_scores_list

    def plot_elbow(
        self,
        k_values: list,
        inertias: list,
        save_path: str = "data/figures/kmeans_elbow.png",
    ) -> None:
        """
        Plot inertia vs K (elbow curve) and save to disk.

        Args:
            k_values (list[int]): Values of K tested.
            inertias (list[float]): Corresponding inertia values.
            save_path (str): Path where the PNG will be saved.
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, inertias, marker='o')
        ax.set_title("Elbow Curve")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_silhouette(
        self,
        k_values: list,
        silhouette_scores: list,
        save_path: str = "data/figures/kmeans_silhouette.png",
    ) -> None:
        """
        Plot silhouette score vs K and save to disk.

        Args:
            k_values (list[int]): Values of K tested.
            silhouette_scores (list[float]): Corresponding silhouette scores.
            save_path (str): Path where the PNG will be saved.
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, silhouette_scores, marker='o')
        max_score = max(silhouette_scores)
        max_index = silhouette_scores.index(max_score)
        ax.axvline(x=k_values[max_index], color='r', linestyle='--')
        ax.set_title("Silhouette Score vs K")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=0)
    c = KMeansClusterer(n_clusters=3)
    c.fit(X)
    assert c.labels_.shape == (60,)
    assert c.inertia_ > 0
    k_values, inertias, sil_scores = c.find_optimal_k(X, k_min=2, k_max=6)
    assert len(k_values) == 5
