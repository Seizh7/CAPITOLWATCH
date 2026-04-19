# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from capitolwatch.analysis.clustering.base import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """
    DBSCAN clustering class around sklearn.cluster.DBSCAN.

    Attributes:
        eps (float): Maximum distance between two samples for one to be
            considered in the neighborhood of the other.
        min_samples (int): Minimum number of samples in a neighborhood to
            form a core point.
        metric (str): Distance metric passed to sklearn DBSCAN.
            Use "euclidean" for dense numerical data or "cosine" for
            sparse frequency/count vectors where direction matters more
            than magnitude.
        labels_ (np.ndarray or None): Cluster labels assigned by fit().
            -1 indicates a noise point (outlier).
        n_clusters_ (int or None): Number of clusters found (noise excluded).
        n_outliers_ (int or None): Number of noise points (label == -1).
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        self.eps = eps
        self.min_samples = min_samples
        # metric controls how DBSCAN measures distances between points
        self.metric = metric
        self._model = None
        self.labels_ = None
        self.n_clusters_ = None
        self.n_outliers_ = None

    def fit(self, matrix: np.ndarray) -> "DBSCANClusterer":
        """
        Train DBSCAN on the feature matrix.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).

        Returns:
            self
        """
        # create and fit the DBSCAN model with the configured distance metric
        self._model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        self._model.fit(matrix)
        # store the labels assigned by DBSCAN
        self.labels_ = self._model.labels_

        # Count clusters and outliers based on labels_
        self.n_clusters_ = len(
            set(self.labels_)
        ) - (1 if -1 in self.labels_ else 0)

        # Count outliers (label == -1)
        self.n_outliers_ = sum(self.labels_ == -1)

        return self

    def get_params(self) -> dict:
        """
        Return the hyperparameters of this clusterer.

        Returns:
            dict: {"eps": float, "min_samples": int}
        """
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
        }

    def grid_search(
        self,
        matrix: np.ndarray,
        eps_values: list = None,
        min_samples_values: list = None,
    ) -> list:
        """
        Test all combinations (eps, min_samples) and calculate the
        silhouette score for each valid configuration.

        A configuration is valid when:
        - At least 2 clusters were found (excluding noise).
        - At least 2 non-noise points exist.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).
            eps_values (list[float]): Epsilon values to try.
                Defaults to [0.3, 0.5, 0.7, 1.0, 1.5, 2.0].
            min_samples_values (list[int]): min_samples values to try.
                Defaults to [3, 5, 7, 10].

        Returns:
            list[dict]: Results sorted by silhouette descending (None last).
                Each dict has keys:
                    "eps", "min_samples", "n_clusters", "n_outliers",
                    "silhouette".
        """
        # apply default grid values if none are provided
        if eps_values is None:
            eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        if min_samples_values is None:
            min_samples_values = [3, 5, 7, 10]

        results = []

        # test all combinations of eps and min_samples
        for eps in eps_values:
            for min_s in min_samples_values:
                # propagate metric so grid search uses the same distance
                temp = DBSCANClusterer(
                    eps=eps,
                    min_samples=min_s,
                    metric=self.metric,
                )
                temp.fit(matrix)
                # mask for non-noise points (label != -1)
                mask = temp.labels_ != -1
                # silhouette_score requires at least 2 distinct cluster labels
                # and at least 2 non-noise samples to compare distances
                if temp.n_clusters_ >= 2 and mask.sum() >= 2:
                    # compute silhouette only on clustered (non-noise) points
                    silhouette = silhouette_score(
                        matrix[mask], temp.labels_[mask]
                    )
                else:
                    # degenerate case: all noise, or single cluster
                    silhouette = None
                # store the results for this parameter combination
                results.append({
                    "eps": eps,
                    "min_samples": min_s,
                    "n_clusters": temp.n_clusters_,
                    "n_outliers": temp.n_outliers_,
                    "silhouette": silhouette
                })

        # Sort: valid scores first (descending), then None entries
        results.sort(
            key=lambda r: (r["silhouette"] is None, -(r["silhouette"] or 0))
        )
        return results

    def find_best_params(
        self,
        grid_results: list,
        n_total: int,
        max_noise_ratio: float = 0.20,
    ) -> dict:
        """
        Return the best valid result: highest silhouette with noise ratio
        below max_noise_ratio.

        A result passes the filter when:
        - silhouette is not None (at least 2 clusters found)
        - n_outliers / n_total <= max_noise_ratio

        Fallback: if no result passes the filter, return grid_results[0]
        (highest silhouette regardless of noise), because that is the
        "least bad" option and signals that feature space may not contain
        dense natural clusters for DBSCAN.

        Args:
            grid_results (list[dict]): Output of grid_search(), already
                sorted by silhouette descending.
            n_total (int): Total number of samples in the dataset.
                Used to compute the noise ratio.
            max_noise_ratio (float): Maximum acceptable fraction of
                outliers. Defaults to 0.20 (20%).

        Returns:
            dict: Best parameter combination satisfying the noise
                constraint, or grid_results[0] as a fallback.
        """
        # iterate over results already sorted by silhouette descending
        for result in grid_results:
            # skip invalid configurations (single cluster or all noise)
            if result["silhouette"] is None:
                continue
            # compute the fraction of points labeled as noise
            noise_ratio = result["n_outliers"] / n_total
            # accept the first result that meets the noise constraint
            if noise_ratio <= max_noise_ratio:
                return result
        # fallback: return the highest silhouette entry regardless of noise
        return grid_results[0] if grid_results else {}

    def get_outliers(self, politician_labels) -> list:
        """
        Return metadata for politicans identified as noise (label == -1).

        Args:
            politician_labels (pd.DataFrame): DataFrame with at least
                columns first_name, last_name, party.

        Returns:
            list[dict]: Each dict has keys index, first_name, last_name,
                party.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.labels_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        outliers = []
        # iterate over all labels and collect noise points (label == -1)
        for i, label in enumerate(self.labels_):
            if label == -1:
                # extract politician metadata from the DataFrame row
                row = politician_labels.iloc[i]
                outliers.append({
                    "index": i,
                    "first_name": row["first_name"],
                    "last_name": row["last_name"],
                    "party": row["party"]
                })
        return outliers

    def plot_grid_search(
        self,
        results: list,
        eps_values: list,
        min_samples_values: list,
        save_path: str = "data/visualizations/dbscan_grid_search.png",
    ) -> None:
        """
        Plot a heatmap of silhouette scores over the (eps, min_samples) grid.

        Cells where silhouette is None (degenerate clustering) are shown
        as zero.

        Args:
            results (list[dict]): Output of grid_search().
            eps_values (list[float]): Epsilon values that were tested.
            min_samples_values (list[int]): min_samples values that were
                tested.
            save_path (str): Path where the PNG will be saved.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # build a 2D score matrix: rows = eps, columns = min_samples
        score_matrix = np.zeros((len(eps_values), len(min_samples_values)))
        for r in results:
            # locate the cell corresponding to this (eps, min_samples) pair
            i = eps_values.index(r["eps"])
            j = min_samples_values.index(r["min_samples"])
            # use 0.0 for degenerate configurations (None silhouette)
            sil = r["silhouette"]
            score_matrix[i, j] = sil if sil is not None else 0.0

        # draw the heatmap: x-axis = min_samples, y-axis = eps
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(score_matrix, cmap="viridis", origin="lower")
        ax.set_xticks(np.arange(len(min_samples_values)))
        ax.set_yticks(np.arange(len(eps_values)))
        ax.set_xticklabels(min_samples_values)
        ax.set_yticklabels(eps_values)
        ax.set_xlabel("min_samples")
        ax.set_ylabel("eps")
        ax.set_title("DBSCAN Grid Search Silhouette Scores")
        fig.colorbar(im, ax=ax, label="Silhouette Score")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_clusters_pca(
        self,
        matrix: np.ndarray,
        politician_labels=None,
        save_path: str = "data/visualizations/dbscan_clusters_pca.png",
    ) -> None:
        """
        Project feature matrix to 2D via PCA and plot clusters.
        Outliers (label == -1) are plotted separately. If politician_labels
        is provided, annotate outliers with their names.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).
            politician_labels (pd.DataFrame or None): Optional DataFrame
                with columns first_name, last_name, party.
            save_path (str): Path where the PNG will be saved.
        """
        if self.labels_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # project the high-dimensional feature matrix down to 2 components
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)

        # collect valid cluster ids, excluding noise label -1
        unique_clusters = sorted(
            c for c in set(self.labels_) if c != -1
        )
        # use tab10 colormap for up to 10 distinct cluster colors
        colors = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(10, 7))

        # draw each cluster with a distinct color
        for idx, cluster_id in enumerate(unique_clusters):
            mask = self.labels_ == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=colors[idx % len(colors)],
                label=f"Cluster {cluster_id}",
                s=60,
                alpha=0.8,
            )

        # draw noise points in black with an 'x' marker
        outlier_mask = self.labels_ == -1
        if outlier_mask.any():
            ax.scatter(
                coords[outlier_mask, 0],
                coords[outlier_mask, 1],
                color="black",
                marker="x",
                s=80,
                linewidths=1.5,
                label="Outliers",
            )
            # annotate each outlier with the politician's name
            if politician_labels is not None:
                for i in np.where(outlier_mask)[0]:
                    row = politician_labels.iloc[i]
                    name = f"{row['first_name']} {row['last_name']}"
                    ax.annotate(
                        name,
                        xy=(coords[i, 0], coords[i, 1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

        ax.set_title("DBSCAN Clusters (PCA 2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
    clusterer = DBSCANClusterer(eps=0.3, min_samples=5)
    clusterer.fit(X)
    print("Cluster labels:", clusterer.labels_)
    print("Number of clusters found:", clusterer.n_clusters_)
    print("Number of outliers found:", clusterer.n_outliers_)
