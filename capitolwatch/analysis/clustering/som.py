# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import os

import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.cluster import KMeans

from capitolwatch.analysis.clustering.base import BaseClusterer


class SOMClusterer(BaseClusterer):
    """
    Self-Organizing Map (SOM) clustering wrapper around minisom.MiniSom.

    Input data must be normalized to [0, 1] (MinMaxScaler) to match the
    default random weight initialization range of MiniSom.

    Attributes:
        m (int): Number of rows in the SOM grid.
        n (int): Number of columns in the SOM grid.
        sigma (float): Initial neighborhood radius; controls how many
            neurons around the BMU are updated during training.
        learning_rate (float): Initial step size for weight updates;
            decays during training.
        n_iterations (int): Total number of training steps.
        random_seed (int): Seed for reproducibility.
        n_clusters (int or None): Number of clusters extracted via K-Means
            on neuron weights. Set by extract_clusters().
        labels_ (np.ndarray or None): Cluster label per sample. Set by
            extract_clusters().
        bmu_coords_ (list[tuple[int, int]] or None): BMU (row, col) for
            each sample. Set by fit().
    """

    def __init__(
        self,
        m: int = 10,
        n: int = 10,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        n_iterations: int = 1000,
        random_seed: int = 42,
    ):
        self.m = m
        self.n = n
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self._som = None
        self.labels_ = None
        self.bmu_coords_ = None
        self.n_clusters = None

    def fit(self, matrix: np.ndarray) -> "SOMClusterer":
        """
        Initialize and train the SOM on the feature matrix.

        Data must be normalized to [0, 1] (MinMaxScaler) so that initial
        random neuron weights (also in [0, 1]) are in the same space as
        the input vectors.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features). Values must be in [0, 1].

        Returns:
            self
        """
        n_features = matrix.shape[1]

        # initialize MiniSom with grid dimensions and training parameters
        self._som = MiniSom(
            x=self.m,
            y=self.n,
            input_len=n_features,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
        )

        # Initialize neuron weights randomly in [0, 1] and train the SOM
        # on the data
        self._som.random_weights_init(matrix)
        self._som.train(matrix, self.n_iterations)

        # Store the BMU coordinate (row, col) for every sample
        self.bmu_coords_ = self.get_bmu_coords(matrix)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Return cluster labels assigned during extract_clusters().

        Args:
            matrix (np.ndarray): Feature matrix (ignored; labels are
                stored from the last extract_clusters() call).

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).
        """
        if self.labels_ is None:
            raise RuntimeError(
                "Labels not set. Call extract_clusters() after fit()."
            )
        return self.labels_

    def get_params(self) -> dict:
        """
        Return the hyperparameters of this clusterer.

        Returns:
            dict: SOM hyperparameters.
        """
        return {
            "m": self.m,
            "n": self.n,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "random_seed": self.random_seed,
        }

    def get_bmu_coords(self, matrix: np.ndarray) -> list:
        """
        Find the Best Matching Unit (BMU) for each sample in matrix.

        The BMU is the neuron whose weight vector is closest (Euclidean
        distance) to the input sample.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).

        Returns:
            list[tuple[int, int]]: (row, col) grid position for each
                sample, in the same order as the rows of matrix.
        """
        if self._som is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # winner(x) returns the (row, col) of the closest neuron to x
        return [self._som.winner(sample) for sample in matrix]

    def compute_umatrix(self) -> np.ndarray:
        """
        Return the Unified Distance Matrix (U-Matrix) of the trained SOM.

        Each cell (i, j) holds the mean Euclidean distance from neuron
        (i, j)'s weight vector to its immediate grid neighbors. High
        values indicate boundaries between clusters; low values indicate
        the core of a cluster.

        Returns:
            np.ndarray: U-Matrix of shape (m, n), values in [0, 1].
        """
        if self._som is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # distance_map() returns a (m, n) array where each cell is the
        # normalized mean Euclidean distance to its grid neighbors
        return self._som.distance_map()

    def extract_clusters(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Assign discrete cluster labels to samples by applying K-Means on
        the neuron weight vectors, then propagating labels to samples via
        their BMU.

        Args:
            n_clusters (int): Number of clusters to form.
            random_state (int): Seed for K-Means reproducibility.

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).
        """
        if self._som is None or self.bmu_coords_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        self.n_clusters = n_clusters

        # get_weights() returns (m, n, n_features); reshape to (m*n,
        # n_features)
        # K-Means will sees one vector per neuron, not one per grid cell
        weights = self._som.get_weights().reshape(self.m * self.n, -1)

        # cluster neuron weight vectors — captures macro topology without
        # individual sample noise
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        kmeans.fit(weights)
        neuron_labels = kmeans.labels_

        # convert each sample's BMU (row, col) to a flat neuron index,
        # then look up which cluster that neuron belongs to
        sample_labels = [
            neuron_labels[row * self.n + col]
            for row, col in self.bmu_coords_
        ]
        self.labels_ = np.array(sample_labels)
        return self.labels_

    def plot_umatrix(
        self,
        matrix: np.ndarray = None,
        politician_labels=None,
        save_path: str = "data/visualizations/som_umatrix.png",
    ) -> None:
        """
        Plot the U-Matrix as a heatmap and optionally overlay each
        politician's BMU position, color-coded by party.

        Args:
            matrix (np.ndarray or None): Feature matrix used to locate
                each politician on the grid.
            politician_labels (pd.DataFrame or None): DataFrame with
                columns first_name, last_name, party.
            save_path (str): Path where the PNG will be saved.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._som is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        umatrix = self.compute_umatrix()

        fig, ax = plt.subplots(figsize=(10, 8))
        # bone_r: dark = cluster core (low distance), light = boundary
        im = ax.imshow(umatrix, cmap="bone_r", origin="lower")
        fig.colorbar(im, ax=ax, label="Mean distance to neighbors")
        ax.set_title("SOM U-Matrix")
        ax.set_xlabel("Neuron column")
        ax.set_ylabel("Neuron row")

        # overlay politician positions color-coded by party
        if matrix is not None and politician_labels is not None:
            party_colors = {"Republican": "red", "Democratic": "blue"}
            bmu_coords = self.get_bmu_coords(matrix)
            rng = np.random.default_rng(seed=0)
            # add jitter to points so overlapping BMUs are distinguishable,
            # and color by party
            for i, (row, col) in enumerate(bmu_coords):
                party = politician_labels.iloc[i]["party"]
                color = party_colors.get(party, "green")
                # small jitter so overlapping BMUs are distinguishable
                jitter = 0.3 * (rng.random(2) - 0.5)
                ax.scatter(
                    col + jitter[1],
                    row + jitter[0],
                    color=color,
                    s=40,
                    alpha=0.7,
                    zorder=3,
                )
            # build a minimal legend for party colors
            for party, color in party_colors.items():
                ax.scatter([], [], color=color, s=40, label=party)
            ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_som_map(
        self,
        matrix: np.ndarray,
        politician_labels=None,
        save_path: str = "data/visualizations/som_map.png",
    ) -> None:
        """
        Visualize the SOM grid: each occupied cell is colored by cluster,
        and politician names are annotated at their BMU position.

        Args:
            matrix (np.ndarray): Feature matrix of shape (n_samples,
                n_features).
            politician_labels (pd.DataFrame or None): DataFrame with
                columns first_name, last_name, party.
            save_path (str): Path where the PNG will be saved.

        Raises:
            RuntimeError: If extract_clusters() has not been called yet.
        """
        if self.labels_ is None:
            raise RuntimeError(
                "Labels not set. Call extract_clusters() after fit()."
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        bmu_coords = self.get_bmu_coords(matrix)
        colors = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(12, 10))

        # draw a colored circle at each politician's BMU, by cluster
        for i, (row, col) in enumerate(bmu_coords):
            cluster_id = self.labels_[i]
            ax.scatter(
                col,
                row,
                color=colors[cluster_id % len(colors)],
                s=300,
                alpha=0.4,
                zorder=1,
            )

        # annotate with abbreviated name (initial + last name)
        if politician_labels is not None:
            for i, (row, col) in enumerate(bmu_coords):
                p = politician_labels.iloc[i]
                name = f"{p['first_name'][0]}. {p['last_name']}"
                ax.annotate(
                    name,
                    xy=(col, row),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    zorder=4,
                )

        ax.set_xlim(-1, self.n)
        ax.set_ylim(-1, self.m)
        ax.set_title("SOM Grid — Politician Positions by Cluster")
        ax.set_xlabel("Neuron column")
        ax.set_ylabel("Neuron row")

        # cluster color legend
        for k in range(self.n_clusters):
            ax.scatter(
                [], [],
                color=colors[k % len(colors)],
                label=f"Cluster {k}",
                s=100,
            )
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler

    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=0)
    # SOM requires [0, 1] input to match the default weight initialization
    X = MinMaxScaler().fit_transform(X)

    clusterer = SOMClusterer(m=5, n=5, n_iterations=500, random_seed=42)
    clusterer.fit(X)
    clusterer.extract_clusters(n_clusters=3)
    print("Labels:", clusterer.labels_)
    print("Unique labels:", np.unique(clusterer.labels_))
    print("U-Matrix shape:", clusterer.compute_umatrix().shape)
