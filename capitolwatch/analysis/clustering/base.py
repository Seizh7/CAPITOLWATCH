# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
import numpy as np


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.

    All clusterers must implement: fit() and get_params().
    """

    @abstractmethod
    def fit(self, matrix: np.ndarray) -> "BaseClusterer":
        """
        Train the clustering model on feature matrix.

        Args:
            matrix (np.ndarray): Feature matrix of shape
                (n_samples, n_features).

        Returns:
            self
        """
        pass

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to samples in matrix.

        Only for K-Means method.

        Args:
            matrix (np.ndarray): Feature matrix of shape
                (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the hyperparameters of the clusterer.

        Returns:
            dict: Hyperparameter names and values.
        """
        pass
