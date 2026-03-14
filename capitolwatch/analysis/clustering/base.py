# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
import numpy as np
import joblib


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.

    All clusterers must implement: fit(), predict(), get_params().
    Common persistence methods save_model() and load_model() are provided.
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

    @abstractmethod
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to samples in matrix.

        Args:
            matrix (np.ndarray): Feature matrix of shape
                (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).
                        Label -1 indicates noise (used by DBSCAN).
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

    def save_model(self, path: str) -> None:
        """
        Serialize the fitted model to disk using joblib.

        Args:
            path (str): File path where the model will be saved.
        """
        joblib.dump(self, path)

    @staticmethod
    def load_model(path: str) -> "BaseClusterer":
        """
        Deserialize a model from disk and return it.

        Args:
            path (str): File path of the serialized model.

        Returns:
            BaseClusterer: The loaded model instance.
        """
        return joblib.load(path)
