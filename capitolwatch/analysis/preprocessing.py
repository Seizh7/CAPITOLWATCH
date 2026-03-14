# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Normalization utilities for feature matrices.

Supported scalers:
    - StandardScaler : mean=0, std=1  (K-Means, DBSCAN)
    - MinMaxScaler   : range [0, 1]   (SOM)

Main functions:
    - normalize_features(matrix, scaler) -> (matrix_scaled_df, fitted_scaler)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize_features(matrix, scaler):
    """
    Apply a scaler to a feature DataFrame (matrix).

    Args:
        matrix (pd.DataFrame): Feature matrix of shape (n_politicians,
            n_features).
        scaler: An sklearn scaler instance (StandardScaler or MinMaxScaler).

    Returns:
        tuple:
            - pd.DataFrame: Scaled matrix, same shape/index/columns as matrix.
            - scaler: The fitted scaler instance.

    Raises:
        TypeError: If matrix is not a DataFrame.
    """
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(matrix)}")

    # Apply scaler.fit_transform on matrix.values
    scaled_values = scaler.fit_transform(matrix.values)

    # Build a DataFrame with same index and columns as matrix
    matrix_scaled = pd.DataFrame(
        scaled_values, index=matrix.index, columns=matrix.columns
    )

    return matrix_scaled, scaler


if __name__ == "__main__":
    # Example data
    df = pd.DataFrame({
        'feature1': [10, 20, 30],
        'feature2': [1, 2, 3]
    }, index=[1, 2, 3])

    scaler = StandardScaler()
    scaled_df, fitted_scaler = normalize_features(df, scaler)
    print(scaled_df)
    print(f"Fitted scaler mean: {fitted_scaler.mean_}")
    print(f"Fitted scaler var: {fitted_scaler.var_}")

    scaler = MinMaxScaler()
    scaled_df, fitted_scaler = normalize_features(df, scaler)
    print(scaled_df)
    print(f"Fitted scaler data min: {fitted_scaler.data_min_}")
    print(f"Fitted scaler data max: {fitted_scaler.data_max_}")
