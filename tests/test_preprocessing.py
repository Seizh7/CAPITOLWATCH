# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from capitolwatch.analysis.preprocessing import normalize_features


SAMPLE_DF = pd.DataFrame(
    {
        'feature1': [10.0, 20.0, 30.0],
        'feature2': [1.0, 2.0, 3.0],
    },
    index=['P1', 'P2', 'P3'],
)


class TestNormalizeFeaturesOutput:

    def test_returns_tuple(self):
        """Function must return a tuple (DataFrame, scaler)."""
        result = normalize_features(SAMPLE_DF, StandardScaler())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_is_dataframe(self):
        """First element of the tuple must be a DataFrame."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        assert isinstance(scaled, pd.DataFrame)

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        assert scaled.shape == SAMPLE_DF.shape

    def test_index_preserved(self):
        """Row index must be unchanged."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        assert list(scaled.index) == list(SAMPLE_DF.index)

    def test_columns_preserved(self):
        """Column names must be unchanged."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        assert list(scaled.columns) == list(SAMPLE_DF.columns)

    def test_returns_fitted_scaler(self):
        """Second element must be the fitted scaler (has mean_ attribute)."""
        _, fitted = normalize_features(SAMPLE_DF, StandardScaler())
        assert hasattr(fitted, 'mean_')


class TestStandardScaler:

    def test_mean_near_zero(self):
        """After StandardScaler, each column mean must be ≈ 0."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        assert np.allclose(scaled.mean(), 0.0, atol=1e-10)

    def test_std_near_one(self):
        """After StandardScaler, each column std must be ≈ 1."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        # ddof=0 (population std) matches sklearn's StandardScaler
        assert np.allclose(scaled.std(ddof=0), 1.0, atol=1e-10)

    def test_known_value(self):
        """feature1=[10,20,30] -> value 10 must scale to ≈ -1.2247."""
        scaled, _ = normalize_features(SAMPLE_DF, StandardScaler())
        assert pytest.approx(scaled.loc['P1', 'feature1'], abs=1e-4) == -1.2247


class TestMinMaxScaler:

    def test_min_is_zero(self):
        """After MinMaxScaler, minimum value per column must be 0."""
        scaled, _ = normalize_features(SAMPLE_DF, MinMaxScaler())
        assert np.allclose(scaled.min(), 0.0, atol=1e-10)

    def test_max_is_one(self):
        """After MinMaxScaler, maximum value per column must be 1."""
        scaled, _ = normalize_features(SAMPLE_DF, MinMaxScaler())
        assert np.allclose(scaled.max(), 1.0, atol=1e-10)

    def test_values_in_range(self):
        """All values must be in [0, 1]."""
        scaled, _ = normalize_features(SAMPLE_DF, MinMaxScaler())
        assert (scaled.values >= 0.0).all()
        assert (scaled.values <= 1.0).all()


class TestNormalizeFeaturesRobustness:

    def test_type_error_on_non_dataframe(self):
        """Must raise TypeError if input is not a DataFrame."""
        with pytest.raises(TypeError):
            normalize_features([[1, 2], [3, 4]], StandardScaler())

    def test_type_error_on_numpy_array(self):
        """Must raise TypeError even for numpy arrays."""
        with pytest.raises(TypeError):
            normalize_features(np.array([[1, 2], [3, 4]]), StandardScaler())

    def test_fitted_scaler_reusable(self):
        """Fitted scaler must be reusable to transform new data consistently."""
        _, fitted = normalize_features(SAMPLE_DF, StandardScaler())
        new_data = np.array([[20.0, 2.0]])
        transformed = fitted.transform(new_data)
        # feature1=20 is the mean -> should transform to 0.0
        assert pytest.approx(transformed[0, 0], abs=1e-10) == 0.0
