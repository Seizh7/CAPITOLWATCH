# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import pandas as pd
import pytest
from capitolwatch.analysis.feature_engineering import (
    get_sorted_subtypes,
    create_frequency_vectors,
    create_weighted_frequency_vectors,
    compute_numerical_features,
    combine_features,
)


# Minimal fixtures shared across tests

SUBTYPES = ['ETF', 'Stock']

POLITICIANS_DF = pd.DataFrame({'id': ['P1', 'P2']})

ASSETS_DF = pd.DataFrame({
    'politician_id': ['P1', 'P1', 'P1', 'P2'],
    'subtype':       ['Stock', 'Stock', 'ETF', 'Stock'],
    'value_numeric': [1000.0, 2000.0, 500.0, 3000.0],
})


class TestGetSortedSubtypes:

    def test_returns_sorted_list(self):
        """Result must be sorted alphabetically."""
        result = get_sorted_subtypes(ASSETS_DF)
        assert result == sorted(result)

    def test_deterministic_across_calls(self):
        """Two calls on the same DataFrame must produce identical lists."""
        assert get_sorted_subtypes(ASSETS_DF) == get_sorted_subtypes(ASSETS_DF)

    def test_ignores_null_subtypes(self):
        """NaN subtypes must not appear in the result."""
        df = ASSETS_DF.copy()
        df.loc[len(df)] = ['P1', None, 0.0]
        result = get_sorted_subtypes(df)
        assert None not in result
        assert all(isinstance(s, str) for s in result)

    def test_no_duplicates(self):
        """Each subtype must appear exactly once."""
        result = get_sorted_subtypes(ASSETS_DF)
        assert len(result) == len(set(result))


class TestCreateFrequencyVectors:

    def test_correct_counts(self):
        """P1 has 2 Stock and 1 ETF, P2 has 0 ETF and 1 Stock."""
        matrix = create_frequency_vectors(POLITICIANS_DF, ASSETS_DF, SUBTYPES)
        assert matrix.loc['P1', 'ETF'] == 1
        assert matrix.loc['P1', 'Stock'] == 2
        assert matrix.loc['P2', 'ETF'] == 0
        assert matrix.loc['P2', 'Stock'] == 1

    def test_shape(self):
        """Matrix must be (n_politicians, n_subtypes)."""
        matrix = create_frequency_vectors(POLITICIANS_DF, ASSETS_DF, SUBTYPES)
        assert matrix.shape == (len(POLITICIANS_DF), len(SUBTYPES))

    def test_politician_without_assets_is_zero_row(self):
        """A politician with no assets must appear as a row of zeros."""
        politicians = pd.DataFrame({'id': ['P1', 'P2', 'P3']})
        matrix = create_frequency_vectors(politicians, ASSETS_DF, SUBTYPES)
        assert matrix.shape[0] == 3
        assert (matrix.loc['P3'] == 0).all()

    def test_column_order_matches_subtypes(self):
        """Columns must follow the exact order of the subtypes argument."""
        matrix = create_frequency_vectors(POLITICIANS_DF, ASSETS_DF, SUBTYPES)
        assert list(matrix.columns) == SUBTYPES

    def test_no_nan_values(self):
        """Matrix must not contain any NaN."""
        matrix = create_frequency_vectors(POLITICIANS_DF, ASSETS_DF, SUBTYPES)
        assert not matrix.isnull().any().any()


class TestCreateWeightedFrequencyVectors:

    def test_correct_sums(self):
        """P1 ETF value = 500, P1 Stock value = 1000 + 2000 = 3000."""
        matrix = create_weighted_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )
        assert matrix.loc['P1', 'ETF'] == pytest.approx(500.0)
        assert matrix.loc['P1', 'Stock'] == pytest.approx(3000.0)
        assert matrix.loc['P2', 'ETF'] == pytest.approx(0.0)
        assert matrix.loc['P2', 'Stock'] == pytest.approx(3000.0)

    def test_shape(self):
        """Matrix must be (n_politicians, n_subtypes)."""
        matrix = create_weighted_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )
        assert matrix.shape == (len(POLITICIANS_DF), len(SUBTYPES))

    def test_no_nan_values(self):
        """Matrix must not contain any NaN."""
        matrix = create_weighted_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )
        assert not matrix.isnull().any().any()


class TestComputeNumericalFeatures:

    def setup_method(self):
        self.freq_matrix = create_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )

    def test_total_assets(self):
        """P1 has 3 assets total, P2 has 1."""
        features = compute_numerical_features(self.freq_matrix)
        assert features.loc['P1', 'total_assets'] == 3
        assert features.loc['P2', 'total_assets'] == 1

    def test_diversity(self):
        """P1 uses 2 subtypes, P2 uses 1."""
        features = compute_numerical_features(self.freq_matrix)
        assert features.loc['P1', 'diversity'] == 2
        assert features.loc['P2', 'diversity'] == 1

    def test_concentration_fully_concentrated(self):
        """A politician with all assets in one subtype must have H = 1.0."""
        features = compute_numerical_features(self.freq_matrix)
        assert features.loc['P2', 'concentration'] == pytest.approx(1.0)

    def test_concentration_range(self):
        """Herfindahl index must always be in [0, 1]."""
        features = compute_numerical_features(self.freq_matrix)
        assert (features['concentration'] >= 0).all()
        assert (features['concentration'] <= 1.0 + 1e-9).all()

    def test_no_division_by_zero(self):
        """A politician with 0 total assets must not raise an exception."""
        zero_matrix = pd.DataFrame(
            {'ETF': [0], 'Stock': [0]},
            index=pd.Index(['P_empty'], name='politician_id')
        )
        features = compute_numerical_features(zero_matrix)
        assert features.loc['P_empty', 'total_assets'] == 0
        assert not features.isnull().any().any()

    def test_output_columns(self):
        """Output must have exactly the 3 expected columns."""
        features = compute_numerical_features(self.freq_matrix)
        assert set(features.columns) == {
            'total_assets', 'diversity', 'concentration'
        }


class TestCombineFeatures:

    def test_shape(self):
        """Combined matrix must have n_subtypes + 3 columns."""
        freq_matrix = create_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )
        numerical_features = compute_numerical_features(freq_matrix)
        combined = combine_features(freq_matrix, numerical_features)
        assert combined.shape == (len(POLITICIANS_DF), len(SUBTYPES) + 3)

    def test_no_nan_values(self):
        """Combined matrix must not contain any NaN."""
        freq_matrix = create_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )
        numerical_features = compute_numerical_features(freq_matrix)
        combined = combine_features(freq_matrix, numerical_features)
        assert not combined.isnull().any().any()

    def test_index_preserved(self):
        """Index (politician_id) must be preserved after combination."""
        freq_matrix = create_frequency_vectors(
            POLITICIANS_DF, ASSETS_DF, SUBTYPES
        )
        numerical_features = compute_numerical_features(freq_matrix)
        combined = combine_features(freq_matrix, numerical_features)
        assert list(combined.index) == list(freq_matrix.index)
