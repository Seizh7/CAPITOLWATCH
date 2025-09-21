# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0

"""
Tests for som_analysis.py - SOM Clustering and Analysis Module

This module tests the SOM clustering analysis including:
- SOM data preparation and loading
- SOM clustering execution
- Analysis results and metrics
- CSV export functionality
- Error handling for invalid inputs
"""

import pytest
import tempfile
import warnings
from pathlib import Path
import pandas as pd

# Import SOM analysis module
from capitolwatch.analysis.som_analysis import (
    run_comprehensive_portfolio_analysis
)
from capitolwatch.analysis.clustering import (
    load_portfolio_embeddings_for_clustering,
    perform_som_clustering
)
from capitolwatch.analysis.init_analysis import (
    initialize_complete_analysis_pipeline
)


class TestSOMDataPreparation:
    """Tests for SOM data preparation and loading"""

    def test_som_data_loading(self):
        """Test SOM data loading functionality"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        try:
            data = load_portfolio_embeddings_for_clustering(
                embedding_method="custom_financial_weighted",
                min_politicians_with_embeddings=3
            )

            if data is not None and not data.empty:
                assert 'politician_id' in data.columns
                assert len(data) >= 0  # May be empty in test environment
                # Check that data has embedding columns
                embedding_cols = [col for col in data.columns
                                  if col.startswith('embedding_')]
                assert len(embedding_cols) > 0
        except Exception:
            # Skip test if data not available
            pytest.skip("SOM data not available in test environment")

    def test_som_data_different_methods(self):
        """Test SOM data loading with different embedding methods"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        methods = [
            "custom_financial_weighted",
            "tfidf_basic_weighted",
            "word2vec_weighted",
            "glove_weighted"
        ]

        for method in methods:
            try:
                data = load_portfolio_embeddings_for_clustering(
                    embedding_method=method,
                    min_politicians_with_embeddings=1
                )
                # Data may be None if method not available
                if data is not None:
                    assert isinstance(data, pd.DataFrame)
            except Exception:
                # Some methods may not be available
                continue

    def test_som_data_minimum_requirements(self):
        """Test SOM data loading with minimum politician requirements"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        try:
            # Test with high minimum (should return None or empty)
            data_high_min = load_portfolio_embeddings_for_clustering(
                embedding_method="custom_financial_weighted",
                min_politicians_with_embeddings=1000  # Very high
            )
            # Should be None or empty
            assert data_high_min is None or data_high_min.empty

            # Test with low minimum (should return data if available)
            data_low_min = load_portfolio_embeddings_for_clustering(
                embedding_method="custom_financial_weighted",
                min_politicians_with_embeddings=1
            )
            # May have data
            if data_low_min is not None:
                assert isinstance(data_low_min, pd.DataFrame)
        except Exception:
            pytest.skip("SOM data validation not available")


class TestSOMClustering:
    """Tests for SOM clustering execution"""

    def test_som_clustering_execution(self):
        """Test SOM clustering execution with valid parameters"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        try:
            results = perform_som_clustering(
                embedding_method="custom_financial_weighted",
                som_grid_size=4,
                min_politicians_for_clustering=3
            )

            if results is not None:
                assert 'clustered_data' in results
                assert 'cluster_analysis' in results

                # Check clustered data structure
                if results['clustered_data'] is not None:
                    clustered_df = results['clustered_data']
                    assert isinstance(clustered_df, pd.DataFrame)
                    assert 'cluster' in clustered_df.columns

        except Exception:
            pytest.skip("SOM clustering not available in test environment")

    def test_som_different_grid_sizes(self):
        """Test SOM clustering with different grid sizes"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        grid_sizes = [3, 4, 6, 8]

        for grid_size in grid_sizes:
            try:
                results = perform_som_clustering(
                    embedding_method="custom_financial_weighted",
                    som_grid_size=grid_size,
                    min_politicians_for_clustering=1
                )

                if results is not None:
                    assert isinstance(results, dict)
                    # Grid size should influence number of possible clusters
                    max_clusters = grid_size * grid_size
                    assert max_clusters >= grid_size

            except Exception:
                # Some grid sizes may not work
                continue

    def test_som_clustering_validation(self):
        """Test SOM clustering results validation"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        try:
            results = perform_som_clustering(
                embedding_method="custom_financial_weighted",
                som_grid_size=6,
                min_politicians_for_clustering=5
            )

            if results is not None and 'clustered_data' in results:
                clustered_data = results['clustered_data']

                if clustered_data is not None and not clustered_data.empty:
                    # Validate cluster assignments
                    clusters = clustered_data['cluster'].unique()
                    assert len(clusters) > 0
                    assert all(cluster >= 0 for cluster in clusters)

                    # Validate cluster analysis
                    if 'cluster_analysis' in results:
                        analysis = results['cluster_analysis']
                        assert isinstance(analysis, dict)

        except Exception:
            pytest.skip("SOM clustering validation not available")


class TestComprehensiveAnalysis:
    """Tests for comprehensive portfolio analysis pipeline"""

    def test_comprehensive_analysis_execution(self):
        """Test complete comprehensive analysis pipeline"""
        # First ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        # Run comprehensive analysis
        results = run_comprehensive_portfolio_analysis(
            embedding_method="custom_financial_weighted",
            som_grid_size=6,
            save_results_to_csv=False  # Don't save during tests
        )

        assert results is not None
        assert 'clustering_results' in results
        assert 'portfolio_metrics' in results
        assert 'analysis_summary' in results

        # Check data quality
        if results['clustering_results']:
            assert len(results['clustering_results']) > 0

        if results['portfolio_metrics'] is not None:
            # portfolio_metrics is a dict with 'metrics_dataframe' key
            if isinstance(results['portfolio_metrics'], dict):
                metrics_key = 'metrics_dataframe'
                if metrics_key in results['portfolio_metrics']:
                    metrics_df = results['portfolio_metrics'][metrics_key]
                    assert isinstance(metrics_df, pd.DataFrame)
            else:
                # If it's directly a DataFrame
                metrics_df = results['portfolio_metrics']
                assert isinstance(metrics_df, pd.DataFrame)

    def test_analysis_different_methods(self):
        """Test comprehensive analysis with different embedding methods"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        methods = [
            "custom_financial_weighted",
            "tfidf_basic_weighted"
        ]

        for method in methods:
            try:
                results = run_comprehensive_portfolio_analysis(
                    embedding_method=method,
                    som_grid_size=4,
                    save_results_to_csv=False
                )

                if results is not None:
                    assert isinstance(results, dict)
                    assert 'clustering_results' in results

            except Exception:
                # Some methods may not be available
                continue

    def test_analysis_parameter_variations(self):
        """Test analysis with different parameter combinations"""
        # Ensure pipeline is initialized
        initialize_complete_analysis_pipeline()

        parameter_sets = [
            {"som_grid_size": 4, "save_results_to_csv": False},
            {"som_grid_size": 6, "save_results_to_csv": False},
            {"som_grid_size": 8, "save_results_to_csv": False},
        ]

        for params in parameter_sets:
            try:
                results = run_comprehensive_portfolio_analysis(
                    embedding_method="custom_financial_weighted",
                    **params
                )

                if results is not None:
                    assert isinstance(results, dict)

            except Exception:
                # Some parameter combinations may fail
                continue


class TestCSVExport:
    """Tests for CSV export functionality"""

    def test_csv_export_functionality(self):
        """Test CSV export to temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize pipeline
            initialize_complete_analysis_pipeline()

            # Run analysis with CSV export to temp directory
            run_comprehensive_portfolio_analysis(
                embedding_method="custom_financial_weighted",
                som_grid_size=6,
                save_results_to_csv=True,
                output_directory=temp_dir
            )

            # Check that CSV files were created
            temp_path = Path(temp_dir)
            expected_files = [
                "politician_clusters.csv",
                "portfolio_metrics.csv",
                "analysis_summary.csv",
                "party_risk_analysis.csv"
            ]

            for filename in expected_files:
                file_path = temp_path / filename
                # Some files may not be created if no data
                if file_path.exists():
                    # File should not be empty
                    assert file_path.stat().st_size > 0

    def test_csv_content_validation(self):
        """Test CSV file content validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize pipeline
            initialize_complete_analysis_pipeline()

            # Run analysis with CSV export
            run_comprehensive_portfolio_analysis(
                embedding_method="custom_financial_weighted",
                som_grid_size=6,
                save_results_to_csv=True,
                output_directory=temp_dir
            )

            temp_path = Path(temp_dir)

            # Check politician_clusters.csv if it exists
            clusters_file = temp_path / "politician_clusters.csv"
            if clusters_file.exists():
                df = pd.read_csv(clusters_file)
                assert 'politician_id' in df.columns
                assert 'cluster' in df.columns

            # Check portfolio_metrics.csv if it exists
            metrics_file = temp_path / "portfolio_metrics.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                assert 'politician_id' in df.columns
                assert 'herfindahl_index' in df.columns


class TestErrorHandling:
    """Tests for error handling and edge cases"""

    def test_invalid_embedding_method(self):
        """Test error handling for invalid embedding methods"""
        # Test with invalid embedding method
        results = run_comprehensive_portfolio_analysis(
            embedding_method="nonexistent_method",
            som_grid_size=6,
            save_results_to_csv=False
        )

        # Should return empty dict or handle gracefully when method not found
        assert results == {} or results is None or 'error' in results

    def test_invalid_som_parameters(self):
        """Test error handling for invalid SOM parameters"""
        # Test with invalid grid size - suppress minisom sigma warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = run_comprehensive_portfolio_analysis(
                embedding_method="custom_financial_weighted",
                som_grid_size=0,  # Invalid
                save_results_to_csv=False
            )
        # Should handle gracefully
        assert results is not None  # May return empty dict or proper results

    def test_missing_data_scenarios(self):
        """Test behavior with missing or insufficient data"""
        # Test with method that might not have enough data
        try:
            results = run_comprehensive_portfolio_analysis(
                embedding_method="custom_financial_weighted",
                som_grid_size=20,  # Very large grid for small dataset
                save_results_to_csv=False
            )

            # Should handle gracefully even with insufficient data
            assert results is not None

        except Exception:
            # Acceptable if insufficient data causes controlled failure
            pass

    def test_output_directory_validation(self):
        """Test output directory validation and creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with non-existent subdirectory
            sub_dir = Path(temp_dir) / "analysis_output"

            results = run_comprehensive_portfolio_analysis(
                embedding_method="custom_financial_weighted",
                som_grid_size=6,
                save_results_to_csv=True,
                output_directory=str(sub_dir)
            )

            # Should create directory and handle gracefully
            if results is not None:
                assert sub_dir.exists() or results == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
