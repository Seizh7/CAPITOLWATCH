# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Integration tests for analysis pipeline.

These tests exercise the full pipeline from raw SQLite data to evaluated
clustering labels.

Pipeline under test:
    SQLite DB → data_loader → feature_engineering → preprocessing
              → clustering algorithm → evaluation metrics
"""

import sqlite3
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@pytest.fixture()
def minimal_db(tmp_path: Path) -> Path:
    """
    Build a minimal SQLite database with enough rows to run clustering.

    Creates 15 politicians spread across 3 parties, each with
    several assets of different subtypes, so clustering algorithms can
    find at least 2 clusters.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        Path: Absolute path to the SQLite file.
    """
    db_path = tmp_path / "test_capitolwatch.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    # Create schema (mirrors init_db.py)
    cur.executescript("""
        CREATE TABLE politicians (
            id VARCHAR(7) PRIMARY KEY,
            last_name TEXT,
            first_name TEXT,
            party TEXT
        );
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            politician_id VARCHAR(7),
            source_file TEXT,
            year INTEGER,
            url TEXT,
            import_timestamp TEXT,
            checksum TEXT,
            encoding TEXT,
            FOREIGN KEY (politician_id) REFERENCES politicians(id)
        );
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            subtype TEXT
        );
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            product_id INTEGER,
            value TEXT,
            income_type TEXT,
            income TEXT,
            FOREIGN KEY (report_id) REFERENCES reports(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
    """)

    # Insert 15 politicians (6R, 6D, 3I) to ensure non-trivial clustering
    politicians = [
        ("P001", "Smith",    "Alice",   "Republican"),
        ("P002", "Johnson",  "Bob",     "Republican"),
        ("P003", "Williams", "Carol",   "Republican"),
        ("P004", "Brown",    "David",   "Republican"),
        ("P005", "Jones",    "Emily",   "Republican"),
        ("P006", "Garcia",   "Frank",   "Republican"),
        ("P007", "Miller",   "Grace",   "Democrat"),
        ("P008", "Davis",    "Henry",   "Democrat"),
        ("P009", "Wilson",   "Iris",    "Democrat"),
        ("P010", "Martinez", "Jack",    "Democrat"),
        ("P011", "Anderson", "Kate",    "Democrat"),
        ("P012", "Taylor",   "Liam",    "Democrat"),
        ("P013", "Thomas",   "Mia",     "Independent"),
        ("P014", "Hernandez", "Noah",   "Independent"),
        ("P015", "Moore",    "Olivia",  "Independent"),
    ]
    cur.executemany(
        "INSERT INTO politicians VALUES (?, ?, ?, ?)",
        politicians,
    )

    # Insert common products with distinct subtypes
    subtypes = ["Stock", "Mutual Fund", "ETF", "Bond", "Real Estate"]
    products = [(f"Product {i}", "Financial", subtypes[i % len(subtypes)])
                for i in range(10)]
    cur.executemany(
        "INSERT INTO products (name, type, subtype) VALUES (?,?,?)", products
    )

    # Insert one report per politician
    for i, (pid, *_) in enumerate(politicians):
        cur.execute(
            "INSERT INTO reports"
            " (politician_id, source_file, year) VALUES (?,?,?)",
            (pid, f"report_{i}.html", 2023),
        )

    # Insert assets: Republicans heavy on Stock+ETF, Democrats on Mutual
    # Fund+Bond
    # This creates two distinguishable clusters
    report_bias = {
        "P001": [1, 1, 1, 3, 3],  # Stock + ETF
        "P002": [1, 1, 3, 3, 3],
        "P003": [1, 1, 1, 3, 4],
        "P004": [1, 3, 3, 3, 3],
        "P005": [1, 1, 3, 3, 3],
        "P006": [1, 1, 1, 3, 3],
        "P007": [2, 2, 2, 4, 4],  # Mutual Fund + Bond
        "P008": [2, 2, 4, 4, 4],
        "P009": [2, 2, 2, 4, 5],
        "P010": [2, 4, 4, 4, 4],
        "P011": [2, 2, 4, 4, 4],
        "P012": [2, 2, 2, 4, 4],
        "P013": [1, 2, 3, 4, 5],  # Mixed
        "P014": [1, 2, 3, 4, 5],
        "P015": [2, 3, 4, 5, 5],
    }
    # Map politician id to their report id (inserted in order, ids 1..15)
    pid_to_report = {pid: i + 1 for i, (pid, *_) in enumerate(politicians)}
    for pid, product_ids in report_bias.items():
        report_id = pid_to_report[pid]
        for prod_id in product_ids:
            cur.execute(
                "INSERT INTO assets"
                " (report_id, product_id, value) VALUES (?,?,?)",
                (report_id, prod_id, "$1,001 - $15,000"),
            )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def patched_config(minimal_db: Path, monkeypatch):
    """
    Override CONFIG.db_path with the temporary database.

    Args:
        minimal_db (Path): Path to the test SQLite database.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        object: The patched config instance.
    """
    from config import CONFIG
    monkeypatch.setattr(CONFIG, "db_path", minimal_db)
    return CONFIG


# 1: data_loader

class TestDataLoaderIntegration:
    """Validate that data_loader reads from the real (patched) database."""

    def test_load_politicians_returns_dataframe(self, patched_config):
        """load_politicians() should return a non-empty DataFrame."""
        from capitolwatch.analysis.data_loader import load_politicians

        df = load_politicians()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15
        assert set(df.columns) >= {"id", "first_name", "last_name", "party"}

    def test_load_politicians_party_distribution(self, patched_config):
        """Party column should contain the three expected parties."""
        from capitolwatch.analysis.data_loader import load_politicians

        df = load_politicians()
        parties = set(df["party"].unique())

        assert "Republican" in parties
        assert "Democrat" in parties
        assert "Independent" in parties

    def test_load_assets_returns_dataframe(self, patched_config):
        """load_assets_with_products() should return enriched assets."""
        from capitolwatch.analysis.data_loader import load_assets_with_products

        df = load_assets_with_products()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "subtype" in df.columns
        assert "value_numeric" in df.columns

    def test_load_assets_no_empty_subtype(self, patched_config):
        """Empty subtypes must be normalized to 'Uncategorized'."""
        from capitolwatch.analysis.data_loader import load_assets_with_products

        df = load_assets_with_products()

        # Empty strings and NaN must not remain
        assert not df["subtype"].isna().any()
        assert not (df["subtype"] == "").any()

    def test_load_assets_value_numeric_positive(self, patched_config):
        """value_numeric should be >= 0 for all rows."""
        from capitolwatch.analysis.data_loader import load_assets_with_products

        df = load_assets_with_products()

        assert (df["value_numeric"] >= 0).all()


# 2: feature_engineering

class TestFeatureEngineeringIntegration:
    """Check that feature matrices have the correct shape and types."""

    def test_frequency_matrix_shape(self, patched_config):
        """Frequency matrix should have one row per politician."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
        )

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        matrix = create_frequency_vectors(politicians, assets, subtypes)

        assert matrix.shape[0] == len(politicians)
        assert matrix.shape[1] == len(subtypes)

    def test_weighted_matrix_non_negative(self, patched_config):
        """Weighted matrix entries should all be >= 0."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_weighted_frequency_vectors,
        )

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        matrix = create_weighted_frequency_vectors(
            politicians, assets, subtypes
        )

        assert (matrix.values >= 0).all()

    def test_combine_features_includes_numerical(self, patched_config):
        """combine_features() should append numerical columns."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes,
            create_frequency_vectors,
            compute_numerical_features,
            combine_features,
        )

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)

        # The combined matrix must be wider than the frequency matrix alone
        assert combined.shape[1] > freq.shape[1]
        assert combined.shape[0] == len(politicians)


# 3: preprocessing

class TestPreprocessingIntegration:
    """Verify normalize_features preserves shape and produces valid output."""

    def test_standard_scaler_preserves_shape(self, patched_config):
        """StandardScaler must return the same shape as the input."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)

        scaled, _ = normalize_features(combined, StandardScaler())

        assert scaled.shape == combined.shape

    def test_minmax_scaler_range(self, patched_config):
        """MinMaxScaler output must be in [0, 1]."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)

        scaled, _ = normalize_features(combined, MinMaxScaler())

        values = scaled.values
        assert values.min() >= -1e-9  # allow floating-point tolerance
        assert values.max() <= 1 + 1e-9


# 4a: KMeans end-to-end

class TestKMeansPipeline:
    """Full KMeans pipeline: DB → features → fit → evaluate."""

    def test_kmeans_produces_valid_labels(self, patched_config):
        """KMeansClusterer should assign an integer label to every sample."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features
        from capitolwatch.analysis.clustering.kmeans import KMeansClusterer

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)
        scaled, _ = normalize_features(combined, StandardScaler())

        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer.fit(scaled.values)

        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(politicians)
        # Labels must be integers in [0, n_clusters-1]
        assert set(clusterer.labels_).issubset({0, 1, 2})

    def test_kmeans_evaluation_returns_finite_metrics(self, patched_config):
        """evaluate_clustering() should return finite metrics for KMeans."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features
        from capitolwatch.analysis.clustering.kmeans import KMeansClusterer
        from capitolwatch.analysis.evaluation import evaluate_clustering

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)
        scaled, _ = normalize_features(combined, StandardScaler())

        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer.fit(scaled.values)

        result = evaluate_clustering(
            scaled.values, clusterer.labels_, "kmeans", "freq_baseline"
        )

        assert result["algo_name"] == "kmeans"
        assert result["n_clusters"] == 3
        assert np.isfinite(result["silhouette"])


# 4b: DBSCAN end-to-end

class TestDBSCANPipeline:
    """Full DBSCAN pipeline: DB → features → fit → evaluate."""

    def test_dbscan_assigns_labels(self, patched_config):
        """DBSCANClusterer.fit() should assign labels to all samples."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features
        from capitolwatch.analysis.clustering.dbscan import DBSCANClusterer

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)
        scaled, _ = normalize_features(combined, StandardScaler())

        clusterer = DBSCANClusterer(eps=1.0, min_samples=2, metric="euclidean")
        clusterer.fit(scaled.values)

        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(politicians)
        # Labels must be integers >= -1
        assert all(label >= -1 for label in clusterer.labels_)

    def test_dbscan_n_clusters_consistent(self, patched_config):
        """n_clusters_ must equal the count of unique labels excluding -1."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features
        from capitolwatch.analysis.clustering.dbscan import DBSCANClusterer

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)
        scaled, _ = normalize_features(combined, StandardScaler())

        clusterer = DBSCANClusterer(eps=1.0, min_samples=2, metric="euclidean")
        clusterer.fit(scaled.values)

        expected = len(set(clusterer.labels_) - {-1})
        assert clusterer.n_clusters_ == expected


# 4c: SOM end-to-end

class TestSOMPipeline:
    """Full SOM pipeline: DB → features → fit → extract_clusters → evaluate."""

    def test_som_assigns_labels_after_extract(self, patched_config):
        """SOM labels must be set after calling extract_clusters()."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features
        from capitolwatch.analysis.clustering.som import SOMClusterer

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)
        # SOM requires MinMaxScaler
        scaled, _ = normalize_features(combined, MinMaxScaler())

        clusterer = SOMClusterer(m=3, n=3, n_iterations=100, random_seed=42)
        clusterer.fit(scaled.values)
        clusterer.extract_clusters(n_clusters=3)

        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(politicians)

    def test_som_bmu_coords_shape(self, patched_config):
        """bmu_coords_ must have one entry per politician after fit()."""
        from capitolwatch.analysis.data_loader import (
            load_politicians, load_assets_with_products,
        )
        from capitolwatch.analysis.feature_engineering import (
            get_sorted_subtypes, create_frequency_vectors,
            compute_numerical_features, combine_features,
        )
        from capitolwatch.analysis.preprocessing import normalize_features
        from capitolwatch.analysis.clustering.som import SOMClusterer

        politicians = load_politicians()
        assets = load_assets_with_products()
        subtypes = get_sorted_subtypes(assets)
        freq = create_frequency_vectors(politicians, assets, subtypes)
        numerical = compute_numerical_features(freq)
        combined = combine_features(freq, numerical)
        scaled, _ = normalize_features(combined, MinMaxScaler())

        clusterer = SOMClusterer(m=3, n=3, n_iterations=100, random_seed=42)
        clusterer.fit(scaled.values)

        assert clusterer.bmu_coords_ is not None
        assert len(clusterer.bmu_coords_) == len(politicians)


# 5: External metrics

class TestExternalMetrics:
    """Verify external metrics run without errors on real data."""

    def test_ari_returns_float(self, patched_config):
        """calculate_ari() should return a float between -1 and 1."""
        from capitolwatch.analysis.data_loader import load_politicians
        from capitolwatch.analysis.evaluation import calculate_ari

        politicians = load_politicians()
        # Use party as ground truth: encode to integer
        party_map = {p: i for i, p in enumerate(politicians["party"].unique())}
        labels_true = politicians["party"].map(party_map).values
        # Simulate predicted labels
        labels_pred = np.zeros(len(politicians), dtype=int)
        labels_pred[len(politicians) // 2:] = 1

        ari = calculate_ari(labels_true, labels_pred)

        assert isinstance(ari, float)
        assert -1.0 <= ari <= 1.0

    def test_nmi_returns_float(self, patched_config):
        """calculate_nmi() should return a float in [0, 1]."""
        from capitolwatch.analysis.data_loader import load_politicians
        from capitolwatch.analysis.evaluation import calculate_nmi

        politicians = load_politicians()
        party_map = {p: i for i, p in enumerate(politicians["party"].unique())}
        labels_true = politicians["party"].map(party_map).values
        labels_pred = np.zeros(len(politicians), dtype=int)
        labels_pred[len(politicians) // 2:] = 1

        nmi = calculate_nmi(labels_true, labels_pred)

        assert isinstance(nmi, float)
        assert 0.0 <= nmi <= 1.0
