# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Feature store: serialization and loading of computed feature matrices.

All matrices are saved as raw DataFrames.

Store layout:
    data/feature_store/
    ├── freq_baseline.pkl      -- subtyp frequency vectors + numerical features
    ├── freq_weighted.pkl      -- subtype weighted vectors + numerical features
    ├── sector_baseline.pkl    -- sector frequency vectors + numerical features
    └── politician_labels.pkl  -- id, first_name, last_name, party

Main functions:
    - build_feature_store() : compute all matrices and persist them
    - load_features(feature_type) : load one matrix from disk
"""

import joblib
from pathlib import Path

from capitolwatch.analysis.data_loader import (
    load_politicians, load_assets_with_products
)
from capitolwatch.analysis.feature_engineering import (
    get_sorted_subtypes,
    get_sorted_sectors,
    create_frequency_vectors,
    create_weighted_frequency_vectors,
    create_sector_frequency_vectors,
    compute_numerical_features,
    combine_features,
)

FEATURE_STORE_DIR = Path("data/feature_store")

FEATURE_FILES = {
    "freq_baseline": FEATURE_STORE_DIR / "freq_baseline.pkl",
    "freq_weighted": FEATURE_STORE_DIR / "freq_weighted.pkl",
    "sector_baseline": FEATURE_STORE_DIR / "sector_baseline.pkl",
    "politician_labels": FEATURE_STORE_DIR / "politician_labels.pkl",
}


def _save(obj, path):
    """
    Serialize obj to disk using joblib.

    Args:
        obj: Any Python object (DataFrame, dict, …).
        path (Path): Destination file path.
    """
    joblib.dump(obj, path)


def build_feature_store():
    """
    Compute all feature matrices and store them to data/feature_store/.

    Steps:
        1. Load politicians + assets from the database
        2. Build freq_baseline (subtype frequency vectors + numerical features)
        3. Build freq_weighted (subtype weighted vectors + numerical features)
        4. Build sectorbaseline (sector frequency vectors + numerical features)
        5. Save each matrix as a .pkl file
        6. Save politician_labels (id, first_name, last_name, party)
    """
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 – load raw data
    politicians = load_politicians()
    assets = load_assets_with_products()
    subtypes = get_sorted_subtypes(assets)
    sectors = get_sorted_sectors(assets)

    # Step 2 – freq_baseline
    freq_matrix = create_frequency_vectors(politicians, assets, subtypes)
    numerical_features = compute_numerical_features(freq_matrix)
    freq_baseline_matrix = combine_features(freq_matrix, numerical_features)

    # Step 3 – freq_weighted
    weighted_matrix = create_weighted_frequency_vectors(
        politicians, assets, subtypes
    )
    freq_weighted_matrix = combine_features(
        weighted_matrix, numerical_features
    )

    # Step 4 – sector_baseline (sector frequency counts + numerical features)
    sector_matrix = create_sector_frequency_vectors(
        politicians, assets, sectors
    )
    sector_baseline_matrix = combine_features(
        sector_matrix, numerical_features
    )

    # Step 5 – persist feature matrices
    _save(freq_baseline_matrix, FEATURE_FILES["freq_baseline"])
    _save(freq_weighted_matrix, FEATURE_FILES["freq_weighted"])
    _save(sector_baseline_matrix, FEATURE_FILES["sector_baseline"])

    # Step 6 – persist politician labels
    politician_labels = politicians[
        ['id', 'first_name', 'last_name', 'party']
    ].copy()
    _save(politician_labels, FEATURE_FILES["politician_labels"])


def load_features(feature_type):
    """
    Load a feature matrix from the store.

    Args:
        feature_type (str): One of "freq_baseline", "freq_weighted",
                            "politician_labels".

    Returns:
        pd.DataFrame: The stored feature matrix.
    """
    if feature_type not in FEATURE_FILES:
        raise KeyError(
            f"Unknown feature type '{feature_type}'. "
            f"Available: {list(FEATURE_FILES.keys())}"
        )

    path = FEATURE_FILES[feature_type]
    if not path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {path}. "
            "Run build_feature_store() first."
        )

    return joblib.load(path)


if __name__ == "__main__":
    # Build the store, then verify by loading each matrix
    build_feature_store()

    for feature_type in ["freq_baseline", "freq_weighted", "sector_baseline"]:
        matrix = load_features(feature_type)
        print(f"{feature_type}: {matrix.shape}")

    labels = load_features("politician_labels")
    print(f"politician_labels: {labels.shape}")
    print(labels.head())
