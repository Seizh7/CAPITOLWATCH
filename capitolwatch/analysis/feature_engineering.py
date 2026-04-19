# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import pandas as pd
from capitolwatch.analysis.data_loader import (
    load_politicians, load_assets_with_products
)


def get_sorted_subtypes(assets_df):
    """
    Extract a sorted list of all unique subtypes.

    Args:
        assets_df (pd.DataFrame): Assets DataFrame with 'subtype' column.

    Returns:
        list[str]: Sorted list of unique subtype names.
    """
    subtypes = assets_df['subtype'].dropna().unique()
    return sorted(subtypes)


def create_frequency_vectors(politicians_df, assets_df, subtypes):
    """
    Build a frequency matrix: count of assets per politician per subtype.

    Args:
        politicians_df (pd.DataFrame): Active politicians [id, ...]
        assets_df (pd.DataFrame): Enriched assets [politician_id, subtype, ...]
        subtypes (list[str]): Ordered list of subtypes (columns)

    Returns:
        pd.DataFrame: Matrix of shape (n_politicians, n_subtypes),
                      indexed by politician_id
    """
    # Count occurrences of each (politician_id, subtype) pair
    freq_matrix = (
        assets_df.groupby(['politician_id', 'subtype'])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all politicians appear
    freq_matrix = freq_matrix.reindex(
        index=politicians_df['id'],
        fill_value=0
    )

    # Ensure all subtypes appear as columns in deterministic order
    freq_matrix = freq_matrix.reindex(
        columns=subtypes,
        fill_value=0
    )

    freq_matrix.index.name = 'politician_id'
    freq_matrix.columns.name = None
    return freq_matrix


def create_weighted_frequency_vectors(politicians_df, assets_df, subtypes):
    """
    Build a weighted matrix: sum of asset values per politician per subtype.

    Args:
        politicians_df (pd.DataFrame): Active politicians [id, ...]
        assets_df (pd.DataFrame): Enriched assets [politician_id, subtype,
            value_numeric]
        subtypes (list[str]): Ordered list of subtypes (columns)

    Returns:
        pd.DataFrame: Matrix of shape (n_politicians, n_subtypes),
                      indexed by politician_id
    """
    # Sum value_numeric per (politician_id, subtype) pair
    weighted_matrix = (
        assets_df.groupby(['politician_id', 'subtype'])['value_numeric']
        .sum()
        .unstack(fill_value=0)
    )

    # Ensure all politicians appear
    weighted_matrix = weighted_matrix.reindex(
        index=politicians_df['id'],
        fill_value=0
    )

    # Ensure all subtypes appear as columns in deterministic order
    weighted_matrix = weighted_matrix.reindex(
        columns=subtypes,
        fill_value=0
    )

    weighted_matrix.index.name = 'politician_id'
    weighted_matrix.columns.name = None
    return weighted_matrix


def compute_numerical_features(freq_matrix):
    """
    Compute 3 summary features per politician:
        - total_assets     : total number of assets
        - diversity        : number of distinct subtypes used
        - concentration    : Herfindahl index (sum of squared proportions)

    Args:
        freq_matrix (pd.DataFrame): Frequency matrix

    Returns:
        pd.DataFrame: Shape (n_politicians, 3), indexed by politician_id
    """
    # Sum of each row in freq_matrix
    total_assets = freq_matrix.sum(axis=1)

    # Count of non-zero values per row
    diversity = (freq_matrix > 0).sum(axis=1)

    # Herfindahl index: sum of squared proportions
    # Replace 0 total with 1 to avoid division by zero
    safe_total = total_assets.replace(0, 1)
    proportions = freq_matrix.div(safe_total, axis=0)
    concentration = (proportions ** 2).sum(axis=1)

    numerical_features = pd.DataFrame({
        'total_assets': total_assets,
        'diversity': diversity,
        'concentration': concentration
    }, index=freq_matrix.index)

    return numerical_features


def combine_features(freq_matrix, numerical_features):
    """
    Concatenate frequency matrix and numerical features into a single matrix.

    Args:
        freq_matrix (pd.DataFrame): Shape (n_politicians, n_subtypes)
        numerical_features (pd.DataFrame): Shape (n_politicians, 3)

    Returns:
        pd.DataFrame: Combined matrix (n_politicians, n_subtypes + 3)
    """
    combined_matrix = pd.concat([freq_matrix, numerical_features], axis=1)
    return combined_matrix


def analyze_sparsity(matrix, name="matrix"):
    """
    Compute and print sparsity statistics of a feature matrix.

    Args:
        matrix (pd.DataFrame): Feature matrix to analyze.
        name (str): Label for display purposes.
    """
    # Total number of values
    total_cells = matrix.size
    # Number of zero values
    zero_cells = (matrix.values == 0).sum()
    # Percentage of zeros
    sparsity = (zero_cells / total_cells) * 100

    print(f"{name}: {matrix.shape}, sparsity = {sparsity:.1f}%")


if __name__ == "__main__":
    # Load data
    politicians = load_politicians()
    assets = load_assets_with_products()

    # Get sorted subtypes
    subtypes = get_sorted_subtypes(assets)
    print(f"Unique subtypes : ({len(subtypes)})\n")
    print(f"10 first subtypes :\n{subtypes[:10]}\n")

    # Create frequency vectors
    freq_matrix = create_frequency_vectors(politicians, assets, subtypes)
    print(f"Frequency matrix shape: {freq_matrix.shape}")
    print(freq_matrix.head(), "\n")

    # Create weighted frequency vectors
    weighted_freq_matrix = create_weighted_frequency_vectors(
        politicians, assets, subtypes
    )
    print(f"Weighted frequency matrix shape: {weighted_freq_matrix.shape}")
    print(weighted_freq_matrix.head(), "\n")

    # Compute numerical features
    numerical_features = compute_numerical_features(freq_matrix)
    print(f"Numerical features shape: {numerical_features.shape}")
    print(numerical_features.head(), "\n")

    # Combine all features
    combined_features = combine_features(freq_matrix, numerical_features)
    print(f"Combined feature matrix shape: {combined_features.shape}")
    print(combined_features.head())

    # Analyze sparsity
    analyze_sparsity(freq_matrix, name="Frequency matrix")
    analyze_sparsity(weighted_freq_matrix, name="Weighted frequency matrix")
