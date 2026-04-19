# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
This module loads data from the SQLite database and prepares it
for clustering analysis.

Main functions:
    - load_politicians() : Load politicians with at least 1 asset
    - load_assets_with_products() : Load enriched assets with product
    informations
    - parse_value_range() : Parse financial value ranges
"""

import re
from config import CONFIG
from capitolwatch.services.analytics import get_active_politicians_dataframe


def parse_value_range(value_str):
    """
    Parse a financial value range and return the arithmetic mean.

    Supported formats:
        - "$1,001 - $15,000" : 8000.50 (mean)
        - "$50,000,001+" : 50000001.0
        - None / "" : 0.0
        - "None (or less than $201)" : 201.0

    Args:
        value_str (str): String representing a value range

    Returns:
        float: Arithmetic mean of the range, or 0 if invalid
    """
    # NULL or empty values
    if not value_str or value_str.strip() == "":
        return 0.0

    # "None (or less than $201)"
    if "less than" in value_str:
        # Extract the threshold value
        threshold_match = re.search(r"\$(\d+[,\d]*)", value_str)
        if threshold_match:
            threshold = float(threshold_match.group(1).replace(",", ""))
            return threshold
        return 0.0

    # "None" without threshold
    if "None" in value_str:
        return 0.0

    cleaned = value_str.replace("$", "").replace(",", "").strip()

    range_pattern = r"(\d+)\s*-\s*(\d+)"

    # If value is in range format
    match = re.search(range_pattern, cleaned)
    if match:
        min_value = float(match.group(1))
        max_value = float(match.group(2))
        return (min_value + max_value) / 2

    # If value is in "plus" format
    plus_pattern = r"(\d+)\+"
    match_plus = re.search(plus_pattern, cleaned)
    if match_plus:
        return float(match_plus.group(1))

    return 0.0


def load_politicians():
    """
    Load active politicians (those with at least 1 asset).

    Returns:
        pd.DataFrame: DataFrame [id, first_name, last_name, party]
    """

    df = get_active_politicians_dataframe(config=CONFIG)

    # Validation: check structure and content
    if len(df) == 0:
        raise ValueError("No active politicians found in database")

    required_columns = ['id', 'first_name', 'last_name', 'party']
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise AssertionError(f"Missing required columns: {missing_columns}")

    return df


def load_assets_with_products():
    """
    Load enriched assets with product information.

    Performs INNER JOIN between:
        - assets table (transactions/holdings)
        - products table (financial products)
        - politicians table (to filter active assets)

    Returns:
        pd.DataFrame: Enriched DataFrame with columns:
                     [asset_id, politician_id, product_id, value,
                      value_numeric, product_name, subtype, sector]
    """
    from capitolwatch.services.analytics import (
        get_assets_with_products_dataframe
    )

    df = get_assets_with_products_dataframe(config=CONFIG)

    # Validate dataset is not empty
    if len(df) == 0:
        raise ValueError("No assets found in database")

    # Normalize empty subtypes to 'Uncategorized'
    df['subtype'] = (df['subtype']
                     .fillna('Uncategorized')
                     .replace('', 'Uncategorized'))

    # Normalize empty sectors to 'Uncategorized'
    df['sector'] = (df['sector']
                    .fillna('Uncategorized')
                    .replace('', 'Uncategorized'))

    # Creates a numeric representation of value ranges for analysis
    df['value_numeric'] = df['value'].apply(parse_value_range)

    # Validation: check required columns exist
    required_columns = [
        'asset_id', 'politician_id', 'product_id',
        'value', 'product_name', 'subtype', 'sector', 'value_numeric'
    ]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise AssertionError(f"Missing required columns: {missing_columns}")

    return df


def get_dataset_summary():
    """
    Return a statistical summary of the loaded dataset.

    Useful for verifying data quality before clustering.

    Returns:
        dict: Dictionary with key statistics:
              - n_politicians: Number of politicians
              - n_assets: Total number of assets
              - n_unique_subtypes: Number of unique subtypes
              - avg_assets_per_politician: Average assets per politician
              - party_distribution: Distribution by party
              - top_subtypes: Most common asset subtypes
              - total_value: Total portfolio value
              - mean_value: Mean asset value
              - median_value: Median asset value
    """
    politicians = load_politicians()
    assets = load_assets_with_products()

    summary = {
        'n_politicians': len(politicians),
        'n_assets': len(assets),
        'n_unique_subtypes': assets['subtype'].nunique(),
        'avg_assets_per_politician': len(assets) / len(politicians),
        'party_distribution': politicians['party'].value_counts().to_dict(),
        'top_subtypes': assets['subtype'].value_counts().head(10).to_dict(),
        'total_value': assets['value_numeric'].sum(),
        'mean_value': assets['value_numeric'].mean(),
        'median_value': assets['value_numeric'].median(),
    }

    return summary


if __name__ == "__main__":
    # Test 1: Load politicians
    print("Test 1: Loading politicians")
    politicians = load_politicians()
    print(politicians.head())
    print()

    # Test 2: Load assets
    print("Test 2: Loading enriched assets")
    assets = load_assets_with_products()
    print(assets.head())
    print()

    # Test 3: Dataset summary
    print("Test 3: Dataset summary")
    summary = get_dataset_summary()
    print(f"Politicians: {summary['n_politicians']}")
    print(f"Assets: {summary['n_assets']}")
    print(f"Unique subtypes: {summary['n_unique_subtypes']}")
    print(f"Avg assets/politician: {summary['avg_assets_per_politician']:.1f}")
    print("\nParty distribution:")
    for party, count in summary['party_distribution'].items():
        print(f"  {party}: {count}")
    print("\nTop 5 subtypes:")
    for subtype, count in list(summary['top_subtypes'].items())[:5]:
        print(f"  {subtype}: {count}")

    # Test 4: Value parsing quality
    print("Test 4: Value parsing quality")
    non_zero_values = (assets['value_numeric'] > 0).sum()
    zero_values = (assets['value_numeric'] == 0).sum()
    pct_non_zero = (non_zero_values / len(assets)) * 100

    print(f"  Assets with value > 0: {non_zero_values} ({pct_non_zero:.1f}%)")
    print(f"  Assets with value = 0: {zero_values} ({100-pct_non_zero:.1f}%)")
    min_val = assets[assets['value_numeric'] > 0]['value_numeric'].min()
    print(f"  Min non-zero value: ${min_val:,.2f}")
    print(f"  Max value: ${assets['value_numeric'].max():,.2f}")
    print()
