# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Portfolio embeddings generator: weighted aggregation of product embeddings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import re

from config import CONFIG
from capitolwatch.services.product_embeddings import get_embeddings
from capitolwatch.services.analytics import get_politicians_with_assets
from capitolwatch.services.portfolio_embeddings import (
    store_portfolio_embedding
)
from capitolwatch.services.assets import get_politician_assets_simple

SUPPORTED_METHODS = ["custom_financial", "tfidf_basic", "word2vec", "glove"]


def parse_asset_value(value_str: str) -> float:
    """
    Convert textual asset value to numeric estimate.

    This function handles various formats of asset values including ranges,
    special cases, and currency symbols to provide consistent numeric values
    for portfolio weighting calculations.

    Args:
        value_str: String representation of asset value

    Returns:
        Float value representing the estimated asset worth, or -1.0 if invalid
    """
    if not value_str:
        return -1.0

    # Clean and normalize the input string
    clean_string = str(value_str).strip().lower()
    if clean_string in {"none", "null", "n/a", ""}:
        return -1.0

    # Handle special cases with known patterns
    if "none (or less than $1,001)" in clean_string:
        return 500.0
    if "over $1,000,000" in clean_string:
        return 1_500_000.0
    if "unascertainable" in clean_string:
        return 50_000.0

    # Handle value ranges (e.g., "$15,001 - $50,000")
    if " - " in clean_string and "$" in clean_string:
        extracted_numbers = re.findall(r"\$?([\d,]+)", value_str)
        if len(extracted_numbers) >= 2:
            try:
                low_value = float(extracted_numbers[0].replace(",", ""))
                high_value = float(extracted_numbers[1].replace(",", ""))
                return (low_value + high_value) / 2  # Use midpoint of range
            except ValueError:
                return 50_000.0

    # Handle single dollar values
    if "$" in clean_string:
        extracted_numbers = re.findall(r"[\d,]+", value_str.replace("$", ""))
        if extracted_numbers:
            try:
                return float(extracted_numbers[0].replace(",", ""))
            except ValueError:
                return 50_000.0

    # Try direct numeric conversion
    try:
        return float(clean_string.replace(",", ""))
    except ValueError:
        return -1.0


def has_child_assets(product_id: str, all_assets: List[Dict]) -> bool:
    """
    Check if a product acts as a parent of other assets.

    This helps identify fund holdings or parent companies that shouldn't
    be double-counted.

    Args:
        product_id: ID of the product to check
        all_assets: List of all assets for comparison

    Returns:
        True if this product appears to be a parent of other assets
    """
    parent_name = str(product_id).lower()
    for asset in all_assets:
        child_product_id = str(asset.get("product_id", "")).lower()
        # Check for name overlap indicating parent-child relationship
        has_overlap = (parent_name != child_product_id and
                       (parent_name in child_product_id or
                        child_product_id in parent_name))
        if has_overlap:
            return True
    return False


def create_portfolio_embedding(
    politician_id: int,
    embedding_method: str = "custom_financial",
    config: Optional[object] = None
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Create a single portfolio embedding for one politician.

    This function aggregates product embeddings weighted by asset values
    to create a representative vector for the politician's entire portfolio.

    Args:
        politician_id: ID of the politician to process
        embedding_method: Method used for product embeddings
        config: Database configuration

    Returns:
        A tuple (embedding_vector, metadata) where:
            embedding_vector: 1-D numpy array or None if failed
            metadata: dictionary with processing statistics
    """
    config = config or CONFIG

    # Get product embeddings for the specified method
    product_data = get_embeddings(embedding_method, config=config)
    if product_data["embeddings"] is None:
        error_message = f"No product embeddings found for '{embedding_method}'"
        return None, {"error": error_message}

    # Create lookup dictionary for product embeddings
    product_embeddings = {
        product_id: product_data["embeddings"][index]
        for index, product_id in enumerate(product_data["product_ids"])
    }

    # Get politician's assets
    raw_assets = get_politician_assets_simple(politician_id, config=config)
    if not raw_assets:
        return None, {"error": "No assets found"}

    # Process asset values and filter valid ones
    valid_assets = []
    for asset in raw_assets:
        # Parse asset value
        value = parse_asset_value(asset["value"])

        # Skip invalid values, but check for parent assets
        if value == -1.0:
            if has_child_assets(asset["product_id"], raw_assets):
                continue  # Skip parent assets to avoid double counting
            value = 25_000.0  # Default value for unspecified assets

        if value > 0:
            valid_assets.append({
                "product_id": asset["product_id"],
                "value": value
            })

    if not valid_assets:
        return None, {"error": "No valid assets with values"}

    # Collect embeddings and weights for portfolio aggregation
    embedding_vectors = []
    asset_weights = []
    total_portfolio_value = 0.0

    for asset in valid_assets:
        # Get embedding for this product
        product_embedding = product_embeddings.get(asset["product_id"])
        if product_embedding is None:
            continue  # Skip products without embeddings

        # Clean embedding and check validity
        clean_embedding = np.nan_to_num(product_embedding, nan=0.0)
        if np.all(clean_embedding == 0):
            continue  # Skip zero embeddings

        # Add to portfolio
        embedding_vectors.append(clean_embedding)
        asset_weight = float(asset["value"])
        asset_weights.append(asset_weight)
        total_portfolio_value += asset_weight

    if not embedding_vectors:
        return None, {"error": "No valid embeddings found for assets"}

    # Create weighted average portfolio embedding
    normalized_weights = np.array(asset_weights) / sum(asset_weights)
    stacked_vectors = np.vstack(embedding_vectors)
    portfolio_vector = np.average(
        stacked_vectors, axis=0, weights=normalized_weights
    )    # Ensure clean output
    portfolio_vector = np.nan_to_num(portfolio_vector, nan=0.0)

    metadata = {
        "method": f"{embedding_method}_weighted",
        "asset_count": len(embedding_vectors),
        "total_value": total_portfolio_value,
        "base_method": embedding_method,
        "aggregation": "weighted_average"
    }

    return portfolio_vector, metadata


def generate_portfolio_embeddings(
    embedding_method: str = "custom_financial",
    config: Optional[object] = None
) -> Dict:
    """
    Generate portfolio embeddings for all politicians with assets.

    This function processes all politicians in the database and creates
    portfolio embeddings by aggregating their product embeddings weighted
    by asset values.

    Args:
        embedding_method: Method to use for product embeddings
        config: Database configuration

    Returns:
        Dictionary with processing statistics and results
    """
    if embedding_method not in SUPPORTED_METHODS:
        error_msg = f"Unsupported method '{embedding_method}'. "
        error_msg += f"Available: {SUPPORTED_METHODS}"
        raise ValueError(error_msg)

    config = config or CONFIG
    politician_ids = get_politicians_with_assets(config=config)

    if not politician_ids:
        print("No politicians with assets found in database.")
        return {}

    politician_count = len(politician_ids)
    print(f"Generating '{embedding_method}' portfolio embeddings...")
    print(f"Processing {politician_count} politicians...")

    results = {
        "method": f"{embedding_method}_weighted",
        "processed": 0,
        "skipped": 0,
        "errors": 0
    }

    for politician_id in politician_ids:
        try:
            # Generate portfolio embedding
            portfolio_vector, metadata = create_portfolio_embedding(
                politician_id, embedding_method, config
            )

            if portfolio_vector is None:
                results["skipped"] += 1
                continue

            # Store the embedding
            store_portfolio_embedding(
                politician_id=politician_id,
                method=metadata["method"],
                embedding_vector=portfolio_vector,
                features_used=[f"product_{embedding_method}_weighted"],
                asset_count=metadata["asset_count"],
                total_value=metadata["total_value"],
                metadata=metadata,
                config=config
            )

            results["processed"] += 1

        except Exception as e:
            print(f"Error processing politician {politician_id}: {e}")
            results["errors"] += 1

    return results


def generate_and_store_portfolio_embeddings(
    methods: Optional[List[str]] = None,
    config: Optional[object] = None
) -> Dict:
    """
    Generate portfolio embeddings using multiple product embedding methods.

    This function runs the portfolio embedding generation process for
    multiple underlying product embedding methods and returns a summary.

    Args:
        methods: List of embedding methods to use
        config: Database configuration

    Returns:
        Dictionary with results for each method
    """
    config = config or CONFIG
    default_methods = ["custom_financial", "tfidf_basic", "word2vec", "glove"]
    methods = methods or default_methods
    results = {}

    print(f"Generating portfolio embeddings for {len(methods)} methods :")

    for method in methods:
        print(f"\nProcessing method: {method.upper()}")
        try:
            method_results = generate_portfolio_embeddings(method, config)
            results[method] = method_results

            if method_results:
                processed_count = method_results.get('processed', 0)
                skipped_count = method_results.get('skipped', 0)
                errors_count = method_results.get('errors', 0)
                print(f"   {processed_count} processed, "
                      f"{skipped_count} skipped, {errors_count} errors")
            else:
                print(f"No results for method '{method}'")

        except Exception as e:
            print(f"Method '{method}' failed: {e}")
            results[method] = {"error": str(e)}

    return results


if __name__ == "__main__":
    print("Testing portfolio embedding generation (no database storage)...")

    try:
        # Get a few politicians for testing
        politician_ids = get_politicians_with_assets(config=CONFIG)
        if not politician_ids:
            print("No politicians with assets found for testing.")
        else:
            # Limit to 3 politicians for the test
            test_politician_ids = politician_ids[:3]
            print(f"Testing with {len(test_politician_ids)} politicians :")

            # Test portfolio embedding generation for each method
            for method in SUPPORTED_METHODS:
                print(f"\nTesting method: {method.upper()}")
                try:
                    successful_count = 0

                    for politician_id in test_politician_ids:
                        (
                            portfolio_vector, metadata
                        ) = create_portfolio_embedding(
                            politician_id, method, CONFIG
                        )

                        if portfolio_vector is not None:
                            successful_count += 1
                            asset_count = metadata.get('asset_count', 'N/A')
                            total_value = metadata.get('total_value', 'N/A')
                            print(f"Politician {politician_id}: "
                                  f"shape {portfolio_vector.shape}, "
                                  f"{asset_count} assets, ${total_value:,.0f}")
                        else:
                            error_msg = metadata.get('error', 'Unknown error')
                            print(f"Politician {politician_id}: {error_msg}")

                except Exception as e:
                    print(f"Method '{method}' failed: {e}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
