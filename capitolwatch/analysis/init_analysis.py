# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Complete analysis initialization: generates all embeddings and metrics.

This module is the main entry point for initializing the entire analysis
pipeline
starting from an empty database. It coordinates the generation of:
1. Product embeddings (4 methods)
2. Portfolio embeddings (weighted aggregations)
3. Portfolio metrics (HHI, diversification, risk profiles)
"""

from typing import List, Optional

from config import CONFIG
from capitolwatch.services.product_embeddings import get_all_statistics
from capitolwatch.services.portfolio_embeddings import (
    list_available_portfolio_methods,
    get_portfolio_embedding_statistics,
)
from capitolwatch.analysis.product_embedding_generator import (
    generate_and_store_products_embeddings,
)
from capitolwatch.analysis.portfolio_embeddings_generator import (
    generate_and_store_portfolio_embeddings,
)
from capitolwatch.analysis.portfolio_metrics_generator import (
    generate_portfolio_metrics,
)


def check_product_embeddings_exist(config: Optional[object] = None) -> bool:
    """
    Check if product embeddings exist in the database.

    This function verifies that the prerequisite product embeddings
    are available before attempting to generate portfolio embeddings.

    Args:
        config: Database configuration

    Returns:
        True if product embeddings exist, False otherwise
    """
    config = config or CONFIG
    product_statistics = get_all_statistics(config=config)
    available_methods = product_statistics.get("methods", [])

    return len(available_methods) > 0


def generate_missing_product_embeddings(
    config: Optional[object] = None
) -> bool:
    """
    Generate all product embeddings using the 4 available methods.

    This function creates product embeddings using all available methods:
    custom_financial, tfidf_basic, word2vec, and glove.

    Args:
        config: Database configuration

    Returns:
        True if embeddings were generated successfully, False otherwise
    """
    config = config or CONFIG

    try:
        print("Generating product embeddings for all methods...")

        # List of all embedding methods to generate
        methods = ["custom_financial", "tfidf_basic", "word2vec", "glove"]
        successful_methods = 0
        total_products = 0

        for method in methods:
            print(f"   Generating '{method}' embeddings...")
            result = generate_and_store_products_embeddings(
                config=config,
                method=method
            )

            if result and result.get("product_count", 0) > 0:
                product_count = result['product_count']
                print(f"   {method}: {product_count} products processed")
                successful_methods += 1
                total_products = product_count  # Same count for all methods
            else:
                print(f"   Warning: Failed to generate {method} embeddings")

        if successful_methods > 0:
            print(
                f"Generated embeddings for {total_products}"
                f"products using {successful_methods}/4 methods"
            )
            return True
        else:
            print("Failed to generate any product embeddings")
            return False

    except Exception as e:
        print(f"Product embedding generation failed: {e}")
        return False


def check_portfolio_embeddings_exist(
    required_methods: List[str],
    config: Optional[object] = None
) -> List[str]:
    """
    Check which portfolio embedding methods are missing.

    This function compares required methods against available methods
    to identify which portfolio embeddings need to be generated.

    Args:
        required_methods: List of required embedding method names
        config: Database configuration

    Returns:
        List of missing method names that need to be generated
    """
    config = config or CONFIG
    available_methods = list_available_portfolio_methods(config=config)
    missing_methods = [
        method for method in required_methods
        if method not in available_methods
    ]

    return missing_methods


def generate_missing_portfolio_embeddings(
    missing_methods: List[str],
    config: Optional[object] = None
) -> bool:
    """
    Generate missing portfolio embeddings for specified methods.

    This function creates portfolio embeddings by combining product
    embeddings with politician asset weights for each missing method.

    Args:
        missing_methods: List of method names to generate
        config: Database configuration

    Returns:
        True if all methods were generated successfully, False otherwise
    """
    config = config or CONFIG

    try:
        for method in missing_methods:
            # Remove '_weighted' suffix to get base method name
            base_method = method.replace("_weighted", "")

            print(f"Generating portfolio embeddings for '{method}'...")
            result = generate_and_store_portfolio_embeddings(
                embedding_method=base_method,
                config=config
            )

            if result and result.get("processed_politicians", 0) > 0:
                processed_count = result['processed_politicians']
                print(f"   {method}: {processed_count} politicians processed")
            else:
                print(f"   Warning: No embeddings generated for {method}")

        return True

    except Exception as e:
        print(f"Portfolio embedding generation failed: {e}")
        return False


def check_portfolio_metrics_exist(config: Optional[object] = None) -> bool:
    """
    Check if portfolio metrics have been calculated and stored.

    This function verifies if portfolio metrics (HHI, diversification,
    risk profiles) exist for the politicians in the database.

    Args:
        config: Database configuration

    Returns:
        True if metrics exist, False otherwise
    """
    config = config or CONFIG

    try:
        # Try to load existing metrics
        metrics_df = generate_portfolio_metrics(config=config)
        return not metrics_df.empty
    except Exception:
        return False


def generate_missing_portfolio_metrics(
    config: Optional[object] = None
) -> bool:
    """
    Generate portfolio metrics for all politicians.

    This function calculates Herfindahl index, diversification scores,
    and risk profiles for all politicians with sufficient asset data.

    Args:
        config: Database configuration

    Returns:
        True if metrics were generated successfully, False otherwise
    """
    config = config or CONFIG

    try:
        print("Generating portfolio metrics...")
        metrics_df = generate_portfolio_metrics(config=config)

        if not metrics_df.empty:
            politicians_count = len(metrics_df)
            avg_hhi = metrics_df['herfindahl_index'].mean()
            print(
                f"Portfolio metrics calculated for "
                f"{politicians_count} politicians"
            )
            print(f"Average HHI: {avg_hhi:.4f}")
            return True
        else:
            print("Failed to generate portfolio metrics")
            return False

    except Exception as e:
        print(f"Portfolio metrics generation failed: {e}")
        return False


def initialize_complete_analysis_pipeline(
    required_methods: Optional[List[str]] = None,
    config: Optional[object] = None
) -> bool:
    """
    Initialize the complete analysis pipeline from scratch.

    This function orchestrates the full pipeline initialization:
    1. Product embeddings generation (4 methods)
    2. Portfolio embeddings generation (weighted aggregations)
    3. Portfolio metrics calculation (HHI, diversification, risk)

    Args:
        required_methods: List of required portfolio embedding methods
        config: Database configuration

    Returns:
        True if complete pipeline initialization was successful, False
        otherwise
    """
    config = config or CONFIG
    default_methods = [
        "custom_financial_weighted",
        "tfidf_basic_weighted",
        "word2vec_weighted",
        "glove_weighted"
    ]
    required_methods = required_methods or default_methods

    print("=" * 70)
    print("INITIALIZING COMPLETE ANALYSIS PIPELINE")
    print("=" * 70)

    # Step 1: Generate product embeddings
    print("Step 1: Product embeddings generation...")
    if not check_product_embeddings_exist(config):
        print("   Product embeddings missing, generating...")
        if not generate_missing_product_embeddings(config):
            print("   ERROR: Failed to generate product embeddings")
            return False
        print("   Product embeddings generated successfully")
    else:
        print("   Product embeddings already exist")

    # Step 2: Generate portfolio embeddings
    print("\nStep 2: Portfolio embeddings generation...")
    missing_methods = check_portfolio_embeddings_exist(
        required_methods, config
    )

    if missing_methods:
        print(f"   Missing portfolio embeddings: {missing_methods}")
        if not generate_missing_portfolio_embeddings(missing_methods, config):
            print("   ERROR: Failed to generate portfolio embeddings")
            return False
        print("   Portfolio embeddings generated successfully")
    else:
        print("   All required portfolio embeddings already exist")

    # Step 3: Generate portfolio metrics
    print("\nStep 3: Portfolio metrics generation...")
    if not check_portfolio_metrics_exist(config):
        print("   Portfolio metrics missing, generating...")
        if not generate_missing_portfolio_metrics(config):
            print("   ERROR: Failed to generate portfolio metrics")
            return False
        print("   Portfolio metrics generated successfully")
    else:
        print("   Portfolio metrics already exist")

    # Step 4: Verify final pipeline status
    print("\nStep 4: Pipeline verification...")
    portfolio_stats = get_portfolio_embedding_statistics(config=config)
    available_methods = list_available_portfolio_methods(config=config)
    metrics_df = generate_portfolio_metrics(config=config)

    print(f"Available embedding methods: {available_methods}")
    total = portfolio_stats.get('total_embeddings', 0)
    print(f"Total portfolio embeddings: {total}")
    print(f"Politicians with metrics: {len(metrics_df)}")

    print("\n" + "=" * 50)
    print("PIPELINE INITIALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 50)

    return True


if __name__ == "__main__":
    print("Initializing complete analysis pipeline...")

    # Execute complete pipeline initialization
    success = initialize_complete_analysis_pipeline()

    if success:
        print("Initialization completed successfully")
    else:
        print("Pipeline initialization failed")
