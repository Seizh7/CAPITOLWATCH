# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Full portfolio analysis with embeddings and method comparison.

This script:
 - loads products from the database,
 - ensures embeddings exist (generates missing ones),
 - compares embedding methods and prints a compact report,
 - returns a small export dictionary containing results and recommendations.

The code is organized into small helper functions for clarity and testability.
"""

import time
import traceback
from typing import Dict, List, Optional

from config import CONFIG
from capitolwatch.analysis.embeddings import (
    generate_embeddings_for_all_products,
    compare_methods
)
from capitolwatch.services.product_embeddings import (
    list_available_methods,
    get_all_statistics
)
from capitolwatch.services.products import get_all_products_for_embeddings


def prepare_products(config) -> List[Dict]:
    """Load products suitable for embedding analysis."""
    products = get_all_products_for_embeddings(config=config)
    print(f"Found {len(products)} products in the database")
    return products


def ensure_embeddings_available(
    config, preferred_method: str = "custom_financial"
) -> List[str]:
    """
    Ensure there is at least one embedding method available.
    If none exist, generate the preferred method.

    Returns the list of available methods after the check.
    """
    methods = list_available_methods(
        config=config
    )
    if not methods:
        print("No embeddings found. Generating default embeddings ")
        print("(custom_financial).")
        result = generate_embeddings_for_all_products(
            config, method=preferred_method
        )
        print(f"Generated {result['product_count']} embeddings ")
        print(f"({result['embedding_dimension']} dimensions).")
        methods = [preferred_method]
    else:
        print(f"Available embedding methods: {methods}")
    return methods


def report_embedding_statistics(config) -> Dict:
    """Fetch and print embedding statistics from the storage service."""
    stats = get_all_statistics(config=config)
    overall = stats.get("_overall", {})
    if overall:
        print(f"Total embeddings: {overall.get('total_embeddings', 0)}")
        print(f"Unique products: {overall.get('unique_products', 0)}")
        print(f"Methods active: {overall.get('methods_count', 0)}")

    # Print per-method summary
    for method_name, method_stats in stats.items():
        if method_name == "_overall":
            continue
        count = method_stats.get("count", 0)
        if count <= 0:
            continue
        dims = method_stats.get("dimension_range", (0, 0))
        print(
            f"  {method_name}: {count} embeddings "
            f"(dims {dims[0]} - {dims[1]})"
        )

    return stats


def generate_and_compare_methods(config, methods: List[str]) -> Dict:
    """
    Compare multiple embedding methods and print a compact comparison report.
    Returns the comparison dictionary returned by compare_methods.
    """
    if len(methods) < 2:
        print("Not enough methods to compare (need at least 2).")
        return {}

    print("Comparing embedding methods...")
    comparison = compare_methods(config, methods)

    for method_name, comp_stats in comparison.items():
        count = comp_stats.get("product_count", 0)
        dims = comp_stats.get("dimension", 0)
        magnitude = comp_stats.get("mean_magnitude", 0.0)
        variance = comp_stats.get("variance_explained", 0.0)
        features = comp_stats.get("features_used", []) or []
        feature_sample = ", ".join(features[:5])
        more_count = len(features) - 5
        if len(features) > 5:
            feature_str = (
                f"{feature_sample}, +{more_count} more"
            )
        else:
            feature_str = feature_sample
        print(f"\nMethod: {method_name}")
        print(f"  Products: {count}")
        print(f"  Dimensions: {dims}")
        print(f"  Mean magnitude: {magnitude:.4f}")
        print(f"  Variance: {variance:.4f}")
        print(f"  Features (sample): {feature_str}")

    return comparison


def run_complete_analysis() -> Optional[Dict]:
    """Main workflow to run the full portfolio analysis."""
    print("CAPITOLWATCH - Complete portfolio analysis")
    print("=" * 60)
    start_time = time.time()

    try:
        # 1. Load products
        products = prepare_products(CONFIG)
        if not products:
            print("No products available. Aborting.")
            return None

        # 2. Ensure we have embeddings (generate default if missing)
        methods = ensure_embeddings_available(
            CONFIG,
            preferred_method="custom_financial"
        )

        # 3. Pick the primary method
        primary_method = (
            "custom_financial"
            if "custom_financial" in methods
            else methods[0]
        )
        print(f"Selected primary method: {primary_method}")

        # Print current embedding statistics
        report_embedding_statistics(CONFIG)

        # Compare methods
        comparison = (
            generate_and_compare_methods(CONFIG, methods)
            if len(methods) > 1 else {}
        )

        # Prepare simple export info
        export_info = {
            "total_products": len(products),
            "embedding_methods": methods,
            "method_comparison": comparison,
            "analysis_timestamp": time.time(),
        }

        # Final timing and summary
        total_time = time.time() - start_time
        print("\nAnalysis finished")
        print(f"Duration: {total_time:.1f} s")
        print(f"Products analyzed: {len(products)}")
        print(f"Embedding methods available: {len(methods)}")

        return export_info

    except Exception as exc:
        print("Fatal error during analysis:", exc)
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_complete_analysis()
