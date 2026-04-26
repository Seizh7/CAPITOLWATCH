# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Run cluster analysis for all 6 experiments and generate Markdown reports.

Uses the same parameters as the evaluation step, so results are consistent.
Saves one report per experiment to data/figures/cluster_profiles/.

Usage:
    python -m capitolwatch.analysis.run_cluster_analysis
"""

from capitolwatch.analysis.cluster_analysis import run_analysis
from capitolwatch.analysis.data_loader import (
    load_assets_with_products,
    load_politicians,
)
from capitolwatch.analysis.run_evaluation import (
    _get_dbscan_labels,
    _get_kmeans_labels,
    _get_som_labels,
)


def run_all_analyses(
    output_dir: str = "data/figures/cluster_profiles",
) -> dict:
    """
    Analyze clusters for all 6 experiments.

    Loads data once, then runs K-Means, DBSCAN, and SOM (each with two
    feature types), generating a report for each.

    Args:
        output_dir (str): Where to save the reports (default:
            data/figures/cluster_profiles/).

    Returns:
        dict: Results keyed by experiment name, e.g. "kmeans/freq_baseline".
    """
    from pathlib import Path

    out_path = Path(output_dir)

    # Load the raw data once
    print("Loading raw data from database")
    politicians_df = load_politicians()
    assets_df = load_assets_with_products()
    print(f"  Politicians: {len(politicians_df)} | Assets: {len(assets_df)}")

    experiments = [
        ("kmeans", "freq_baseline", _get_kmeans_labels),
        ("kmeans", "freq_weighted", _get_kmeans_labels),
        ("dbscan", "freq_baseline", _get_dbscan_labels),
        ("dbscan", "freq_weighted", _get_dbscan_labels),
        ("som", "freq_baseline", _get_som_labels),
        ("som", "freq_weighted", _get_som_labels),
    ]

    all_profiles = {}

    for algo_name, feature_type, loader_fn in experiments:
        _, labels = loader_fn(feature_type)

        profiles = run_analysis(
            labels=labels,
            politicians_df=politicians_df,
            assets_df=assets_df,
            algo_name=algo_name,
            feature_type=feature_type,
            output_dir=out_path,
        )
        all_profiles[f"{algo_name}/{feature_type}"] = profiles

    print(f"\nAll reports saved to: {out_path.resolve()}")
    return all_profiles


if __name__ == "__main__":
    run_all_analyses()
