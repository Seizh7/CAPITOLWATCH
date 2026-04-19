# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
DBSCAN sector analysis script.

Runs the DBSCAN experiment on the sector_baseline feature set and compares
it against freq_baseline.
Results are exported to data/visualizations/evaluation_results_sector.csv.

Usage:
    python -m capitolwatch.analysis.clustering.run_dbscan_sector
"""

import pandas as pd

from capitolwatch.analysis.clustering.run_dbscan import (
    print_results,
    run_dbscan_experiment,
)
from capitolwatch.analysis.evaluation import (
    build_comparison_table, export_results
)

_OUTPUT_PATH = "data/visualizations/evaluation_results_sector.csv"

# eps range are same as the main DBSCAN study for comparison
_EPS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _experiment_to_eval_dict(experiment: dict) -> dict:
    """
    Convert the output of run_dbscan_experiment() to the evaluate_clustering()
    format expected by build_comparison_table().

    Args:
        experiment (dict): Output of run_dbscan_experiment().

    Returns:
        dict: Keys — algo_name, feature_type, n_clusters, n_outliers,
            silhouette.
    """
    return {
        "algo_name": "dbscan",
        "feature_type": experiment["feature_type"],
        "n_clusters": experiment["n_clusters"],
        "n_outliers": experiment["n_outliers"],
        "silhouette": experiment["best_silhouette"],
    }


def run_sector_comparison(
    output_path: str = _OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Run DBSCAN on sector_baseline and freq_baseline, then export comparison.

    Steps:
    1. Run DBSCAN experiment on sector_baseline (eps grid: 0.1 -> 0.6)
    2. Run DBSCAN experiment on freq_baseline
    3. Build sorted comparison table
    4. Export table to CSV

    Args:
        output_path (str): Path for the exported CSV file.

    Returns:
        pd.DataFrame: Comparison table with one row per experiment.
    """
    print("Running DBSCAN on sector_baseline ...")
    sector_result = run_dbscan_experiment("sector_baseline")
    print_results(sector_result)

    # freq_baseline is the reference point
    print("\nRunning DBSCAN on freq_baseline (reference) ...")
    freq_baseline_result = run_dbscan_experiment("freq_baseline")
    print_results(freq_baseline_result)

    eval_rows = [
        _experiment_to_eval_dict(sector_result),
        _experiment_to_eval_dict(freq_baseline_result),
    ]

    df = build_comparison_table(eval_rows)
    export_results(df, output_path)
    print(f"\nSector comparison exported to: {output_path}")

    _print_comparison(sector_result, freq_baseline_result)

    return df


def _print_comparison(sector: dict, freq_baseline: dict) -> None:
    """
    Print a side-by-side comparison of sector_baseline vs freq_baseline.

    Args:
        sector (dict): run_dbscan_experiment() output for sector_baseline.
        freq_baseline (dict): run_dbscan_experiment() output for freq_baseline.
    """
    print("\n=== Comparison: sector_baseline vs freq_baseline ===")
    rows = [sector, freq_baseline]
    headers = ["feature_type", "n_clusters", "n_outliers", "silhouette"]
    widths = [20, 12, 12, 12]

    header_line = "".join(
        h.ljust(w) for h, w in zip(headers, widths)
    )
    print(header_line)
    print("-" * sum(widths))

    for row in rows:
        sil = row["best_silhouette"]
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        values = [
            row["feature_type"],
            str(row["n_clusters"]),
            str(row["n_outliers"]),
            sil_str,
        ]
        print("".join(v.ljust(w) for v, w in zip(values, widths)))

    # outliers in common
    sector_outlier_names = [
        f"{o['first_name']} {o['last_name']}"
        for o in sector.get("outliers", [])
    ]
    fb_outlier_names = [
        f"{o['first_name']} {o['last_name']}"
        for o in freq_baseline.get("outliers", [])
    ]

    common = set(sector_outlier_names) & set(fb_outlier_names)
    print(f"\nOutliers in common (sector_baseline ∩ freq_baseline): "
          f"{sorted(common) if common else 'none'}")


if __name__ == "__main__":
    run_sector_comparison()
