# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Convert cluster labels into human-readable financial profiles.

After clustering assigns a label to each politician, the module analyzes
what those labels mean by looking at their investment patterns. This module
computes per-cluster statistics and generates reports.

Main functions:
    - compute_cluster_profiles() : build a ClusterProfile per unique label
    - generate_cluster_report() : render profiles as a Markdown document
    - save_cluster_report() : persist a report
    - run_analysis() : end-to-end pipeline for one experiment
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Default directory for reports
CLUSTER_PROFILES_DIR = Path("data/visualizations/cluster_profiles")


class ClusterProfile:
    """
    Store the statistics calculated for one cluster.

    Attributes:
        cluster_id (int): Label for this cluster (-1 = DBSCAN outliers).
        experiment (str): Which algorithm and feature type produced this,
            e.g. "kmeans / freq_weighted".
        size (int): How many politicians are in this cluster.
        politicians (pd.DataFrame): Names and parties of cluster members.
        top_subtypes (list): Top investment types in this cluster, ranked
            by average value.
        mean_total_value (float): Average portfolio value per politician
            in this cluster (in dollars).
        mean_diversity (float): Average number of different investment types
            per politician.
        party_distribution (dict): Percentage of Republicans and Democrats.
    """

    def __init__(self, cluster_id: int, experiment: str):
        self.cluster_id = cluster_id
        self.experiment = experiment
        self.size: int = 0
        self.politicians: pd.DataFrame = pd.DataFrame()
        self.top_subtypes: list = []
        self.mean_total_value: float = 0.0
        self.mean_diversity: float = 0.0
        self.party_distribution: dict = {}


def compute_cluster_profiles(
    labels: np.ndarray,
    politicians_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    experiment: str,
    top_n_subtypes: int = 5,
) -> list:
    """
    Calculate statistics for each cluster.

    Creates one ClusterProfile per cluster label. DBSCAN outliers (-1) are
    included as their own profile. Investment types are sorted by average
    value, not by frequency, because value matters more than count.

    Args:
        labels (np.ndarray): Cluster assignment for each politician.
        politicians_df (pd.DataFrame): Politician data, rows aligned with
            labels.
        assets_df (pd.DataFrame): Investment data with politician_id, subtype,
            value_numeric columns.
        experiment (str): Name of this run, e.g. "dbscan / freq_weighted".
        top_n_subtypes (int): How many top investment types to include
            (default: 5).

    Returns:
        list: One ClusterProfile per cluster, with outliers last.
    """
    unique_labels = sorted(set(labels))
    # Put -1 (outliers) at the end so regular clusters come first
    if -1 in unique_labels:
        unique_labels = [lbl for lbl in unique_labels if lbl != -1] + [-1]

    profiles = []

    for cid in unique_labels:
        profile = ClusterProfile(cluster_id=cid, experiment=experiment)
        mask = labels == cid

        # 1. Politicians in this cluster
        # politicians_df is positionally aligned with labels, so boolean mask
        # works
        profile.politicians = politicians_df[mask].reset_index(drop=True)
        profile.size = int(mask.sum())

        # 2. Party distribution
        party_counts = profile.politicians["party"].value_counts()
        profile.party_distribution = {
            party: round(count / profile.size * 100, 1)
            for party, count in party_counts.items()
        }

        # 3. Assets for these politicians
        member_ids = profile.politicians["id"].tolist()
        member_assets = assets_df[assets_df["politician_id"].isin(member_ids)]

        # 4. Mean total portfolio value per politician
        # Sum all values per politician, then take the mean across politicians
        total_per_politician = (
            member_assets.groupby("politician_id")["value_numeric"].sum()
        )
        profile.mean_total_value = float(total_per_politician.mean())

        # 5. Mean diversity (distinct subtypes per politician)
        diversity_per_politician = (
            member_assets.groupby("politician_id")["subtype"].nunique()
        )
        profile.mean_diversity = float(diversity_per_politician.mean())

        # 6. Top subtypes by mean invested value
        # Group all assets by subtype, compute mean value across all members,
        # then take the top-N.
        if not member_assets.empty:
            subtype_values = (
                member_assets.groupby("subtype")["value_numeric"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n_subtypes)
            )
            profile.top_subtypes = list(subtype_values.items())
        else:
            profile.top_subtypes = []

        profiles.append(profile)

    return profiles


def _suggest_cluster_name(
    profile: ClusterProfile,
) -> str:
    """
    Create a readable name for a cluster based on its main investment type.

    Args:
        profile (ClusterProfile): A cluster with computed statistics.

    Returns:
        str: A short name like "Mutual Fund Investors" or "Outliers".
    """
    if profile.cluster_id == -1:
        return "Outliers (Atypical Portfolios)"

    if not profile.top_subtypes:
        return f"Cluster {profile.cluster_id}"

    # Use the dominant subtype as the primary descriptor
    dominant_subtype, _ = profile.top_subtypes[0]

    # Map common subtype names
    name_map = {
        "Mutual Fund": "Mutual Fund Investors",
        "Stock": "Stock-Heavy Investors",
        "Exchange Traded Fund/Note": "ETF-Focused Investors",
        "Uncategorized": "Diversified Investors",
        "Municipal Security": "Bond/Municipal Security Investors",
        "Corporate Bond": "Corporate Bond Investors",
        "Real Estate": "Real Estate Investors",
    }

    return name_map.get(dominant_subtype, f"{dominant_subtype} Investors")


def generate_cluster_report(
    profiles: list,
    algo_name: str,
    feature_type: str,
) -> str:
    """
    Convert cluster statistics into a Markdown report.

    Returns a summary table followed by detailed pages per cluster.

    Args:
        profiles (list): List of ClusterProfiles from
            compute_cluster_profiles().
        algo_name (str): Algorithm name, e.g. "dbscan".
        feature_type (str): Feature type used, e.g. "freq_weighted".

    Returns:
        str: Markdown text ready to save as a file.
    """
    def _format_value(value: float) -> str:
        # Format values with space as thousand separator
        return f"{int(value):,}".replace(",", " ") + " $"

    lines = []

    # Header
    lines.append(f"# Cluster report — {algo_name} / {feature_type}")
    lines.append("")
    lines.append(
        f"Algorithm: {algo_name.upper()} | "
        f"Feature type: {feature_type} | "
        f"Politicians analysed: {sum(p.size for p in profiles)}"
    )
    lines.append("")

    # Summary table — one row per cluster
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Cluster | Suggested name | Size | Mean value | "
        "Mean diversity | Top subtype |"
    )
    lines.append(
        "------------------------------------------------"
        "-------------------------------"
    )
    for p in profiles:
        name = _suggest_cluster_name(p)
        top_sub = p.top_subtypes[0][0] if p.top_subtypes else "—"
        label = "Outliers" if p.cluster_id == -1 else str(p.cluster_id)
        lines.append(
            f"| {label} | {name} | {p.size} | "
            f"{_format_value(p.mean_total_value)} | "
            f"{p.mean_diversity:.1f} | {top_sub} |"
        )

    lines.append("")

    # Detailed section for each cluster
    for p in profiles:
        label = "Outliers" if p.cluster_id == -1 else f"Cluster {p.cluster_id}"
        suggested = _suggest_cluster_name(p)

        lines.append(f"## {label} — {suggested}")
        lines.append("")
        lines.append(f"Size: {p.size} politicians")
        lines.append("")

        # Party distribution
        lines.append("Party distribution:")
        for party, pct in p.party_distribution.items():
            lines.append(f"- {party}: {pct}%")
        lines.append("")

        # Financial profile
        lines.append(
            f"Mean portfolio value: {_format_value(p.mean_total_value)}"
        )
        lines.append(
            f"Mean diversity (distinct subtypes): {p.mean_diversity:.1f}"
        )
        lines.append("")

        # Top subtypes
        if p.top_subtypes:
            lines.append("Top investment types by mean value:")
            lines.append("")
            lines.append("| # | Subtype | Mean value |")
            lines.append("----------------------------")
            for rank, (subtype, mean_val) in enumerate(p.top_subtypes, 1):
                lines.append(
                    f"| {rank} | {subtype} | {_format_value(mean_val)} |"
                )
            lines.append("")

        # List all politicians in this cluster
        lines.append("Politicians in this cluster:")
        lines.append("")
        for _, row in p.politicians.iterrows():
            lines.append(
                f"- {row['first_name']} {row['last_name']} ({row['party']})"
            )
        lines.append("")

    return "\n".join(lines)


def save_cluster_report(
    report_content: str,
    algo_name: str,
    feature_type: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save a Markdown report.

    Filename is {algo_name}_{feature_type}.md. Creates the output directory
    if needed.

    Args:
        report_content (str): Markdown text from generate_cluster_report().
        algo_name (str): Algorithm name for the filename.
        feature_type (str): Feature type for the filename.
        output_dir (Path, optional): Where to save. Defaults to
            data/visualizations/cluster_profiles/.

    Returns:
        Path: The file that was written.
    """
    if output_dir is None:
        output_dir = CLUSTER_PROFILES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{algo_name}_{feature_type}.md"
    filepath = output_dir / filename
    filepath.write_text(report_content, encoding="utf-8")
    return filepath


def run_analysis(
    labels: np.ndarray,
    politicians_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    algo_name: str,
    feature_type: str,
    output_dir: Optional[Path] = None,
    top_n_subtypes: int = 5,
) -> list:
    """
    Analyze clusters and generate a Markdown report.

    Computes statistics, creates a readable report, saves it, and prints
    a summary.

    Args:
        labels (np.ndarray): Cluster assignment per politician.
        politicians_df (pd.DataFrame): Politician names/parties,
            rows aligned with labels.
        assets_df (pd.DataFrame): Investment data for each politician.
        algo_name (str): Algorithm name, e.g. "kmeans".
        feature_type (str): Feature type, e.g. "freq_weighted".
        output_dir (Path, optional): Where to save the report (default:
            data/visualizations/cluster_profiles/).
        top_n_subtypes (int): How many investment types to show (default: 5).

    Returns:
        list: One ClusterProfile per cluster.
    """
    experiment = f"{algo_name} / {feature_type}"
    print(f"\n  Analysing {experiment} :", flush=True)

    # 1. Compute profiles
    profiles = compute_cluster_profiles(
        labels=labels,
        politicians_df=politicians_df,
        assets_df=assets_df,
        experiment=experiment,
        top_n_subtypes=top_n_subtypes,
    )

    # 2. Generate Markdown
    report = generate_cluster_report(profiles, algo_name, feature_type)

    # 3. Save to disk
    filepath = save_cluster_report(report, algo_name, feature_type, output_dir)
    print(f"    Report saved: {filepath}")

    # 4. Print a compact summary to stdout
    for p in profiles:
        label = "outliers" if p.cluster_id == -1 else f"cluster {p.cluster_id}"
        top = p.top_subtypes[0][0] if p.top_subtypes else "—"
        print(f"    {label:12s} | size={p.size:3d} | top_subtype={top!r}")

    return profiles
