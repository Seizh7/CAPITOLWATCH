# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""Portfolio metrics generator: HHI, diversification, risk classification."""

import pandas as pd
from typing import Dict, List, Optional

from config import CONFIG
from capitolwatch.services.analytics import (
    get_sector_distribution_for_politician,
    get_politicians_with_assets
)
from capitolwatch.services.politicians import get_politician_basic_info


def calculate_herfindahl_index(sector_weights: List[float]) -> float:
    """
    Calculate the Herfindahl-Hirschman Index for portfolio concentration.

    This metric measures portfolio concentration where 0 indicates perfect
    diversification and 1 indicates complete concentration in one sector.

    Args:
        sector_weights: List of sector weight values

    Returns:
        Float value between 0 and 1 representing portfolio concentration
    """
    if not sector_weights or sum(sector_weights) == 0:
        return 0.0

    # Normalize weights to sum to 1
    total_weight = sum(sector_weights)
    normalized_weights = [weight / total_weight for weight in sector_weights]

    # Calculate HHI as sum of squared normalized weights
    hhi_value = sum(weight ** 2 for weight in normalized_weights)
    return hhi_value


def calculate_diversification_score(sector_weights: List[float]) -> float:
    """
    Calculate a simple diversification score from 0 to 100.

    This score combines portfolio concentration (HHI) with sector count
    to provide an intuitive measure where higher values indicate better
    diversification across multiple sectors.

    Args:
        sector_weights: List of sector weight values

    Returns:
        Float value between 0 and 100 representing diversification level
    """
    if not sector_weights:
        return 0.0

    # Calculate concentration component (70% weight)
    hhi_value = calculate_herfindahl_index(sector_weights)
    concentration_score = (1 - hhi_value) * 70

    # Calculate sector diversity component (30% weight)
    active_sectors = len([weight for weight in sector_weights if weight > 0])
    diversity_score = min(active_sectors / 8.0, 1.0) * 30

    return round(concentration_score + diversity_score, 1)


def classify_risk_profile(
    hhi_value: float,
    sector_count: int,
    dominant_sector_weight: float
) -> str:
    """
    Classify investment risk profile based on portfolio metrics.

    This function categorizes portfolios into risk profiles based on
    concentration metrics and diversification patterns.

    Args:
        hhi_value: Herfindahl-Hirschman Index value
        sector_count: Number of different sectors in portfolio
        dominant_sector_weight: Weight of the largest sector

    Returns:
        String classification: "Aggressive", "Conservative", or "Balanced"
    """
    # Aggressive: high concentration or single dominant sector
    if dominant_sector_weight > 0.6 or hhi_value > 0.5:
        return "Aggressive"

    # Conservative: well diversified across many sectors
    elif sector_count >= 5 and hhi_value < 0.3:
        return "Conservative"

    # Balanced: moderate diversification
    else:
        return "Balanced"


def calculate_politician_metrics(politician_id: str) -> Optional[Dict]:
    """
    Calculate portfolio metrics for a single politician.

    This function retrieves politician data and sector distribution to
    compute concentration, diversification, and risk classification metrics.

    Args:
        politician_id: Unique identifier for the politician

    Returns:
        Dictionary with calculated metrics or None if no data available
    """
    # Get basic politician information
    politician_info = get_politician_basic_info(politician_id, config=CONFIG)
    if not politician_info:
        return None

    # Get sector distribution for portfolio analysis
    sector_distribution = get_sector_distribution_for_politician(
        politician_id, config=CONFIG
    )

    # Handle case with no portfolio data
    if not sector_distribution:
        full_name = (
            f"{politician_info.get('first_name', '')} "
            f"{politician_info.get('last_name', '')}"
        ).strip()
        return {
            'politician_id': politician_id,
            'name': full_name or "Unknown",
            'party': politician_info.get('party', 'Unknown'),
            'total_assets': 0,
            'herfindahl_index': 0.0,
            'diversification_score': 0.0,
            'risk_profile': 'Unknown',
            'sector_count': 0,
            'dominant_sector_weight': 0.0
        }

    # Calculate portfolio metrics
    sector_values = list(sector_distribution.values())
    total_assets = sum(sector_values)
    sector_weights = [value / total_assets for value in sector_values]

    hhi_value = calculate_herfindahl_index(sector_values)
    diversification_score = calculate_diversification_score(sector_values)
    dominant_weight = max(sector_weights) if sector_weights else 0.0
    sector_count = len(sector_weights)

    risk_profile = classify_risk_profile(
        hhi_value, sector_count, dominant_weight
    )

    # Format politician name
    full_name = (
        f"{politician_info.get('first_name', '')} "
        f"{politician_info.get('last_name', '')}"
    ).strip()

    return {
        'politician_id': politician_id,
        'name': full_name or "Unknown",
        'party': politician_info.get('party', 'Unknown'),
        'total_assets': total_assets,
        'herfindahl_index': round(hhi_value, 4),
        'diversification_score': diversification_score,
        'risk_profile': risk_profile,
        'sector_count': sector_count,
        'dominant_sector_weight': round(dominant_weight, 4)
    }


def generate_portfolio_metrics(
    config: Optional[object] = None
) -> pd.DataFrame:
    """
    Generate portfolio metrics for all politicians with assets.

    This function processes all politicians in the database and calculates
    comprehensive portfolio metrics including concentration, diversification,
    and risk classification measures.

    Args:
        config: Database configuration

    Returns:
        DataFrame containing metrics for all politicians with portfolios
    """
    config = config or CONFIG
    politician_ids = get_politicians_with_assets(config=config)

    if not politician_ids:
        print("No politicians with assets found in database.")
        return pd.DataFrame()

    print(f"Portfolio metrics for {len(politician_ids)} politicians")

    # Calculate metrics for each politician
    metrics_list = []
    for politician_id in politician_ids:
        politician_metrics = calculate_politician_metrics(politician_id)
        if politician_metrics:
            metrics_list.append(politician_metrics)

    # Create dataframe with results
    metrics_dataframe = pd.DataFrame(metrics_list)
    print(f"Metrics calculated for {len(metrics_dataframe)} politicians")

    return metrics_dataframe


def analyze_party_patterns(metrics_df: pd.DataFrame) -> Dict:
    """
    Analyze portfolio patterns by political party.

    This function aggregates portfolio metrics by party affiliation to
    identify patterns and differences in investment behavior across
    political groups.

    Args:
        metrics_df: DataFrame containing politician portfolio metrics

    Returns:
        Dictionary with party-level analysis and statistics
    """
    party_analysis = {}

    for party in metrics_df['party'].unique():
        if pd.isna(party):
            continue

        # Filter data for current party
        party_data = metrics_df[metrics_df['party'] == party]

        # Calculate party-level statistics
        party_analysis[party] = {
            'politician_count': len(party_data),
            'average_herfindahl': round(
                party_data['herfindahl_index'].mean(),
                4
            ),
            'average_diversification': round(
                party_data['diversification_score'].mean(), 1
            ),
            'average_assets': round(
                party_data['total_assets'].mean(), 1
            ),
            'risk_profile_distribution': (
                party_data['risk_profile'].value_counts().to_dict()
            )
        }

    return party_analysis


if __name__ == "__main__":
    print("Testing portfolio metrics generation...")

    try:
        # Generate metrics for all politicians
        metrics_df = generate_portfolio_metrics()

        if not metrics_df.empty:
            # Display basic statistics
            print(f"Total politicians analyzed: {len(metrics_df)}")
            print(f"Average HHI: {metrics_df['herfindahl_index'].mean():.4f}")
            print("Average diversification:")
            print(f"{metrics_df['diversification_score'].mean():.1f}")
            # Show risk profile distribution
            risk_distribution = metrics_df['risk_profile'].value_counts()
            print("Risk profile distribution:")
            for profile, count in risk_distribution.items():
                percentage = count / len(metrics_df) * 100
                print(f"     {profile}: {count} ({percentage:.1f}%)")

            # Analyze party patterns
            party_analysis = analyze_party_patterns(metrics_df)
            print(f"Party analysis for {len(party_analysis)} parties")
        else:
            print("No portfolio metrics generated")

    except Exception as e:
        print(f"Portfolio metrics test failed: {e}")
        import traceback
        traceback.print_exc()
