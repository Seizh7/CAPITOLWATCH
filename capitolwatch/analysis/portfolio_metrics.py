# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Portfolio analysis and metrics calculation for politicians' portfolios.

This module provides functions to:
- Calculate diversification metrics (Herfindahl Index, Shannon Entropy)
- Analyze sector concentration and risk profiles
- Generate portfolio feature vectors for clustering
- Classify investment strategies (Conservative, Aggressive, Balanced)
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter

from config import CONFIG
from capitolwatch.services.politicians import get_politician_basic_info
from capitolwatch.services.products import get_all_sectors
from capitolwatch.services.analytics import (
    get_sector_distribution_for_politician,
    get_industry_distribution_for_politician,
    get_politicians_with_assets
)


def calculate_herfindahl_index(weights: List[float]) -> float:
    """
    Calculate Herfindahl-Hirschman Index for portfolio concentration.

    Args:
        weights: List of portfolio weights (should sum to 1.0)

    Returns:
        float: HHI value between 0 (perfectly diversified) and 1 (concentrated)
    """
    if not weights or sum(weights) == 0:
        return 0.0

    # Normalize weights
    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    # Calculate HHI: sum of squared weights
    hhi = sum(w ** 2 for w in normalized_weights)
    return hhi


def calculate_shannon_entropy(weights: List[float]) -> float:
    """
    Calculate Shannon entropy for portfolio diversification.

    Args:
        weights: List of portfolio weights

    Returns:
        float: Shannon entropy (higher = more diversified)
    """
    if not weights or sum(weights) == 0:
        return 0.0

    # Normalize weights
    total = sum(weights)
    normalized_weights = [w / total for w in weights if w > 0]

    # Calculate Shannon entropy: -sum(p * log(p))
    entropy = -sum(w * np.log(w) for w in normalized_weights)
    return entropy


def get_politician_portfolio_data(politician_id: str, config=None) -> Dict:
    """
    Retrieve portfolio data for a specific politician.

    Args:
        politician_id: ID of the politician
        config: Configuration object (optional)

    Returns:
        Dict containing portfolio information
    """
    if config is None:
        config = CONFIG

    # Get basic politician info
    politician_info = get_politician_basic_info(politician_id, config=config)
    if not politician_info:
        return {
            'politician_id': politician_id,
            'portfolio_data': pd.DataFrame(),
            'sector_weights': {},
            'industry_weights': {},
            'total_assets': 0
        }

    # Get sector and industry distributions
    sector_counts = get_sector_distribution_for_politician(
        politician_id,
        config=config
    )
    industry_counts = get_industry_distribution_for_politician(
        politician_id, config=config
    )

    if not sector_counts:
        return {
            'politician_id': politician_id,
            'politician_name': politician_info['politician_name'],
            'party': politician_info['party'],
            'portfolio_data': pd.DataFrame(),
            'sector_weights': {},
            'industry_weights': {},
            'total_assets': 0
        }

    # Calculate weights
    total_assets = sum(sector_counts.values())
    sector_weights = {
        sector: count / total_assets
        for sector, count in sector_counts.items()
    }

    total_industry_assets = sum(industry_counts.values())
    industry_weights = {
        industry: count / total_industry_assets
        for industry, count in industry_counts.items()
    }

    return {
        'politician_id': politician_id,
        'politician_name': politician_info['politician_name'],
        'party': politician_info['party'],
        'portfolio_data': pd.DataFrame(),
        'sector_weights': sector_weights,
        'industry_weights': industry_weights,
        'total_assets': int(total_assets)
    }


def calculate_portfolio_metrics(portfolio_data: Dict) -> Dict:
    """
    Calculate comprehensive portfolio metrics for a politician.

    Args:
        portfolio_data: Portfolio data dictionary from
        get_politician_portfolio_data

    Returns:
        Dict with calculated metrics
    """
    if portfolio_data['total_assets'] == 0:
        return {
            'politician_id': portfolio_data['politician_id'],
            'total_assets': 0,
            'sector_diversification': {
                'herfindahl_index': 0.0,
                'shannon_entropy': 0.0,
                'num_sectors': 0
            },
            'industry_diversification': {
                'herfindahl_index': 0.0,
                'shannon_entropy': 0.0,
                'num_industries': 0
            },
            'risk_profile': 'Unknown',
            'dominant_sector': None,
            'concentration_ratio': 0.0
        }

    sector_weights = list(portfolio_data['sector_weights'].values())
    industry_weights = list(portfolio_data['industry_weights'].values())

    # Sector diversification metrics
    sector_hhi = calculate_herfindahl_index(sector_weights)
    sector_entropy = calculate_shannon_entropy(sector_weights)

    # Industry diversification metrics
    industry_hhi = calculate_herfindahl_index(industry_weights)
    industry_entropy = calculate_shannon_entropy(industry_weights)

    # Dominant sector and concentration
    dominant_sector = max(
        portfolio_data['sector_weights'].items(),
        key=lambda x: x[1]
    )
    concentration_ratio = dominant_sector[1]  # Weight of largest sector

    # Risk profile classification
    risk_profile = classify_risk_profile(
        sector_hhi,
        len(sector_weights),
        concentration_ratio
    )

    return {
        'politician_id': portfolio_data['politician_id'],
        'politician_name': portfolio_data.get('politician_name', 'Unknown'),
        'party': portfolio_data.get('party', 'Unknown'),
        'total_assets': portfolio_data['total_assets'],
        'sector_diversification': {
            'herfindahl_index': round(sector_hhi, 4),
            'shannon_entropy': round(sector_entropy, 4),
            'num_sectors': len(sector_weights)
        },
        'industry_diversification': {
            'herfindahl_index': round(industry_hhi, 4),
            'shannon_entropy': round(industry_entropy, 4),
            'num_industries': len(industry_weights)
        },
        'risk_profile': risk_profile,
        'dominant_sector': dominant_sector[0],
        'concentration_ratio': round(concentration_ratio, 4),
        'sector_breakdown': portfolio_data['sector_weights']
    }


def classify_risk_profile(
        herfindahl_index: float,
        num_sectors: int,
        concentration_ratio: float
) -> str:
    """
    Classify investment risk profile based on diversification metrics.

    Args:
        herfindahl_index: Portfolio concentration (0-1)
        num_sectors: Number of different sectors
        concentration_ratio: Weight of dominant sector

    Returns:
        str: Risk profile classification
    """
    if concentration_ratio > 0.7 or herfindahl_index > 0.5:
        return "Concentrated"
    elif num_sectors >= 6 and herfindahl_index < 0.25:
        return "Diversified"
    elif concentration_ratio > 0.4:
        return "Focused"
    else:
        return "Balanced"


def generate_portfolio_feature_vector(
    portfolio_data: Dict,
    all_sectors: List[str] = None,
    config=None
) -> np.ndarray:
    """
    Generate feature vector for clustering based on portfolio composition.

    Args:
        portfolio_data: Portfolio data dictionary
        all_sectors: List of all possible sectors for consistent vector size
        config: Configuration object (optional)

    Returns:
        np.ndarray: Feature vector for clustering
    """
    if config is None:
        config = CONFIG

    if all_sectors is None:
        all_sectors = get_all_sectors(config=config)

    # Initialize feature vector
    features = []

    # Sector weights (one-hot encoded style)
    sector_weights = portfolio_data['sector_weights']
    for sector in all_sectors:
        features.append(sector_weights.get(sector, 0.0))

    # Diversification metrics
    metrics = calculate_portfolio_metrics(portfolio_data)
    features.extend([
        metrics['sector_diversification']['herfindahl_index'],
        metrics['sector_diversification']['shannon_entropy'],
        metrics['sector_diversification']['num_sectors'] / 10.0,  # Normalize
        metrics['concentration_ratio'],
        np.log1p(metrics['total_assets']) / 10.0  # Log-scaled asset count
    ])

    return np.array(features)


def get_all_portfolio_metrics(config=None) -> pd.DataFrame:
    """
    Calculate portfolio metrics for all politicians with investment data.

    Args:
        config: Configuration object (optional)

    Returns:
        pd.DataFrame: Portfolio metrics for all politicians
    """
    if config is None:
        config = CONFIG

    # Get all politicians with investment data using the service
    politician_ids = get_politicians_with_assets(config=config)

    print(
        f"Calculating portfolio metrics for {len(politician_ids)} politicians"
    )

    # Calculate metrics for each politician
    all_metrics = []
    for politician_id in politician_ids:
        portfolio_data = get_politician_portfolio_data(politician_id, config)
        metrics = calculate_portfolio_metrics(portfolio_data)
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def analyze_party_differences(metrics_df: pd.DataFrame) -> Dict:
    """
    Analyze differences in investment patterns between political parties.

    Args:
        metrics_df: DataFrame with portfolio metrics for all politicians

    Returns:
        Dict: Analysis results by party
    """
    party_analysis = {}

    for party in metrics_df['party'].unique():
        if pd.isna(party):
            continue

        party_data = metrics_df[metrics_df['party'] == party]

        party_analysis[party] = {
            'count': len(party_data),
            'avg_diversification': {
                'herfindahl_index': party_data['sector_diversification'].apply(
                    lambda x: x['herfindahl_index']
                ).mean(),
                'shannon_entropy': party_data['sector_diversification'].apply(
                    lambda x: x['shannon_entropy']
                ).mean(),
                'num_sectors': party_data['sector_diversification'].apply(
                    lambda x: x['num_sectors']
                ).mean()
            },
            'avg_total_assets': party_data['total_assets'].mean(),
            'risk_profile_distribution': (
                party_data['risk_profile'].value_counts().to_dict()
            ),
            'top_sectors': Counter([
                item
                for sublist in party_data['sector_breakdown'].apply(
                    lambda x: [
                        k for k, v in x.items() if v > 0.1
                    ]  # Sectors with >10% weight
                ).tolist()
                for item in sublist
            ]).most_common(5)
        }

    return party_analysis


if __name__ == "__main__":
    # Test the portfolio metrics calculation
    print("CAPITOLWATCH - Portfolio Metrics Analysis")

    # Calculate metrics for all politicians
    metrics_df = get_all_portfolio_metrics()
    print(f"Portfolio metrics calculated for {len(metrics_df)} politicians")

    # Show summary statistics
    print("\nOverall Statistics:")
    herfindahl_series = metrics_df['sector_diversification'].map(
        lambda x: x['herfindahl_index']
    )
    num_sectors_series = metrics_df['sector_diversification'].map(
        lambda x: x['num_sectors']
    )
    avg_portfolio_size = metrics_df['total_assets'].mean()
    avg_herfindahl_index = herfindahl_series.mean()
    avg_num_sectors = num_sectors_series.mean()

    print(f"Average portfolio size: {avg_portfolio_size:.1f} assets")
    print(f"Average Herfindahl Index: {avg_herfindahl_index:.3f}")
    print(f"Average number of sectors: {avg_num_sectors:.1f}")

    # Risk profile distribution
    print("\nRisk Profile Distribution:")
    risk_dist = metrics_df['risk_profile'].value_counts()
    for profile, count in risk_dist.items():
        print(
            f"  {profile}: {count} politicians "
            f"({count/len(metrics_df)*100:.1f}%)"
        )

    # Party analysis
    print("\nParty Analysis:")
    party_analysis = analyze_party_differences(metrics_df)
    for party, analysis in party_analysis.items():
        avg_div = analysis['avg_diversification']
        top_sectors = [sector for sector, _ in analysis['top_sectors'][:3]]
        print(f"\n{party} ({analysis['count']} politicians):")
        print(f"  Avg diversification: {avg_div['herfindahl_index']:.3f}")
        print(f"  Avg sectors: {avg_div['num_sectors']:.1f}")
        print(f"  Top sectors: {top_sectors}")
