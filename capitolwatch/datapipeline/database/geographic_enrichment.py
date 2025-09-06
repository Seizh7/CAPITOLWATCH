# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Minimaliste geographic enrichment module.
All database interactions handled by services layer.
"""

import re
from typing import Optional, Dict


def classify_by_ticker_pattern(ticker: str) -> Optional[str]:
    """
    Classify a product geographically based on known ticker patterns.

    Args:
        ticker (str): Product ticker symbol.

    Returns:
        Optional[str]: Geographic classification or None if no match.
    """
    if not ticker:
        return None

    ticker = ticker.upper()

    # Common ETF ticker patterns
    us_etf_patterns = [
        r'^VT[ISMW]',              # Vanguard Total (VTI, VTS, VTW, etc.)
        r'^SPY$', r'^QQQ$', r'^IWM$',  # Large-cap US
        r'^EFA$', r'^VEA$',        # Developed markets ex-US
        r'^VWO$', r'^EEM$',        # Emerging markets
    ]

    for pattern in us_etf_patterns:
        if re.match(pattern, ticker):
            if any(x in ticker for x in ['EFA', 'VEA', 'VWO', 'EEM']):
                return 'International'
            return 'US-Focused'

    # Regional suffix patterns
    if ticker.endswith('TO') or ticker.endswith('TSE'):
        return 'Canada-Focused'
    elif any(
        suffix in ticker
        for suffix in ['.L', '.LON', '.PA', '.FR', '.DE']
    ):
        return 'Europe-Focused'

    return None


def classify_by_name_analysis(name: str) -> Optional[str]:
    """
    Classify a product geographically based on its name.

    Args:
        name (str): Full product name.

    Returns:
        Optional[str]: Geographic classification or None if no match.
    """
    if not name:
        return None

    name_lower = name.lower()

    # International keywords
    international_keywords = [
        'international', 'global', 'world', 'emerging', 'developed markets',
        'ex-us', 'foreign', 'overseas', 'europe', 'asia', 'pacific',
        'emerging markets', 'eafe', 'msci world'
    ]
    if any(keyword in name_lower for keyword in international_keywords):
        return 'International'

    # US-focused keywords
    us_keywords = [
        'us ', 'u.s.', 'united states', 'america', 'domestic',
        'nasdaq', 's&p 500', 'russell', 'dow jones'
    ]
    if any(keyword in name_lower for keyword in us_keywords):
        return 'US-Focused'

    # Regional specifics
    if any(word in name_lower for word in ['canada', 'canadian', 'toronto']):
        return 'Canada-Focused'
    elif any(
        word in name_lower
        for word in ['europe', 'european', 'eurozone']
    ):
        return 'Europe-Focused'

    return None


def classify_product_geography(product_name: str) -> Dict[str, str]:
    """
    Determine a product's geographic classification based on its ticker or
    name.

    Args:
        product_name (str): Product name.

    Returns:
        Dict[str, str]: Dictionary containing:
            - international_exposure (Yes/No)
            - geographic_classification (str)
    """
    # Extract ticker if available at the start of the product name
    ticker_match = re.match(r'^([A-Z]{1,5})\s*-', product_name)
    ticker = ticker_match.group(1) if ticker_match else None

    # Try classification by ticker
    classification = None
    if ticker:
        classification = classify_by_ticker_pattern(ticker)

    # Fallback to classification by name
    if not classification:
        classification = classify_by_name_analysis(product_name)

    # Default classification
    if not classification:
        classification = 'Unknown'

    # Determine international exposure flag
    international_exposure = (
        'Yes' if classification == 'International' else 'No'
    )

    return {
        'international_exposure': international_exposure,
        'geographic_classification': classification
    }
