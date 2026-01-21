# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Geographic enrichment module for financial products.
Determines if a product is domestic (US-focused) or international.
"""

from typing import Optional, Dict


# Keywords indicating international exposure in fund/ETF names
INTERNATIONAL_KEYWORDS = [
    'international', 'intl', 'global', 'world',
    'ex-us', 'ex us', 'ex-usa', 'ex usa', 'non-us', 'non us',
    'europe', 'european', 'asia', 'pacific',
    'latam', 'latin america', 'africa', 'middle east',
    'china', 'japan', 'india', 'brazil', 'uk', 'germany', 'france',
    'eafe', 'msci world', 'msci eafe',
    'foreign', 'overseas', 'developed markets',
    'emerging', 'emerging markets', 'em markets',
    'non-us', 'non us', 'ex-united states'
]

# Keywords indicating US domestic focus
US_DOMESTIC_KEYWORDS = [
    "s&p", "sp500", "500", "nasdaq", "dow jones", "russell",
    "total stock market", "qqq", "us ", "u.s.", "usa", "america"
    "domestic", "us equity", "u.s. equity",
]


# Funds with US managers but international focus (exceptions)
US_MANAGER_INTERNATIONAL_FUNDS = [
    'europacific',
    "new perspective",
    "new world",
]


def is_international_fund(name: str) -> bool:
    """
    Check if a fund/ETF name indicates international exposure.

    Args:
        name: Product name.

    Returns:
        True if the fund appears to have international focus.
    """
    if not name:
        return False

    name_lower = name.lower()

    # Check for international keywords
    for keyword in INTERNATIONAL_KEYWORDS:
        if keyword in name_lower:
            return True

    return False


def is_us_manager_international_fund(name: str) -> bool:
    """
    Check if a fund has a US manager but international focus.

    Args:
        name: Product name.

    Returns:
        True if the fund is managed by US firm but invests internationally.
    """
    if not name:
        return False

    name_lower = name.lower()

    for keyword in US_MANAGER_INTERNATIONAL_FUNDS:
        if keyword in name_lower:
            return True

    return False


def is_us_focused_fund(name: str) -> bool:
    """
    Check if a fund/ETF name indicates US domestic focus.

    Args:
        name: Product name.

    Returns:
        True if the fund appears to be US-focused.
    """
    if not name:
        return False

    name_lower = name.lower()

    # Check for US-focused keywords
    for keyword in US_DOMESTIC_KEYWORDS:
        if keyword in name_lower:
            return True

    return False


def determine_is_domestic(product: Dict) -> Optional[bool]:
    """
    Determine if a product is domestic (US-focused) or international.

    Logic:
    1. For stocks/bonds: Use country field from Yahoo Finance
       - country == 'United States' -> is_domestic = True
       - country != 'United States' -> is_domestic = False

    2. For ETFs/Mutual Funds: Analyze the fund name
       - International keywords -> is_domestic = False
       - US manager with international focus -> is_domestic = False
       - US-focused keywords -> is_domestic = True
       - US fund manager (without intl keywords) -> is_domestic = True
       - Otherwise, use country field as fallback

    Args:
        product: Dict with product data (name, country, is_etf,
            is_mutual_fund)

    Returns:
        True if domestic (US-focused), False if international, None if unknown.
    """
    name = product.get('name', '')
    country = product.get('country', '')
    is_etf = product.get('is_etf', False)
    is_mutual_fund = product.get('is_mutual_fund', False)

    # For funds/ETFs: analyze the name first
    if is_etf or is_mutual_fund:
        # Priority 1: Check for explicit international keywords
        if is_international_fund(name):
            return False

        # Priority 2: Check for US manager with international mandate
        if is_us_manager_international_fund(name):
            return False

        # Priority 3: Check for US-focused keywords (S&P 500, Russell, etc.)
        if is_us_focused_fund(name):
            return True

        # Fallback to country if available
        if country:
            return country == 'United States'

        return None

    # For individual stocks/bonds: use country directly
    if country:
        return country == 'United States'

    return None


def enrich_product_geography(product: Dict) -> Dict[str, any]:
    """
    Geographic enrichment for a single product.

    Args:
        product: Dict with product data (name, country, is_etf, is_mutual_fund)

    Returns:
        Dict with geographic enrichment: {'is_domestic': bool or None}
    """
    is_domestic = determine_is_domestic(product)

    if is_domestic is not None:
        return {'is_domestic': is_domestic}

    return {}
