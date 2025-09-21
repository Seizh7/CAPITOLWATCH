# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Comprehensive geographic enrichment module for financial products.
Provides unified geographic enrichment including country inference,
geographic classification, and international exposure determination.
"""

import re
from typing import Optional, Dict


# ---------- Country Inference Methods ----------

def infer_country_from_exchange(
    exchange: str, ticker: str = "", name: str = ""
) -> Optional[str]:
    """Infer country from exchange code, but only for domestic products."""
    if not exchange:
        return None

    # Only infer US for clearly US-domestic products on US exchanges
    if exchange.upper() in ['US', 'NYSE', 'NASDAQ', 'NYSEARCA', 'BATS']:
        # Check if it's likely a foreign ADR or international listing
        if ticker and name:
            # ADR patterns (foreign companies on US exchanges)
            adr_patterns = [
                r'ADR|American.*Depositary',
                r'Nestle|ASML|SAP|Toyota|Sony|Samsung',  # Foreign companies
                r'\.L$|\.TO$|\.PA$|\.F$',  # Foreign ticker suffixes
                r'Royal.*Dutch|Unilever|Novartis'
            ]

            combined_text = f"{ticker} {name}"
            for pattern in adr_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return None  # Don't assume US for foreign companies

        # For genuine US domestic products
        return 'United States'

    # Direct mapping for other exchanges (domestic companies)
    exchange_mapping = {
        'TSX': 'Canada',
        'LSE': 'United Kingdom',
        'FRA': 'Germany',
        'PAR': 'France',
        'AMS': 'Netherlands',
        'SWX': 'Switzerland',
        'TYO': 'Japan',
        'HKG': 'Hong Kong',
        'ASX': 'Australia'
    }

    return exchange_mapping.get(exchange.upper())


def infer_country_from_ticker_pattern(
    ticker: str, name: str = ""
) -> Optional[str]:
    """Infer country from ticker patterns and company/fund names."""
    if not ticker:
        return None

    ticker = ticker.upper()

    # Specific company patterns by country
    country_patterns = {
        'United States': [
            # Major US indices and ETFs
            r'SPY|VOO|IVV|SPDR.*S&P|S&P.*500',
            r'QQQ|NASDAQ.*100',
            r'IWM|Russell.*2000',
            r'DIA|DOW.*JONES',

            # US sector ETFs
            r'XL[EKFIVPUYB]|SPDR.*Select.*Sector',
            r'VT[IEB]|Vanguard.*(Total.*Stock|FTSE.*US)',

            # Major US companies
            r'AAPL|Apple|MSFT|Microsoft|GOOGL|Alphabet|AMZN|Amazon',
            r'TSLA|Tesla|META|Facebook|NVDA|Nvidia',
            r'JPM|JP.*Morgan|BAC|Bank.*America|WFC|Wells.*Fargo',
            r'JNJ|Johnson.*Johnson|PFE|Pfizer|KO|Coca.*Cola',
            r'WMT|Walmart|HD|Home.*Depot|MCD|McDonald',

            # US mutual fund families
            r'Vanguard(?!.*Europe|.*International)',
            r'Fidelity|American.*Funds|T\..*Rowe.*Price',
            r'BlackRock.*US|iShares.*US|Invesco.*US',

            # Explicit US terms
            r'\bUS\b|USA|United.*States|American(?!.*Funds.*Europe)',
            r'Dollar|Treasury|Federal.*Reserve'
        ],

        'Switzerland': [
            r'Nestle|Nestlé|NSRGY',
            r'Novartis|NVS',
            r'Roche|RHHBY',
            r'UBS|Credit.*Suisse',
            r'Swiss|Switzerland|CHF'
        ],

        'Netherlands': [
            r'ASML|ASMLY',
            r'Royal.*Dutch|Shell.*Netherlands',
            r'ING|ABN.*AMRO',
            r'Unilever.*Netherlands'
        ],

        'Germany': [
            r'SAP|Siemens|BMW|Mercedes|Volkswagen',
            r'Deutsche|Bayer|BASF',
            r'Adidas|Allianz'
        ],

        'Japan': [
            r'Toyota|Sony|Nintendo|SoftBank',
            r'Honda|Panasonic|Canon',
            r'Nikkei|Tokyo|Japan|JPY'
        ],

        'Canada': [
            r'Royal.*Bank.*Canada|RBC',
            r'Shopify|SHOP|BlackBerry|Canadian',
            r'Toronto|TSX|CAD.*Dollar'
        ],

        'United Kingdom': [
            r'BP|British.*Petroleum|Vodafone',
            r'HSBC|Barclays|Lloyds',
            r'British|UK|United.*Kingdom|GBP'
        ],

        'South Korea': [
            r'Samsung|LG|Hyundai|SK.*Telecom'
        ],

        'Taiwan': [
            r'TSMC|Taiwan.*Semi|Foxconn'
        ]
    }

    # Check patterns for each country
    combined_text = f"{ticker} {name}" if name else ticker

    for country, patterns in country_patterns.items():
        for pattern in patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return country

    return None


def infer_country_from_currency(currency: str) -> Optional[str]:
    """Infer country from currency (fallback method)."""
    if not currency:
        return None

    currency_mapping = {
        'USD': 'United States',
        'CAD': 'Canada',
        'EUR': 'Germany',  # Default eurozone
        'GBP': 'United Kingdom',
        'JPY': 'Japan',
        'CHF': 'Switzerland',
        'AUD': 'Australia',
        'HKD': 'Hong Kong'
    }

    return currency_mapping.get(currency.upper())


# ---------- Geographic Classification Methods ----------

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


# ---------- Unified Enrichment Functions ----------

def infer_country_for_product(product: Dict) -> Optional[str]:
    """
    Infer country for a product using all available methods.

    Args:
        product: Dict with product data (ticker, name, exchange, currency).

    Returns:
        Inferred country or None.
    """
    # Skip if country already exists
    if product.get('country'):
        return product['country']

    ticker = product.get('ticker', '')
    name = product.get('name', '')
    exchange = product.get('exchange', '')
    currency = product.get('currency', '')

    # Method 1: Pattern-based inference (most reliable for company origin)
    country = infer_country_from_ticker_pattern(ticker, name)
    if country:
        return country

    # Method 2: Exchange-based inference (only for domestic products)
    country = infer_country_from_exchange(exchange, ticker, name)
    if country:
        return country

    # Method 3: Currency-based inference (fallback)
    country = infer_country_from_currency(currency)
    if country:
        return country

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


def enrich_product_geography(product: Dict) -> Dict[str, any]:
    """
    Comprehensive geographic enrichment for a single product.

    Args:
        product: Dict with product data (id, name, ticker, exchange,
                 currency, country).

    Returns:
        Dict with all geographic enrichment data:
        - country (if not already present)
        - international_exposure
        - geographic_classification
    """
    enrichment = {}

    # 1. Country inference
    country = infer_country_for_product(product)
    if country and not product.get('country'):
        enrichment['country'] = country

    # 2. Geographic classification based on name
    product_name = product.get('name', '')
    if product_name:
        geo_classification = classify_product_geography(product_name)
        enrichment.update(geo_classification)

    return enrichment
