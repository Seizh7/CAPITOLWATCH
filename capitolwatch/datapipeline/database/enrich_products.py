# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Pipeline for comprehensive product enrichment using financial and geographic
data.
Enhances database products with financial metadata and geographic information.
"""

import re
import json
import requests
import yfinance as yf
from datetime import datetime, timezone
from typing import Optional, Dict

from config import CONFIG
from capitolwatch.db import get_connection
from capitolwatch.services.products import (
    get_products_without_enrichment,
    enrich_product
)
from capitolwatch.datapipeline.database.geographic_enrichment import (
    enrich_product_geography
)


# ---------- Product Type Filtering ----------
# Relevant types for financial investment analysis
# These products have enrichable data (ticker, sector, etc.)
ANALYZABLE_PRODUCT_TYPES = {
    # Listed stocks
    'Corporate Securities',
    'American Depository Receipt',

    # Investment funds
    'Mutual Funds',

    # Bonds (partially enrichable)
    'Government Securities',
    'Foreign Bonds',
}

# Types excluded from analysis (not relevant for clustering)
# - Non-financial assets (real estate, farms)
# - Containers without detailed content (IRA, 401k, managed accounts)
# - Insurance and annuity products
# - Private unlisted holdings
EXCLUDED_PRODUCT_TYPES = {
    # Bank deposits and accounts
    'Bank Deposit',
    'Brokerage/Managed Account',

    # Insurance and annuities
    'Life Insurance',
    'Annuity',

    # Real estate and physical assets
    'Real Estate',
    'Farm',
    'Personal Property',

    # Private business entities
    'Business Entity',

    # Retirement plans (containers)
    'Retirement Plans',

    # Deferred compensation
    'Deferred Compensation',

    # Trusts and fiduciaries
    'Trust',

    # Education savings
    'Education Savings Plans',
    'UGMA/UTMA',

    # Private funds (not enrichable via API)
    'Investment Fund',

    # Other
    'Accounts Receivable',
    'Intellectual Property',
    'Cryptocurrency',
    'Other',
}


def is_product_analyzable(product_type: str) -> bool:
    """
    Determines if a product type is relevant for analysis.

    Args:
        product_type: Financial product type.

    Returns:
        True if the product should be enriched and analyzed.
    """
    if not product_type:
        return False

    # Explicit check in lists
    if product_type in ANALYZABLE_PRODUCT_TYPES:
        return True
    if product_type in EXCLUDED_PRODUCT_TYPES:
        return False

    # By default, include unknown types to avoid missing
    # potentially interesting data
    return True


def extract_ticker(name: str) -> Optional[str]:
    """
    Extracts a ticker from a product name.

    Args:
        name (str): Product name.
    Returns:
        Optional[str]: Ticker, or None if not extractable.
    """
    if not name:
        return None

    # Main pattern: "TICKER - Description" or "TICKER-Description"
    pattern = r'^([A-Z]{1,5})\s*-\s*.+'
    match = re.match(pattern, name)
    if match:
        ticker = match.group(1)
        # Validation: 1-5 alphabetic characters
        if 1 <= len(ticker) <= 5 and ticker.isalpha():
            return ticker

    return None


def get_openfigi_session(api_key: str) -> requests.Session:
    """
    Creates an HTTP session for OpenFIGI.

    Args:
        api_key (str): OpenFIGI API key.
    Returns:
        requests.Session: Configured session.
    """
    session = requests.Session()
    session.headers.update({
        'Content-Type': 'application/json',
        'X-OPENFIGI-APIKEY': api_key
    })
    return session


def get_openfigi_security_info(
    session: requests.Session,
    ticker: str
) -> Optional[Dict]:
    """
    Retrieves security information from OpenFIGI.

    Args:
        session (requests.Session): Configured HTTP session.
        ticker (str): Ticker symbol.
    Returns:
        Optional[Dict]: FIGI data or None.
    """
    url = "https://api.openfigi.com/v3/mapping"
    query = [{"idType": "TICKER", "idValue": ticker, "exchCode": "US"}]

    try:
        response = session.post(url, data=json.dumps(query))
        response.raise_for_status()
        result = response.json()
        if (result and len(result) > 0 and
                'data' in result[0] and result[0]['data']):
            data = result[0]['data'][0]
            return {
                'figi': data.get('figi'),
                'ticker': data.get('ticker'),
                'exchange': data.get('exchCode'),
                'security_type': data.get('securityType'),
                'market_sector': data.get('marketSector')
            }
        return None

    except Exception as e:
        print(f"  Error querying OpenFIGI for {ticker}: {e}")
        return None


def get_yahoo_security_info(ticker: str) -> Optional[Dict]:
    """
    Fetches financial information from Yahoo Finance.

    Args:
        ticker (str): Ticker symbol.
    Returns:
        Optional[Dict]: Yahoo Finance data or None.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'symbol' not in info:
            return None

        return {
            'symbol': info.get('symbol'),
            'name': info.get('longName') or info.get('shortName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'country': info.get('country'),
            'currency': info.get('currency'),
            'market_cap': info.get('marketCap'),
            'beta': info.get('beta'),
            'dividend_yield': info.get('dividendYield'),
            'expense_ratio': info.get('annualReportExpenseRatio'),
            'asset_class_yahoo': info.get('quoteType'),
            'fund_family': info.get('fundFamily'),
            'category': info.get('category')
        }

    except Exception as e:
        print(f"  Error querying Yahoo Finance for {ticker}: {e}")
        return None


def classify_asset_class(openfigi_data: Dict, yahoo_data: Dict) -> str:
    """
    Determines asset class from provided data.

    Args:
        openfigi_data (Dict): FIGI data dictionary.
        yahoo_data (Dict): Yahoo Finance data dictionary.
    Returns:
        str: Asset class.
    """
    # Yahoo Finance classification takes priority
    if yahoo_data and yahoo_data.get('asset_class_yahoo'):
        yahoo_class = yahoo_data['asset_class_yahoo']
        mapping = {
            'EQUITY': 'Equity',
            'ETF': 'ETF',
            'MUTUALFUND': 'Mutual Fund',
            'MONEYMARKET': 'Cash'
        }
        return mapping.get(yahoo_class, 'Investment Fund')

    # Fallback to OpenFIGI
    if openfigi_data and openfigi_data.get('security_type'):
        figi_type = openfigi_data['security_type']
        if 'Stock' in figi_type or 'Common Stock' in figi_type:
            return 'Equity'
        elif 'ETP' in figi_type:
            return 'ETF'
        elif 'Open-End Fund' in figi_type:
            return 'Mutual Fund'

    return 'Unknown'


def calculate_market_cap_tier(market_cap: Optional[int]) -> Optional[str]:
    """
    Calculates market capitalization tier.

    Args:
        market_cap (Optional[int]): Market capitalization.
    Returns:
        Optional[str]: Market cap tier.
    """
    if not market_cap:
        return None
    if market_cap >= 200_000_000_000:
        return 'Mega'
    elif market_cap >= 10_000_000_000:
        return 'Large'
    elif market_cap >= 2_000_000_000:
        return 'Mid'
    elif market_cap >= 300_000_000:
        return 'Small'
    else:
        return 'Micro'


def calculate_risk_rating(
    sector: str,
    asset_class: str,
    beta: Optional[float]
) -> str:
    """
    Calculates risk rating based on sector, asset class, and beta.

    Args:
        sector (str): Industry sector.
        asset_class (str): Asset class.
        beta (Optional[float]): Beta coefficient.
    Returns:
        str: Risk rating.
    """
    # Conservative assets
    if asset_class in ['Cash', 'Money Market']:
        return 'Conservative'
    # Sector-based
    if sector:
        conservative_sectors = ['Utilities', 'Consumer Staples']
        aggressive_sectors = ['Technology', 'Biotechnology', 'Energy']
        if sector in conservative_sectors:
            return 'Conservative'
        elif sector in aggressive_sectors:
            return 'Aggressive'
    # Beta-based
    if beta:
        if beta < 0.8:
            return 'Conservative'
        elif beta > 1.3:
            return 'Aggressive'
    return 'Moderate'


def determine_fund_flags(name: str, asset_class: str) -> Dict[str, bool]:
    """
    Determines fund flags based on name and API-derived asset class.

    Args:
        name (str): Product name.
        asset_class (str): Asset class from API data (more reliable than HTML).

    Returns:
        Dict[str, bool]: Flags for is_etf, is_mutual_fund, and is_index_fund.
    """
    name_lower = name.lower() if name else ""

    # Use asset_class (from APIs) to override HTML-based classification
    # as it's more accurate and reliable
    if asset_class == 'ETF':
        is_etf = True
        is_mutual_fund = False
    elif asset_class == 'Mutual Fund':
        is_etf = False
        is_mutual_fund = True
    else:
        # For non-fund assets, both flags are False
        is_etf = False
        is_mutual_fund = False

    return {
        'is_etf': is_etf,
        'is_mutual_fund': is_mutual_fund,
        'is_index_fund': 'index' in name_lower
    }


def enrich_single_product(
    product: Dict,
    openfigi_session: requests.Session
) -> Optional[Dict]:
    """
    Enrich a single product with financial and geographic data.

    Args:
        product (Dict): Product dictionary with 'id', 'name', 'type'.
        openfigi_session (requests.Session): OpenFIGI HTTP session.

    Returns:
        Dict: Enrichment data, or None if product type is excluded.
    """
    product_id = product['id']
    name = product['name']
    product_type = product.get('type', '')

    # Skip non-analyzable product types
    if not is_product_analyzable(product_type):
        return None

    print(f"[{product_id}] Processing product: {name}")

    # Initialize enrichment data
    enrichment = {
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'is_analyzable': True  # Mark as analyzable
    }

    # --- Ticker extraction ---
    ticker = extract_ticker(name)
    if not ticker:
        print("  No extractable ticker -> marked as non-tradeable")
        enrichment['data_source'] = 'Manual_NonTradeable'

        # Attempt geographic enrichment even for non-tradeable products
        try:
            # Create product dict for geographic enrichment
            product_dict = {
                'name': name,
                'country': None,
                'is_etf': False,
                'is_mutual_fund': False
            }
            geo_data = enrich_product_geography(product_dict)
            if geo_data:
                enrichment.update(geo_data)
                print("  Geographic enrichment applied")
        except Exception as e:
            print(f"  Geographic enrichment failed: {e}")

        return enrichment

    print(f"  Extracted ticker: {ticker}")
    enrichment['ticker'] = ticker

    # --- Financial data retrieval ---
    openfigi_data = get_openfigi_security_info(openfigi_session, ticker)
    yahoo_data = get_yahoo_security_info(ticker)

    openfigi_success = openfigi_data is not None
    yahoo_success = yahoo_data is not None

    if openfigi_success and yahoo_success:
        status = "Both sources available"
    elif openfigi_success or yahoo_success:
        status = "Partial success"
    else:
        status = "No data available"

    print(
        f"  Financial status: {status} "
        f"(OpenFIGI: {'OK' if openfigi_success else 'Failed'} | "
        f"Yahoo: {'OK' if yahoo_success else 'Failed'})"
    )

    if not openfigi_success and not yahoo_success:
        enrichment['data_source'] = 'API_Failed'
    else:
        enrichment['data_source'] = 'OpenFIGI+Yahoo'

        # Add OpenFIGI data
        if openfigi_data:
            enrichment.update({
                'figi': openfigi_data.get('figi'),
                'exchange': openfigi_data.get('exchange')
            })

        # Add Yahoo Finance data
        if yahoo_data:
            enrichment.update({
                'sector': yahoo_data.get('sector'),
                'industry': yahoo_data.get('industry'),
                'country': yahoo_data.get('country'),
                'currency': yahoo_data.get('currency', 'USD'),
                'market_cap': yahoo_data.get('market_cap'),
                'beta': yahoo_data.get('beta'),
                'dividend_yield': yahoo_data.get('dividend_yield'),
                'expense_ratio': yahoo_data.get('expense_ratio'),
                'fund_family': yahoo_data.get('fund_family'),
                'category': yahoo_data.get('category')
            })

        # Derived classifications
        asset_class = classify_asset_class(
            openfigi_data or {},
            yahoo_data or {}
        )
        enrichment['asset_class'] = asset_class

        market_cap_tier = calculate_market_cap_tier(
            enrichment.get('market_cap')
        )
        if market_cap_tier:
            enrichment['market_cap_tier'] = market_cap_tier

        risk_rating = calculate_risk_rating(
            enrichment.get('sector', ''),
            asset_class,
            enrichment.get('beta')
        )
        enrichment['risk_rating'] = risk_rating

        # Fund flags: Use API-derived asset_class to correct
        # HTML-based classification. This overrides is_etf/is_mutual_fund
        # from HTML since API data is more reliable
        fund_flags = determine_fund_flags(name, asset_class)
        enrichment.update(fund_flags)

        is_etf_different = (
            fund_flags['is_etf'] != product.get('is_etf', False)
        )
        is_mutual_different = (
            fund_flags['is_mutual_fund'] !=
            product.get('is_mutual_fund', False)
        )
        if is_etf_different or is_mutual_different:
            print(
                f"  Corrected fund type: "
                f"ETF={fund_flags['is_etf']}, "
                f"Mutual={fund_flags['is_mutual_fund']} "
                f"(was ETF={product.get('is_etf', False)}, "
                f"Mutual={product.get('is_mutual_fund', False)})"
            )

    # --- Geographic enrichment ---
    try:
        # Create product dict for geographic enrichment
        # Use existing is_etf/is_mutual_fund from product (HTML source)
        product_dict = {
            'name': name,
            'country': enrichment.get('country'),
            'is_etf': product.get('is_etf', False),
            'is_mutual_fund': product.get('is_mutual_fund', False)
        }
        geo_data = enrich_product_geography(product_dict)
        if geo_data:
            enrichment.update(geo_data)
            print("  Geographic enrichment applied")
    except Exception as e:
        print(f"  Geographic enrichment failed: {e}")

    return enrichment


def run_enrichment_pipeline(
    limit: Optional[int] = None,
    start_from: int = 0
) -> Dict:
    """
    Run the product enrichment pipeline.

    Args:
        limit (Optional[int]): Maximum number of products to process.
        start_from (int): Starting index. Defaults to 0.

    Returns:
        Dict: Statistics dictionary.
    """
    print("Starting product enrichment pipeline")

    # Retrieve products to enrich
    conn = get_connection(CONFIG)
    products = get_products_without_enrichment(connection=conn)

    total_products = len(products)
    print(f"{total_products} products to process")

    if start_from > 0:
        products = products[start_from:]
        print(f"Starting from index {start_from}")

    if limit:
        products = products[:limit]
        print(f"Limiting to {limit} products")

    # Create OpenFIGI session
    openfigi_session = get_openfigi_session(CONFIG.openfigi_api_key)

    # Initialize statistics
    stats = {
        'total_processed': 0,
        'enriched': 0,
        'failed': 0,
        'non_tradeable': 0,
        'geographic_enriched': 0,
        'skipped_non_analyzable': 0,
        'start_time': datetime.now()
    }

    # Process each product
    for i, product in enumerate(products, 1):
        print(f"\n[{i}/{len(products)}] {product['name'][:50]}...")

        try:
            enrichment_data = enrich_single_product(product, openfigi_session)

            if enrichment_data is None:
                stats['skipped_non_analyzable'] += 1
                stats['total_processed'] += 1
                continue

            # Update database
            enrich_product(product['id'], enrichment_data, connection=conn)

            # Update stats
            data_source = enrichment_data.get('data_source', '')
            if data_source == 'Manual_NonTradeable':
                stats['non_tradeable'] += 1
            elif 'Failed' in data_source:
                stats['failed'] += 1
            else:
                stats['enriched'] += 1
                if enrichment_data.get('is_domestic') is not None:
                    stats['geographic_enriched'] += 1

        except Exception as e:
            print(f"  Error: {e}")
            stats['failed'] += 1

        stats['total_processed'] += 1

        # Commit every 100 products
        if i % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    # Final report
    duration = datetime.now() - stats['start_time']

    print("\n=== Enrichment Report ===")
    print(f"Duration: {duration}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Skipped (non-analyzable): {stats['skipped_non_analyzable']}")
    print(f"Enriched: {stats['enriched']}")
    print(f"Non-tradeable: {stats['non_tradeable']}")
    print(f"Failed: {stats['failed']}")

    return stats


def main():
    """
    Main entry point for the enrichment pipeline.
    """
    try:
        stats = run_enrichment_pipeline(
            limit=None,   # No limit by default
            start_from=0
        )

        print("\nEnrichment completed successfully")
        print(
            f"Final stats: {stats['enriched']} enriched, "
            f"{stats['failed']} failed"
        )

    except KeyboardInterrupt:
        print("\nEnrichment interrupted by user")
        return 1
    except Exception as e:
        print(f"\nFatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
