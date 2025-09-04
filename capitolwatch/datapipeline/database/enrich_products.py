# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Pipeline for financial product enrichment using OpenFIGI and Yahoo Finance.
Enhances database products with financial metadata.
"""

import re
import time
import json
import requests
import yfinance as yf
from datetime import datetime, timezone
from typing import Optional, Dict

from config.development import Config
from capitolwatch.db import get_connection
from capitolwatch.services.products import (
    get_products_without_enrichment,
    enrich_product
)


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
            'asset_class_yahoo': info.get('quoteType')  # EQUITY, ETF
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
    Determines fund type flags.

    Args:
        name (str): Product name.
        asset_class (str): Asset class.
    Returns:
        Dict[str, bool]: Flags indicating fund types.
    """
    name_lower = name.lower() if name else ""
    return {
        'is_etf': asset_class == 'ETF',
        'is_mutual_fund': asset_class == 'Mutual Fund',
        'is_index_fund': 'index' in name_lower
    }


def enrich_single_product(
    product: Dict,
    openfigi_session: requests.Session
) -> Dict:
    """
    Enriches a single product.

    Args:
        product (Dict): Product dictionary with 'id', 'name', 'type'.
        openfigi_session (requests.Session): OpenFIGI HTTP session.
    Returns:
        Dict: Enrichment data.
    """
    product_id = product['id']
    name = product['name']

    print(f"[{product_id}] Product: {name}")

    # Ticker extraction
    ticker = extract_ticker(name)
    if not ticker:
        print("  No extractable ticker - marked as non-tradeable")
        return {
            'data_source': 'Manual_NonTradeable',
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    print(f"  Ticker: {ticker}")

    # Data retrieval
    openfigi_data = get_openfigi_security_info(openfigi_session, ticker)
    yahoo_data = get_yahoo_security_info(ticker)

    openfigi_success = openfigi_data is not None
    yahoo_success = yahoo_data is not None

    status = (
        "Both sources OK" if openfigi_success and yahoo_success
        else "Partial success" if openfigi_success or yahoo_success
        else "Failed"
    )
    print(
        (
            f"  Status: {status} | OpenFIGI: {'OK' if openfigi_success else 'Failed'} | "
            f"Yahoo: {'OK' if yahoo_success else 'Failed'}"
        )
    )

    if not openfigi_success and not yahoo_success:
        return {
            'ticker': ticker,
            'data_source': 'API_Failed',
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    # Build enrichment data
    enrichment = {
        'ticker': ticker,
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'data_source': 'OpenFIGI+Yahoo'
    }
    # OpenFIGI data
    if openfigi_data:
        enrichment.update({
            'figi': openfigi_data.get('figi'),
            'exchange': openfigi_data.get('exchange')
        })
    # Yahoo Finance data
    if yahoo_data:
        enrichment.update({
            'sector': yahoo_data.get('sector'),
            'industry': yahoo_data.get('industry'),
            'country': yahoo_data.get('country'),
            'currency': yahoo_data.get('currency', 'USD'),
            'market_cap': yahoo_data.get('market_cap'),
            'beta': yahoo_data.get('beta'),
            'dividend_yield': yahoo_data.get('dividend_yield'),
            'expense_ratio': yahoo_data.get('expense_ratio')
        })
    # Calculated classifications
    asset_class = classify_asset_class(openfigi_data or {}, yahoo_data or {})
    enrichment['asset_class'] = asset_class

    market_cap_tier = calculate_market_cap_tier(enrichment.get('market_cap'))
    if market_cap_tier:
        enrichment['market_cap_tier'] = market_cap_tier

    risk_rating = calculate_risk_rating(
        enrichment.get('sector', ''),
        asset_class,
        enrichment.get('beta')
    )
    enrichment['risk_rating'] = risk_rating

    # Fund flags
    fund_flags = determine_fund_flags(name, asset_class)
    enrichment.update(fund_flags)

    return enrichment


def run_enrichment_pipeline(
    config: Config,
    limit: Optional[int] = None,
    start_from: int = 0
) -> Dict:
    """
    Runs the financial product enrichment pipeline.

    Args:
        config (Config): Application configuration.
        limit (Optional[int]): Maximum number of products to process.
        start_from (int): Starting index.
    Returns:
        Dict: Statistics dictionary.
    """
    print("STARTING PRODUCT ENRICHMENT PIPELINE")

    # Retrieve products to enrich
    conn = get_connection(config)
    products = get_products_without_enrichment(connection=conn)
    conn.close()

    total_products = len(products)
    print(f"{total_products} products to enrich")

    if start_from > 0:
        products = products[start_from:]
        print(f"Starting from index {start_from}")

    if limit:
        products = products[:limit]
        print(f"Limiting to {limit} products")

    # Create OpenFIGI session
    openfigi_session = get_openfigi_session(config.openfigi_api_key)

    # Statistics
    stats = {
        'total_processed': 0,
        'enriched': 0,
        'failed': 0,
        'non_tradeable': 0,
        'start_time': datetime.now()
    }

    # Processing loop
    for i, product in enumerate(products, 1):
        print(f"\n[{i}/{len(products)}]", end=" ")

        try:
            enrichment_data = enrich_single_product(product, openfigi_session)

            # Update in database
            conn = get_connection(config)
            success = enrich_product(
                product['id'],
                enrichment_data,
                connection=conn
            )
            conn.commit()
            conn.close()

            if success:
                data_source = enrichment_data.get('data_source')
                if data_source == 'Manual_NonTradeable':
                    stats['non_tradeable'] += 1
                elif data_source == 'API_Failed':
                    stats['failed'] += 1
                else:
                    stats['enriched'] += 1
            else:
                stats['failed'] += 1
                print("  Database update failed")

        except Exception as e:
            print(f"  Error: {e}")
            stats['failed'] += 1

        stats['total_processed'] += 1

        # Rate limiting
        time.sleep(0.5)

    # Final report
    duration = datetime.now() - stats['start_time']

    print(f"\n\n{'='*60}")
    print("PRODUCT ENRICHMENT REPORT")
    print(f"Duration: {duration}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Successfully enriched: {stats['enriched']}")
    print(f"Non-tradeable: {stats['non_tradeable']}")
    print(f"Failures: {stats['failed']}")

    if stats['total_processed'] > 0:
        success_rate = (stats['enriched'] / stats['total_processed']) * 100
        print(f"Success rate: {success_rate:.1f}%")

    return stats


def main():
    """
    Main entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Financial Product Enrichment Pipeline"
    )
    parser.add_argument('--limit', type=int,
                        help="Maximum number of products to process")
    parser.add_argument('--start-from', type=int, default=0,
                        help="Starting index")
    parser.add_argument('--test', action='store_true',
                        help="Test mode (first 10 products only)")

    args = parser.parse_args()

    config = Config()
    limit = 10 if args.test else args.limit

    try:
        run_enrichment_pipeline(
            config=config,
            limit=limit,
            start_from=args.start_from
        )
        print("\nEnrichment completed!")

    except KeyboardInterrupt:
        print("\n\nEnrichment interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
