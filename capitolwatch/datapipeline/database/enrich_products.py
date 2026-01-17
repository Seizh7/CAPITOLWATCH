# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Pipeline for comprehensive product enrichment using financial and geographic
data.
Enhances database products with financial metadata and geographic information.
"""

import re
import time
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
# Product types relevant for financial investment analysis
# These products contain enrichable data (ticker, sector, etc.)
ANALYZABLE_PRODUCT_TYPES = {
    # Actions cotées en bourse
    'Corporate SecuritiesStock',
    'Corporate SecuritiesStock Option',
    'American Depository Receipt',

    # Fonds d'investissement
    'Mutual FundsMutual Fund',
    'Mutual FundsExchange Traded Fund/Note',
    'Mutual FundsStable Value Fund',

    # Obligations (partiellement enrichissables)
    'Corporate SecuritiesCorporate Bond',
    'Government SecuritiesUS Treasury/Agency Security',
    'Government SecuritiesMunicipal Security',
    'Foreign BondsForeign Bond',
}

# Product types excluded from analysis (not relevant for clustering)
# - Non-financial assets (real estate, farms)
# - Containers without detailed contents (IRA, 401k, managed accounts)
# - Insurance products and annuities
# - Private, non-listed holdings
EXCLUDED_PRODUCT_TYPES = {
    # Dépôts et comptes bancaires
    'Bank Deposit',
    'Brokerage/Managed Account',

    # Assurance et rentes
    'Life InsuranceWhole',
    'Life InsuranceUniversal',
    'Life InsuranceVariable',
    'AnnuityFixed',
    'AnnuityVariable Annuity',

    # Immobilier et actifs physiques
    'Real EstateResidential',
    'Real EstateCommercial',
    'Real EstateUnimproved Land',
    'Real EstateMineral Rights',
    'Real EstateREIT',
    'Farm',
    'Personal PropertyOther Property',

    # Entités commerciales privées
    'Business EntityLimited Liability Company (LLC)',
    'Business EntityLimited Partnership (LP)',
    'Business EntityLimited Liability Limited Partnership (LLLP)',
    'Business EntityGeneral Partnership',
    'Business EntitySole Proprietorship',
    'Corporate SecuritiesNon-Public Stock',

    # Plans de retraite (containers)
    'Retirement PlansIRA',
    'Retirement Plans401(k), 403(b), or other Defined Contribution Plan.',
    'Retirement PlansDefined Benefit Pension Plan',
    'Retirement PlansDeferred Compensation',

    # Compensation différée
    'Deferred Compensation',
    'Deferred CompensationDeferred Compensation - Other',
    'Deferred CompensationDeferred Compensation - Cash',

    # Trusts et fiducies
    'TrustGeneral Trust',
    'TrustBlind',
    'TrustExcepted',

    # Épargne éducation
    'Education Savings Plans529 College Savings Plan',
    'UGMA/UTMA',

    # Fonds privés (non enrichissables via API)
    'Investment FundPrivate Equity Fund',
    'Investment FundHedge Fund',
    'Investment FundInvestment Club',

    # Autres
    'Accounts ReceivableFrom a Business',
    'Accounts ReceivableFrom an Individual',
    'Intellectual PropertyCopyrights',
    'Cryptocurrency',
    'Other',
}


def is_product_analyzable(product_type: str) -> bool:
    """
    Determine whether a product type is relevant for analysis.

    Args:
        product_type: The financial product type.

    Returns:
        True if the product should be enriched and analyzed.
    """
    if not product_type:
        return False

    # Vérification explicite dans les listes
    if product_type in ANALYZABLE_PRODUCT_TYPES:
        return True
    if product_type in EXCLUDED_PRODUCT_TYPES:
        return False

    # Par défaut, on inclut les types inconnus pour éviter de manquer
    # des données potentiellement intéressantes
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
        'last_updated': datetime.now(timezone.utc).isoformat()
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
                'id': product_id,
                'name': name,
                'ticker': None,
                'exchange': None,
                'currency': None,
                'country': None
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
                'expense_ratio': yahoo_data.get('expense_ratio')
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

        # Fund flags
        enrichment.update(determine_fund_flags(name, asset_class))

    # --- Geographic enrichment ---
    try:
        # Create comprehensive product dict for geographic enrichment
        product_dict = {
            'id': product_id,
            'name': name,
            'ticker': enrichment.get('ticker'),
            'exchange': enrichment.get('exchange'),
            'currency': enrichment.get('currency'),
            'country': enrichment.get('country')
        }
        geo_data = enrich_product_geography(product_dict)
        if geo_data:
            enrichment.update(geo_data)
            print("  Geographic enrichment applied")

            # Update data source to reflect both financial and geographic
            if enrichment['data_source'] != 'API_Failed':
                enrichment['data_source'] = (
                    enrichment['data_source'] + '+Geographic'
                )
    except Exception as e:
        print(f"  Geographic enrichment failed: {e}")

    return enrichment


def run_enrichment_pipeline(
    limit: Optional[int] = None,
    start_from: int = 0
) -> Dict:
    """
    Run the comprehensive product enrichment pipeline with financial and
    geographic data.

    Args:
        limit (Optional[int]): Maximum number of products to process. Defaults
        to None.
        start_from (int): Starting index. Defaults to 0.

    Returns:
        Dict: Statistics dictionary containing counts and processing metadata.
    """
    print("Starting comprehensive product enrichment pipeline")

    # Retrieve products to enrich
    conn = get_connection(CONFIG)
    products = get_products_without_enrichment(connection=conn)
    conn.close()

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

    # Processing loop
    for i, product in enumerate(products, 1):
        print(f"\n[{i}/{len(products)}] Processing product {product['id']}")

        try:
            # Perform enrichment (financial + geographic)
            enrichment_data = enrich_single_product(
                product,
                openfigi_session
            )

            # Skip if product type is not analyzable
            if enrichment_data is None:
                stats['skipped_non_analyzable'] += 1
                stats['total_processed'] += 1
                continue

            # Update in database
            conn = get_connection(CONFIG)
            success = enrich_product(
                product['id'],
                enrichment_data,
                connection=conn
            )
            conn.commit()
            conn.close()

            if success:
                data_source = enrichment_data.get('data_source', '')
                if data_source == 'Manual_NonTradeable':
                    stats['non_tradeable'] += 1
                elif 'Failed' in data_source:
                    stats['failed'] += 1
                else:
                    stats['enriched'] += 1
                    if 'Geographic' in data_source:
                        stats['geographic_enriched'] += 1
            else:
                stats['failed'] += 1
                print("  Database update failed")

        except Exception as e:
            print(f"  Error during enrichment: {e}")
            stats['failed'] += 1

        stats['total_processed'] += 1

        # Rate limiting to avoid API overload
        time.sleep(0.5)

    # Final report
    duration = datetime.now() - stats['start_time']

    print("\nComprehensive Product Enrichment Report")
    print(f"Duration: {duration}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Skipped (non-analyzable types): {stats['skipped_non_analyzable']}")
    print(f"Successfully enriched: {stats['enriched']}")
    print(f"Geographic enriched: {stats['geographic_enriched']}")
    print(f"Non-tradeable: {stats['non_tradeable']}")
    print(f"Failures: {stats['failed']}")

    skipped = stats['skipped_non_analyzable']
    analyzable_total = stats['total_processed'] - skipped
    if analyzable_total > 0:
        success_rate = (stats['enriched'] / analyzable_total) * 100
        print(f"Success rate (analyzable only): {success_rate:.1f}%")

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
