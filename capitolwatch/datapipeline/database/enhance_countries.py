#!/usr/bin/env python3
"""
Post-processing script to enhance country data for financial products.
Fills missing country information using intelligent rules and patterns.
"""

import re
from typing import Optional, Dict
from config.development import Config
from capitolwatch.db import get_connection


# ============================================================================
# COUNTRY INFERENCE RULES
# ============================================================================

def infer_country_from_exchange(exchange: str, ticker: str = "", name: str = "") -> Optional[str]:
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
                r'Nestle|ASML|SAP|Toyota|Sony|Samsung',  # Known foreign companies
                r'\.L$|\.TO$|\.PA$|\.F$',  # Foreign ticker suffixes
                r'Royal.*Dutch|Unilever|Novartis'
            ]
            
            combined_text = f"{ticker} {name}"
            for pattern in adr_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return None  # Don't assume US for likely foreign companies
        
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


def infer_country_from_ticker_pattern(ticker: str, name: str) -> Optional[str]:
    """Infer country from ticker patterns and company/fund names."""
    if not ticker or not name:
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
    combined_text = f"{ticker} {name}"
    
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


# ============================================================================
# MAIN ENHANCEMENT LOGIC
# ============================================================================

def enhance_product_country(product: Dict) -> Optional[str]:
    """
    Enhance country information for a single product.
    
    Args:
        product: Dict with product data
        
    Returns:
        Inferred country or None
    """
    # Skip if country already exists
    if product.get('country'):
        return None
    
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


def run_country_enhancement():
    """Run the country enhancement process."""
    print("🌍 STARTING COUNTRY ENHANCEMENT")
    print("=" * 50)
    
    config = Config()
    conn = get_connection(config)
    
    # Get products missing country information
    cursor = conn.execute('''
        SELECT id, ticker, name, exchange, currency, country
        FROM products 
        WHERE data_source = 'OpenFIGI+Yahoo'
        AND (country IS NULL OR country = '')
        AND ticker IS NOT NULL
    ''')
    
    products = [dict(row) for row in cursor.fetchall()]
    total_missing = len(products)
    
    print(f"📊 Products missing country: {total_missing:,}")
    
    if total_missing == 0:
        print("✅ All products already have country information!")
        conn.close()
        return
    
    # Statistics
    stats = {
        'enhanced': 0,
        'exchange_based': 0,
        'pattern_based': 0,
        'currency_based': 0,
        'unchanged': 0
    }
    
    # Process each product
    for i, product in enumerate(products, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{total_missing} ({(i/total_missing)*100:.1f}%)")
        
        # Get enhancement method details for statistics
        inferred_country = None
        method = None
        
        # Try pattern method first (most reliable for company origin)
        inferred_country = infer_country_from_ticker_pattern(
            product.get('ticker'), product.get('name')
        )
        if inferred_country:
            method = 'pattern_based'
        
        # Try exchange method (only for domestic products)
        if not inferred_country:
            inferred_country = infer_country_from_exchange(
                product.get('exchange'), 
                product.get('ticker'), 
                product.get('name')
            )
            if inferred_country:
                method = 'exchange_based'
        
        # Currency method
        if not inferred_country:
            inferred_country = infer_country_from_currency(product.get('currency'))
            if inferred_country:
                method = 'currency_based'
        
        # Update database if country inferred
        if inferred_country:
            cursor = conn.execute('''
                UPDATE products 
                SET country = ? 
                WHERE id = ?
            ''', (inferred_country, product['id']))
            
            stats['enhanced'] += 1
            stats[method] += 1
            
            # Debug output for first 10
            if i <= 10:
                print(f"  [{product['id']:3}] {product['ticker']:6} -> {inferred_country:15} (via {method})")
        else:
            stats['unchanged'] += 1
    
    conn.commit()
    conn.close()
    
    # Final report
    print(f"\n{'='*50}")
    print("🌍 COUNTRY ENHANCEMENT REPORT")
    print(f"{'='*50}")
    print(f"📦 Total processed: {total_missing:,}")
    print(f"✅ Successfully enhanced: {stats['enhanced']:,}")
    print(f"⚪ Unchanged: {stats['unchanged']:,}")
    
    if stats['enhanced'] > 0:
        success_rate = (stats['enhanced'] / total_missing) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        print(f"\n🔍 Enhancement methods:")
        print(f"  Exchange-based: {stats['exchange_based']:,}")
        print(f"  Pattern-based: {stats['pattern_based']:,}")
        print(f"  Currency-based: {stats['currency_based']:,}")
    
    return stats


def main():
    """Entry point."""
    try:
        stats = run_country_enhancement()
        print(f"\n🎉 Country enhancement completed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
