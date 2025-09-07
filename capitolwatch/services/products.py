"""
Product services: CRUD and helpers for the products table.

Implements add_product returning a generated INTEGER PRIMARY KEY id,
with enrichment support for OpenFIGI and Yahoo Finance data.
"""

from typing import Optional

from capitolwatch.db import get_connection
from config import CONFIG


# ---------- Utilities ----------

def normalize_ticker(ticker: Optional[str]) -> Optional[str]:
    """
    Normalize a ticker by removing spaces and uppercasing.
    """
    if ticker is None:
        return None
    return ticker.replace(" ", "").upper().strip()


# ---------- Read API (get*) ----------

def get_id_by_ticker(
    ticker: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[int]:
    """
    Return a id given a ticker.

    Args:
        ticker: Product ticker symbol.
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        The id if found, else None.
    """
    ticker = normalize_ticker(ticker)

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT id FROM products WHERE ticker = ? LIMIT 1",
            (ticker,),
        )
        row = cur.fetchone()
        return int(row["id"]) if row else None
    finally:
        if close:
            connection.close()


def get_product(
    id: int,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[dict]:
    """
    Fetch a single product by its primary key.

    Returns: Full product dict or None
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT id, name, type, figi, ticker, exchange, sector, industry,
                   country, asset_class, beta, dividend_yield, expense_ratio,
                   market_cap, currency, is_etf, is_mutual_fund, is_index_fund,
                   market_cap_tier, risk_rating, last_updated, data_source
            FROM products
            WHERE id = ?
            """,
            (id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if close:
            connection.close()


def get_products_without_enrichment(
    *,
    config: Optional[object] = None,
    connection=None
) -> list[dict]:
    """
    Retrieve all products that have no enrichment data.

    Args:
        config: Optional config override (defaults to global CONFIG).
        connection: Optional existing SQLite connection to reuse.

    Returns:
        list[dict]: List of products with keys {id, name, type}.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT id, name, type
            FROM products
            WHERE data_source = 'Manual' AND ticker IS NULL
            """
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            connection.close()


def get_geographic_enrichment_stats(
    *,
    config: Optional[object] = None,
    connection=None
) -> dict:
    """
    Get geographic enrichment statistics for products.

    Returns:
        dict: Statistics about geographic enrichment coverage.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()

        # Basic statistics
        cur.execute('''
            SELECT
                COUNT(*) as total_products,
                COUNT(international_exposure) as geo_enriched,
                COUNT(geographic_classification) as region_classified
            FROM products
            WHERE ticker IS NOT NULL AND ticker != "Manual"
        ''')

        total, geo_enriched, region_classified = cur.fetchone()

        # Distribution by region
        cur.execute('''
            SELECT
                geographic_classification,
                COUNT(*) as count
            FROM products
            WHERE geographic_classification IS NOT NULL
            GROUP BY geographic_classification
            ORDER BY count DESC
        ''')

        regions = dict(cur.fetchall())

        return {
            'total_products': total,
            'geo_enriched': geo_enriched,
            'region_classified': region_classified,
            'regions': regions,
            'coverage_rate': (geo_enriched / total * 100) if total > 0 else 0
        }

    except Exception as e:
        return {
            'error': str(e),
            'total_products': 0,
            'geo_enriched': 0,
            'region_classified': 0,
            'regions': {},
            'coverage_rate': 0
        }
    finally:
        if close:
            connection.close()


def get_all_products_for_embeddings(
    *,
    config: Optional[object] = None,
    connection=None,
) -> list[dict]:
    """
    Retrieve all products from database for embedding generation.

    Args:
        config: Optional config override (defaults to global CONFIG).
        connection: Optional existing SQLite connection to reuse.

    Returns:
        list[dict]: List of products with all relevant fields for embeddings.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute("""
            SELECT id, name, type, sector, industry, asset_class,
                   country, market_cap, beta, dividend_yield, expense_ratio,
                   market_cap_tier, risk_rating, currency,
                   is_etf, is_mutual_fund, is_index_fund
            FROM products
            ORDER BY id
        """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        if close:
            connection.close()


# ---------- Write API (add*/update*) ----------

def add_product(
    product: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Insert a single product and return its generated id.

    Expected keys in `product`:
      - name (str, required)
      - type (str, required)
      - All other fields are optional (figi, ticker, etc.)

    Behavior:
      - Uses INTEGER PRIMARY KEY auto-generation for id.
      - If a conflicting row already exists (same name,type),
        the existing id is returned.

    Returns:
      id (int)
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    name = product.get("name")
    product_type = product.get("type")
    if not name or not product_type:
        raise ValueError("'name' and 'type' are required to create a product")

    ticker = normalize_ticker(product.get("ticker"))

    try:
        cur = connection.cursor()

        # Check for existing by ticker first (if provided)
        if ticker:
            existing_id = get_id_by_ticker(
                ticker, config=config, connection=connection
            )
            if existing_id is not None:
                return existing_id

        # Fallback: look for existing by (name, type)
        cur.execute(
            (
                "SELECT id FROM products "
                "WHERE name = ? AND type = ? LIMIT 1"
            ),
            (name, product_type),
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])

        # Insert new product with basic fields only
        cur.execute(
            """
            INSERT INTO products (name, type, ticker, data_source)
            VALUES (?, ?, ?, ?)
            """,
            (name, product_type, ticker, "Manual"),
        )
        new_id = int(cur.lastrowid)
        if close:
            connection.commit()
        return new_id
    finally:
        if close:
            connection.close()


def enrich_product(
    product_id: int,
    enrichment_data: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Enrich a product with API data from OpenFIGI and Yahoo Finance.
    Includes automatic geographic enrichment when applicable.

    Args:
        product_id: Target product id to enrich.
        enrichment_data: Dict with enrichment fields (figi, sector, etc.)
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        True if product was updated, False otherwise.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        # Build dynamic UPDATE statement based on provided fields
        valid_fields = {
            'figi', 'ticker', 'exchange', 'sector', 'industry', 'country',
            'asset_class', 'beta', 'dividend_yield', 'expense_ratio',
            'market_cap', 'currency', 'is_etf', 'is_mutual_fund',
            'is_index_fund', 'market_cap_tier', 'risk_rating',
            'last_updated', 'data_source', 'international_exposure',
            'geographic_classification'
        }

        sets = []
        params = []

        for field, value in enrichment_data.items():
            if field in valid_fields and value is not None:
                sets.append(f"{field} = ?")
                params.append(value)

        if not sets:
            return False

        params.append(product_id)
        sql = f"UPDATE products SET {', '.join(sets)} WHERE id = ?"

        cur = connection.cursor()
        cur.execute(sql, tuple(params))

        updated = cur.rowcount > 0
        if close:
            connection.commit()
        return updated
    finally:
        if close:
            connection.close()
