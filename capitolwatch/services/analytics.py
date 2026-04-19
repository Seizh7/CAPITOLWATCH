# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Analytics service for multi-table database operations.

This service handles complex queries that join multiple tables
for portfolio analysis and clustering purposes.
"""

import pandas as pd
from typing import Optional

from capitolwatch.db import get_connection
from config import CONFIG


# ---------- Read API (get*) ----------

def get_politicians_with_assets(
    *,
    config: Optional[object] = None,
    connection=None,
) -> list:
    """
    Get list of politician IDs who have investment assets.

    Args:
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        List of politician IDs
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT DISTINCT p.id
            FROM politicians p
            JOIN reports r ON p.id = r.politician_id
            JOIN assets a ON r.id = a.report_id
            """
        )
        return [row['id'] for row in cur.fetchall()]
    finally:
        if close:
            connection.close()


def get_politician_portfolio_raw_data(
    politician_id: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> pd.DataFrame:
    """
    Get raw portfolio data for a politician with product details.

    Args:
        politician_id: ID of the politician
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        DataFrame with portfolio data
    """

    query = """
    SELECT
        p.id as politician_id,
        (p.first_name || ' ' || p.last_name) as politician_name,
        p.party,
        pr.sector,
        pr.industry,
        pr.asset_class,
        pr.market_cap_tier,
        pr.name as product_name,
        COUNT(*) as asset_count,
        a.value as asset_value,
        a.income_type,
        a.income
    FROM politicians p
    JOIN reports r ON p.id = r.politician_id
    JOIN assets a ON r.id = a.report_id
    JOIN products pr ON a.product_id = pr.id
    WHERE p.id = ? AND pr.sector IS NOT NULL
    GROUP BY p.id, pr.sector, pr.industry, pr.asset_class, pr.market_cap_tier,
        pr.id
    ORDER BY asset_count DESC
    """

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        return pd.read_sql_query(query, connection, params=(politician_id,))
    finally:
        if close:
            connection.close()


def get_sector_distribution_for_politician(
    politician_id: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> dict:
    """
    Get sector distribution (asset counts) for a specific politician.

    Args:
        politician_id: ID of the politician
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        Dict mapping sector names to asset counts
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT
                pr.sector,
                COUNT(*) as asset_count
            FROM politicians p
            JOIN reports r ON p.id = r.politician_id
            JOIN assets a ON r.id = a.report_id
            JOIN products pr ON a.product_id = pr.id
            WHERE p.id = ? AND pr.sector IS NOT NULL
            GROUP BY pr.sector
            ORDER BY asset_count DESC
            """,
            (politician_id,),
        )
        return {row['sector']: row['asset_count'] for row in cur.fetchall()}
    finally:
        if close:
            connection.close()


def get_industry_distribution_for_politician(
    politician_id: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> dict:
    """
    Get industry distribution (asset counts) for a specific politician.

    Args:
        politician_id: ID of the politician
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        Dict mapping industry names to asset counts
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT
                pr.industry,
                COUNT(*) as asset_count
            FROM politicians p
            JOIN reports r ON p.id = r.politician_id
            JOIN assets a ON r.id = a.report_id
            JOIN products pr ON a.product_id = pr.id
            WHERE p.id = ? AND pr.industry IS NOT NULL
            GROUP BY pr.industry
            ORDER BY asset_count DESC
            """,
            (politician_id,),
        )
        return {row['industry']: row['asset_count'] for row in cur.fetchall()}
    finally:
        if close:
            connection.close()


def get_portfolio_summary_by_party(
    *,
    config: Optional[object] = None,
    connection=None,
) -> pd.DataFrame:
    """
    Get portfolio summary statistics grouped by political party.

    Args:
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        DataFrame with party-level portfolio statistics
    """

    query = """
    SELECT
        p.party,
        pr.sector,
        COUNT(*) as investment_count,
        COUNT(DISTINCT p.id) as politician_count,
        COUNT(DISTINCT pr.id) as unique_products
    FROM assets a
    JOIN reports r ON a.report_id = r.id
    JOIN politicians p ON r.politician_id = p.id
    JOIN products pr ON a.product_id = pr.id
    WHERE p.party IS NOT NULL AND pr.sector IS NOT NULL
    GROUP BY p.party, pr.sector
    ORDER BY p.party, investment_count DESC
    """

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        return pd.read_sql_query(query, connection)
    finally:
        if close:
            connection.close()


def get_politician_asset_counts(
    *,
    config: Optional[object] = None,
    connection=None,
) -> pd.DataFrame:
    """
    Get asset counts for all politicians.

    Args:
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        DataFrame with politician asset counts
    """

    query = """
    SELECT
        p.id as politician_id,
        (p.first_name || ' ' || p.last_name) as politician_name,
        p.party,
        COUNT(a.id) as total_assets,
        COUNT(DISTINCT pr.sector) as unique_sectors,
        COUNT(DISTINCT pr.industry) as unique_industries
    FROM politicians p
    JOIN reports r ON p.id = r.politician_id
    JOIN assets a ON r.id = a.report_id
    JOIN products pr ON a.product_id = pr.id
    WHERE pr.sector IS NOT NULL
    GROUP BY p.id, p.first_name, p.last_name, p.party
    ORDER BY total_assets DESC
    """

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        return pd.read_sql_query(query, connection)
    finally:
        if close:
            connection.close()


def get_active_politicians_dataframe(
    *,
    config: Optional[object] = None,
    connection=None,
) -> pd.DataFrame:
    """
    Get all politicians who have at least one asset.

    This is the base dataset for clustering analysis.

    Args:
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        DataFrame with columns: [id, first_name, last_name, party]
    """
    query = """
        SELECT DISTINCT
            p.id,
            p.first_name,
            p.last_name,
            p.party
        FROM politicians p
        INNER JOIN reports r ON p.id = r.politician_id
        INNER JOIN assets a ON r.id = a.report_id
        ORDER BY p.last_name, p.first_name
    """

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        return pd.read_sql_query(query, connection)
    finally:
        if close:
            connection.close()


def get_assets_with_products_dataframe(
    *,
    config: Optional[object] = None,
    connection=None,
) -> pd.DataFrame:
    """
    Get all assets enriched with product information for active politicians.

    Performs INNER JOIN between:
        - assets table
        - products table (product details)
        - reports table (link to politicians)
        - politicians table (filter active only)

    Args:
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        DataFrame with columns:
            - asset_id: ID of the asset
            - politician_id: ID of the politician
            - product_id: ID of the product
            - value: String representation of value range
            - product_name: Name of the product
            - subtype: Product subtype (e.g., 'Stock', 'Mutual Fund')
            - sector: Economic sector (e.g., 'Technology', 'Healthcare')

        Returns all assets with values for active politicians, including:
            - Standalone assets (bank accounts, pension plans)
            - Child assets (mutual funds within IRAs)
        Excludes only empty asset containers (IRAs, 401k without direct values)
    """
    query = """
        SELECT
            a.id AS asset_id,
            r.politician_id,
            a.product_id,
            a.value,
            pr.name AS product_name,
            pr.subtype,
            pr.sector
        FROM assets a
        INNER JOIN products pr ON a.product_id = pr.id
        INNER JOIN reports r ON a.report_id = r.id
        INNER JOIN politicians p ON r.politician_id = p.id
        WHERE p.id IN (
            SELECT DISTINCT r2.politician_id
            FROM reports r2
            INNER JOIN assets a2 ON r2.id = a2.report_id
        )
        AND a.value IS NOT NULL
        AND a.value != ''
        ORDER BY r.politician_id, a.id
    """

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        return pd.read_sql_query(query, connection)
    finally:
        if close:
            connection.close()
