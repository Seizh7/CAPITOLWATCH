# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Optional
from capitolwatch.db import get_connection
from config import CONFIG


# ---------- Read API (get*) ----------

def get_assets_for_report(
    report_id: int,
    *,
    config: Optional[object] = None,
    connection=None,
) -> list[dict]:
    """
    Retrieve all financial assets linked to a given report.

    Args:
        report_id (int): Target report ID (foreign key in `assets`).
        config (Optional[object]): Optional config override.
        connection: Optional existing SQLite connection to reuse.

    Returns:
        list[dict]: List of asset rows with fields:
            - id
            - owner
            - value
            - income_type
            - income
            - comment
            - parent_asset_id
            - product_name (from products table)
            - product_type (from products table)
            - isin (from products table)
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT a.id, a.owner, a.value, a.income_type, a.income, a.comment,
                   a.parent_asset_id,
                   pr.name AS product_name,
                   pr.type AS product_type,
                   pr.isin
            FROM assets a
            JOIN products pr ON a.product_id = pr.product_id
            WHERE a.report_id = ?
            ORDER BY a.id
            """,
            (report_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            connection.close()


def get_politician_assets(
    politician_id: int,
    *,
    config: Optional[object] = None
) -> list[dict]:
    """
    Retrieve all financial assets for a given politician.

    Args:
        politician_id (int): Target politician ID.
        config (Optional[object]): Optional config override.

    Returns:
        list[dict]: List of asset rows with fields:
            - id: asset ID
            - product_id: product ID
            - value: asset value
            - owner: asset owner
            - income_type: type of income
            - product_name: name of the product
            - product_type: type of product
    """
    if config is None:
        config = CONFIG

    with get_connection(config) as conn:
        cur = conn.execute(
            """
            SELECT a.id, a.product_id, a.value, a.owner, a.income_type,
                   a.income, a.comment,
                   pr.name AS product_name,
                   pr.type AS product_type,
                   pr.isin
            FROM assets a
            JOIN products pr ON a.product_id = pr.id
            WHERE a.politician_id = ? AND a.value > 0
            ORDER BY a.value DESC
            """,
            (politician_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_politician_assets_simple(
    politician_id: int,
    *,
    config: Optional[object] = None
) -> list[dict]:
    """
    Retrieve simple asset data (product_id, value) for a given politician.

    This function is optimized for embedding computation and returns
    only the essential fields without joining with products table.

    Args:
        politician_id (int): Target politician ID.
        config (Optional[object]): Optional config override.

    Returns:
        list[dict]: List of asset rows with fields:
            - product_id: product ID
            - value: asset value
    """
    if config is None:
        config = CONFIG

    with get_connection(config) as conn:
        cur = conn.execute(
            """
            SELECT product_id, value
            FROM assets
            WHERE politician_id = ?
              AND value IS NOT NULL
              AND value != ''
            ORDER BY id
            """,
            (politician_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------- Write API (add*) ----------

def add_asset(
    report_id: int,
    asset: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Insert a new asset row for a given report and return the inserted row id.

    Args:
        report_id (int): Target report ID (foreign key in `assets`).
        asset (dict): Asset information with keys:
            - politician_id (str, optional, FK to politicians.politician_id)
            - product_id (int, required, FK to products.product_id)
            - owner (str)
            - value (float or str)
            - income_type (str)
            - income (str or float)
            - income_subtype (str, optional)
            - comment (str)
            - parent_asset_id (int | None)
        config (Optional[object]): Optional config override.
        connection: Optional existing DB connection.

    Returns:
        int: The primary key (id) of the inserted asset row.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            INSERT INTO assets (
                report_id, politician_id, product_id, owner, value,
                income_type, income, income_subtype, comment, parent_asset_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                asset.get("politician_id"),
                asset["product_id"],
                asset.get("owner"),
                asset.get("value"),
                asset.get("income_type"),
                asset.get("income"),
                asset.get("income_subtype"),
                asset.get("comment"),
                asset.get("parent_asset_id"),
            ),
        )
        inserted_id = cur.lastrowid
        if close:
            connection.commit()
        return int(inserted_id)
    finally:
        if close:
            connection.close()
