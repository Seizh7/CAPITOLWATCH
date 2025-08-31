# Copyright (c) 2025 Seizh7
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


# ---------- Write API (add*) ----------

def add_asset(
    report_id: int,
    asset: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Insert a new asset row for a given report and return the inserted row id.

    Args:
        report_id (int): Target report ID (foreign key in `assets`).
        asset (dict): Asset information with keys:
            - product_id (int, required, FK to products.product_id)
            - owner (str)
            - value (float or str)
            - income_type (str)
            - income (str or float)
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
                report_id, product_id, owner, value,
                income_type, income, comment, parent_asset_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                asset["product_id"],
                asset.get("owner"),
                asset.get("value"),
                asset.get("income_type"),
                asset.get("income"),
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
