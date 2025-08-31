# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Optional

from capitolwatch.db import get_connection
from config import CONFIG


# ---------- Utilities ----------

def normalize_isin(isin: Optional[str]) -> Optional[str]:
    """
    Normalize an ISIN by removing spaces and uppercasing.
    """
    if isin is None:
        return None
    return isin.replace(" ", "").upper().strip()


# ---------- Read API (get*) ----------

def get_product_id_by_isin(
    isin: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[int]:
    """
    Return a product_id given an ISIN (exact match after normalization).

    Args:
        isin: Product ISIN (case-insensitive).
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        The product_id if found, else None.
    """
    isin = normalize_isin(isin)

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT product_id FROM products WHERE isin = ? LIMIT 1",
            (isin,),
        )
        row = cur.fetchone()
        return int(row["product_id"]) if row else None
    finally:
        if close:
            connection.close()


def get_product(
    product_id: int,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[dict]:
    """
    Fetch a single product by its primary key.

    Returns: {product_id, name, isin, type, details} or None
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT product_id, name, isin, type, details
            FROM products
            WHERE product_id = ?
            """,
            (product_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if close:
            connection.close()


def get_products(
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    product_type: Optional[str] = None,
    name_like: Optional[str] = None,
    config: Optional[object] = None,
    connection=None,
) -> list[dict]:
    """
    Return a (paginated) list of products, with optional filters.

    Args:
        limit: Optional page size. If None, returns all rows.
        offset: Offset for pagination (ignored if limit is None).
        product_type: Optional exact match on products.type.
        name_like: Optional substring match on products.name
            (case-insensitive).
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        where: list[str] = []
        params: list[object] = []

        if product_type:
            where.append("type = ?")
            params.append(product_type)

        if name_like:
            where.append("LOWER(name) LIKE ?")
            params.append(f"%{name_like.lower()}%")

        sql = (
            "SELECT product_id, name, isin, type, details FROM products"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY name"

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cur = connection.cursor()
        cur.execute(sql, tuple(params))
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            connection.close()


# ---------- Write API (add*) ----------

def add_product(
    product: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Insert a single product if not present.

    Expected keys in `product`:
      - name (str)
      - isin (str)
      - type (str)
      - details (str, optional)

    Returns:
      True if inserted, False if already existing (ignored).
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    name = product.get("name")
    isin = normalize_isin(product.get("isin"))
    ptype = product.get("type")
    details = product.get("details")

    try:
        cur = connection.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO products (name, isin, type, details)
            VALUES (?, ?, ?, ?)
            """,
            (name, isin, ptype, details),
        )
        if close:
            connection.commit()
        return cur.rowcount > 0
    finally:
        if close:
            connection.close()


def add_products(
    products: list[dict],
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Bulk insert products (idempotent). Returns the number of rows inserted.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    inserted = 0
    try:
        for prod in products:
            if add_product(prod, config=config, connection=connection):
                inserted += 1
        if close:
            connection.commit()
        return inserted
    finally:
        if close:
            connection.close()


def get_or_create_product_id(
    product: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Convenience helper: return the product_id for a product spec, inserting
    it if missing. Uses ISIN as the uniqueness key if provided, otherwise
    falls back to (name, type) uniqueness heuristic.

    Returns:
        product_id (int)
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        isin = normalize_isin(product.get("isin"))
        if isin:
            pid = get_product_id_by_isin(
                isin, config=config, connection=connection
            )
            if pid is not None:
                return pid
            # Not found -> insert
            add_product(product, config=config, connection=connection)
            pid = get_product_id_by_isin(
                isin, config=config, connection=connection
            )
            if pid is None:
                raise RuntimeError("Failed to create product for ISIN")
            return pid

        # Fallback: try to find by (name, type)
        cur = connection.cursor()
        cur.execute(
            (
                "SELECT product_id FROM products "
                "WHERE name = ? AND type = ? LIMIT 1"
            ),
            (product.get("name"), product.get("type")),
        )
        row = cur.fetchone()
        if row:
            return int(row["product_id"])

        add_product(product, config=config, connection=connection)

        cur.execute(
            (
                "SELECT product_id FROM products "
                "WHERE name = ? AND type = ? LIMIT 1"
            ),
            (product.get("name"), product.get("type")),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create product (no ISIN)")
        return int(row["product_id"])
    finally:
        if close:
            connection.close()
