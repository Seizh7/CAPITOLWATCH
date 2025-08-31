"""
Product services: CRUD and helpers for the products table.

Implements add_product returning a generated INTEGER PRIMARY KEY id,
with optional ISIN and details, plus a function to update ISIN/details.
"""

from typing import Optional
import sqlite3

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

def get_id_by_isin(
    isin: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[int]:
    """
    Return a id given an ISIN (exact match after normalization).

    Args:
        isin: Product ISIN (case-insensitive).
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        The id if found, else None.
    """
    isin = normalize_isin(isin)

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT id FROM products WHERE isin = ? LIMIT 1",
            (isin,),
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

    Returns: {id, name, isin, type, details} or None
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT id, name, isin, type, details
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
      - isin (str, optional)
      - details (str, optional)

    Behavior:
      - Uses INTEGER PRIMARY KEY auto-generation for id.
      - ISIN is optional. If provided, it is normalized and must be unique.
      - Details is optional.
      - If a conflicting row already exists (same ISIN or same (name,type)
        when ISIN is missing), the existing id is returned.

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

    isin = normalize_isin(product.get("isin"))
    details = product.get("details")

    try:
        cur = connection.cursor()

        if isin:
            existing_id = get_id_by_isin(
                isin, config=config, connection=connection
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

        # Insert new product
        cur.execute(
            """
            INSERT INTO products (name, type, isin, details)
            VALUES (?, ?, ?, ?)
            """,
            (name, product_type, isin, details),
        )
        new_id = int(cur.lastrowid)
        if close:
            connection.commit()
        return new_id
    finally:
        if close:
            connection.close()


def update_product(
    id: int,
    *,
    isin: Optional[str] = None,
    details: Optional[str] = None,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Update a product to add/set its ISIN and/or details.

    Args:
        id: Target product id to update.
        isin: Optional ISIN to set (normalized, must be unique if provided).
        details: Optional free-form details.

    Returns:
        True if at least one column was updated, False otherwise.

    Raises:
        ValueError on invalid input or ISIN uniqueness conflicts.
    """
    if isin is None and details is None:
        return False

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        sets = []
        params: list[object] = []

        if isin is not None:
            norm_isin = normalize_isin(isin)
            sets.append("isin = ?")
            params.append(norm_isin)

        if details is not None:
            sets.append("details = ?")
            params.append(details)

        params.append(id)

        sql = f"UPDATE products SET {', '.join(sets)} WHERE id = ?"

        cur = connection.cursor()
        try:
            cur.execute(sql, tuple(params))
        except sqlite3.IntegrityError as e:
            # Likely ISIN UNIQUE constraint violation
            raise ValueError(str(e))

        updated = cur.rowcount > 0
        if close:
            connection.commit()
        return updated
    finally:
        if close:
            connection.close()
