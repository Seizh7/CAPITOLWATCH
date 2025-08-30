# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import re
from typing import Optional, Iterable
from capitolwatch.db import get_connection
from config import CONFIG


def normalize_name(name: str) -> str:
    """
    Normalizes a personal name by removing punctuation, converting to
    lowercase, and collapsing extra spaces.
    """
    if not name:
        return ""
    name = name.lower()                  # Lowercase for consistent comparison
    name = re.sub(r"[.']", "", name)     # Remove dots and apostrophes
    name = re.sub(r"[-]", " ", name)     # Replace hyphens with space
    name = re.sub(r"\s+", " ", name)     # Collapse multiple spaces into one
    name = name.strip(", ")              # Remove commas and spaces
    return name.strip()


def get_politician_id(
    first_name: str,
    last_name: str,
    *,
    config: Optional[object] = None,
    connection=None
) -> Optional[str]:
    """
    Look up a politician ID by first/last name.

    Args:
        first_name (str): First name (will be normalized).
        last_name (str): Last name (will be normalized).
        config (object, optional): Optional config override.
        connection (sqlite3.Connection, optional): Reuse an existing
            DB connection.

    Returns:
        Optional[str]: Politician ID if found, else None.
    """
    first_name = normalize_name(first_name)
    last_name = normalize_name(last_name)

    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT id FROM politicians
            WHERE (first_name=? AND last_name=?)
               OR (first_name=? AND last_name=?)
            LIMIT 1
            """,
            (first_name, last_name, last_name, first_name),
        )
        row = cur.fetchone()
        return row["id"] if row else None
    finally:
        if close:
            connection.close()


def list_politicians(
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    config: Optional[object] = None,
    connection=None
) -> list[dict]:
    """
    Return a list of politicians from the DB.

    Args:
        limit (int | None): Max number of rows (None = no limit).
        offset (int): Offset for pagination.
        config (object, optional): Optional config override.
        connection (sqlite3.Connection, optional): Reuse an existing
            DB connection.

    Returns:
        list[dict]: List of politicians with {id, first_name, last_name, party}
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        sql = """
            SELECT id, first_name, last_name, party
            FROM politicians
            ORDER BY last_name, first_name
        """
        params: Iterable = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (limit, offset)

        cur = connection.cursor()
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            connection.close()


def get_politician_by_id(
    politician_id: str,
    *,
    config: Optional[object] = None,
    connection=None
) -> Optional[dict]:
    """
    Fetch a single politician record by ID.

    Args:
        politician_id (str): The unique politician ID.
        config (object, optional): Optional config override.
        connection (sqlite3.Connection, optional): Reuse an existing
            DB connection.

    Returns:
        dict | None: Politician record if found, else None.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            SELECT id, first_name, last_name, party
            FROM politicians
            WHERE id = ?
            """,
            (politician_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if close:
            connection.close()


__all__ = ["get_politician_id", "list_politicians", "get_politician_by_id"]
