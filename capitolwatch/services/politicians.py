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


def add_politicians(
    politicians,
    *,
    config: Optional[object] = None,
    connection=None
):
    """
    Inserts the list of senators into the 'politicians' table in the database.

    Args:
        politicians (list of dict): List of politician information to insert.
    """
    return add_politician_list(
        politicians, config=config, connection=connection
    )


def add_politician(
    politician: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Insert a single politician into the database if it doesn't already exist.

    Expected keys in `politician`:
      - first_name (str)
      - last_name (str)
      - party (str)
      - bioguide_id (str) -> stored as `id` in DB

    Returns:
      - bool: True if inserted, False if ignored (already present).
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO politicians (
                first_name, last_name, party, id
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                politician["first_name"],
                politician["last_name"],
                politician["party"],
                politician["bioguide_id"],
            ),
        )

        if close:
            connection.commit()
        return cur.rowcount > 0
    finally:
        if close:
            connection.close()


def add_politician_list(
    politicians,
    *,
    config: Optional[object] = None,
    connection=None,
):
    """
    Insert a list of politicians with add_politician
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        for person in politicians:
            add_politician(person, config=config, connection=connection)
        if close:
            connection.commit()
    finally:
        if close:
            connection.close()
