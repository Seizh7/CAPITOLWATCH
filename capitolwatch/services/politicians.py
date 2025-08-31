# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import re
from typing import Optional, Iterable

from capitolwatch.db import get_connection
from config import CONFIG


# ---------- Utilities ----------

def normalize_name(name: str) -> str:
    """
    Normalize a personal name: lowercase, remove punctuation, replace hyphens
    by spaces, collapse multiple spaces, strip commas/edges.
    """
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[.']", "", name)
    name = re.sub(r"-", " ", name)
    name = re.sub(r"\s+", " ", name).strip(", ")
    return name.strip()


# ---------- Read API (get*) ----------

def get_politician_id_by_name(
    first_name: str,
    last_name: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[str]:
    """
    Return a politician ID given normalized first/last names (exact match).

    Args:
        first_name: Raw first name (will be normalized).
        last_name: Raw last name (will be normalized).
        config: Optional config override.
        connection: Optional existing DB connection.

    Returns:
        Politician ID if found, else None.
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
            SELECT id
            FROM politicians
            WHERE (first_name = ? AND last_name = ?)
               OR (first_name = ? AND last_name = ?)
            LIMIT 1
            """,
            (first_name, last_name, last_name, first_name),
        )
        row = cur.fetchone()
        return row["id"] if row else None
    finally:
        if close:
            connection.close()


def get_politicians(
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    config: Optional[object] = None,
    connection=None,
) -> list[dict]:
    """
    Return a (paginated) list of politicians.

    Returns:
        [{id, first_name, last_name, party}, ...]
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


def get_politician(
    politician_id: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[dict]:
    """
    Return a single politician by ID, or None if not found.
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


# ---------- Write API (add*) ----------

def add_politician(
    politician: dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Insert a single politician if not present.

    Expected keys in `politician`:
      - first_name (str)
      - last_name (str)
      - party (str)
      - bioguide_id (str)  -> stored as `id` in DB

    Returns:
      True if inserted, False if already existing (ignored).
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


def add_politicians(
    politicians: list[dict],
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Bulk insert a list of politicians (idempotent with INSERT OR IGNORE).

    Returns:
        Number of rows actually inserted.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    inserted = 0
    try:
        for person in politicians:
            if add_politician(person, config=config, connection=connection):
                inserted += 1
        if close:
            connection.commit()
        return inserted
    finally:
        if close:
            connection.close()
