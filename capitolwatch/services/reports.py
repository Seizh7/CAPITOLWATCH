# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Optional

from capitolwatch.db import get_connection
from config import CONFIG


def get_reports_for_politician(
    politician_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    config: Optional[object] = None,
    connection=None,
) -> list[dict]:
    """
    Return a paginated list of reports for a given politician.

    Args:
        politician_id: Target politician ID (foreign key in `reports`).
        limit: Optional page size. If None, returns all rows.
        offset: Offset for pagination (ignored if limit is None).
        config: Optional config override (defaults to global CONFIG).
        connection: Optional existing DB connection to reuse.

    Returns:
        A list of dict rows with keys:
        {id, year, source_file, import_timestamp, url}.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        sql = (
            "SELECT id, year, source_file, import_timestamp, url "
            "FROM reports WHERE politician_id = ? "
            "ORDER BY year DESC"
        )
        params = [politician_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]

        cur = connection.cursor()
        cur.execute(sql, tuple(params))
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            connection.close()


def get_report(
    report_id: int,
    *,
    config: Optional[object] = None,
    connection=None,
) -> dict | None:
    """
    Fetch a single report by ID, joined with the politician’s basic info.

    Args:
        report_id: Report primary key.
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        dict with keys {id, year, politician_id, first_name, last_name}
        or None if not found.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            (
                "SELECT r.id, r.year, r.politician_id, "
                "p.first_name, p.last_name "
                "FROM reports r JOIN politicians p "
                "ON r.politician_id = p.id "
                "WHERE r.id = ?"
            ),
            (report_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if close:
            connection.close()


def update_report_match(
    report_id: int,
    politician_id: str,
    *,
    year: int | None = None,
    url: str | None = None,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Set the `politician_id` (and optionally `year` and/or `url`) on a row.

    Args:
        report_id: Report primary key to update.
        politician_id: Target politician ID to set.
        year: Optional year override.
        url: Optional URL override.
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        True if at least one row was updated, else False.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        sets: list[str] = ["politician_id = ?"]
        params: list[object] = [politician_id]

        if year is not None:
            sets.append("year = ?")
            params.append(year)
        if url is not None:
            sets.append("url = ?")
            params.append(url)

        sql = f"UPDATE reports SET {', '.join(sets)} WHERE id = ?"
        params.append(report_id)

        cur = connection.cursor()
        cur.execute(sql, tuple(params))

        # Auto-commit only if we opened the connection here
        if close:
            connection.commit()

        return cur.rowcount > 0
    finally:
        if close:
            connection.close()
