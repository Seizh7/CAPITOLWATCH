# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Optional

from capitolwatch.db import get_connection
from config import CONFIG
from datetime import datetime, timezone


# ---------- Read API (get*) ----------

def get_reports_by_politician(
    politician_id: str,
    *,
    limit: Optional[int] = None,
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


def get_report_by_id(
    report_id: int,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[dict]:
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


def get_politician_id(
    report_id: int,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[str]:
    """
    Return the politician_id for a given report ID, or None if not found.

    Args:
        report_id: Report primary key.
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        The politician_id as a string, or None if the report does not exist
        or has no politician linked.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT politician_id FROM reports WHERE id = ?",
            (report_id,),
        )
        row = cur.fetchone()
        return row["politician_id"] if row else None
    finally:
        if close:
            connection.close()


def get_report_by_checksum(
    checksum: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[dict]:
    """
    Return a report by its checksum, or None if not found.

    Args:
        checksum: SHA-1 checksum of the HTML content.
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        dict with report data or None if not found.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True
    try:
        cur = connection.cursor()
        cur.execute(
            (
                "SELECT id, checksum, source_file, encoding, "
                "import_timestamp, url, politician_id, year "
                "FROM reports WHERE checksum = ? LIMIT 1"
            ),
            (checksum,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if close:
            connection.close()


# ---------- Update API (update*) ----------

def update_report_fields(
    report_id: int,
    politician_id: str,
    *,
    year: Optional[int] = None,
    url: Optional[str] = None,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Update report fields: set `politician_id` and optionally :`year` and `url`.

    Args:
        report_id: Report primary key to update.
        politician_id: Target politician ID to set.
        year: Optional year.
        url: Optional URL.
        config: Optional config.
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

        if close:
            connection.commit()

        return cur.rowcount > 0
    finally:
        if close:
            connection.close()


# ---------- Write API (add*) ----------

def add_report(
    *,
    checksum: str,
    source_file: str = "",
    encoding: str = "utf-8",
    import_timestamp: Optional[str] = None,
    url: Optional[str] = None,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Insert a new report row with auto-generated ID, or return existing ID if
    checksum already exists.

    Args:
        checksum: SHA-1 checksum of the HTML content.
        source_file: Relative path to the stored HTML file.
        encoding: File encoding label (default "utf-8").
        import_timestamp: ISO timestamp (defaults to now in UTC if None).
        url: Original URL if known (optional).
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        The auto-generated report ID.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        # Check if report with this checksum already exists
        existing = get_report_by_checksum(
            checksum, config=config, connection=connection
        )
        if existing:
            return existing["id"]

        # Insert new report
        if import_timestamp is None:
            import_timestamp = datetime.now(timezone.utc).isoformat()

        cur = connection.cursor()
        cur.execute(
            (
                "INSERT INTO reports (url, import_timestamp, checksum, "
                "encoding, source_file) VALUES (?, ?, ?, ?, ?)"
            ),
            (url, import_timestamp, checksum, encoding, source_file),
        )
        generated_id = cur.lastrowid
        if close:
            connection.commit()
        return generated_id
    finally:
        if close:
            connection.close()


def update_report_source_file(
    report_id: int,
    source_file: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Update the source_file field for an existing report.

    Args:
        report_id: Report primary key to update.
        source_file: Relative path to the stored HTML file.
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        True if the row was updated, else False.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            "UPDATE reports SET source_file = ? WHERE id = ?",
            (source_file, report_id),
        )

        if close:
            connection.commit()
        return cur.rowcount > 0
    finally:
        if close:
            connection.close()
