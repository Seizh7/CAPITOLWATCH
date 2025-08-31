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

def upsert_report_import_metadata(
    report_id: int,
    *,
    checksum: str,
    source_file: str,
    encoding: str = "utf-8",
    import_timestamp: Optional[str] = None,
    url: Optional[str] = None,
    config: Optional[object] = None,
    connection=None,
) -> str:
    """
    Insert a new report row with import metadata or update an existing one.
      - If report exists (by id), update checksum, source_file, encoding,
        import_timestamp, and url (url only if provided).
      - If not, insert a new row with provided values (url can be None).

    Args:
        report_id: Primary key of the report (filename stem as int).
        checksum: SHA-1 (or other) checksum of the HTML content.
        source_file: Relative path to the stored HTML file.
        encoding: File encoding label (default "utf-8").
        import_timestamp: ISO timestamp (defaults to now in UTC if None).
        url: Original URL if known (kept as-is when updating if None).
        config: Optional config override.
        connection: Optional existing DB connection to reuse.

    Returns:
        "updated" if an existing row was updated, "inserted" if a new row
        was created.
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    if import_timestamp is None:
        import_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        cur = connection.cursor()

        # Try update first (don't overwrite URL with None)
        cur.execute(
            (
                "UPDATE reports SET "
                "checksum = ?, source_file = ?, encoding = ?, "
                "import_timestamp = ?, "
                "url = COALESCE(?, url) "
                "WHERE id = ?"
            ),
            (
                checksum,
                source_file,
                encoding,
                import_timestamp,
                url,
                report_id,
            ),
        )

        if cur.rowcount > 0:
            if close:
                connection.commit()
            return "updated"

        # Not found -> insert
        cur.execute(
            (
                "INSERT INTO reports (id, url, import_timestamp, checksum, "
                "encoding, source_file) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (
                report_id,
                url,
                import_timestamp,
                checksum,
                encoding,
                source_file,
            ),
        )

        if close:
            connection.commit()
        return "inserted"
    finally:
        if close:
            connection.close()
