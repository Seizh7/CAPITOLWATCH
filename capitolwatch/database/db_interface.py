"""
Copyright (c) 2025 Seizh7
Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0)
"""

from config import CONFIG
from namematching import compare_names


def get_politician_id(cur, first_name, last_name):
    """
    Finds a politician by first and last name
    Args:
        cur (sqlite3.Cursor): Database cursor.
        first_names (str): First names.
        last_name (str): Last name.

    Returns:
        str or None: Politician ID if found, else None.
    """
    cur.execute("SELECT id, first_name, last_name FROM politicians")

    first_name = " ".join(first_name)
    last_name = " ".join(last_name)
    report_name = f"{first_name} {last_name}"

    for pid, db_first, db_last in cur.fetchall():
        database_name = " ".join([db_first, db_last])
        score = compare_names(report_name, database_name)
        if score > 0.8:
            return pid
    return None


def get_report_id(cur, source_file):
    """
    Returns the ID of the report based on the source file path.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        source_file (str): Path to the report HTML file.
    Returns:
        int or None: Report ID if found, else None.
    """
    cur.execute("""
        SELECT id FROM reports WHERE source_file = ?
    """, (source_file,))
    row = cur.fetchone()
    return row[0] if row else None


def get_reports_by_politician(cur, politician_id):
    """
    Returns all reports for a given politician.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        politician_id (str): Politician ID.
    Returns:
        list: List of report rows for the given politician.
    """
    cur.execute("""
        SELECT * FROM reports
        WHERE politician_id = ?
    """, (politician_id,))
    return cur.fetchall()


def get_assets_by_report(cur, report_id):
    """
    Returns all assets linked to a report.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        report_id (int): Report ID.
    Returns:
        list: List of asset rows for the given report.
    """
    cur.execute("""
        SELECT * FROM assets
        WHERE report_id = ?
    """, (report_id,))
    return cur.fetchall()


def get_product_by_name(cur, name):
    """
    Returns the financial product by its name.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        name (str): Product name.
    Returns:
        tuple or None: Product row if found, else None.
    """
    cur.execute("""
        SELECT * FROM products
        WHERE name = ?
    """, (name,))
    return cur.fetchone()


def update_report(cur, report_id, politician_id, year):
    """
    Updates the report row with the given politician ID and year.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        report_id (int): Report ID.
        politician_id (str): Politician ID.
        year (int): Reporting year.
    Returns:
        None
    """
    cur.execute("""
        UPDATE reports
        SET politician_id = ?, year = ?
        WHERE id = ?
    """, (politician_id, year, report_id))


def insert_or_get_product(cur, name):
    """
    Inserts a new product if not already present. Returns the product's ID.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        name (str): Product name.
    Returns:
        int: Product ID.
    """
    cur.execute("""
        SELECT product_id FROM products WHERE name = ?
    """, (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("""
        INSERT INTO products (name, isin, type, details)
        VALUES (?, NULL, NULL, NULL)
    """, (name,))
    return cur.lastrowid


def insert_assets(cur, assets, report_id, politician_id):
    """
    Inserts all assets in the list, handling parent/child hierarchy.
    Returns the number of inserted assets.

    Args:
        cur (sqlite3.Cursor): Database cursor.
        assets (list): List of asset dictionaries.
        report_id (int): Report ID.
        politician_id (str): Politician ID.
    Returns:
        int: Number of inserted assets.
    """
    index_to_id = {}

    for asset in assets:
        product_id = insert_or_get_product(cur, asset["name"])
        parent_asset_id = None
        if asset.get("parent_index"):
            parent_asset_id = index_to_id.get(asset["parent_index"])

        cur.execute("""
            INSERT INTO assets
                (report_id, politician_id, product_id, owner, value,
                    income_type, income, comment, parent_asset_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id, politician_id, product_id,
            asset["owner"], asset["value"],
            asset["income_type"], asset["income"],
            asset["comment"], parent_asset_id
        ))
        asset_id = cur.lastrowid
        index_to_id[asset["index"]] = asset_id

    return len(assets)


if __name__ == "__main__":
    import sqlite3

    # Example usage
    conn = sqlite3.connect(CONFIG.db_path)
    cur = conn.cursor()

    # Example: Get politician ID
    politician_id = get_politician_id(cur, ['a', "mitchell"], ["mcconnell"])
    print(f"Mitch McConnell ID: {politician_id}")

    # Close the connection
    conn.close()
