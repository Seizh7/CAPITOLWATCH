# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
CLI script to parse a report HTML and insert its assets/products into the DB.

- Input: a report id (int) or a path to the HTML file.
- Behavior: if only report id is provided, we try to locate the HTML via the
  reports.source_file field; otherwise, the provided path is used directly.

This script uses the services layer for all DB interactions.
"""

from typing import Optional

from bs4 import BeautifulSoup

from config import CONFIG
from capitolwatch.db import get_connection
from capitolwatch.services.assets import add_asset
from capitolwatch.services.products import add_product
from capitolwatch.datapipeline.parsing.extractor import extract_assets
from capitolwatch.services.reports import get_politician_id
from capitolwatch.datapipeline.database.matching_workflow import (
    parse_report_id
)


def sort_key(a: dict) -> tuple:
    """
    Sort key for asset items based on their hierarchical index.
    """
    idx = str(a.get("index", ""))
    try:
        return tuple(int(p) for p in idx.split("."))
    except Exception:
        return (float("inf"),)


def process_assets_parsing(html_file_path: str) -> Optional[str]:
    """
    Parse a stored HTML report, extract assets info, create products if needed,
    and insert assets into the DB with parent-child relationships.

    Args:
        html_file_path: Path to the HTML report to process.

    Returns:
        A status string like "inserted: <count>"; or None on fatal error.
    """
    conn = get_connection(CONFIG)
    # Use a single connection/transaction for all inserts

    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")

        report_id = parse_report_id(html_file_path)
        if report_id is None:
            print(f"Could not parse report id from filename: {html_file_path}")
            return None

        # Ensure report exists and fetch politician link if any (optional)
        try:
            politician_id = get_politician_id(report_id, connection=conn)
        except Exception:
            politician_id = None

        assets = extract_assets(soup)
        if not assets:
            print(f"No assets found in report {report_id} ({html_file_path})")
            return "inserted: 0"

        # Sort by hierarchical index (e.g., 3 before 3.1) to ensure parents
        # are inserted before children.
        assets_sorted = sorted(assets, key=sort_key)

        # Map extracted index -> inserted asset id
        index_to_id: dict[str, int] = {}
        inserted = 0

        for asset in assets_sorted:
            name = (asset.get("name") or "").strip()
            product_type = (asset.get("type") or "Unknown").strip()
            if not name:
                continue

            # 1) Ensure product exists and get product_id
            product_id = add_product(
                {
                    "name": name,
                    "type": product_type,
                },
                connection=conn,
                config=CONFIG,
            )

            # 2) Resolve parent asset id if any
            parent_idx = asset.get("parent_index")
            parent_asset_id = (
                index_to_id.get(parent_idx) if parent_idx else None
            )

            # 3) Insert asset
            asset_row = {
                "product_id": product_id,
                "politician_id": politician_id,
                "owner": asset.get("owner"),
                "value": asset.get("value"),
                "income_type": asset.get("income_type"),
                "income": asset.get("income"),
                "comment": asset.get("comment"),
                "parent_asset_id": parent_asset_id,
            }

            inserted_id = add_asset(
                report_id,
                asset_row,
                connection=conn,
                config=CONFIG,
            )

            idx = asset.get("index")
            if isinstance(idx, str) and idx:
                index_to_id[idx] = inserted_id
            inserted += 1

        # Commit once
        conn.commit()
        status = f"inserted: {inserted}"
        print(f"Report {report_id}: {status}")
        return status
    finally:
        conn.close()


def process_reports_assets(folder_path, db_path, project_root) -> None:
    """
    Imports assets from HTML report files in a folder and saves them in the
    database (assets table), one report per file.

    Args:
        folder_path (str or Path): Directory containing .html files
        db_path (str): Path to the SQLite database (unused, for symmetry)
        project_root (str or Path): Project root for relative paths (unused)
    """
    from pathlib import Path

    folder = Path(folder_path)
    processed = 0
    for file in sorted(folder.glob("*.html")):
        try:
            report_id = int(file.stem)
        except Exception:
            report_id = None

        try:
            status = process_assets_parsing(str(file))
            if report_id is not None:
                print(f"Report {report_id} {status}.")
            else:
                print(f"{file.name}: {status}")
        except Exception as e:
            if report_id is not None:
                print(f"Error processing report {report_id}: {e}")
            else:
                print(f"Error processing {file.name}: {e}")
        processed += 1

    print("Import finished.")


def main():
    process_reports_assets(
        CONFIG.output_folder,
        CONFIG.db_path,
        CONFIG.project_root,
    )


if __name__ == "__main__":
    main()
