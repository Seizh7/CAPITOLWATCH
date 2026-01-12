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
from pathlib import Path

from bs4 import BeautifulSoup

from config import CONFIG
from capitolwatch.db import get_connection
from capitolwatch.services.assets import add_asset
from capitolwatch.services.products import add_product
from capitolwatch.datapipeline.database.extractor import extract_assets
from capitolwatch.services.reports import get_politician_id
from capitolwatch.datapipeline.database.matching_workflow import (
    parse_report_id
)


def sort_key(asset: dict) -> tuple:
    """
    Sorting key for asset items based on hierarchical index.

    Args:
        asset (dict): Asset dictionary expected to contain an 'index' field,
            e.g. "1.2.3". Missing or malformed indices are pushed to the end.

    Returns:
        tuple: A tuple of integers for sorting (e.g. "1.2.3" -> (1, 2, 3)),
               or (inf,) if the index is missing/invalid.
    """
    idx = str(asset.get("index", "")).strip()
    try:
        parts = [
            int(p) for p in idx.split(".") if p.strip().isdigit()
        ]
        if not parts:
            raise ValueError("empty index parts")
        return tuple(parts)
    except Exception:
        return (float("inf"),)


def process_assets_parsing(html_file_path: str) -> Optional[str]:
    """
    Parse a stored HTML report, extract assets, ensure products exist,
    and insert assets with parent-child relationships.

    Args:
        html_file_path: Path to the HTML report to process.

    Returns:
        A status string like "inserted: <count>"; or None on fatal error.
    """
    conn = get_connection(CONFIG)

    try:
        # Parse HTML
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")

        # Resolve report id from filename
        report_id = parse_report_id(html_file_path)
        if report_id is None:
            print(f"Could not parse report id from filename: {html_file_path}")
            return None

        # Read politician_id linked to the report
        try:
            politician_id = get_politician_id(report_id, connection=conn)
        except Exception:
            politician_id = None

        # Extract asset blocks
        assets = extract_assets(soup)
        if not assets:
            print(f"No assets found in report {report_id} ({html_file_path})")
            return "inserted: 0"

        # Sort parents before children (e.g., 3 before 3.1 before 3.1.1)
        assets_sorted = sorted(assets, key=sort_key)

        # Map: extracted index (e.g., "3.1") -> inserted asset_id
        index_to_id: dict[str, int] = {}
        inserted = 0

        for asset in assets_sorted:
            name = (asset.get("name") or "").strip()
            product_type = (asset.get("type") or "Unknown").strip()
            if not name:
                continue  # skip nameless rows

            # 1) Ensure product exists and get product_id
            product_id = add_product(
                {"name": name, "type": product_type},
                connection=conn,
                config=CONFIG,
            )

            # 2) Resolve parent asset id
            parent_idx = asset.get("parent_index")
            parent_asset_id = (
                index_to_id.get(parent_idx) if parent_idx else None
            )

            # 3) Insert asset row; schema inferred from your SELECT
            asset_row = {
                "politician_id": politician_id,
                "product_id": product_id,
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

            # Record mapping for children resolution
            idx = asset.get("index")
            if isinstance(idx, str) and idx:
                index_to_id[idx] = inserted_id
            inserted += 1

        conn.commit()
        status = f"inserted: {inserted}"
        print(f"Report {report_id}: {status}")
        return status
    finally:
        conn.close()


def process_reports_assets(folder_path: str) -> None:
    """
    Import assets from HTML report files stored in a folder and persist them
    into the database.

    Args:
        folder_path: Directory containing .html report files.
    """
    folder = Path(folder_path)

    files = sorted(folder.glob("*.html"))

    processed = 0
    succeeded = 0
    failed = 0

    for file in files:
        report_id = parse_report_id(file)

        try:
            status = process_assets_parsing(str(file))

            # Normalize the printed line
            label = f"Report {report_id}"
            if status is None:
                print(f"{label}: [FAILED]")
                failed += 1
            else:
                print(f"{label}: {status}")
                succeeded += 1

        except Exception as exc:
            # Keep going on individual failures
            label = f"Report {report_id}"
            print(f"{label}: [ERROR] {exc}")
            failed += 1

        processed += 1

    # Final summary
    print("\nImport finished.")
    print(
        f"Processed: {processed} | Succeeded: {succeeded} | Failed: {failed}"
    )


def main():
    process_reports_assets(CONFIG.output_folder)


if __name__ == "__main__":
    main()
