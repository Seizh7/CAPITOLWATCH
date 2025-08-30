# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import os
from typing import Optional, Tuple

from bs4 import BeautifulSoup

from capitolwatch.services.politician_matcher import get_enhanced_politician_id
from capitolwatch.db import get_connection
from capitolwatch.datapipeline.parsing.extractor import extract_politician_name
from config import CONFIG


def resolve_politician(
    cursor, soup: BeautifulSoup
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract first/last names from an HTML soup and resolve to a politician ID
    using the enhanced matching helper, reusing an existing DB cursor.

    Returns:
        (politician_id, first_names, last_name)
        -> Any part can be None if extraction fails or no match is found.
    """
    # Extract the politician's first/last names from the page
    first_names, last_name = extract_politician_name(soup)
    if not first_names or not last_name:
        return None, first_names, last_name

    # Resolve to a politician ID (HIGH/MEDIUM/override only)
    politician_id = get_enhanced_politician_id(
        cursor, first_names.split(), last_name.split()
    )
    return politician_id, first_names, last_name


def process_report_matching(html_file_path: str) -> Optional[str]:
    """
    Parse a stored HTML report, extract the politician name, and resolve it
    to a politician ID using the enhanced matching pipeline.

    Args:
        html_file_path: Path to the HTML report to process.

    Returns:
        The matched politician ID if strong enough; otherwise None.
    """
    conn = get_connection(CONFIG)
    cur = conn.cursor()

    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")

        politician_id, first_names, last_name = resolve_politician(
            cur, soup
        )

        if not first_names or not last_name:
            print(f"Could not extract name from {html_file_path}")
            return None

        if politician_id:
            print(f"Matched: {first_names} {last_name} → {politician_id}")
        else:
            print(f"No match: {first_names} {last_name}")

        return politician_id
    finally:
        conn.close()


def main_processing_workflow() -> dict:
    """
    Walk through all HTML reports, try to resolve each to a politician ID
    using the enhanced matching helper, and collect simple run statistics.

    Returns:
        {
            "processed": int,
            "matched": int,
            "needs_review": list[{"file": str, "name": str}],
        }
    """
    reports_dir = CONFIG.output_folder

    stats = {"processed": 0, "matched": 0, "needs_review": []}

    conn = get_connection(CONFIG)
    cur = conn.cursor()

    try:
        for filename in sorted(os.listdir(reports_dir)):
            # Skip non-HTML
            if not filename.endswith(".html"):
                continue

            file_path = os.path.join(reports_dir, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                soup = BeautifulSoup(content, "html.parser")

                # Correct function name
                politician_id, first_names, last_name = resolve_politician(
                    cur, soup
                )

                if first_names and last_name and politician_id:
                    stats["matched"] += 1

                    # Optionally print the canonical DB name next to the ID
                    try:
                        cur.execute(
                            (
                                "SELECT first_name, last_name "
                                "FROM politicians WHERE id = ?"
                            ),
                            (politician_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            if isinstance(row, dict) or hasattr(row, "keys"):
                                db_first = row["first_name"]
                                db_last = row["last_name"]
                            else:
                                db_first, db_last = row
                            db_name = f"{db_first} {db_last}"
                        else:
                            db_name = "?"
                    except Exception:
                        db_name = "?"

                    print(
                        f"{filename}: {first_names} {last_name} → "
                        f"{politician_id} ({db_name})"
                    )
                elif first_names and last_name:
                    stats["needs_review"].append(
                        {
                            "file": filename,
                            "name": f"{first_names} {last_name}"
                        }
                    )
                    print(
                        f"{filename}: {first_names} {last_name} needs review"
                    )

                stats["processed"] += 1

            except Exception as e:  # keep simple for script usage
                print(f"Error processing {filename}: {e}")

    finally:
        conn.close()

    print("\nProcessing Summary:")
    print(f"Processed: {stats['processed']}")
    print(f"Matched: {stats['matched']}")
    print(f"Need review: {len(stats['needs_review'])}")

    return stats


if __name__ == "__main__":
    main_processing_workflow()
