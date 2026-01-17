# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import hashlib
from datetime import datetime, timezone
from config import CONFIG
from capitolwatch.services.reports import (
    add_report,
    update_report_source_file,
    get_report_by_checksum,
)

# Minimum file size in KB for valid reports (error pages are smaller)
MIN_FILE_SIZE_KB = 15


def import_reports(folder_path, project_root):
    """
    Imports HTML report files downloaded by downloader.py and saves
    their metadata in the database.

    Process:
    1. Reads *.html files
    2. Computes checksum
    3. Inserts in database (auto-generates ID)
    4. Renames file with the generated ID

    Args:
        folder_path (str or Path): Directory containing *.html files
        db_path (str): Path to the SQLite database
        project_root (str or Path): Project root for relative paths
    """

    # Iterate through HTML files only
    files = list(folder_path.glob("*.html"))

    if not files:
        print("No HTML files found to import.")
        return

    print(f"Found {len(files)} HTML file(s) to import.")

    imported_count = 0
    skipped_count = 0
    error_count = 0

    for file in files:
        # Check file size (skip small files that are likely error pages)
        file_size_kb = file.stat().st_size / 1024
        if file_size_kb < MIN_FILE_SIZE_KB:
            print(
                f"Skipping {file.name}: too small ({file_size_kb:.1f} KB, "
                f"minimum {MIN_FILE_SIZE_KB} KB)."
            )
            error_count += 1
            continue

        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Compute SHA-1 checksum of the HTML content
        checksum = hashlib.sha1(html_content.encode("utf-8")).hexdigest()

        # Check if report already exists
        existing = get_report_by_checksum(checksum)
        if existing:
            print(
                f"Report {existing['id']} already exists "
                f"(skipping {file.name})."
            )
            skipped_count += 1
            continue

        # Insert new report with auto-generated ID
        report_id = add_report(
            checksum=checksum,
            source_file="",  # Will be set after rename
            encoding="utf-8",
            import_timestamp=datetime.now(timezone.utc).isoformat(),
            url=None,
        )

        # Rename file with the generated ID
        new_filename = file.parent / f"{report_id}.html"
        file.rename(new_filename)

        # Update source_file path
        relative_path = str(new_filename.relative_to(project_root))
        update_report_source_file(report_id, relative_path)

        print(f"Report {report_id} imported (renamed from {file.name}).")
        imported_count += 1

    print(
        f"Import finished: {imported_count} imported, "
        f"{skipped_count} duplicates skipped, "
        f"{error_count} error pages skipped."
    )


if __name__ == "__main__":
    import_reports(
        CONFIG.output_folder,
        CONFIG.db_path,
        CONFIG.project_root
    )
