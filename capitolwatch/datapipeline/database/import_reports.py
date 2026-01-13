# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import hashlib
from datetime import datetime, timezone
from config import CONFIG
from capitolwatch.services.reports import add_report, update_report_source_file


def import_reports(folder_path, project_root):
    """
    Imports temporary HTML report files downloaded by downloader.py and saves
    their metadata in the database.

    Process:
    1. Reads temp_*.html files
    2. Computes checksum
    3. Inserts in database (auto-generates ID)
    4. Renames file with the generated ID

    Args:
        folder_path (str or Path): Directory containing temp_*.html files
        db_path (str): Path to the SQLite database
        project_root (str or Path): Project root for relative paths
    """

    # Iterate through temporary files only
    temp_files = list(folder_path.glob("temp_*.html"))

    if not temp_files:
        print("No temporary files found to import.")
        return

    print(f"Found {len(temp_files)} temporary file(s) to import.")

    for file in temp_files:
        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Compute SHA-1 checksum of the HTML content
        checksum = hashlib.sha1(html_content.encode("utf-8")).hexdigest()

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

    print("Import finished.")


if __name__ == "__main__":
    import_reports(
        CONFIG.output_folder,
        CONFIG.db_path,
        CONFIG.project_root
    )
