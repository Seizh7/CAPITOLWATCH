# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import hashlib
from datetime import datetime, timezone
from config import CONFIG
from capitolwatch.services.reports import upsert_report_import_metadata


def import_reports(folder_path, db_path, project_root):
    """
    Imports HTML report files from a folder and saves their metadata in the
    database (reports table).

    Args:
        folder_path (str or Path): Directory containing .html files
        db_path (str): Path to the SQLite database
        project_root (str or Path): Project root for relative paths
    """
    # All DB interactions are delegated to the reports service layer.

    # Iterate through each .html file in the folder
    for file in folder_path.glob("*.html"):
        report_id = int(file.stem)  # Extract report ID from filename
        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Compute SHA-1 checksum of the HTML content
        checksum = hashlib.sha1(html_content.encode("utf-8")).hexdigest()

        relative_path = str(file.relative_to(project_root))

        # Upsert via service layer; this will insert or update as needed
        status = upsert_report_import_metadata(
            report_id,
            checksum=checksum,
            source_file=relative_path,
            encoding="utf-8",
            import_timestamp=datetime.now(timezone.utc).isoformat(),
            url=None,
        )
        print(f"Report {report_id} {status}.")

    print("Import finished.")


if __name__ == "__main__":
    import_reports(
        CONFIG.output_folder,
        CONFIG.db_path,
        CONFIG.project_root
    )
