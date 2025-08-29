# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import hashlib
import sqlite3
from config import CONFIG
from datetime import datetime, timezone


def import_reports(folder_path, db_path, project_root):
    """
    Imports HTML report files from a folder and saves their metadata in the
    database (reports table).

    Args:
        folder_path (str or Path): Directory containing .html files
        db_path (str): Path to the SQLite database
        project_root (str or Path): Project root for relative paths
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Iterate through each .html file in the folder
    for file in folder_path.glob("*.html"):
        report_id = int(file.stem)  # Extract report ID from filename
        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Compute SHA-1 checksum of the HTML content
        checksum = hashlib.sha1(html_content.encode("utf-8")).hexdigest()

        relative_path = str(file.relative_to(project_root))

        # Check if the report already exists in the database
        cur.execute("""
            SELECT id FROM reports WHERE id = ?
        """, (report_id,))
        row = cur.fetchone()

        if row:
            # Update the existing report record with new metadata
            cur.execute("""
                UPDATE reports
                SET checksum = ?, source_file = ?, encoding = ?,
                        import_timestamp = ?
                WHERE id = ?
            """, (
                checksum,
                relative_path,
                "utf-8",
                datetime.now(timezone.utc).isoformat(),
                report_id
            ))
            print(f"Report {report_id} updated.")
        else:
            # Insert a new report record
            cur.execute("""
                INSERT INTO reports (id, url, import_timestamp, checksum,
                        encoding, source_file)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                None,
                datetime.now(timezone.utc).isoformat(),
                checksum,
                "utf-8",
                relative_path
            ))
            print(f"Report {report_id} added.")

    conn.commit()
    conn.close()
    print("Import finished.")


if __name__ == "__main__":
    import_reports(
        CONFIG.output_folder,
        CONFIG.db_path,
        CONFIG.project_root
    )
