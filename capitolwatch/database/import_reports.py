"""
The MIT License (MIT)

Copyright (c) 2025-present Seizh7

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import hashlib
import sqlite3
from config import CONFIG
from datetime import datetime, timezone

def import_reports(folder_path, db_path, project_root):
    """
    Imports HTML report files from a folder and saves their metadata in the database (reports table).

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
                SET checksum = ?, source_file = ?, encoding = ?, import_timestamp = ?
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
                INSERT INTO reports (id, url, import_timestamp, checksum, encoding, source_file)
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