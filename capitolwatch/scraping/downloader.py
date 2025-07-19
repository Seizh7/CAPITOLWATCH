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

import time
import hashlib
import sqlite3
from datetime import datetime, timezone

def download_report(driver, url, config):
    """
    Downloads a US Senate report HTML page, save, and fills the 'reports'
    table with metadata.

    Args:
        driver (selenium.webdriver.Chrome): Active Selenium instance.
        url (str): Relative or absolute URL of the report.
        config (Config): Configuration instance containing paths and settings.

    Returns:
        None
    """
    # Build the absolute URL if needed
    full_url = url if url.startswith("http") else "https://efdsearch.senate.gov" + url

    driver.get(full_url)
    time.sleep(2)  # Wait for the page to load completely

    # Get HTML content
    html_content = driver.page_source

    # Compute SHA-1 checksum of the HTML content
    checksum = hashlib.sha1(html_content.encode("utf-8")).hexdigest()

    # Insert metadata in DB (without source_file yet)
    conn = sqlite3.connect(config.db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports (url, import_timestamp, checksum, encoding)
        VALUES (?, ?, ?, ?)
    """, (
        full_url,
        datetime.now(timezone.utc).isoformat(),
        checksum,
        "utf-8"
    ))
    report_id = cur.lastrowid  # Get the auto-incremented ID

    filename = config.output_folder / f"{report_id}.html"
    relative_path = filename.relative_to(config.project_root)

    # Save the HTML file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML saved: {filename}")

    # Update the DB row with the source_file
    cur.execute("""
        UPDATE reports
        SET source_file = ?
        WHERE id = ?
    """, (str(relative_path), report_id))

    conn.commit()
    conn.close()

    time.sleep(1)
