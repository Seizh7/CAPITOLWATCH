# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

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
    if url.startswith("http"):
        pass
    else:
        url = "https://efdsearch.senate.gov" + url

    driver.get(url)
    time.sleep(1)  # Wait for the page to load completely

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
        url,
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
