# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import time
import hashlib
from datetime import datetime, timezone
from capitolwatch.services.reports import (
    add_report,
    update_report_source_file
)


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

    # Create report in database first to get auto-generated ID
    report_id, status = add_report(
        report_id=None,  # Auto-generate ID
        checksum=checksum,
        encoding="utf-8",
        import_timestamp=datetime.now(timezone.utc).isoformat(),
        url=url,
        config=config
    )

    # Create filename using the generated ID
    filename = config.output_folder / f"{report_id}.html"
    relative_path = filename.relative_to(config.project_root)

    # Save the HTML file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML saved: {filename}")

    # Update the report with the source file path
    update_report_source_file(
        report_id=report_id,
        source_file=str(relative_path),
        config=config
    )

    print(f"Report {status} in database with ID: {report_id}")

    time.sleep(1)
