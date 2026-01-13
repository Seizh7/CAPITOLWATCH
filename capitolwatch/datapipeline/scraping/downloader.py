# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import time


def download_report(driver, url, config):
    """
    Downloads a US Senate report HTML page and saves it with a temporary name.

    The file is saved with a temporary name. It should be imported into the
    database later using import_reports module, which will rename it with
    the proper database-generated ID.

    Args:
        driver (selenium.webdriver.Chrome): Active Selenium instance.
        url (str): Relative or absolute URL of the report.
        config (Config): Configuration instance containing paths and settings.

    Returns:
        tuple: (filename, url)
    """
    # Build the absolute URL if needed
    if url.startswith("http"):
        pass
    else:
        url = "https://efdsearch.senate.gov" + url

    driver.get(url)
    time.sleep(1)  # Wait for the page to load completely

    html_content = driver.page_source

    # Generate temporary filename using timestamp (milliseconds)
    temp_id = int(time.time() * 1000)
    filename = config.output_folder / f"temp_{temp_id}.html"

    # Save the HTML file with temporary name
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Downloaded: {filename}")

    time.sleep(1)

    return filename, url
