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

def download_report(driver, url, output_folder):
    """
    Saves the raw HTML page of an individual US Senate report.

    Args:
        driver (selenium.webdriver.Chrome): Active Selenium instance.
        url (str): Relative or absolute URL of the report.
        output_folder (Path): Folder where files will be saved.

    Returns:
        None
    """
    # Build the absolute URL if needed
    full_url = url if url.startswith("http") else "https://efdsearch.senate.gov" + url

    driver.get(full_url)
    time.sleep(2)  # Wait for the page to load completely

    # Search the ID after 'annual'
    parts = url.split("/annual/")
    if len(parts) > 1:
        report_id = parts[1].split("/")[0]
    else:
        report_id = "unknown_id"

    # Filename derived from the URL
    filename = output_folder / (report_id + ".html")
    
    # Save the raw HTML content of the page
    with open(filename, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print(f"HTML saved: {filename}")

    time.sleep(1)
