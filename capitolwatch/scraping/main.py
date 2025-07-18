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

from capitolwatch.scraping.driver import setup_driver
from capitolwatch.scraping.scraper import submit_search_form, get_all_links
from capitolwatch.scraping.downloader import download_report
from config import START_DATE, END_DATE, YEAR, OUTPUT_FOLDER
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def main():
    """
    Launches the automation of scraping annual reports.
    """
    # Initialize the Selenium browser
    driver = setup_driver()
    
    # Submit the search form for reports from the selected period
    submit_search_form(driver, START_DATE, END_DATE)
    
    # Wait for the results table to load
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "filedReports")))
    
    # Create the output folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Collect all report links for the target year
    all_report_links = get_all_links(driver, YEAR)
    print(f"Total links collected : {len(all_report_links)}")
    
    # Iterate and download each individual report
    for i, report_url in enumerate(all_report_links):
        print(f"[{i+1}/{len(all_report_links)}] Report : {report_url}")
        try:
            download_report(driver, report_url, OUTPUT_FOLDER)
        except Exception as e:
            print("Error while downloading : ", e)
    
    # Properly close the browser
    driver.quit()
    print("Downloads completed.")

if __name__ == "__main__":
    main()
