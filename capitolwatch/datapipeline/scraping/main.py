# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from capitolwatch.datapipeline.scraping.driver import setup_driver
from capitolwatch.datapipeline.scraping.scraper import (
    submit_search_form, get_all_links
)
from capitolwatch.datapipeline.scraping.downloader import download_report
from config import CONFIG
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
    submit_search_form(driver, CONFIG.start_date, CONFIG.end_date)

    # Wait for the results table to load
    wait = WebDriverWait(driver, 5)
    wait.until(EC.presence_of_element_located((By.ID, "filedReports")))

    # Create the output folder
    CONFIG.output_folder.mkdir(parents=True, exist_ok=True)

    # Collect all report links for the target year
    all_report_links = get_all_links(driver, CONFIG.year)
    print(f"Total links collected : {len(all_report_links)}")

    # Iterate and download each individual report
    for i, report_url in enumerate(all_report_links):
        print(f"[{i+1}/{len(all_report_links)}] Report : {report_url}")
        try:
            download_report(driver, report_url, CONFIG)
        except Exception as e:
            print("Error while downloading : ", e)

    # Properly close the browser
    driver.quit()
    print("Downloads completed.")


if __name__ == "__main__":
    main()
