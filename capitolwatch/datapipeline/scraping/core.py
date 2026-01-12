#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Core scraping logic - separated from CLI for reusability.

This module contains the business logic for scraping Senate reports,
independent of the CLI interface.
"""

from pathlib import Path

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from capitolwatch.datapipeline.scraping.driver import setup_driver
from capitolwatch.datapipeline.scraping.scraper import (
    submit_search_form,
    get_all_links
)
from capitolwatch.datapipeline.scraping.downloader import download_report


def run_scraping(
    year: str,
    start_date: str = None,
    end_date: str = None,
    output_dir: Path = None,
    config: object = None
):
    """
    Execute the scraping workflow.

    Args:
        year: Target year for annual reports (e.g., "2023").
        start_date: Start date in MM/DD/YYYY format (default: from config).
        end_date: End date in MM/DD/YYYY format (default: from config).
        output_dir: Output directory for downloaded reports.
        config: Configuration object (default: global CONFIG).

    Returns:
        dict: Summary with 'total_found', 'downloaded', 'errors'.
    """
    if config is None:
        from config import CONFIG
        config = CONFIG

    # Use config defaults if not provided
    start_date = start_date or config.start_date
    end_date = end_date or config.end_date
    output_dir = output_dir or config.output_folder

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("CAPITOLWATCH - Senate Report Scraper")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Target year: {year}")
    print(f"Output directory: {output_dir}")
    print()

    driver = None
    errors = []

    try:
        # Initialize Selenium driver
        print("Starting browser...")
        driver = setup_driver()

        # Submit search form
        print("Submitting search form...")
        submit_search_form(driver, start_date, end_date)

        # Wait for results table to load
        print("Waiting for results...")
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "filedReports")))

        # Collect all report links
        print(f"Collecting report links for year {year}...")
        all_report_links = get_all_links(driver, year)
        print(f"Found {len(all_report_links)} total reports")
        print()

        if not all_report_links:
            print("No reports found. Try adjusting your search parameters.")
            return {"total_found": 0, "downloaded": 0, "errors": []}

        # Download each report
        downloaded = 0
        for i, link in enumerate(all_report_links):
            print(f"[{i}/{len(all_report_links)}] Downloading: {link}")
            try:
                download_report(driver, link, config)
                print("Success")
                downloaded += 1
            except Exception as e:
                print(f"Error: {e}")
                errors.append({"link": link, "error": str(e)})

        print("Scraping complete.")
        if errors:
            print(f"{len(errors)} errors occurred")

        return {
            "total_found": len(all_report_links),
            "downloaded": downloaded,
            "errors": errors
        }

    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        return {
            "total_found": 0,
            "downloaded": 0,
            "errors": ["Interrupted by user"]
        }

    except Exception as e:
        print(f"\nError during scraping: {e}")
        import traceback
        traceback.print_exc()
        return {"total_found": 0, "downloaded": 0, "errors": [str(e)]}

    finally:
        if driver:
            print("Closing browser...")
            driver.quit()
