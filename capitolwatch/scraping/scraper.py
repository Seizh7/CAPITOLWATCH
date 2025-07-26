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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from bs4 import BeautifulSoup


def extract_links(soup, year):
    """
    Searches for all annual report links for a given year in the results page.

    Args:
        soup (BeautifulSoup): BeautifulSoup object of the HTML page.
        year (str): Target year.

    Returns:
        list: List of relative URLs of found annual reports.
    """
    links = []
    # Search for the reports table
    table = soup.find("table", {"id": "filedReports"})
    if table:
        # Loop through all links in the table and filter on the target text
        for a in table.find_all("a", href=True):
            if f"Annual Report for CY {year}" in a.text:
                links.append(a["href"])
    return links


def submit_search_form(driver, start_date, end_date):
    """
    Fills in and submits the search form.

    Args:
        driver (selenium.webdriver.Chrome): Active Selenium instance.
        start_date (str): Start date (MM/DD/YYYY).
        end_date (str): End date (MM/DD/YYYY).

    Returns:
        None
    """
    # User agreement must be accepted before access to search.
    driver.get("https://efdsearch.senate.gov/search/")
    try:
        # Find the agreement checkbox and activate it
        agree_checkbox = WebDriverWait(driver, 5).until(
            expected_conditions.presence_of_element_located(
                (By.ID, "agree_statement")
            )
        )
        agree_checkbox.click()
        time.sleep(1)  # Wait for agreement to be submitted
    except Exception:
        # In case the agreement is already validated
        pass

    # Wait for the search form to load
    WebDriverWait(driver, 5).until(
        expected_conditions.presence_of_element_located((By.ID, "searchForm"))
    )

    # Select "Senator" option
    senator_selector = "input.senator_filer"
    senator_checkbox = driver.find_element(By.CSS_SELECTOR, senator_selector)
    if not senator_checkbox.is_selected():
        senator_checkbox.click()

    # Select "Annual" option
    annual_selector = "input.report_types_annual"
    annual_checkbox = driver.find_element(By.CSS_SELECTOR, annual_selector)
    if not annual_checkbox.is_selected():
        annual_checkbox.click()

    # Fill in the search dates
    driver.find_element(By.ID, "fromDate").clear()
    driver.find_element(By.ID, "fromDate").send_keys(start_date)
    driver.find_element(By.ID, "toDate").clear()
    driver.find_element(By.ID, "toDate").send_keys(end_date)

    # Submit the search form
    submit_button = "button[type='submit']"
    search_button = driver.find_element(By.CSS_SELECTOR, submit_button)
    search_button.click()
    time.sleep(2)


def get_all_links(driver, year):
    """
    Loops through all result pages to collect all annual report links.

    Args:
        driver (selenium.webdriver.Chrome): Active Selenium instance.
        year (str): Target year.

    Returns:
        set: Set of all found relative URLs.
    """
    all_links = set()
    while True:
        time.sleep(2)
        # Use BeautifulSoup to parse the current page
        soup = BeautifulSoup(driver.page_source, "html.parser")
        page_links = extract_links(soup, year)
        print(f"Found {len(page_links)} reports on this page.")

        all_links.update(page_links)

        # Look for the "Next" button for pagination
        try:
            next_button = driver.find_element(By.ID, "filedReports_next")
            if "disabled" in next_button.get_attribute("class"):
                print("End of pages.")
                break
            else:
                next_button.click()
                print("Next page clicked.")
        except Exception as e:
            print("No Next button or error :", e)
            break
    return all_links
