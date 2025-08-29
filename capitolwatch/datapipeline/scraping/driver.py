# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def setup_driver(headless=True):
    """
    Initializes a Chrome WebDriver.

    Args:
        headless (bool): If True, launches the browser in headless mode.

    Returns:
        selenium.webdriver.Chrome: Configured Chrome browser instance.
    """
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    return driver
