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

from pathlib import Path

# Path to the root of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to directories
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = PROJECT_ROOT / "capitolwatch"
SCRAPING_DIR = SOURCE_DIR / "scraping"

# Target and searching year for the reports
YEAR = "2023"
SEARCH_YEAR = str(int(YEAR) + 1)

# Output folder for saving the annual reports
OUTPUT_FOLDER = DATA_DIR / f"annual_reports_{YEAR}"

# Search period for collecting the reports
START_DATE = f"01/01/{SEARCH_YEAR}"
END_DATE = f"12/31/{SEARCH_YEAR}"

# Base URL for the US Senate EFD search
BASE_URL = "https://efdsearch.senate.gov"
SEARCH_URL = BASE_URL + "/search/"
