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
from dotenv import load_dotenv
import os


class Config:
    def __init__(self, year="2023"):
        # Root of the project (this file is at the project root)
        self.project_root = Path(__file__).resolve().parent

        # Data and results directories
        self.data_dir = self.project_root / "data"
        self.source_dir = self.project_root / "capitolwatch"
        self.scraping_dir = self.source_dir / "scraping"

        # Target and searching year for the reports
        self.year = year
        self.search_year = str(int(self.year) + 1)

        # Output folder for saving the annual reports
        self.output_folder = self.data_dir / f"annual_reports_{self.year}"

        # Search period for collecting the reports
        self.start_date = f"01/01/{self.search_year}"
        self.end_date = f"12/31/{self.search_year}"

        # Base URL for the US Senate EFD search
        self.base_url = "https://efdsearch.senate.gov"
        self.search_url = self.base_url + "/search/"

        # Path to the database
        self.db_path = self.data_dir / "capitolwatch.db"

        # Load environment variables from .env file
        self.api_key = os.getenv("CONGRESS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing CONGRESS_API_KEY in .env")

load_dotenv()
CONFIG = Config()
