# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path
import os


class Config:
    def __init__(self, year="2023"):
        self.project_root = Path(__file__).resolve().parent.parent

        self.data_dir = self.project_root / "data"
        self.source_dir = self.project_root / "capitolwatch"

        self.datapipeline_dir = self.source_dir / "datapipeline"
        self.scraping_dir = self.datapipeline_dir / "scraping"
        self.parsing_dir = self.datapipeline_dir / "parsing"
        self.database_dir = self.datapipeline_dir / "database"

        self.year = year
        self.search_year = str(int(self.year) + 1)
        self.output_folder = self.data_dir / f"annual_reports_{self.year}"
        self.start_date = f"01/01/{self.search_year}"
        self.end_date = f"12/31/{self.search_year}"

        self.base_url = "https://efdsearch.senate.gov"
        self.search_url = self.base_url + "/search/"

        self.db_path = self.data_dir / "capitolwatch_dev.db"

        self.api_key = os.getenv("CONGRESS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing CONGRESS_API_KEY (development)")

        self.debug = True
