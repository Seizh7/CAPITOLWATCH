# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path
import os


class Config:
    """Application configuration loaded from environment variables.

    All values that differ between environments are driven by env variables.

    Attributes:
        project_root (Path): Absolute path to the repository root.
        data_dir (Path): Directory where data files and the database are.
        source_dir (Path): Root of the source code.
        database_dir (Path): Alias for data_dir.
        datapipeline_dir (Path): Path to the datapipeline sub-package.
        services_dir (Path): Path to the services sub-package.
        scraping_dir (Path): Path to the scraping sub-package.
        parsing_dir (Path): Path to the parsing sub-package.
        year (str): Reference year for report search.
        search_year (str): Calendar year used in EFD search queries (year + 1).
        output_folder (Path): Destination folder for downloaded HTML reports.
        start_date (str): Search window start date (MM/DD/YYYY).
        end_date (str): Search window end date (MM/DD/YYYY).
        base_url (str): Base URL of the EFD search portal.
        search_url (str): Full URL of the EFD search endpoint.
        db_path (Path): Path to the SQLite database file.
        congress_api_key (str): Congress.gov API key (from env).
        openfigi_api_key (str): OpenFIGI API key (from env).
        debug (bool): Enable verbose/debug output when True.
    """

    def __init__(
        self, year: str = "2023", project_root: Path | None = None
    ) -> None:
        """Initialize configuration from environment variables.

        Args:
            year (str): Reference year for report ingestion. Defaults to "2023"
            project_root (Path | None): Repository root path.

        Raises:
            ValueError: If CONGRESS_API_KEY or OPEN_FIGI_API_KEY are missing.
        """
        # Accept an externally resolved root to avoid recomputing it when
        # __init__.py has already done so — falls back to self-resolution.
        self.project_root = (
            project_root or Path(__file__).resolve().parent.parent
        )

        self.data_dir = self.project_root / "data"
        self.source_dir = self.project_root / "capitolwatch"
        self.database_dir = self.data_dir

        self.datapipeline_dir = self.source_dir / "datapipeline"
        self.services_dir = self.source_dir / "services"
        self.scraping_dir = self.datapipeline_dir / "scraping"
        self.parsing_dir = self.datapipeline_dir / "parsing"

        self.year = year
        self.search_year = str(int(self.year) + 1)
        self.output_folder = self.data_dir / f"annual_reports_{self.year}"
        self.start_date = f"01/01/{self.search_year}"
        self.end_date = f"12/31/{self.search_year}"

        self.base_url = "https://efdsearch.senate.gov"
        self.search_url = self.base_url + "/search/"

        self.db_path = self.data_dir / "capitolwatch.db"

        self.congress_api_key = os.getenv("CONGRESS_API_KEY")
        if not self.congress_api_key:
            raise ValueError("Missing environment variable: CONGRESS_API_KEY")

        self.openfigi_api_key = os.getenv("OPEN_FIGI_API_KEY")
        if not self.openfigi_api_key:
            raise ValueError("Missing environment variable: OPEN_FIGI_API_KEY")

        # debug is True unless APP_ENV is set to "production"
        app_env = os.getenv("APP_ENV", "development").lower()
        self.debug = app_env != "production"
