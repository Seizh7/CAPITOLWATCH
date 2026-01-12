# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Allows running the scraping CLI with:
python -m capitolwatch.datapipeline.scraping --year 2023
"""

from capitolwatch.datapipeline.scraping.cli import app

if __name__ == "__main__":
    app()
