# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Allows running the database CLI with:
python -m capitolwatch.datapipeline.database init
python -m capitolwatch.datapipeline.database import --folder data/reports
python -m capitolwatch.datapipeline.database match
python -m capitolwatch.datapipeline.database parse --folder data/reports
python -m capitolwatch.datapipeline.database enrich
python -m capitolwatch.datapipeline.database pipeline --folder data/reports
"""

from capitolwatch.datapipeline.database.cli import app

if __name__ == "__main__":
    app()
