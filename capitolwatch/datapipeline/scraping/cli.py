#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
CLI for scraping US Senate annual reports using Typer.

This module provides a command-line interface for the scraping functionality.
Can be used standalone or integrated into the main CAPITOLWATCH CLI.
"""

import re
from pathlib import Path
from typing import Optional

import typer

from capitolwatch.datapipeline.scraping.core import run_scraping
from config import CONFIG


app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    year: str = typer.Option(
        ...,
        "--year",
        "-y",
        help="Target year for annual reports (e.g., 2023)"
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        "-s",
        help="Start date in MM/DD/YYYY format (default: from config)"
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        "-e",
        help="End date in MM/DD/YYYY format (default: from config)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for downloaded reports (default: from config)"
    )
):
    """
    Scrape US Senate annual reports from efdsearch.senate.gov.

    Examples:
        python -m capitolwatch.datapipeline.scraping --year 2023
    """
    # Validate date formats if provided
    date_pattern = r'^\d{2}/\d{2}/\d{4}$'

    if start and not re.match(date_pattern, start):
        msg = f"Error: Invalid start date format: {start}. Expected MM/DD/YYYY"
        typer.echo(msg, err=True)
        raise typer.Exit(1)

    if end and not re.match(date_pattern, end):
        msg = f"Error: Invalid end date format: {end}. Expected MM/DD/YYYY"
        typer.echo(msg, err=True)
        raise typer.Exit(1)

    # Run scraping
    result = run_scraping(
        year=year,
        start_date=start,
        end_date=end,
        output_dir=output_dir,
        config=CONFIG
    )

    # Exit with error code if there were errors
    if result["errors"] and result["downloaded"] == 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
