#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
CLI for database construction using Typer.

This module provides a command-line interface for building the CAPITOLWATCH
database with step-by-step operations or complete pipeline execution.
Can be used standalone or integrated into the main CAPITOLWATCH CLI.
"""

from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(
    help="Build and manage the CAPITOLWATCH database",
    no_args_is_help=True,
    add_completion=False
)


@app.command()
def init():
    """
    Initialize database and add senators.

    This creates all necessary tables and populates the politicians
    table with current senators from the Congress API.

    Example:
        python -m capitolwatch.datapipeline.database init
    """
    from capitolwatch.datapipeline.database.core import initialize_db
    from config import CONFIG

    try:
        result = initialize_db(CONFIG)
        typer.secho(
            f"\nInitialization complete: "
            f"{result['senators_added']} senators added",
            fg=typer.colors.GREEN,
            bold=True
        )
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("import")
def import_reports(
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Folder containing HTML report files to import "
        "(default: from config)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )
):
    """
    Import reports from a local folder.

    Reads HTML files from the specified folder, computes checksums,
    and stores metadata in the database.

    Example:
        python -m capitolwatch.datapipeline.database import
        python -m capitolwatch.datapipeline.database import \\
            --folder data/reports
    """
    from capitolwatch.datapipeline.database.core import (
        import_reports_from_folder
    )
    from config import CONFIG

    folder_path = folder or CONFIG.output_folder
    try:
        result = import_reports_from_folder(folder_path, CONFIG)
        typer.secho(
            f"\nImport complete: {result['imported_count']} "
            f"report(s) imported",
            fg=typer.colors.GREEN,
            bold=True
        )
    except (FileNotFoundError, ValueError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def match():
    """
    Match politicians to their reports.

    Analyzes HTML reports to identify the politician who filed each report
    and updates the database with the politician_id.

    Example:
        python -m capitolwatch.datapipeline.database match
    """
    from capitolwatch.datapipeline.database.core import (
        match_politicians_to_reports
    )
    from config import CONFIG

    try:
        stats = match_politicians_to_reports(CONFIG)

        if stats['needs_review']:
            typer.secho(
                f"\nWarning: {len(stats['needs_review'])} "
                f"report(s) require manual review",
                fg=typer.colors.YELLOW
            )

        typer.secho(
            f"\nMatching complete: "
            f"{stats['matched']}/{stats['processed']} matched",
            fg=typer.colors.GREEN,
            bold=True
        )
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("parse")
def parse_assets(
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Folder containing HTML report files (default: from config)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )
):
    """
    Parse assets from HTML reports.

    Extracts asset and product information from HTML reports
    and stores them in the database.

    Example:
        python -m capitolwatch.datapipeline.database parse
        python -m capitolwatch.datapipeline.database parse \\
            --folder data/reports
    """
    from capitolwatch.datapipeline.database.core import parse_report_assets
    from config import CONFIG

    folder_path = folder or CONFIG.output_folder
    try:
        result = parse_report_assets(folder_path, CONFIG)
        typer.secho(
            f"\nParsing complete: {result['processed_count']} "
            f"report(s) processed",
            fg=typer.colors.GREEN,
            bold=True
        )
    except (FileNotFoundError, ValueError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def enrich():
    """
    Enrich products with financial data.

    Fetches additional information (stock prices, company details,
    geographic data) for products using external APIs.

    Example:
        python -m capitolwatch.datapipeline.database enrich
    """
    from capitolwatch.datapipeline.database.core import enrich_products_data
    from config import CONFIG

    try:
        stats = enrich_products_data(CONFIG)

        total = stats.get('total_processed', 0)
        enriched = stats.get('enriched', 0)

        if total > 0:
            success_rate = enriched / total * 100
            if success_rate < 80:
                typer.secho(
                    f"\nWarning: Low success rate ({success_rate:.1f}%)",
                    fg=typer.colors.YELLOW
                )

        typer.secho(
            f"\nEnrichment complete: {enriched}/{total} products enriched",
            fg=typer.colors.GREEN,
            bold=True
        )
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def pipeline(
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Folder containing HTML reports (default: from config)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )
):
    """
    Run the complete database construction pipeline.

    Executes all steps in order:
    1. Initialize database and add senators
    2. Import reports from folder
    3. Match politicians to reports
    4. Parse assets from reports
    5. Enrich products with financial data

    Example:
        python -m capitolwatch.datapipeline.database pipeline \\
            --folder data/reports
    """
    from capitolwatch.datapipeline.database.core import run_complete_pipeline
    from config import CONFIG

    try:
        results = run_complete_pipeline(folder, CONFIG)

        # Check if pipeline completed successfully
        if "enrichment" in results:
            typer.secho(
                "\nComplete pipeline finished successfully!",
                fg=typer.colors.GREEN,
                bold=True
            )
        else:
            typer.secho(
                "\nPipeline completed with warnings",
                fg=typer.colors.YELLOW,
                bold=True
            )
            raise typer.Exit(1)

    except KeyboardInterrupt:
        typer.secho(
            "\nPipeline interrupted by user",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(130)
    except Exception as e:
        typer.secho(f"Pipeline error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def main():
    """Entry point for CLI when called directly."""
    app()


if __name__ == "__main__":
    main()
