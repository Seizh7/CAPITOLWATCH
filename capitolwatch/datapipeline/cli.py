#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Main CLI for CAPITOLWATCH data pipeline orchestration.

This CLI coordinates the scraping and database modules, providing
both individual module access and orchestrated workflows.
"""

from pathlib import Path
from typing import Optional

import typer

from capitolwatch.datapipeline.scraping import cli as scraping_cli
from capitolwatch.datapipeline.database import cli as database_cli
from config import CONFIG


app = typer.Typer(
    help="CAPITOLWATCH Data Pipeline - Orchestrate scraping and "
         "database operations",
    no_args_is_help=True,
    add_completion=False
)

# Add sub-CLIs as separate commands
app.add_typer(
    scraping_cli.app,
    name="scraping",
    help="Scrape US Senate annual reports"
)
app.add_typer(
    database_cli.app,
    name="database",
    help="Build and manage the database"
)


@app.command()
def full_pipeline(
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
        help="Start date in MM/DD/YYYY format for scraping"
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        "-e",
        help="End date in MM/DD/YYYY format for scraping"
    ),
    skip_scraping: bool = typer.Option(
        False,
        "--skip-scraping",
        help="Skip the scraping step (use existing HTML files)"
    ),
    skip_init: bool = typer.Option(
        False,
        "--skip-init",
        help="Skip database initialization (use existing database)"
    ),
):
    """
    Run the complete data pipeline: scrape → import → match → parse.

    This orchestrates the entire workflow from scraping reports to
    building the complete database with all enrichments.

    Example:
        python -m capitolwatch.datapipeline full-pipeline --year 2023
    """
    typer.secho(
        "\nStarting CAPITOLWATCH Full Pipeline",
        fg=typer.colors.CYAN,
        bold=True
    )

    # Step 1: Scraping (unless skipped)
    if not skip_scraping:
        typer.secho(
            "\nStep 1/5: Scraping annual reports...",
            fg=typer.colors.BLUE,
            bold=True
        )
        from capitolwatch.datapipeline.scraping.core import run_scraping
        try:
            run_scraping(
                year=year,
                start_date=start,
                end_date=end,
                output_folder=CONFIG.output_folder
            )
            typer.secho("Scraping completed", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"Scraping failed: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    else:
        typer.secho("\n⏭Step 1/5: Scraping skipped", fg=typer.colors.YELLOW)

    # Step 2: Database initialization (unless skipped)
    if not skip_init:
        typer.secho(
            "\nStep 2/5: Initializing database...",
            fg=typer.colors.BLUE,
            bold=True
        )
        from capitolwatch.datapipeline.database.core import initialize_db
        try:
            initialize_db(CONFIG)
            typer.secho("Database initialized", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(
                f"✗ Initialization failed: {e}",
                fg=typer.colors.RED,
                err=True
            )
            raise typer.Exit(code=1)
    else:
        typer.secho(
            "\n⏭Step 2/5: Initialization skipped",
            fg=typer.colors.YELLOW
        )

    # Step 3: Import reports
    typer.secho(
        "\nStep 3/5: Importing reports to database...",
        fg=typer.colors.BLUE,
        bold=True
    )
    from capitolwatch.datapipeline.database.core import (
        import_reports_from_folder
    )
    try:
        import_reports_from_folder(CONFIG)
        typer.secho("Reports imported", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Import failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Step 4: Match politicians
    typer.secho(
        "\nStep 4/5: Matching politicians to reports...",
        fg=typer.colors.BLUE,
        bold=True,
    )
    from capitolwatch.datapipeline.database.core import (
        match_politicians_to_reports
    )
    try:
        match_politicians_to_reports(CONFIG)
        typer.secho("Politicians matched", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Matching failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Step 5: Parse assets
    typer.secho(
        "\nStep 5/5: Parsing report assets...",
        fg=typer.colors.BLUE,
        bold=True
    )
    from capitolwatch.datapipeline.database.core import parse_report_assets
    try:
        parse_report_assets(CONFIG)
        typer.secho("Assets parsed", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Parsing failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Final summary
    typer.secho(
        "Full Pipeline Completed Successfully",
        fg=typer.colors.GREEN,
        bold=True
    )
    typer.secho(f"Database: {CONFIG.db_path}", fg=typer.colors.CYAN)
    typer.secho(f"Reports: {CONFIG.output_folder}", fg=typer.colors.CYAN)


@app.command()
def quick_update(
    year: str = typer.Option(
        ...,
        "--year",
        "-y",
        help="Target year for annual reports"
    ),
):
    """
    Quick update: scrape new reports → import → match (skip init).

    Use this when the database already exists and you just want to
    update it with newly scraped reports.

    Example:
        python -m capitolwatch.datapipeline quick-update --year 2024
    """
    typer.secho("\n⚡ Quick Update Mode", fg=typer.colors.CYAN, bold=True)

    # Call full-pipeline with skip-init enabled
    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["full-pipeline", "--year", year, "--skip-init"]
    )
    raise typer.Exit(code=result.exit_code)


@app.command()
def enrich_products():
    """
    Enrich products with financial and geographic data.

    Fetches financial metadata from OpenFIGI and Yahoo Finance,
    adds geographic information, and updates the database.
    Only processes analyzable product types (stocks, ETFs, bonds, etc.).

    Example:
        python -m capitolwatch.datapipeline enrich-products
    """
    typer.secho(
        "\nProduct Enrichment Pipeline",
        fg=typer.colors.CYAN,
        bold=True
    )

    from capitolwatch.datapipeline.database.enrich_products import (
        run_enrichment_pipeline
    )

    try:
        stats = run_enrichment_pipeline()

        # Display summary
        typer.secho(
            "\nEnrichment Completed",
            fg=typer.colors.GREEN,
            bold=True
        )
        typer.secho(
            f"  Total processed: {stats['total_processed']}",
            fg=typer.colors.WHITE
        )
        typer.secho(
            f"  Skipped (non-analyzable): {stats['skipped_non_analyzable']}",
            fg=typer.colors.YELLOW
        )
        typer.secho(
            f"  Successfully enriched: {stats['enriched']}",
            fg=typer.colors.GREEN
        )
        typer.secho(
            f"  Geographic enriched: {stats['geographic_enriched']}",
            fg=typer.colors.CYAN
        )
        typer.secho(
            f"  Non-tradeable: {stats['non_tradeable']}",
            fg=typer.colors.BLUE
        )
        typer.secho(
            f"  Failures: {stats['failed']}",
            fg=typer.colors.RED
        )

        # Calculate success rate
        skipped = stats['skipped_non_analyzable']
        analyzable_total = stats['total_processed'] - skipped
        if analyzable_total > 0:
            success_rate = (stats['enriched'] / analyzable_total) * 100
            typer.secho(
                f"  Success rate: {success_rate:.1f}%",
                fg=typer.colors.GREEN
            )

    except KeyboardInterrupt:
        typer.secho(
            "\nEnrichment interrupted by user",
            fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"\nEnrichment failed: {e}",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(code=1)


@app.command()
def status():
    """
    Display pipeline status and statistics.

    Shows database statistics, report counts, and configuration.
    """
    typer.secho(
        "\nCAPITOLWATCH Pipeline Status",
        fg=typer.colors.CYAN,
        bold=True
    )

    # Check database
    from capitolwatch.db import get_connection
    try:
        conn = get_connection(CONFIG)
        cur = conn.cursor()

        # Count politicians
        cur.execute("SELECT COUNT(*) as count FROM politicians")
        politician_count = cur.fetchone()["count"]

        # Count reports
        cur.execute("SELECT COUNT(*) as count FROM reports")
        total_reports = cur.fetchone()["count"]

        # Count matched reports
        cur.execute(
            "SELECT COUNT(*) as count FROM reports "
            "WHERE politician_id IS NOT NULL"
        )
        matched_reports = cur.fetchone()["count"]

        # Count assets
        cur.execute("SELECT COUNT(*) as count FROM assets")
        asset_count = cur.fetchone()["count"]

        conn.close()

        typer.secho(f"\nDatabase: {CONFIG.db_path}", fg=typer.colors.GREEN)
        typer.secho(
            f"  Politicians: {politician_count}",
            fg=typer.colors.WHITE
        )
        typer.secho(
            f"  Reports: {total_reports} (matched: {matched_reports})",
            fg=typer.colors.WHITE
        )
        typer.secho(f"  Assets: {asset_count}", fg=typer.colors.WHITE)

    except Exception as e:
        typer.secho(f"\nDatabase error: {e}", fg=typer.colors.RED)

    # Check reports folder
    reports_dir = Path(CONFIG.output_folder)
    if reports_dir.exists():
        html_files = list(reports_dir.glob("*.html"))
        typer.secho(f"\nReports folder: {reports_dir}", fg=typer.colors.GREEN)
        typer.secho(f"  HTML files: {len(html_files)}", fg=typer.colors.WHITE)
    else:
        typer.secho(
            f"\nReports folder not found: {reports_dir}",
            fg=typer.colors.RED
        )


if __name__ == "__main__":
    app()
