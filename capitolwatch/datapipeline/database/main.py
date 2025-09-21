# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Database CLI - Step-by-step database construction interface.

This module provides a command-line interface for initializing and building
the CAPITOLWATCH database with the following steps:
1. Database initialization
2. Report collection via scraping
3. Report import from local folder (optional)
4. Politician-to-report matching
"""

import argparse
import sys
from pathlib import Path

from config import CONFIG
from capitolwatch.services.init_db import initialize_database
import capitolwatch.datapipeline.database.congress_api as congress_api
from capitolwatch.services.politicians import add_politicians
from capitolwatch.datapipeline.scraping.main import main as scraping_main
from capitolwatch.datapipeline.database.import_reports import import_reports
from capitolwatch.datapipeline.database.matching_workflow import (
    main as matching_main
)


def step_1_initialize_database():
    """
    Step 1: Initialize database and add senators.
    """
    print("=" * 60)
    print("STEP 1: DATABASE INITIALIZATION")
    print("=" * 60)

    print("Initializing database tables...")
    initialize_database(CONFIG)
    print("Database tables created successfully")

    print("\nFetching current senators...")
    senators = congress_api.get_current_senators(CONFIG)
    print(f"Total senators found: {len(senators)}")

    added_count = add_politicians(senators, config=CONFIG)
    print(f"{added_count} senators added to database")

    print("\nSTEP 1 COMPLETED: Database initialized\n")


def step_2_scrape_reports():
    """
    Step 2: Launch scraping to collect reports.
    """
    print("=" * 60)
    print("STEP 2: REPORT COLLECTION (SCRAPING)")
    print("=" * 60)

    print("Starting report scraping...")
    print(f"Period: {CONFIG.start_date} to {CONFIG.end_date}")
    print(f"Target year: {CONFIG.year}")
    print(f"Output folder: {CONFIG.output_folder}")

    try:
        scraping_main()
        print("\nSTEP 2 COMPLETED: Reports collected via scraping\n")
    except Exception as e:
        print(f"\nERROR during scraping: {e}")
        print("You can continue with step 3 if you already have reports\n")


def step_3_import_reports(folder_path=None):
    """
    Step 3: Import reports from local folder.

    Args:
        folder_path: Path to folder containing HTML reports
    """
    print("=" * 60)
    print("STEP 3: REPORT IMPORT")
    print("=" * 60)

    if folder_path is None:
        folder_path = CONFIG.output_folder

    folder = Path(folder_path)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder_path}")
        return False

    html_files = list(folder.glob("*.html"))
    if not html_files:
        print(f"ERROR: No HTML files found in: {folder_path}")
        return False

    print(f"Source folder: {folder_path}")
    print(f"HTML files found: {len(html_files)}")

    try:
        import_reports(folder, CONFIG.db_path, CONFIG.project_root)
        print("\nSTEP 3 COMPLETED: Reports imported to database\n")
        return True
    except Exception as e:
        print(f"\nERROR during import: {e}\n")
        return False


def step_4_match_politicians():
    """
    Step 4: Launch politician-to-report matching.
    """
    print("=" * 60)
    print("STEP 4: POLITICIAN-TO-REPORT MATCHING")
    print("=" * 60)

    print("Starting matching process...")
    print("Analyzing HTML reports to identify politicians...")

    try:
        stats = matching_main()

        print("\nMATCHING RESULTS:")
        print(f"   Reports processed: {stats['processed']}")
        print(f"   Successfully matched: {stats['matched']}")
        print(f"   Database updated: {stats['updated']}")
        print(f"   Require review: {len(stats['needs_review'])}")

        if stats['needs_review']:
            print("\nReports requiring manual review:")
            # Show only first 5
            for item in stats['needs_review'][:5]:
                print(f"   - {item['file']}: {item['name']}")
            if len(stats['needs_review']) > 5:
                print(f"   ... and {len(stats['needs_review']) - 5} others")

        print("\nSTEP 4 COMPLETED: Politician matching finished\n")
        return stats
    except Exception as e:
        print(f"\nERROR during matching: {e}\n")
        return None


def run_complete_pipeline(import_folder=None):
    """
    Run complete database construction pipeline.

    Args:
        import_folder: Optional folder for importing existing reports
    """
    print("LAUNCHING COMPLETE DATABASE CONSTRUCTION PIPELINE")
    print("=" * 80)

    # Step 1: Initialization
    step_1_initialize_database()

    # Step 2: Scraping (may fail, continue anyway)
    step_2_scrape_reports()

    # Step 3: Report import
    if not step_3_import_reports(import_folder):
        print("WARNING: Cannot continue without imported reports")
        return False

    # Step 4: Matching
    stats = step_4_match_politicians()
    if stats is None:
        return False

    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)
    print("Database is now ready for use.")
    print(f"Database file: {CONFIG.db_path}")

    return True


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="CLI for building CAPITOLWATCH database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

  # Complete pipeline (recommended)
  python -m capitolwatch.datapipeline.database.main --complete

  # Complete pipeline with import from specific folder
  python -m capitolwatch.datapipeline.database.main --complete \\
    --import-folder /path/to/reports

  # Individual steps
  python -m capitolwatch.datapipeline.database.main --init
  python -m capitolwatch.datapipeline.database.main --scrape
  python -m capitolwatch.datapipeline.database.main --import-reports \\
    --folder /path/to/reports
  python -m capitolwatch.datapipeline.database.main --match
        """
    )

    # Main actions
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Run complete pipeline (init + scrape + import + match)"
    )

    # Individual steps
    parser.add_argument(
        "--init",
        action="store_true",
        help="Step 1: Initialize database and add senators"
    )

    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Step 2: Launch scraping to collect reports"
    )

    parser.add_argument(
        "--import-reports",
        action="store_true",
        dest="import_reports",
        help="Step 3: Import reports from folder"
    )

    parser.add_argument(
        "--match",
        action="store_true",
        help="Step 4: Launch politician-to-report matching"
    )

    # Options
    parser.add_argument(
        "--folder",
        "--import-folder",
        dest="folder",
        type=str,
        help="Folder containing HTML reports to import "
             "(default: configured output folder)"
    )

    args = parser.parse_args()

    # Verify at least one action is specified
    actions = [args.complete, args.init, args.scrape,
               args.import_reports, args.match]
    if not any(actions):
        parser.print_help()
        sys.exit(1)

    try:
        # Complete pipeline
        if args.complete:
            success = run_complete_pipeline(args.folder)
            sys.exit(0 if success else 1)

        # Individual steps
        if args.init:
            step_1_initialize_database()

        if args.scrape:
            step_2_scrape_reports()

        if args.import_reports:
            success = step_3_import_reports(args.folder)
            if not success:
                sys.exit(1)

        if args.match:
            stats = step_4_match_politicians()
            if stats is None:
                sys.exit(1)

        print("Operation(s) completed successfully")

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
