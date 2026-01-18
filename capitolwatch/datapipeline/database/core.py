#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Core database pipeline logic - separated from CLI for reusability.

This module contains the business logic for database operations,
independent of the CLI interface.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from capitolwatch.services.init_db import initialize_database
import capitolwatch.datapipeline.database.congress_api as congress_api
from capitolwatch.services.politicians import add_politicians
from capitolwatch.datapipeline.database.import_reports import import_reports
from capitolwatch.datapipeline.database.matching_workflow import (
    main as matching_main
)
from capitolwatch.datapipeline.database.parse_report_assets import (
    process_reports_assets
)
from capitolwatch.datapipeline.database.enrich_products import (
    run_enrichment_pipeline
)


def initialize_db(config: object) -> Dict[str, Any]:
    """
    Initialize database and add senators.

    Args:
        config: Configuration object.

    Returns:
        dict: Summary with 'senators_added'.
    """
    print("Initializing database tables...")
    initialize_database(config)
    print("Database tables created successfully")

    print("\nFetching current senators...")
    senators = congress_api.get_current_senators(config)
    print(f"Found {len(senators)} senators")

    added_count = add_politicians(senators, config=config)
    print(f"{added_count} senators added to database")

    return {"senators_added": added_count}


def import_reports_from_folder(
    folder_path: Path,
    config: object
) -> Dict[str, Any]:
    """
    Import reports from local folder.

    Args:
        folder_path: Path to folder containing HTML reports.
        config: Configuration object.

    Returns:
        dict: Summary with 'imported_count'.

    Raises:
        FileNotFoundError: If folder doesn't exist.
        ValueError: If no HTML files found.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    html_files = list(folder.glob("*.html"))
    if not html_files:
        raise ValueError(f"No HTML files found in: {folder_path}")

    print(f"Found {len(html_files)} HTML file(s)")
    print("Importing reports to database...")

    imported_count = import_reports(folder, config.project_root)

    print("Reports imported successfully")

    return {"imported_count": imported_count}


def match_politicians_to_reports(config: object) -> Dict[str, Any]:
    """
    Launch politician-to-report matching.

    Args:
        config: Configuration object.

    Returns:
        dict: Matching statistics.
    """
    print("Starting politician matching process...")
    print("Analyzing HTML reports to identify politicians...")

    stats = matching_main()

    print("MATCHING RESULTS")
    print(f"Reports processed:     {stats['processed']}")
    print(f"Successfully matched:  {stats['matched']}")
    print(f"Database updated:      {stats['updated']}")
    print(f"Require review:        {len(stats['needs_review'])}")

    if stats['needs_review']:
        print("\nReports requiring manual review:")
        for item in stats['needs_review'][:5]:
            print(f"  • {item['file']}: {item['name']}")
        if len(stats['needs_review']) > 5:
            print(f"  ... and {len(stats['needs_review']) - 5} more")

    return stats


def parse_report_assets(
    folder_path: Path,
    config: object
) -> Dict[str, Any]:
    """
    Parse assets from HTML reports and insert into database.

    Args:
        folder_path: Path to folder containing HTML reports.
        config: Configuration object.

    Returns:
        dict: Summary with 'processed_count'.

    Raises:
        FileNotFoundError: If folder doesn't exist.
        ValueError: If no HTML files found.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    html_files = list(folder.glob("*.html"))
    if not html_files:
        raise ValueError(f"No HTML files found in: {folder_path}")

    print(f"Found {len(html_files)} HTML file(s)")
    print("Parsing assets and products from reports...")

    process_reports_assets(folder_path)

    print("Report assets parsed and imported")

    return {"processed_count": len(html_files)}


def enrich_products_data(config: object) -> Dict[str, Any]:
    """
    Enrich products with financial and geographic data.

    Args:
        config: Configuration object.

    Returns:
        dict: Enrichment statistics.
    """
    print("Starting product enrichment with financial APIs...")
    print("This may take several minutes depending on the number of products")

    stats = run_enrichment_pipeline()

    print("ENRICHMENT RESULTS")
    print(f"Total processed:       {stats.get('total_processed', 0)}")
    print(f"Successfully enriched: {stats.get('enriched', 0)}")
    print(f"Geographic enriched:   {stats.get('geographic_enriched', 0)}")
    print(f"Non-tradeable:         {stats.get('non_tradeable', 0)}")
    print(f"Failed:                {stats.get('failed', 0)}")

    if stats.get('total_processed', 0) > 0:
        enriched = stats.get('enriched', 0)
        total = stats['total_processed']
        success_rate = enriched / total * 100
        print(f"Success rate:          {success_rate:.1f}%")

    return stats


def run_complete_pipeline(
    import_folder: Optional[Path],
    config: object
) -> Dict[str, Any]:
    """
    Run complete database construction pipeline.

    Args:
        import_folder: Optional folder for importing existing reports.
        config: Configuration object.

    Returns:
        dict: Complete pipeline summary.
    """
    print("COMPLETE DATABASE CONSTRUCTION PIPELINE")

    results = {}

    # Step 1: Initialization
    print("\n[STEP 1/5] Database Initialization")
    results["init"] = initialize_db(config)

    # Step 2: Report import
    print("\n[STEP 2/5] Report Import")
    folder = import_folder or config.output_folder
    try:
        results["import"] = import_reports_from_folder(folder, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Cannot continue without imported reports")
        return results

    # Step 3: Matching
    print("\n[STEP 3/5] Politician Matching")
    results["matching"] = match_politicians_to_reports(config)

    # Step 4: Asset parsing
    print("\n[STEP 4/5] Asset Parsing")
    try:
        results["parsing"] = parse_report_assets(folder, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Cannot continue without parsed assets")
        return results

    # Step 5: Product enrichment
    print("\n[STEP 5/5] Product Enrichment")
    results["enrichment"] = enrich_products_data(config)

    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print(f"Database is now ready for use at: {config.db_path}")

    return results
