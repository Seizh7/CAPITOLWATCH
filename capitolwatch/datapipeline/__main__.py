#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Main entry point for CAPITOLWATCH data pipeline CLI.

Usage:
    python -m capitolwatch.datapipeline [COMMAND] [OPTIONS]

Examples:
    # Show help
    python -m capitolwatch.datapipeline --help

    # Run full pipeline
    python -m capitolwatch.datapipeline full-pipeline --year 2023

    # Quick update
    python -m capitolwatch.datapipeline quick-update --year 2024

    # Check status
    python -m capitolwatch.datapipeline status

    # Access scraping sub-CLI
    python -m capitolwatch.datapipeline scraping --year 2023

    # Access database sub-CLI
    python -m capitolwatch.datapipeline database init
"""

from capitolwatch.datapipeline.cli import app


if __name__ == "__main__":
    app()
