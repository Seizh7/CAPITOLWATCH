#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Main entry point for CAPITOLWATCH analysis CLI.

Usage:
    python -m capitolwatch.analysis [COMMAND] [OPTIONS]

Examples:
    # Show help
    python -m capitolwatch.analysis --help

    # Build the feature store
    python -m capitolwatch.analysis features

    # Run internal + external evaluation
    python -m capitolwatch.analysis evaluate

    # Run cluster analysis (Markdown reports)
    python -m capitolwatch.analysis analyze

    # Generate all visualization plots
    python -m capitolwatch.analysis visualize

    # Run the complete pipeline
    python -m capitolwatch.analysis full-pipeline
"""

from capitolwatch.analysis.cli import app


if __name__ == "__main__":
    app()
