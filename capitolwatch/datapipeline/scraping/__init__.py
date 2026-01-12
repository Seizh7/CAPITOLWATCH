# capitolwatch/pipeline/scraping/__init__.py

"""
Package scraping : collects and downloads US Senate annual reports.
"""

from capitolwatch.datapipeline.scraping.core import run_scraping

__all__ = ['run_scraping']
