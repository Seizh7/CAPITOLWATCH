# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from capitolwatch.datapipeline.parsing.cleaner import clean_html_string
import pathlib


def test_clean_html():
    with open(pathlib.Path(__file__).parent / "test.html", "r") as f:
        html = f.read()
    cleaned = clean_html_string(html)
    # Check for essential information in the cleaned text (anonymized)
    assert "Jeanne M" in cleaned
    assert "DUPONT" in cleaned
    assert "Hammond Hardware Company" in cleaned
    assert "Corporate Securities" in cleaned
    assert "Self" in cleaned
    assert "$1,000,001 - $5,000,000" in cleaned
    assert "Dividends" in cleaned
    assert "None" in cleaned

    # Negative controls: removed HTML tags and classes
    assert "<div" not in cleaned
    assert "</td>" not in cleaned
    assert "noWrap" not in cleaned
