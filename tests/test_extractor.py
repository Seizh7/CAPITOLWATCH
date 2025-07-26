"""
The MIT License (MIT)

Copyright (c) 2025-present Seizh7

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import capitolwatch.parsing.extractor as extractor
from bs4 import BeautifulSoup
import pathlib


def test_extract_politician_name_standard():
    with open(pathlib.Path(__file__).parent / "test.html", "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    first, last = extractor.extract_politician_name(soup)
    assert first == "jeanne m"
    assert last == "dupont"


def test_extract_report_year_found():
    with open(pathlib.Path(__file__).parent / "test.html", "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    year = extractor.extract_report_year(soup)
    assert year == 2023


def test_extract_assets_basic():
    with open(pathlib.Path(__file__).parent / "test.html", "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    assets = extractor.extract_assets(soup)
    assert len(assets) == 44

    # First row
    assert assets[0]["index"] == "1"
    assert assets[0]["parent_index"] is None
    assert assets[0]["name"] == "Arp & Hammond Hardware Company"
    assert assets[0]["type"] == "Corporate SecuritiesNon-Public Stock"
    assert assets[0]["owner"] == "Self"
    assert assets[0]["value"] == "$1,000,001 - $5,000,000"
    assert assets[0]["income_type"] == "Dividends"
    assert assets[0]["income"] == "None (or less than $201)"
    assert assets[0]["comment"] == ""

    # Second row
    assert assets[5]["index"] == "4.1"
    assert assets[5]["parent_index"] == "4"
    assert assets[5]["name"] == "Sweetgrass"
    assert assets[5]["type"] == "Real EstateResidential"
    assert assets[5]["owner"] == "Self"
    assert assets[5]["value"] == "$1,000,001 - $5,000,000"
    assert assets[5]["income_type"] is None
    assert assets[5]["income"] == "None (or less than $201)"
    assert assets[5]["comment"] == ""
