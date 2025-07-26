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

from capitolwatch.parsing.cleaner import clean_html_string
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
