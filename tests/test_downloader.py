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

from unittest.mock import MagicMock, patch
from capitolwatch.scraping.downloader import download_report

def test_download_report(tmp_path):
    """
    Tests that download_report saves the HTML content to the correct filename,
    using the report's unique ID from the URL.
    """
    # Mock Selenium driver with a fake page_source
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Fake Report</body></html>"
    
    # Example relative URL (same format as production)
    url = "/search/view/annual/abc123xyz-4567-8901-2345-6789example/"
    output_folder = tmp_path  # pytest provides a temporary directory
    
    # Patch the print function to suppress output
    with patch("builtins.print"):
        download_report(mock_driver, url, output_folder)
    
    # Build the expected filename
    expected_file = output_folder / "abc123xyz-4567-8901-2345-6789example.html"
    # Assert that the file was created
    assert expected_file.exists()
    # Assert the contents are as expected
    content = expected_file.read_text(encoding="utf-8")
    assert "<html><body>Fake Report</body></html>" in content

def test_download_absolute_url(tmp_path):
    """
    Tests download_report with an absolute URL.
    """
    mock_driver = MagicMock()
    mock_driver.page_source = "<html>Absolute</html>"
    url = "https://efdsearch.senate.gov/search/view/annual/9876abcd-1234-5678-efgh-9876ijklmnop/"
    output_folder = tmp_path

    with patch("builtins.print"):
        download_report(mock_driver, url, output_folder)

    expected_file = output_folder / "9876abcd-1234-5678-efgh-9876ijklmnop.html"
    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8")
    assert "Absolute" in content