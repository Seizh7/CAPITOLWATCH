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
import sqlite3


def test_download_report(tmp_path):
    # Mock du driver Selenium
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Fake Report</body></html>"

    url = "/search/view/annual/abc123xyz-4567-8901-2345-6789example/"

    # Préparation du mock config
    mock_config = MagicMock()
    mock_config.db_path = tmp_path / "test.db"
    mock_config.output_folder = tmp_path / "output"
    mock_config.output_folder.mkdir()  # Créer le dossier de sortie
    mock_config.project_root = tmp_path

    # Création de la base de données
    conn = sqlite3.connect(mock_config.db_path)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            import_timestamp TEXT,
            checksum TEXT,
            encoding TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()

    with patch("builtins.print"):
        download_report(mock_driver, url, mock_config)

    # On rouvre la DB pour récupérer le chemin du fichier
    conn = sqlite3.connect(mock_config.db_path)
    cur = conn.cursor()
    cur.execute("SELECT source_file FROM reports")
    row = cur.fetchone()
    conn.close()

    source_file = row[0]
    expected_file = tmp_path / source_file

    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8")
    assert "Fake Report" in content


def test_download_absolute_url(tmp_path):
    mock_driver = MagicMock()
    mock_driver.page_source = "<html>Absolute</html>"

    url = (
        "https://efdsearch.senate.gov/search/view/annual/"
        "9876abcd-1234-5678-efgh-9876ijklmnop/"
    )

    mock_config = MagicMock()
    mock_config.db_path = tmp_path / "test.db"
    mock_config.output_folder = tmp_path / "output"
    mock_config.output_folder.mkdir()
    mock_config.project_root = tmp_path

    conn = sqlite3.connect(mock_config.db_path)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            import_timestamp TEXT,
            checksum TEXT,
            encoding TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()

    with patch("builtins.print"):
        download_report(mock_driver, url, mock_config)

    conn = sqlite3.connect(mock_config.db_path)
    cur = conn.cursor()
    cur.execute("SELECT source_file FROM reports")
    row = cur.fetchone()
    conn.close()

    source_file = row[0]
    expected_file = tmp_path / source_file

    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8")
    assert "Absolute" in content
