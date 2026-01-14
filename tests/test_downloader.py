# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from unittest.mock import MagicMock, patch
from capitolwatch.datapipeline.scraping.downloader import download_report


def test_download_report(tmp_path):
    # Mock du driver Selenium
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Fake Report</body></html>"

    url = "/search/view/annual/abc123xyz-4567-8901-2345-6789example/"

    # Préparation du mock config
    mock_config = MagicMock()
    mock_config.output_folder = tmp_path / "output"
    mock_config.output_folder.mkdir()  # Créer le dossier de sortie
    mock_config.project_root = tmp_path

    # Appel de la fonction — patcher le sleep pour accélérer
    with patch("builtins.print"), patch(
        "capitolwatch.datapipeline.scraping.downloader.time.sleep",
        return_value=None
    ):
        filename, returned_url = download_report(mock_driver, url, mock_config)

    # Vérifications sur le fichier créé et le contenu
    assert filename.exists()
    content = filename.read_text(encoding="utf-8")
    assert "Fake Report" in content
    assert returned_url == "https://efdsearch.senate.gov" + url


def test_download_absolute_url(tmp_path):
    mock_driver = MagicMock()
    mock_driver.page_source = "<html>Absolute</html>"

    url = (
        "https://efdsearch.senate.gov/search/view/annual/"
        "9876abcd-1234-5678-efgh-9876ijklmnop/"
    )

    mock_config = MagicMock()
    mock_config.output_folder = tmp_path / "output"
    mock_config.output_folder.mkdir()
    mock_config.project_root = tmp_path

    with patch("builtins.print"), patch(
        "capitolwatch.datapipeline.scraping.downloader.time.sleep",
        return_value=None
    ):
        filename, returned_url = download_report(mock_driver, url, mock_config)

    assert filename.exists()
    content = filename.read_text(encoding="utf-8")
    assert "Absolute" in content
    assert returned_url == url
