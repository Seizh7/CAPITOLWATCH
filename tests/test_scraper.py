# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)


from bs4 import BeautifulSoup
from capitolwatch.datapipeline.scraping.scraper import extract_links


def test_extract_links():
    """
    Checks that the extract_links function correctly extracts
    all annual report links for the target year (2023)
    from a realistic HTML table.

    This function also ensures that no link from another year is present.
    """
    # Simulated HTML table snippet containing various reports
    html = (
        '<table id="filedReports">'
        '<tr role="row" class="odd"><td>Angela D</td><td>Alsobrooks</td>'
        '<td class=" noWrap">Candidate (Candidate)</td><td>'
        '<a href="/search/view/annual/f59a2eae-946c-419d-b6da-2c1723385614/'
        '" target="_blank">Candidate Report  (Amendment 1)</a></td>'
        '<td>08/29/2024</td></tr>'
        '<tr role="row" class="even"><td>John A</td><td>Barrasso</td>'
        '<td class=" noWrap">Barrasso, John (Senator)</td><td>'
        '<a href="/search/view/annual/139241f0-2606-43e7-a61d-c4b597660cf9/'
        '" target="_blank">Annual Report for CY 2023</a></td>'
        '<td>05/15/2024</td></tr>'
        '<tr role="row" class="odd"><td>Maria</td><td>Cantwell</td>'
        '<td class=" noWrap">Cantwell, Maria (Senator)</td><td>'
        '<a href="/search/view/annual/4c6dee46-2f1c-41ba-bd01-1c8dce1349bf/'
        '" target="_blank">Annual Report for CY 2023</a></td>'
        '<td>05/15/2024</td></tr>'
        '<tr role="row" class="even"><td>Christopher A</td><td>Coons</td>'
        '<td class=" noWrap">Coons, Chris (Senator)</td><td>'
        '<a href="/search/view/annual/dd3245e1-b4c3-4196-882a-4012cf2bd524/'
        '" target="_blank">Annual Report for CY 2023</a></td>'
        '<td>08/12/2024</td></tr>'
        '<tr role="row" class="odd"><td>RICHARD </td><td>BLUMENTHAL</td>'
        '<td class=" noWrap">Senator</td><td>'
        '<a href="/search/view/paper/42f1a7a7-dcf2-45b5-8ece-ac19724aaeb6/'
        '" target="_blank">Annual Report (Amendment)</a></td>'
        '<td>08/13/2024</td></tr>'
        '<tr role="row" class="even"><td>Katie</td><td>Britt</td>'
        '<td class=" noWrap">Britt, Katie (Senator)</td><td>'
        '<a href="/search/view/annual/ae8aa928-7d29-4e74-aac7-c8c9211713c7/'
        '" target="_blank">Annual Report for CY 2023</a></td>'
        '<td>07/15/2024</td></tr>'
        '<tr role="row" class="odd"><td>Michael F</td><td>Bennet</td>'
        '<td class=" noWrap">Bennet, Michael (Senator)</td><td>'
        '<a href="/search/view/annual/786a8aff-781-458e-a9be-ccfd5ea46bfc/'
        '" target="_blank">Annual Report for CY 2023</a></td>'
        '<td>07/15/2024</td></tr>'
        '<tr role="row" class="even"><td>Alexis</td><td>Smith</td>'
        '<td class=" noWrap">Candidate (Candidate)</td><td>'
        '<a href="/search/view/annual/1111eeee-2222-3333-4444-5555aaaa6666/'
        '" target="_blank">Candidate Report</a></td>'
        '<td>06/30/2024</td></tr>'
        '<tr role="row" class="odd"><td>Emma</td><td>Johnson</td>'
        '<td class=" noWrap">Johnson, Emma (Senator)</td><td>'
        '<a href="/search/view/annual/7777bbbb-8888-9999-aaaa-bbbbccccdddd/'
        '" target="_blank">Annual Report for CY 2023</a></td>'
        '<td>05/09/2024</td></tr>'
        '<tr role="row" class="even"><td>Patrick</td><td>O\'Malley</td>'
        '<td class=" noWrap">O\'Malley, Patrick (Senator)</td><td>'
        '<a href="/search/view/annual/abcd1234-5678-90ab-cdef-1234567890ab/'
        '" target="_blank">Annual Report for CY 2022</a></td>'
        '<td>04/28/2023</td></tr>'
        '</table>'
    )

    soup = BeautifulSoup(html, "html.parser")

    # Call the function to test
    links = extract_links(soup, year="2023")

    # Check that only 2023 report links are extracted (6 links expected)
    assert links == [
        "/search/view/annual/139241f0-2606-43e7-a61d-c4b597660cf9/",
        "/search/view/annual/4c6dee46-2f1c-41ba-bd01-1c8dce1349bf/",
        "/search/view/annual/dd3245e1-b4c3-4196-882a-4012cf2bd524/",
        "/search/view/annual/ae8aa928-7d29-4e74-aac7-c8c9211713c7/",
        "/search/view/annual/786a8aff-781-458e-a9be-ccfd5ea46bfc/",
        "/search/view/annual/7777bbbb-8888-9999-aaaa-bbbbccccdddd/",
    ]

    # Additional check: no link from another year should be present in the list
    assert all("2022" not in link for link in links)
