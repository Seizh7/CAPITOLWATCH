# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import re
from capitolwatch.datapipeline.parsing.extractor import normalize_name


def extract_politician_name(soup):
    """
    Extracts names from the <title> tag.

    Args:
        soup (BeautifulSoup): Parsed HTML soup object.

    Returns:
        tuple: (first_names, last_name) as strings.
    """
    # Get the <title> tag from the HTML
    title = soup.title
    if not title or not title.text:
        return "", ""

    # Split the title text by hyphen ('-')
    parts = title.text.split('-')
    if len(parts) < 2:
        return "", ""

    # The last part should contain the name, separated by a comma
    name_part = parts[-1].strip()

    # Split on the first comma only (format: Lastname, Firstname)
    if "," in name_part:
        last, first = name_part.split(",", 1)
        last_name = last.strip()
        # Join in case of multiple first names
        first_names = " ".join(first.split())
        return normalize_name(first_names), normalize_name(last_name)
    return "", ""


def extract_report_year(soup):
    """
    Extracts the report year from the <title> tag.

    Args:
        soup (BeautifulSoup): Parsed HTML soup object.

    Returns:
        int or None: The year of the report if found, otherwise None.
    """
    # Get the <title> tag from the HTML
    title = soup.title
    if not title or not title.text:
        return None

    # Use regex to find a year after 'Annual Report for'
    match = re.search(r"Annual Report for (\d{4})", title.text)
    if match:
        return int(match.group(1))

    return None


def clean_text(text):
    """
    Cleans and normalizes extracted text from HTML.

    Args:
        text (str): The raw text extracted from HTML.

    Returns:
        str or None: The cleaned text, or None.
    """
    # Normalize whitespace and strip leading/trailing commas and spaces
    text = re.sub(r'\s+', ' ', text).strip(", ")
    return text if text not in ("", "None") else None


def extract_assets(soup):
    """
    Parses the 'Part 3. Assets' table from the HTML soup and returns a list of
    dictionaries, each representing an asset.

    Args:
        soup (BeautifulSoup): Parsed HTML soup object.

    Returns:
        list of dict: List of extracted asset dictionaries.
    """
    assets = []

    # Find the <h3> of "Part 3. Assets" title
    h3 = soup.find("h3", string=re.compile("Part 3. Assets"))
    if not h3:
        return assets

    # Locate <tbody> within the parent <section>
    table = h3.find_parent("section").find("tbody")
    if not table:
        return assets

    # Iterate over each row
    for row in table.find_all("tr"):
        cols = row.find_all("td")

        # Asset index (e.g. "3", "3.1")
        idx = cols[0].get_text(strip=True)

        # Determine parent index for this asset
        if "." in idx:
            parent_index = idx.split(".")[0]
        else:
            parent_index = None

        # Extract datas
        strong = cols[1].find("strong")
        asset_name = clean_text(strong.get_text())
        asset_type = clean_text(cols[2].get_text())
        owner = clean_text(cols[3].get_text())
        value = clean_text(cols[4].get_text())
        income_type = clean_text(cols[5].get_text())
        income = clean_text(cols[6].get_text())

        # Optionally extract a comment (if present) from <em> tags
        comment = ""
        comment_div = cols[1].find("description")
        if comment_div:
            comment = comment_div.get_text(strip=True)

        asset = {
            "index": idx,
            "parent_index": parent_index,
            "name": asset_name,
            "type": asset_type,
            "owner": owner,
            "value": value,
            "income_type": income_type,
            "income": income,
            "comment": comment
        }
        assets.append(asset)

    return assets
