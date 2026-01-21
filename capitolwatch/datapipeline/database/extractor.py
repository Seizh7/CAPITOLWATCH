# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import re
from capitolwatch.services.politicians import normalize_name


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


def extract_main_text(element):
    """
    Extract only the main text from an element, ignoring nested divs.

    This is useful for cells like:
        <td>Mutual Funds<div class="muted">Mutual Fund</div></td>
    Where we only want "Mutual Funds", not "Mutual FundsMutual Fund".

    Args:
        element: BeautifulSoup element.

    Returns:
        str or None: The main text content, cleaned.
    """
    if not element:
        return None

    # Get only the direct text content (not from child elements)
    texts = []
    for child in element.children:
        if isinstance(child, str):
            texts.append(child.strip())
        # Stop at the first div (which contains the sub-type)
        elif child.name == 'div':
            break

    main_text = ' '.join(t for t in texts if t)
    if main_text:
        return clean_text(main_text)
    return clean_text(element.get_text())


def extract_type_with_subtype(element):
    """
    Extract both main type and subtype from a cell.

    For cells like:
        <td>Mutual Funds<div class="muted">Exchange Traded Fund/Note</div></td>

    Returns:
        tuple: (main_type, subtype) - subtype may be None
    """
    if not element:
        return None, None

    # Get the main text (before any div)
    main_texts = []
    for child in element.children:
        if isinstance(child, str):
            main_texts.append(child.strip())
        elif child.name == 'div':
            break

    main_type = clean_text(' '.join(t for t in main_texts if t))

    # Get the subtype from the muted div
    muted_div = element.find('div', class_='muted')
    subtype = clean_text(muted_div.get_text()) if muted_div else None

    # If no main text was found, use full text as fallback
    if not main_type:
        main_type = clean_text(element.get_text())

    return main_type, subtype


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
        # For asset_type, extract both main type and subtype
        # (e.g., "Mutual Funds" + "Exchange Traded Fund/Note")
        asset_type, asset_subtype = extract_type_with_subtype(cols[2])
        owner = clean_text(cols[3].get_text())
        value = clean_text(cols[4].get_text())
        income_type = extract_main_text(cols[5])
        # Income may also have a sub-value in muted div
        income, income_subtype = extract_type_with_subtype(cols[6])

        # Optionally extract a filer comment if present, e.g.:
        # <div class="muted"><em>Filer comment: </em>Your text...</div>
        comment = ""
        try:
            for div in cols[1].find_all("div", class_="muted"):
                em = div.find("em")
                if not em:
                    continue
                label = (em.get_text(strip=True) or "").lower()
                if "filer comment" in label:
                    # Collect text after the <em> label within this div
                    parts = []
                    for sib in em.next_siblings:
                        try:
                            text = (
                                sib if isinstance(sib, str)
                                else sib.get_text(" ", strip=True)
                            )
                        except Exception:
                            text = None
                        if text:
                            parts.append(text.strip())
                    comment = clean_text(" ".join(p for p in parts if p)) or ""
                    if comment:
                        break
        except Exception:
            comment = comment or ""

        asset = {
            "index": idx,
            "parent_index": parent_index,
            "name": asset_name,
            "type": asset_type,
            "subtype": asset_subtype,
            "owner": owner,
            "value": value,
            "income_type": income_type,
            "income": income,
            "income_subtype": income_subtype,
            "comment": comment
        }
        assets.append(asset)

    return assets
