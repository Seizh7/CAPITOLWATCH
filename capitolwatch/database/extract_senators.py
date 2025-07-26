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

import re
import requests
import sqlite3


def normalize_name(name):
    """
    Normalizes a personal name by removing punctuation, converting to
    lowercase, and collapsing extra spaces.

    Args:
        name (str): The name to normalize.

    Returns:
        str: Normalized version of the name.
    """
    if not name:
        return ""
    name = name.lower()                  # Lowercase for consistent comparison
    name = re.sub(r"[.']", "", name)     # Remove dots and apostrophes
    name = re.sub(r"[-]", " ", name)     # Replace hyphens with space
    name = re.sub(r"\s+", " ", name)     # Collapse multiple spaces into one
    name = name.strip(", ")              # Remove commas and spaces
    return name.strip()


def get_current_senators(config):
    """
    Get all the current US senators from the Congress.gov API.

    Args:
        config (Config): Configuration object containing API key and settings.

    Returns:
        list of dict: Each dictionary contains 'first_name', 'last_name',
        'bioguide_id', 'party'.
    """
    senators = []
    offset = 0
    limit = 100  # Number of results per API page (& number of senators)

    while True:
        # Build the API request URL and parameters
        url = "https://api.congress.gov/v3/member"
        parameters = {
            "api_key": config.api_key,
            "format": "json",
            "currentMember": "true",
            "limit": limit,
            "offset": offset
        }

        # Request the current batch of members from the Congress API
        response = requests.get(url, parameters)
        response.raise_for_status()  # Stop if request fails
        data = response.json()
        members = data.get("members", [])

        # Stop looping if there are no more members returned
        if not members:
            break

        for member in members:
            try:
                # Extract list of terms (service periods)
                terms = member.get("terms", {}).get("item", [])
                if not terms:
                    continue

                # Check if the member has served in the Senate
                is_senator = any(
                    term.get("chamber") == "Senate"
                    for term in terms
                )
                if not is_senator:
                    continue

                # Parse the full name
                full_name = member["name"]
                party = member.get("partyName", "Unknown")
                bioguide_id = member.get("bioguideId", "")

                # Split into last and first names
                if "," in full_name:
                    last_name, first_names = map(
                        str.strip,
                        full_name.split(",", 1)
                    )
                else:
                    last_name = ""
                    first_names = full_name

                last_name_normalized = normalize_name(last_name)
                first_names_normalized = normalize_name(first_names)

                senators.append({
                    "first_name": last_name_normalized,
                    "last_name": first_names_normalized,
                    "bioguide_id": bioguide_id,
                    "party": party
                })

            except Exception as e:
                # Log and skip any errors on a member (e.g. missing fields)
                print(f"Error for member {member}: {e}")

        # Move to next page of API results
        offset += limit

    return senators


def add_senators_to_db(senators, config):
    """
    Inserts the list of senators into the 'politicians' table in the database.

    Args:
        senators (list of dict): List of senator information to insert.
    """
    conn = sqlite3.connect(config.db_path)
    cur = conn.cursor()

    for senator in senators:
        # Insert the senator if not already present
        cur.execute("""
            INSERT OR IGNORE INTO politicians (first_name, last_name, party,
                    ID)
            VALUES (?, ?, ?, ?)
        """, (
            senator["first_name"],
            senator["last_name"],
            senator["party"],
            senator["bioguide_id"]
        ))

    conn.commit()
    conn.close()
