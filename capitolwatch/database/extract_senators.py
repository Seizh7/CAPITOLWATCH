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
import requests
from config import CONFIG
import sqlite3

def get_current_senators():
    """
    Get all the current US senators from the Congress.gov API.

    Returns:
        list of dict: Each dictionary contains 'first_name', 'last_name', 'bioguide_id', and 'party'.
    """
    senators = []
    offset = 0
    limit = 100  # Number of results per API call (& number of senators)

    while True:
        # Build the API request URL and parameters
        url = "https://api.congress.gov/v3/member"
        parameters = {
            "api_key": CONFIG.api_key,
            "format": "json",
            "currentMember": "true",
            "limit": limit,
            "offset": offset
        }

        # Send GET request to the API
        response = requests.get(url, parameters)
        response.raise_for_status()  # Raise error if request failed
        data = response.json()
        members = data.get("members", [])

        if not members:
            break

        for member in members:
            try:
                # Extract the list of terms (service periods)
                terms = member.get("terms", {}).get("item", [])
                if not terms:
                    continue

                # Check if the member is a senator
                is_senator = any(term.get("chamber") == "Senate" for term in terms)
                if not is_senator:
                    continue

                full_name = member["name"]
                party = member.get("partyName", "Unknown")
                bioguide_id = member.get("bioguideId", "")

                # Split name into last and first names if comma present
                if "," in full_name:
                    last_name, first_names = map(str.strip, full_name.split(",", 1))
                else:
                    last_name = ""
                    first_names = full_name

                senators.append({
                    "first_name": first_names,
                    "last_name": last_name,
                    "bioguide_id": bioguide_id,
                    "party": party
                })

            except Exception as e:
                print(f"Error for member {member}: {e}")

        # Move to the next page of results
        offset += limit

    return senators

def add_senators_to_db(senators):
    """
    Inserts the list of senators into the 'politicians' table in the database.

    Args:
        senators (list of dict): List of senator information to insert.
    """
    conn = sqlite3.connect(CONFIG.db_path)
    cur = conn.cursor()

    for senator in senators:
        # Insert the senator if not already present
        cur.execute("""
            INSERT OR IGNORE INTO politicians (first_name, last_name, party, ID)
            VALUES (?, ?, ?, ?)
        """, (
            senator["first_name"],
            senator["last_name"],
            senator["party"],
            senator["bioguide_id"]
        ))

    conn.commit()
    conn.close()