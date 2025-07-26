# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import requests
from capitolwatch.services.politicians import normalize_name


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
                    "first_name": first_names_normalized,
                    "last_name": last_name_normalized,
                    "bioguide_id": bioguide_id,
                    "party": party
                })

            except Exception as e:
                # Log and skip any errors on a member (e.g. missing fields)
                print(f"Error for member {member}: {e}")

        # Move to next page of API results
        offset += limit

    return senators
