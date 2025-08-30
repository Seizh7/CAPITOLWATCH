# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from config import CONFIG
from capitolwatch.datapipeline.database.init_db import initialize_database
import capitolwatch.datapipeline.database.congress_api as es
from capitolwatch.services.politicians import add_politician_list


def main():
    """
    Initializes the database for the application.
    """
    initialize_database(CONFIG)

    senators = es.get_current_senators(CONFIG)
    print(f"Total senators found: {len(senators)}")

    add_politician_list(senators, config=CONFIG)
    print("Senators have been added to the database.")


if __name__ == "__main__":
    main()
