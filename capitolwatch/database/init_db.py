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

import sqlite3


def initialize_database(config):
    """
    Initializes the SQLite database by creating all required tables if they do
    not already exist.

    This function enables foreign key support, creates the core tables for
    politicians, reports, products, and assets (including self-referencing
    parent_asset_id for hierarchical assets).

    Returns:
        None
    """

    conn = sqlite3.connect()
    cur = conn.cursor()

    cur.execute("PRAGMA foreign_keys = ON;")

    # Stores basic info about politicians
    cur.execute("""
    CREATE TABLE IF NOT EXISTS politicians (
        id VARCHAR(7) PRIMARY KEY,
        last_name TEXT,
        first_name TEXT,
        party TEXT
    )
    """)

    # Stores metadata for each financial disclosure report
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,   -- Unique report ID
        politician_id VARCHAR(7),               -- Foreign key to politicians
        source_file TEXT,                       -- Path to the HTML file
        year INTEGER,                           -- Reporting year
        url TEXT,                               -- Source URL of the report
        import_timestamp TEXT,                  -- Timestamp when imported
        checksum TEXT,                          -- SHA-1 checksum of the HTML
        encoding TEXT,                          -- File encoding
        FOREIGN KEY (politician_id) REFERENCES politicians(id)
    )
    """)

    # Describes each unique financial product
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,         -- Unique ID for the product
        name TEXT,                              -- Name of the product
        isin TEXT UNIQUE,                       -- ISIN code
        type TEXT,                              -- Product type
        details TEXT                            -- Any additional details
    )
    """)

    # Links politicians/reports to the financial products they own
    cur.execute("""
    CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,   -- Unique asset ID
        report_id INTEGER,                      -- Foreign key to reports
        politician_id VARCHAR(7),               -- Foreign key to politicians
        product_id INTEGER,                     -- Foreign key to products
        owner TEXT,                             -- Who owns the asset
        value TEXT,                             -- Value or value range
        income_type TEXT,                       -- Type of income generated
        income TEXT,                            -- Amount of income
        comment TEXT,                           -- Any comments or notes
        parent_asset_id INTEGER,                -- Incase of hierarchical asset
        FOREIGN KEY (report_id) REFERENCES reports(id),
        FOREIGN KEY (politician_id) REFERENCES politicians(id),
        FOREIGN KEY (product_id) REFERENCES products(id),
        FOREIGN KEY (parent_asset_id) REFERENCES assets(id)
    )
    """)

    print(f"Database initialized at {config.db_path.absolute()}")
    conn.commit()
    conn.close()
