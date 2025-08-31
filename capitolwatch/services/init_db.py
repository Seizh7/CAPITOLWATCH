# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import sqlite3


def initialize_database(config):
    """
    Initializes the SQLite database by creating all required tables if they do
    not already exist.

    This function enables foreign key support, creates the core tables for
    politicians, reports, products, and assets (including self-referencing
    parent_asset_id for hierarchical assets).

    Args:
        config (Config): Configuration instance containing paths and settings.
    Returns:
        None
    """

    conn = sqlite3.connect(config.db_path)
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
        id INTEGER PRIMARY KEY,         -- Unique ID for the product
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
