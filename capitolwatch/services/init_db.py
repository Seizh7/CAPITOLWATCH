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

    # Describes each unique financial product with enriched fields
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,

        -- Financial identifiers
        figi TEXT,                          -- Bloomberg Global Identifier
        ticker TEXT,                        -- Main trading symbol
        exchange TEXT,                      -- Primary exchange

        -- Sector classification
        sector TEXT,                        -- GICS sector
        industry TEXT,                      -- GICS industry
        country TEXT,                       -- Country of origin
        asset_class TEXT,                   -- Asset class (EQUITY, ETF, etc.)

        -- Financial metrics
        beta REAL,                          -- Beta coefficient vs. market
        dividend_yield REAL,                -- Annual dividend yield (%)
        expense_ratio REAL,                 -- Expense ratio (for funds)
        market_cap BIGINT,                  -- Market capitalization

        -- Trading metadata
        currency TEXT DEFAULT 'USD',        -- Main currency
        is_etf BOOLEAN DEFAULT 0,           -- ETF flag
        is_mutual_fund BOOLEAN DEFAULT 0,   -- Mutual fund flag
        is_index_fund BOOLEAN DEFAULT 0,    -- Index fund flag

        -- Computed classification
        market_cap_tier TEXT,               -- Size (Large/Mid/Small/Micro)
        risk_rating TEXT,                   -- Risk level

        -- Enrichment metadata
        last_updated TEXT,                  -- Last update timestamp
        data_source TEXT,                   -- Data source (e.g., API, Manual)

        -- Constraints
        UNIQUE(name, type)                  -- Prevent duplicates
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

    # Create indexes for better query performance
    indexes = [
        ("idx_products_sector", "products", "sector"),
        ("idx_products_asset_class", "products", "asset_class"),
        ("idx_products_ticker", "products", "ticker"),
        ("idx_products_risk_rating", "products", "risk_rating"),
        ("idx_products_name_type", "products", "name, type"),
        ("idx_assets_product_id", "assets", "product_id"),
        ("idx_assets_politician_id", "assets", "politician_id"),
        ("idx_assets_report_id", "assets", "report_id")
    ]

    for index_name, table, columns in indexes:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table}({columns});"
        )

    print(f"Database initialized at {config.db_path.absolute()}")
    conn.commit()
    conn.close()
