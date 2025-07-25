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
from config import CONFIG

# Connect to the SQLite database
conn = sqlite3.connect(CONFIG.db_path)
cur = conn.cursor()

# Enable foreign key constraint enforcement in SQLite
cur.execute("PRAGMA foreign_keys = ON;")

# Table: politicians
cur.execute("""
CREATE TABLE IF NOT EXISTS politicians (
    id VARCHAR(7) PRIMARY KEY,
    last_name TEXT,
    first_name TEXT,
    party TEXT
)
""")

# Table: reports (financial disclosure reports)
cur.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    politician_id INTEGER,
    source_file TEXT,
    year INTEGER,
    url TEXT,
    import_timestamp TEXT,
    checksum TEXT,
    encoding TEXT,
    FOREIGN KEY (politician_id) REFERENCES politicians(id)
)
""")

# Table: products (financial products)
cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    isin TEXT UNIQUE,
    type TEXT,
    details TEXT
)
""")

# Table: assets (ownership of financial products by politicians)
cur.execute("""
CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    politician_id INTEGER,
    product_id INTEGER,
    owner TEXT,
    value TEXT,
    income_type TEXT,
    income TEXT,
    comment TEXT,
    parent_asset_id INTEGER,
    FOREIGN KEY (report_id) REFERENCES reports(id),
    FOREIGN KEY (politician_id) REFERENCES politicians(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (parent_asset_id) REFERENCES assets(id)
)
""")

print(f"Database initialized at {CONFIG.db_path.absolute()}")
conn.commit()
conn.close()
