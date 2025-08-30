# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import sqlite3
from config import CONFIG


def get_connection(config):
    """
    Create and return a SQLite database connection with sensible defaults.

    Args:
        config (object, optional): Configuration object.

    Returns:
        sqlite3.Connection: A ready-to-use SQLite connection with:
            - Row access by column name (`sqlite3.Row`)
            - Foreign key enforcement enabled
    """
    cfg = config or CONFIG
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
