# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)
"""
Portfolio embeddings service - Database CRUD operations only.

Provides database operations for storing and retrieving portfolio embeddings.
Note: Table creation is handled by init_db.py
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from config import CONFIG
from capitolwatch.db import get_connection


# ---------- Write API (store embeddings) ----------

def store_portfolio_embedding(
    politician_id: int,
    method: str,
    embedding_vector: Union[np.ndarray, List[float]],
    features_used: Optional[List[str]] = None,
    asset_count: Optional[int] = None,
    total_value: Optional[float] = None,
    metadata: Optional[Dict] = None,
    *,
    config: Optional[object] = None
) -> int:
    """
    Store a portfolio embedding in the database.

    Args:
        politician_id: ID of the politician
        method: Embedding method name
        embedding_vector: The embedding vector
        features_used: List of features used to generate the embedding
        asset_count: Number of assets in the portfolio
        total_value: Total value of the portfolio
        metadata: Additional metadata
        config: Configuration object

    Returns:
        ID of the inserted/updated record
    """

    # Convert numpy array to bytes
    if isinstance(embedding_vector, np.ndarray):
        embedding_blob = embedding_vector.tobytes()
        vector_dimension = len(embedding_vector)
    else:
        embedding_array = np.array(embedding_vector, dtype=np.float32)
        embedding_blob = embedding_array.tobytes()
        vector_dimension = len(embedding_array)

    # Convert lists and dicts to JSON
    features_json = json.dumps(features_used) if features_used else None
    metadata_json = json.dumps(metadata) if metadata else None

    sql = """
    INSERT OR REPLACE INTO portfolio_embeddings
    (politician_id, method, embedding, vector_dimension, features_used,
     asset_count, total_value, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    with get_connection(config or CONFIG) as conn:
        cursor = conn.execute(sql, (
            politician_id, method, embedding_blob, vector_dimension,
            features_json, asset_count, total_value, metadata_json
        ))
        conn.commit()
        return cursor.lastrowid


# ---------- Read API (get embeddings) ----------

def get_portfolio_embedding(
    politician_id: int,
    method: str,
    *,
    config: Optional[object] = None
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Get a specific portfolio embedding from the database.

    Args:
        politician_id: ID of the politician
        method: Embedding method name
        config: Configuration object

    Returns:
        Tuple of (embedding_vector, metadata) or None if not found
    """

    sql = """
    SELECT embedding, features_used, asset_count, total_value, metadata,
           created_at
    FROM portfolio_embeddings
    WHERE politician_id = ? AND method = ?
    """

    with get_connection(config or CONFIG) as conn:
        cursor = conn.execute(sql, (politician_id, method))
        row = cursor.fetchone()

        if row is None:
            return None

        (
            embedding_blob, features_used, asset_count,
            total_value, metadata, created_at
        ) = row
        embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)

        # Parse JSON fields
        features_list = json.loads(features_used) if features_used else []
        metadata_dict = json.loads(metadata) if metadata else {}

        # Combine metadata
        full_metadata = {
            'features_used': features_list,
            'asset_count': asset_count,
            'total_value': total_value,
            'created_at': created_at,
            **metadata_dict
        }

        return embedding_vector, full_metadata


# ---------- Query API (list/get multiple embeddings) ----------

def query_portfolio_embeddings(
    method: Optional[str] = None,
    politician_ids: Optional[List[int]] = None,
    *,
    config: Optional[object] = None
) -> Dict[int, Tuple[np.ndarray, Dict]]:
    """
    Query portfolio embeddings with optional filters.

    Args:
        method: If specified, filter by method
        politician_ids: If specified, filter by politician IDs
        config: Configuration object

    Returns:
        Dict mapping politician_id -> (embedding_vector, metadata)
    """

    where_clauses = []
    params = []

    if method:
        where_clauses.append("method = ?")
        params.append(method)

    if politician_ids:
        placeholders = ",".join("?" for _ in politician_ids)
        where_clauses.append(f"politician_id IN ({placeholders})")
        params.extend(politician_ids)

    where_clause = (
        " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    )

    sql = f"""
    SELECT politician_id, embedding, features_used, asset_count, total_value,
           metadata, created_at
    FROM portfolio_embeddings{where_clause}
    """

    result: Dict[int, Tuple[np.ndarray, Dict]] = {}

    with get_connection(config or CONFIG) as conn:
        cursor = conn.execute(sql, params)

        for row in cursor.fetchall():
            (
                politician_id, embedding_blob, features_used,
                asset_count, total_value, metadata, created_at
            ) = row

            embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            features_list = json.loads(features_used) if features_used else []
            metadata_dict = json.loads(metadata) if metadata else {}

            full_metadata = {
                'features_used': features_list,
                'asset_count': asset_count,
                'total_value': total_value,
                'created_at': created_at,
                **metadata_dict
            }

            result[politician_id] = (embedding_vector, full_metadata)

    return result


# ------ Metadata / Utility API ----------

def list_available_portfolio_methods(
    *, config: Optional[object] = None
) -> List[str]:
    """
    List all available portfolio embedding methods in the database.

    Args:
        config: Configuration object

    Returns:
        List of method names
    """
    sql = "SELECT DISTINCT method FROM portfolio_embeddings ORDER BY method"

    with get_connection(config or CONFIG) as conn:
        cursor = conn.execute(sql)
        return [row[0] for row in cursor.fetchall()]


def count_portfolio_embeddings(
    method: Optional[str] = None,
    *,
    config: Optional[object] = None
) -> int:
    """
    Count portfolio embeddings in the database.

    Args:
        method: If specified, count only this method
        config: Configuration object

    Returns:
        Number of embeddings
    """
    if method:
        sql = "SELECT COUNT(*) FROM portfolio_embeddings WHERE method = ?"
        params = [method]
    else:
        sql = "SELECT COUNT(*) FROM portfolio_embeddings"
        params = []

    with get_connection(config or CONFIG) as conn:
        cursor = conn.execute(sql, params)
        return cursor.fetchone()[0]


def delete_portfolio_embeddings(
    method: Optional[str] = None,
    politician_id: Optional[int] = None,
    *,
    config: Optional[object] = None
) -> int:
    """
    Delete portfolio embeddings from the database.

    Args:
        method: If specified, delete only this method
        politician_id: If specified, delete only this politician
        config: Configuration object

    Returns:
        Number of deleted rows
    """
    where_clauses = []
    params = []

    if method:
        where_clauses.append("method = ?")
        params.append(method)

    if politician_id:
        where_clauses.append("politician_id = ?")
        params.append(politician_id)

    where_clause = (
        f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    )

    with get_connection(config or CONFIG) as conn:
        cursor = conn.execute(
            f"DELETE FROM portfolio_embeddings{where_clause}", params
        )
        conn.commit()
        return cursor.rowcount


def get_portfolio_embedding_statistics(
    *, config: Optional[object] = None
) -> Dict[str, Union[int, List[str]]]:
    """
    Get basic statistics about portfolio embeddings in the database.

    Returns:
        dict: Statistics including total count, methods, and politicians
    """

    with get_connection(config or CONFIG) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM portfolio_embeddings")
        total_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT DISTINCT method FROM portfolio_embeddings ORDER BY method"
        )
        methods = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            "SELECT COUNT(DISTINCT politician_id) FROM portfolio_embeddings"
        )
        unique_politicians = cursor.fetchone()[0]

        return {
            'total_embeddings': total_count,
            'available_methods': methods,
            'unique_politicians': unique_politicians
        }
