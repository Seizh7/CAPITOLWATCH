# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Service for managing aggregated asset embeddings

This service handles the creation, storage, and retrieval of asset-level
embeddings that are aggregated from individual product embeddings weighted
by asset values.
"""

import json
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import CONFIG
from capitolwatch.db import get_connection


# ---------- Write API (store embeddings) ----------

def create_asset_embeddings_table(*, config=None):
    """
    Create the asset_embeddings table if it doesn't exist.

    Args:
        config: Configuration object
    """
    if config is None:
        config = CONFIG

    with get_connection(config) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_embeddings (
                politician_id INTEGER,
                method TEXT NOT NULL,
                embedding BLOB NOT NULL,
                vector_dimension INTEGER NOT NULL,
                features_used TEXT,
                asset_count INTEGER,
                total_value REAL,
                created_at TEXT,
                metadata TEXT,
                PRIMARY KEY (politician_id, method),
                FOREIGN KEY (politician_id)
                    REFERENCES politicians(id) ON DELETE CASCADE
            )
        """)

        # Indexes for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_embeddings_method
            ON asset_embeddings(method)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_embeddings_created
            ON asset_embeddings(created_at)
        """)

        conn.commit()


def store_asset_embedding(
    politician_id: int,
    method: str,
    embedding_vector: np.ndarray,
    features_used: List[str],
    asset_count: int,
    total_value: float,
    metadata: Optional[Dict] = None,
    *,
    config=None
):
    """
    Store an asset embedding in the database.

    Args:
        politician_id: ID of the politician
        method: Embedding method used (e.g., 'custom_financial_weighted')
        embedding_vector: The aggregated asset embedding vector
        features_used: List of features used in the embedding
        asset_count: Number of assets included
        total_value: Total value of the assets
        metadata: Additional metadata as dictionary
        config: Configuration object
    """
    if config is None:
        config = CONFIG

    # Serialize the embedding vector
    embedding_blob = pickle.dumps(embedding_vector)

    # Prepare metadata
    metadata_json = json.dumps(metadata or {})
    features_json = json.dumps(features_used)
    created_at = datetime.now().isoformat()

    with get_connection(config) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO asset_embeddings (
                politician_id, method, embedding, vector_dimension,
                features_used, asset_count, total_value, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            politician_id, method, embedding_blob, len(embedding_vector),
            features_json, asset_count, total_value, created_at, metadata_json
        ))
        conn.commit()


# ---------- Read API (get embeddings) ----------

def get_asset_embedding(
    politician_id: int,
    method: str,
    *,
    config=None
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Retrieve an asset embedding from the database.

    Args:
        politician_id: ID of the politician
        method: Embedding method
        config: Configuration object

    Returns:
        Tuple of (embedding_vector, metadata) or None if not found
    """
    if config is None:
        config = CONFIG

    with get_connection(config) as conn:
        cursor = conn.execute("""
            SELECT embedding, metadata, features_used, asset_count, total_value
            FROM asset_embeddings
            WHERE politician_id = ? AND method = ?
        """, (politician_id, method))

        row = cursor.fetchone()
        if row is None:
            return None

        (
            embedding_blob,
            metadata_json,
            features_json,
            asset_count,
            total_value
        ) = row

        # Deserialize
        embedding_vector = pickle.loads(embedding_blob)
        metadata = json.loads(metadata_json)
        features_used = json.loads(features_json)

        # Add additional info to metadata
        metadata.update({
            'features_used': features_used,
            'asset_count': asset_count,
            'total_value': total_value
        })

        return embedding_vector, metadata


def get_all_asset_embeddings(
    method: str,
    *,
    config=None
) -> Dict[int, Tuple[np.ndarray, Dict]]:
    """
    Get all asset embeddings for a specific method.

    Args:
        method: Embedding method
        config: Configuration object

    Returns:
        Dictionary mapping politician_id to (embedding_vector, metadata)
    """
    if config is None:
        config = CONFIG

    result = {}

    with get_connection(config) as conn:
        cursor = conn.execute("""
            SELECT politician_id, embedding, metadata, features_used,
                   asset_count, total_value
            FROM asset_embeddings
            WHERE method = ?
        """, (method,))

        for row in cursor.fetchall():
            (
                politician_id,
                embedding_blob,
                metadata_json,
                features_json,
                asset_count,
                total_value
            ) = row

            # Deserialize
            embedding_vector = pickle.loads(embedding_blob)
            metadata = json.loads(metadata_json)
            features_used = json.loads(features_json)

            # Add additional info to metadata
            metadata.update({
                'features_used': features_used,
                'asset_count': asset_count,
                'total_value': total_value
            })

            result[politician_id] = (embedding_vector, metadata)

    return result


def list_available_asset_methods(*, config=None) -> List[str]:
    """
    List all available asset embedding methods.

    Args:
        config: Configuration object

    Returns:
        List of method names
    """
    if config is None:
        config = CONFIG

    with get_connection(config) as conn:
        cursor = conn.execute("""
            SELECT DISTINCT method FROM asset_embeddings
            ORDER BY method
        """)
        return [row[0] for row in cursor.fetchall()]


# ---------- Metadata and Statistics API ----------

def get_asset_embedding_statistics(*, config=None) -> Dict:
    """
    Get statistics about asset embeddings in the database.

    Args:
        config: Configuration object

    Returns:
        Dictionary with statistics
    """
    if config is None:
        config = CONFIG

    stats = {}

    with get_connection(config) as conn:
        # Count by method
        cursor = conn.execute("""
            SELECT method, COUNT(*) as count, AVG(vector_dimension) as avg_dim,
                   AVG(asset_count) as avg_assets,
                   AVG(total_value) as avg_value
            FROM asset_embeddings
            GROUP BY method
        """)

        method_stats = {}
        for row in cursor.fetchall():
            method, count, avg_dim, avg_assets, avg_value = row
            method_stats[method] = {
                'count': count,
                'avg_dimension': int(avg_dim) if avg_dim else 0,
                'avg_assets': avg_assets,
                'avg_value': avg_value
            }

        stats['by_method'] = method_stats

        # Total counts
        cursor = conn.execute("SELECT COUNT(*) FROM asset_embeddings")
        stats['total_embeddings'] = cursor.fetchone()[0]

        cursor = conn.execute("""
            SELECT COUNT(DISTINCT politician_id) FROM asset_embeddings
        """)
        stats['unique_politicians'] = cursor.fetchone()[0]

    return stats


# ---------- Deletion API (delete embeddings) ----------

def delete_asset_embeddings(
    method: Optional[str] = None,
    politician_id: Optional[int] = None,
    *,
    config=None
):
    """
    Delete asset embeddings based on criteria.

    Args:
        method: If specified, delete only this method
        politician_id: If specified, delete only this politician
        config: Configuration object
    """
    if config is None:
        config = CONFIG

    where_clauses = []
    params = []

    if method:
        where_clauses.append("method = ?")
        params.append(method)

    if politician_id:
        where_clauses.append("politician_id = ?")
        params.append(politician_id)

    if where_clauses:
        where_clause = " WHERE " + " AND ".join(where_clauses)
    else:
        where_clause = ""

    with get_connection(config) as conn:
        conn.execute(f"DELETE FROM asset_embeddings{where_clause}", params)
        conn.commit()
