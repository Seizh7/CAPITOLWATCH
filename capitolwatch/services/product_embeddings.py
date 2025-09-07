# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Product Embeddings Service - Database operations for product_embeddings table.

Handles storage, retrieval, and management of vectorized product
representations for machine learning clustering and similarity analysis.
"""

import json
import pickle
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

from capitolwatch.db import get_connection
from config import CONFIG


# ---------- Write API (store embeddings) ----------

def store_embeddings(
    product_ids: List[int],
    embeddings: np.ndarray,
    method: str,
    features_used: List[str],
    metadata: Dict,
    *,
    config: Optional[object] = None,
    connection=None,
) -> None:
    """
    Store product embeddings in the database.

    Args:
        product_ids: List of product IDs
        embeddings: Numpy array of embeddings (shape: [n_products
            n_dimensions])
        method: Name of the embedding method
        features_used: List of feature names used for embedding generation
        metadata: Additional metadata about the embedding process
        config: Optional config override
        connection: Optional existing DB connection to reuse
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        timestamp = datetime.now().isoformat()

        # Prepare batch data
        batch_data = []
        for i, product_id in enumerate(product_ids):
            embedding_blob = pickle.dumps(embeddings[i])
            batch_data.append((
                product_id,
                embedding_blob,
                method,
                json.dumps(features_used),
                embeddings.shape[1],
                timestamp,
                json.dumps(metadata)
            ))

        # Batch insert/update
        cur.executemany("""
            INSERT OR REPLACE INTO product_embeddings
            (product_id, embedding, method, features_used,
             vector_dimension, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, batch_data)

        if close:
            connection.commit()
        print(f"Stored {len(product_ids)} embeddings using '{method}' method")

    except Exception as e:
        if close:
            connection.rollback()
        raise e
    finally:
        if close:
            connection.close()


# ---------- Read API (get embeddings) ----------

def get_embeddings(
    method: str,
    product_ids: List[int] = None,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Dict:
    """
    Retrieve product embeddings from the database.

    Args:
        method: Embedding method to retrieve
        product_ids: Optional list of specific product IDs to retrieve
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        Dict with keys:
        - product_ids: List of product IDs
        - embeddings: Numpy array of embeddings (None if no data)
        - metadata: Metadata dict from embedding generation
        - features_used: List of features used for generation
        - vector_dimension: Dimension of embeddings
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()

        if product_ids:
            placeholders = ','.join(['?'] * len(product_ids))
            query = f"""
                SELECT product_id, embedding, features_used,
                       vector_dimension, metadata, updated_at
                FROM product_embeddings
                WHERE method = ? AND product_id IN ({placeholders})
                ORDER BY product_id
            """
            cur.execute(query, [method] + product_ids)
        else:
            cur.execute("""
                SELECT product_id, embedding, features_used,
                       vector_dimension, metadata, updated_at
                FROM product_embeddings
                WHERE method = ?
                ORDER BY product_id
            """, (method,))

        rows = cur.fetchall()

        if not rows:
            return {
                "product_ids": [],
                "embeddings": None,
                "metadata": {},
                "features_used": [],
                "vector_dimension": 0
            }

        # Extract data
        product_ids_result = []
        embeddings_list = []

        for row in rows:
            product_ids_result.append(row["product_id"])
            embedding = pickle.loads(row["embedding"])
            embeddings_list.append(embedding)

        # Use metadata from the first row (should be consistent across method)
        first_row = rows[0]
        metadata = (
            json.loads(first_row["metadata"])
            if first_row["metadata"] else {}
        )
        features_used = (
            json.loads(first_row["features_used"])
            if first_row["features_used"] else []
        )

        return {
            "product_ids": product_ids_result,
            "embeddings": np.array(embeddings_list),
            "metadata": metadata,
            "features_used": features_used,
            "vector_dimension": first_row["vector_dimension"]
        }

    except Exception as e:
        raise e
    finally:
        if close:
            connection.close()


def get_embedding_by_product_and_method(
    product_id: int,
    method: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[Dict]:
    """
    Get a single product embedding by product ID and method.

    Args:
        product_id: Product ID
        method: Embedding method
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        Dict with embedding data or None if not found
    """
    result = get_embeddings(
        method,
        [product_id],
        config=config,
        connection=connection
    )

    if result["embeddings"] is not None and len(result["embeddings"]) > 0:
        return {
            "product_id": product_id,
            "embedding": result["embeddings"][0],
            "metadata": result["metadata"],
            "features_used": result["features_used"],
            "vector_dimension": result["vector_dimension"]
        }
    return None


# ---------- Metadata and Statistics API ----------

def list_available_methods(
    *,
    config: Optional[object] = None,
    connection=None,
) -> List[str]:
    """
    List all embedding methods stored in database.

    Args:
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        List of available embedding methods
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT DISTINCT method "
            "FROM product_embeddings "
            "ORDER BY method"
        )
        methods = [row["method"] for row in cur.fetchall()]
        return methods
    finally:
        if close:
            connection.close()


def get_method_statistics(
    method: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Dict:
    """
    Get statistics about stored embeddings for a specific method.

    Args:
        method: Embedding method
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        Dict with statistics including count, dimensions, date range
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute("""
            SELECT COUNT(*) as count,
                   MIN(vector_dimension) as min_dim,
                   MAX(vector_dimension) as max_dim,
                   MIN(updated_at) as oldest,
                   MAX(updated_at) as newest
            FROM product_embeddings
            WHERE method = ?
        """, (method,))

        row = cur.fetchone()
        if row and row["count"] > 0:
            return {
                "method": method,
                "count": row["count"],
                "dimension_range": (row["min_dim"], row["max_dim"]),
                "date_range": (row["oldest"], row["newest"])
            }
        else:
            return {"method": method, "count": 0}
    finally:
        if close:
            connection.close()


def get_all_statistics(
    *,
    config: Optional[object] = None,
    connection=None,
) -> Dict:
    """
    Get comprehensive statistics about all embeddings in the database.

    Args:
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        Dict with statistics for all methods
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        methods = list_available_methods(config=config, connection=connection)
        stats = {}

        for method in methods:
            stats[method] = get_method_statistics(
                method, config=config, connection=connection
            )

        # Overall statistics
        cur = connection.cursor()
        cur.execute("""
            SELECT COUNT(*) as total_embeddings,
                   COUNT(DISTINCT product_id) as unique_products,
                   COUNT(DISTINCT method) as methods_count
            FROM product_embeddings
        """)

        row = cur.fetchone()
        stats["_overall"] = {
            "total_embeddings": row["total_embeddings"],
            "unique_products": row["unique_products"],
            "methods_count": row["methods_count"]
        }

        return stats
    finally:
        if close:
            connection.close()


# ---------- Delete API ----------

def delete_embeddings(
    method: str,
    product_ids: List[int] = None,
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Delete product embeddings from database.

    Args:
        method: Embedding method to delete
        product_ids: Optional list of specific product IDs to delete
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        Number of deleted rows
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()

        if product_ids:
            placeholders = ','.join(['?'] * len(product_ids))
            query = f"""
                DELETE FROM product_embeddings
                WHERE method = ? AND product_id IN ({placeholders})
            """
            cur.execute(query, [method] + product_ids)
        else:
            cur.execute("""
                DELETE FROM product_embeddings
                WHERE method = ?
            """, (method,))

        deleted_count = cur.rowcount
        if close:
            connection.commit()
        print(f"Deleted {deleted_count} embeddings for method '{method}'")
        return deleted_count

    except Exception as e:
        if close:
            connection.rollback()
        raise e
    finally:
        if close:
            connection.close()


def delete_all_embeddings(
    *,
    config: Optional[object] = None,
    connection=None,
) -> int:
    """
    Delete all embeddings from the database (use with caution).

    Args:
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        Number of deleted rows
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute("DELETE FROM product_embeddings")

        deleted_count = cur.rowcount
        if close:
            connection.commit()
        print(f"Deleted all {deleted_count} embeddings from database")
        return deleted_count

    except Exception as e:
        if close:
            connection.rollback()
        raise e
    finally:
        if close:
            connection.close()


# ---------- Utility Functions ----------

def embedding_exists(
    product_id: int,
    method: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> bool:
    """
    Check if an embedding exists for a specific product and method.

    Args:
        product_id: Product ID
        method: Embedding method
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        True if embedding exists, False otherwise
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute("""
            SELECT 1 FROM product_embeddings
            WHERE product_id = ? AND method = ?
            LIMIT 1
        """, (product_id, method))

        return cur.fetchone() is not None
    finally:
        if close:
            connection.close()


def get_products_without_embeddings(
    method: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> List[int]:
    """
    Get list of product IDs that don't have embeddings for a specific method.

    Args:
        method: Embedding method to check
        config: Optional config override
        connection: Optional existing DB connection to reuse

    Returns:
        List of product IDs missing embeddings
    """
    close = False
    if connection is None:
        connection, close = get_connection(config or CONFIG), True

    try:
        cur = connection.cursor()
        cur.execute("""
            SELECT p.id
            FROM products p
            LEFT JOIN product_embeddings pe ON p.id = pe.product_id AND
                pe.method = ?
            WHERE pe.product_id IS NULL
            ORDER BY p.id
        """, (method,))

        return [row["id"] for row in cur.fetchall()]
    finally:
        if close:
            connection.close()
