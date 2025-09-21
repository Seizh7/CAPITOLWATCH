# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""Clustering analysis: SOM clustering for politician portfolio embeddings."""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from capitolwatch.services.portfolio_embeddings import (
    query_portfolio_embeddings
)
from capitolwatch.services.politicians import get_politician_basic_info


def load_portfolio_embeddings_for_clustering(
    embedding_method: str = "custom_financial_weighted",
    config: Optional[object] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load and prepare portfolio embeddings for clustering analysis.

    This function retrieves portfolio embeddings from the database and
    filters out politicians with insufficient asset data to ensure
    meaningful clustering results.

    Args:
        embedding_method: Method used for portfolio embeddings
        config: Database configuration

    Returns:
        Tuple of (feature_matrix, politician_dataframe) ready for clustering
    """
    config = config or CONFIG

    # Load portfolio embeddings from database
    embeddings_data = query_portfolio_embeddings(
        method=embedding_method, config=config
    )

    if not embeddings_data:
        raise ValueError(
            f"No portfolio embeddings found for '{embedding_method}'"
        )

    # Process embeddings and build dataset
    feature_vectors = []
    politician_records = []

    for politician_id, (embedding_vector, metadata) in embeddings_data.items():
        # Filter out politicians with too few assets
        asset_count = metadata.get("asset_count", 0)
        if asset_count < 3:
            continue

        # Clean embedding vector
        clean_vector = np.nan_to_num(embedding_vector, nan=0.0)
        if np.all(clean_vector == 0):
            continue

        # Get politician information
        politician_info = get_politician_basic_info(
            politician_id, config=config
        )

        # Build politician record
        politician_records.append({
            "politician_id": politician_id,
            "name": politician_info.get(
                "politician_name", f"Politician_{politician_id}"
            ),
            "party": politician_info.get("party", "Unknown"),
            "asset_count": asset_count,
        })

        feature_vectors.append(clean_vector)

    # Convert to arrays and dataframe
    feature_matrix = np.array(feature_vectors)
    politician_dataframe = pd.DataFrame(politician_records)

    print(f"Prepared data: {len(politician_dataframe)} politicians, "
          f"embedding dimension: {feature_matrix.shape[1]}")

    return feature_matrix, politician_dataframe


def train_som_model(
    feature_matrix: np.ndarray,
    som_grid_size: int = 6
) -> Tuple[MiniSom, StandardScaler]:
    """
    Train a Self-Organizing Map (SOM) on portfolio embeddings.

    This function normalizes the input features and trains a SOM to create
    a 2D representation of politician portfolios that preserves similarity
    relationships in the high-dimensional embedding space.

    Args:
        feature_matrix: Array of portfolio embedding vectors
        som_grid_size: Size of the SOM grid (som_grid_size x som_grid_size)

    Returns:
        Tuple of (trained_som_model, feature_scaler)
    """
    # Prepare and normalize features
    clean_features = np.nan_to_num(feature_matrix, nan=0.0)

    # Apply standard scaling
    scaler = StandardScaler()
    try:
        normalized_features = scaler.fit_transform(clean_features)
        normalized_features = np.nan_to_num(normalized_features, nan=0.0)
    except Exception as e:
        print(f"Normalization failed ({e}), using raw features")
        normalized_features = clean_features

    # Initialize and train SOM
    som_model = MiniSom(
        som_grid_size, som_grid_size, normalized_features.shape[1],
        sigma=1.0,          # Initial radius of influence
        learning_rate=0.5,  # Learning rate for weight updates
        random_seed=42      # For reproducible results
    )

    # Train the SOM with normalized features
    som_model.train(normalized_features, num_iteration=500)

    print(f"SOM training completed: {som_grid_size}x{som_grid_size} grid")

    return som_model, scaler


def assign_clusters(
    feature_matrix: np.ndarray,
    politician_dataframe: pd.DataFrame,
    som_model: MiniSom,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Assign each politician to a SOM cluster based on portfolio similarity.

    This function uses the trained SOM to find the best matching unit
    for each politician's portfolio embedding and assigns cluster labels.

    Args:
        feature_matrix: Array of portfolio embedding vectors
        politician_dataframe: DataFrame with politician information
        som_model: Trained SOM model
        scaler: Fitted feature scaler

    Returns:
        DataFrame with politician data and cluster assignments
    """
    # Normalize features using fitted scaler
    normalized_features = scaler.transform(feature_matrix)

    # Find best matching units for each politician
    cluster_assignments = []
    for index, embedding_vector in enumerate(normalized_features):
        # Get coordinates of winning neuron
        winner_x, winner_y = som_model.winner(embedding_vector)

        # Convert 2D coordinates to single cluster ID
        cluster_id = winner_x * som_model.get_weights().shape[0] + winner_y

        cluster_assignments.append({
            "politician_id": politician_dataframe.iloc[index]["politician_id"],
            "cluster_id": cluster_id,
            "cluster_x": winner_x,
            "cluster_y": winner_y,
        })

    # Merge cluster assignments with politician data
    assignment_dataframe = pd.DataFrame(cluster_assignments)
    clustered_politicians = politician_dataframe.merge(
        assignment_dataframe, on="politician_id"
    )

    unique_clusters = clustered_politicians['cluster_id'].nunique()
    print(f"Cluster assignment completed: {unique_clusters} clusters formed")

    return clustered_politicians


def analyze_cluster_composition(clustered_dataframe: pd.DataFrame) -> Dict:
    """
    Analyze the composition and characteristics of each cluster.

    This function computes descriptive statistics for each cluster including
    size, average asset counts, party distribution, and member lists.

    Args:
        clustered_dataframe: DataFrame with politician data and cluster
        assignments

    Returns:
        Dictionary with detailed analysis for each cluster
    """
    cluster_analysis = {}

    for cluster_id in clustered_dataframe["cluster_id"].unique():
        # Filter politicians in current cluster
        cluster_members = clustered_dataframe[
            clustered_dataframe["cluster_id"] == cluster_id
        ]

        # Calculate cluster statistics
        cluster_analysis[cluster_id] = {
            "member_count": len(cluster_members),
            "average_asset_count": round(
                cluster_members["asset_count"].mean(), 1
            ),
            "party_distribution": (
                cluster_members["party"].value_counts().to_dict()
            ),
            "politician_names": cluster_members["name"].tolist(),
        }

    return cluster_analysis


def validate_clustering_results(clustered_dataframe: pd.DataFrame) -> Dict:
    """
    Compute validation metrics for clustering quality assessment.

    This function calculates various metrics to assess the quality and
    characteristics of the clustering results including cluster balance
    and party distribution patterns.

    Args:
        clustered_dataframe: DataFrame with politician data and cluster
        assignments

    Returns:
        Dictionary with validation metrics and statistics
    """
    total_politicians = len(clustered_dataframe)
    cluster_count = clustered_dataframe["cluster_id"].nunique()

    # Calculate cluster size statistics
    cluster_sizes = clustered_dataframe["cluster_id"].value_counts()
    average_cluster_size = (
        round(total_politicians / cluster_count, 1) if cluster_count > 0 else 0
    )

    # Analyze party distribution by cluster
    party_by_cluster = {}
    for cluster_id, cluster_group in clustered_dataframe.groupby("cluster_id"):
        party_by_cluster[cluster_id] = (
            cluster_group["party"].value_counts().to_dict()
        )

    validation_metrics = {
        "total_politicians": total_politicians,
        "cluster_count": cluster_count,
        "average_cluster_size": average_cluster_size,
        "largest_cluster_id": (cluster_sizes.idxmax()
                               if not cluster_sizes.empty else None),
        "smallest_cluster_id": (cluster_sizes.idxmin()
                                if not cluster_sizes.empty else None),
        "largest_cluster_size": (cluster_sizes.max()
                                 if not cluster_sizes.empty else 0),
        "smallest_cluster_size": (cluster_sizes.min()
                                  if not cluster_sizes.empty else 0),
        "party_distribution_by_cluster": party_by_cluster,
    }

    return validation_metrics


def perform_som_clustering(
    embedding_method: str = "custom_financial_weighted",
    som_grid_size: int = 6,
    config: Optional[object] = None
) -> Dict:
    """
    Perform complete SOM clustering analysis on politician portfolios.

    This function executes the full clustering pipeline including data loading,
    SOM training, cluster assignment, and analysis of results.

    Args:
        embedding_method: Method used for portfolio embeddings
        som_grid_size: Size of the SOM grid (grid_size x grid_size)
        config: Database configuration

    Returns:
        Dictionary with clustering results and analysis
    """
    config = config or CONFIG

    try:
        # Step 1: Load and prepare data
        (
            feature_matrix, politician_dataframe
        ) = load_portfolio_embeddings_for_clustering(
            embedding_method, config
        )

        # Step 2: Train SOM model
        som_model, scaler = train_som_model(feature_matrix, som_grid_size)

        # Step 3: Assign clusters
        clustered_politicians = assign_clusters(
            feature_matrix, politician_dataframe, som_model, scaler
        )

        # Step 4: Analyze results
        cluster_analysis = analyze_cluster_composition(clustered_politicians)
        validation_metrics = validate_clustering_results(clustered_politicians)

        return {
            "clustered_politicians": clustered_politicians,
            "cluster_analysis": cluster_analysis,
            "validation_metrics": validation_metrics,
            "clustering_parameters": {
                "embedding_method": embedding_method,
                "som_grid_size": som_grid_size,
                "feature_dimensions": feature_matrix.shape[1]
            },
        }

    except Exception as e:
        print(f"SOM clustering failed: {e}")
        return {}


if __name__ == "__main__":
    try:
        # Perform clustering with default parameters
        clustering_results = perform_som_clustering(
            embedding_method="custom_financial_weighted",
            som_grid_size=6
        )

        if clustering_results:
            # Display clustering summary
            validation = clustering_results["validation_metrics"]
            parameters = clustering_results["clustering_parameters"]

            print(f"Politicians clustered: {validation['total_politicians']}")
            print(f"Clusters formed: {validation['cluster_count']}")
            print(f"Average cluster size:{validation['average_cluster_size']}")
            print(f"Embedding method: {parameters['embedding_method']}")
            print(f"Feature dimensions: {parameters['feature_dimensions']}")

            # Show largest and smallest clusters
            largest_size = validation['largest_cluster_size']
            smallest_size = validation['smallest_cluster_size']
            print(f"Cluster size range: {smallest_size} to {largest_size}")
        else:
            print("Test failed")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
