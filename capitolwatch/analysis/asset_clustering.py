# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Self-Organizing Map (SOM) clustering for politicians' investment portfolios.

This module implements SOM-based clustering to identify groups of politicians
with similar investment strategies and portfolio compositions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from config import CONFIG
from capitolwatch.analysis.portfolio_metrics import (
    get_all_portfolio_metrics,
    get_politician_portfolio_data,
    generate_portfolio_feature_vector
)


class PortfolioSOMClustering:
    """
    Self-Organizing Map clustering for politician portfolios.
    """

    def __init__(self, som_x: int = 10, som_y: int = 10, 
                 sigma: float = 1.0, learning_rate: float = 0.5):
        """
        Initialize SOM clustering.

        Args:
            som_x: Width of the SOM grid
            som_y: Height of the SOM grid  
            sigma: Neighborhood function spread
            learning_rate: Learning rate for training
        """
        self.som_x = som_x
        self.som_y = som_y
        self.sigma = sigma
        self.learning_rate = learning_rate
 
        self.som = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.politician_positions = {}
        self.cluster_assignments = {}
 
    def prepare_data(self, config=None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare portfolio data for SOM clustering.

        Returns:
            Tuple of (feature_matrix, politician_dataframe)
        """
        if config is None:
            config = CONFIG

        print("Preparing portfolio data for clustering...")

        # Get portfolio metrics for all politicians
        metrics_df = get_all_portfolio_metrics(config)

        # Filter politicians with sufficient data
        valid_politicians = metrics_df[
            (metrics_df['total_assets'] >= 3) &  # At least 3 assets
            (metrics_df['risk_profile'] != 'Unknown')  # Valid risk profile
        ].copy()

        print(
            f"Found {len(valid_politicians)} politicians"
            " with sufficient data"
        )

        if len(valid_politicians) == 0:
            raise ValueError("No politicians with sufficient portfolio data")

        # Get all unique sectors for feature vector consistency
        all_sectors = set()
        for _, row in valid_politicians.iterrows():
            all_sectors.update(row['sector_breakdown'].keys())
        all_sectors = sorted(list(all_sectors))

        print(f"Portfolio features based on {len(all_sectors)} sectors")

        # Generate feature vectors for each politician
        feature_vectors = []
        politician_data = []

        for _, row in valid_politicians.iterrows():
            politician_id = row['politician_id']

            # Get detailed portfolio data
            portfolio_data = get_politician_portfolio_data(
                politician_id,
                config
            )

            # Generate feature vector
            feature_vector = generate_portfolio_feature_vector(
                portfolio_data, all_sectors
            )

            feature_vectors.append(feature_vector)
            politician_data.append({
                'politician_id': politician_id,
                'politician_name': row['politician_name'],
                'party': row['party'],
                'risk_profile': row['risk_profile'],
                'total_assets': row['total_assets'],
                'dominant_sector': row['dominant_sector'],
                'herfindahl_index': row['sector_diversification']['herfindahl_index']
            })

        # Create feature matrix
        feature_matrix = np.array(feature_vectors)
        politician_df = pd.DataFrame(politician_data)

        # Store feature names for interpretation
        self.feature_names = (
            all_sectors + 
            ['herfindahl_index', 'shannon_entropy', 'num_sectors_norm', 
             'concentration_ratio', 'log_assets_norm']
        )

        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Features: {len(self.feature_names)} dimensions")

        return feature_matrix, politician_df

    def train_som(self, feature_matrix: np.ndarray, iterations: int = 1000):
        """
        Train the Self-Organizing Map.

        Args:
            feature_matrix: Normalized feature matrix
            iterations: Number of training iterations
        """
        print(f"Training SOM ({self.som_x}x{self.som_y}) for {iterations} iterations...")

        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_matrix)

        # Initialize SOM
        input_len = normalized_features.shape[1]
        self.som = MiniSom(
            self.som_x, self.som_y, input_len,
            sigma=self.sigma, learning_rate=self.learning_rate,
            random_seed=42
        )

        # Train SOM
        self.som.train(normalized_features, iterations)

        print("SOM training completed")

        return normalized_features

    def assign_clusters(self, feature_matrix: np.ndarray, 
                       politician_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign politicians to SOM clusters.

        Args:
            feature_matrix: Feature matrix
            politician_df: Politician dataframe

        Returns:
            DataFrame with cluster assignments
        """
        print("Assigning politicians to clusters...")

        normalized_features = self.scaler.transform(feature_matrix)

        # Get winning neurons for each politician
        cluster_assignments = []
        for i, politician_vector in enumerate(normalized_features):
            winning_neuron = self.som.winner(politician_vector)
            cluster_id = winning_neuron[0] * self.som_y + winning_neuron[1]

            self.politician_positions[politician_df.iloc[i]['politician_id']] = winning_neuron

            cluster_assignments.append({
                'politician_id': politician_df.iloc[i]['politician_id'],
                'cluster_x': winning_neuron[0],
                'cluster_y': winning_neuron[1], 
                'cluster_id': cluster_id
            })

        cluster_df = pd.DataFrame(cluster_assignments)
        result_df = politician_df.merge(cluster_df, on='politician_id')

        print(f"Assigned {len(result_df)} politicians to {len(result_df['cluster_id'].unique())} clusters")

        return result_df

    def analyze_clusters(self, clustered_df: pd.DataFrame) -> Dict:
        """
        Analyze the characteristics of each cluster.

        Args:
            clustered_df: DataFrame with cluster assignments

        Returns:
            Dict with cluster analysis
        """
        print("Analyzing cluster characteristics...")

        cluster_analysis = {}

        for cluster_id in clustered_df['cluster_id'].unique():
            cluster_politicians = clustered_df[clustered_df['cluster_id'] == cluster_id]

            # Basic statistics
            party_dist = cluster_politicians['party'].value_counts().to_dict()
            risk_dist = cluster_politicians['risk_profile'].value_counts().to_dict()
            dominant_sectors = cluster_politicians['dominant_sector'].value_counts().head(3).to_dict()

            cluster_analysis[cluster_id] = {
                'size': len(cluster_politicians),
                'politicians': cluster_politicians[['politician_name', 'party', 'risk_profile']].to_dict('records'),
                'party_distribution': party_dist,
                'risk_profile_distribution': risk_dist,
                'dominant_sectors': dominant_sectors,
                'avg_portfolio_size': cluster_politicians['total_assets'].mean(),
                'avg_diversification': cluster_politicians['herfindahl_index'].mean(),
                'cluster_x': cluster_politicians['cluster_x'].iloc[0],
                'cluster_y': cluster_politicians['cluster_y'].iloc[0]
            }

        return cluster_analysis

    def visualize_som_map(self, clustered_df: pd.DataFrame, 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of the SOM map with politician clusters.

        Args:
            clustered_df: DataFrame with cluster assignments
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Distance matrix (U-Matrix)
        distance_map = self.som.distance_map()
        im1 = ax1.imshow(distance_map, cmap='bone_r', alpha=0.8)
        ax1.set_title('SOM Distance Matrix (U-Matrix)', fontsize=14)
        ax1.set_xlabel('SOM X')
        ax1.set_ylabel('SOM Y')
        plt.colorbar(im1, ax=ax1)

        # Plot 2: Politician distribution by party
        party_colors = {'Republican': 'red', 'Democratic': 'blue', 'Independent': 'green'}

        for _, politician in clustered_df.iterrows():
            x, y = politician['cluster_x'], politician['cluster_y']
            party = politician['party']
            color = party_colors.get(party, 'gray')

            ax2.scatter(x, y, c=color, s=100, alpha=0.7, 
                       edgecolor='black', linewidth=0.5)

        ax2.set_xlim(-0.5, self.som_x - 0.5)
        ax2.set_ylim(-0.5, self.som_y - 0.5)
        ax2.set_title('Politicians by Party on SOM Grid', fontsize=14)
        ax2.set_xlabel('SOM X')
        ax2.set_ylabel('SOM Y')
        ax2.grid(True, alpha=0.3)

        # Add legend
        for party, color in party_colors.items():
            if party in clustered_df['party'].values:
                ax2.scatter([], [], c=color, s=100, label=party)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SOM visualization saved to {save_path}")

        return fig

    def create_cluster_heatmap(
        self,
        clustered_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap of cluster characteristics.

        Args:
            clustered_df: DataFrame with cluster assignments
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure
        """
        # Prepare data for heatmap
        cluster_stats = []

        for cluster_id in sorted(clustered_df['cluster_id'].unique()):
            cluster_data = clustered_df[
                clustered_df['cluster_id'] == cluster_id
            ]

            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'republican_pct': (
                    cluster_data['party'] == 'Republican'
                ).mean() * 100,
                'democratic_pct': (
                    cluster_data['party'] == 'Democratic'
                ).mean() * 100,
                'avg_portfolio_size': cluster_data['total_assets'].mean(),
                'avg_diversification': cluster_data['herfindahl_index'].mean(),
                'concentrated_pct': (
                    cluster_data['risk_profile'] == 'Concentrated'
                ).mean() * 100,
                'diversified_pct': (
                    cluster_data['risk_profile'] == 'Diversified').mean() * 100
            }
            cluster_stats.append(stats)

        heatmap_df = pd.DataFrame(cluster_stats).set_index('cluster_id')

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(heatmap_df.T, annot=True, fmt='.1f', cmap='viridis',
                    ax=ax, cbar_kws={'label': 'Value'})

        ax.set_title('Cluster Characteristics Heatmap', fontsize=16)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Characteristics')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster heatmap saved to {save_path}")

        return fig

    def save_model(self, filepath: str):
        """Save the trained SOM model and scaler."""
        model_data = {
            'som': self.som,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'som_params': {
                'som_x': self.som_x,
                'som_y': self.som_y,
                'sigma': self.sigma,
                'learning_rate': self.learning_rate
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"SOM model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained SOM model and scaler."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.som = model_data['som']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        params = model_data['som_params']
        self.som_x = params['som_x']
        self.som_y = params['som_y']
        self.sigma = params['sigma']
        self.learning_rate = params['learning_rate']

        print(f"SOM model loaded from {filepath}")


def run_full_som_analysis(
    config: Optional[object] = None,
    save_outputs: bool = True
) -> Dict:
    """
    Run complete SOM clustering analysis on politician portfolios.

    Args:
        config: Configuration object
        save_outputs: Whether to save plots, results, and the model

    Returns:
        Dict with analysis results
    """
    print("=" * 60)
    print("CAPITOLWATCH - SOM Clustering Analysis")
    print("=" * 60)

    if config is None:
        config = CONFIG

    som_clustering = PortfolioSOMClustering(som_x=8, som_y=8)

    try:
        # Step 1: Prepare input data
        feature_matrix, politician_df = som_clustering.prepare_data(config)

        # Step 2: Train SOM
        som_clustering.train_som(
            feature_matrix,
            iterations=1000
        )

        # Step 3: Assign clusters
        clustered_df = som_clustering.assign_clusters(
            feature_matrix,
            politician_df
        )

        # Step 4: Analyze clusters
        cluster_analysis = som_clustering.analyze_clusters(clustered_df)

        # Step 5: Print summary
        print("\n" + "=" * 50)
        print("CLUSTER ANALYSIS SUMMARY")
        print("=" * 50)

        for cluster_id, analysis in cluster_analysis.items():
            print(f"\nCluster {cluster_id} (Position: "
                  f"{analysis['cluster_x']},{analysis['cluster_y']}):")
            print(f" Size: {analysis['size']} politicians")
            print(f" Party distribution: {analysis['party_distribution']}")
            print(f" Risk profiles: {analysis['risk_profile_distribution']}")
            print(f" Avg portfolio size: {analysis['avg_portfolio_size']:.1f}")
            print(f" Avg diversification (HHI): "
                  f"{analysis['avg_diversification']:.3f}")
            print(f" Top sectors: "
                  f"{list(analysis['dominant_sectors'].keys())[:3]}")

        # Step 6: Save outputs
        if save_outputs:
            output_dir = Path(config.data_dir) / "som_analysis"
            output_dir.mkdir(exist_ok=True)

            som_clustering.visualize_som_map(
                clustered_df,
                save_path=output_dir / "som_map.png"
            )

            som_clustering.create_cluster_heatmap(
                clustered_df,
                save_path=output_dir / "cluster_heatmap.png"
            )

            som_clustering.save_model(output_dir / "som_model.pkl")

            clustered_df.to_csv(
                output_dir / "politician_clusters.csv",
                index=False
            )

            print(f"\nOutputs saved to: {output_dir}")

        # Step 7: Return results
        return {
            "clustered_politicians": clustered_df,
            "cluster_analysis": cluster_analysis,
            "som_model": som_clustering,
            "feature_matrix": feature_matrix,
            "num_clusters": len(cluster_analysis),
        }

    except Exception as e:
        print(f"Error in SOM analysis: {e}")
        return {}


if __name__ == "__main__":
    results = run_full_som_analysis()

    if results:
        print("\nSOM Analysis completed successfully")
        print(f"Politicians clustered:{len(results['clustered_politicians'])}")
        print(f"Clusters identified: {results['num_clusters']}")
        print("Results saved to data/som_analysis/")
    else:
        print("\nSOM Analysis failed.")
