# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""Complete SOM analysis pipeline: clustering, metrics, and reporting."""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from config import CONFIG
from capitolwatch.analysis.clustering import perform_som_clustering
from capitolwatch.analysis.portfolio_metrics_generator import (
    generate_portfolio_metrics,
    analyze_party_patterns
)


def save_analysis_results_to_csv(
    analysis_results: Dict,
    output_directory: Path
) -> None:
    """
    Save analysis results to CSV files for external use.

    This function exports clustering results, portfolio metrics, and summary
    statistics to separate CSV files in the specified directory.

    Args:
        analysis_results: Dictionary containing all analysis outputs
        output_directory: Path where CSV files will be saved
    """
    print(f"Saving analysis results to: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Export politician clustering results
    if 'clustering_results' in analysis_results:
        clustering_data = analysis_results['clustering_results']

        if 'clustered_politicians' in clustering_data:
            politician_clusters_file = (
                output_directory / "politician_clusters.csv"
            )
            clustering_data['clustered_politicians'].to_csv(
                politician_clusters_file, index=False
            )
            print(f"   Saved politician clusters: {politician_clusters_file}")

    # Export portfolio metrics and risk analysis
    if 'portfolio_metrics' in analysis_results:
        metrics_data = analysis_results['portfolio_metrics']

        if 'metrics_dataframe' in metrics_data:
            portfolio_metrics_file = output_directory / "portfolio_metrics.csv"
            metrics_data['metrics_dataframe'].to_csv(
                portfolio_metrics_file, index=False
            )
            print(f"   Saved portfolio metrics: {portfolio_metrics_file}")

    # Export analysis summary statistics
    if 'analysis_summary' in analysis_results:
        summary_dataframe = pd.DataFrame(
            [analysis_results['analysis_summary']]
        )
        summary_file = output_directory / "analysis_summary.csv"
        summary_dataframe.to_csv(summary_file, index=False)
        print(f"   Saved analysis summary: {summary_file}")

    # Export party-level analysis if available
    if 'party_analysis' in analysis_results:
        party_dataframe = pd.DataFrame(analysis_results['party_analysis'])
        party_file = output_directory / "party_risk_analysis.csv"
        party_dataframe.to_csv(party_file, index=False)
        print(f"   Saved party analysis: {party_file}")

    print("CSV export completed successfully")


def run_comprehensive_portfolio_analysis(
    config: Optional[object] = None,
    embedding_method: str = "custom_financial_weighted",
    som_grid_size: int = 6,
    save_results_to_csv: bool = True,
    output_directory: Optional[str] = None
) -> Dict:
    """
    Execute complete portfolio analysis pipeline including clustering and
    metrics.

    This function runs the full analysis workflow: verifies embeddings exist,
    performs SOM clustering on politician portfolios, calculates portfolio
    metrics, analyzes party patterns, and optionally exports results to CSV.

    Args:
        config: Database configuration object
        embedding_method: Method used for portfolio embeddings
        som_grid_size: Size of the SOM grid for clustering
        save_results_to_csv: Whether to save results to CSV files
        output_directory: Directory for saving CSV outputs

    Returns:
        Dictionary containing all analysis results and summaries
    """
    print("=" * 70)
    print("CAPITOLWATCH - COMPREHENSIVE PORTFOLIO ANALYSIS")
    print("=" * 70)

    config = config or CONFIG

    try:
        # Step 1: Perform SOM clustering analysis
        print("Step 1: Running SOM clustering analysis...")

        clustering_results = perform_som_clustering(
            embedding_method=embedding_method,
            som_grid_size=som_grid_size,
            config=config
        )

        if not clustering_results:
            print("SOM clustering analysis failed")
            return {}

        # Extract clustering summary
        validation_metrics = clustering_results.get('validation_metrics', {})
        clustered_politicians = clustering_results.get(
            'clustered_politicians', pd.DataFrame()
        )

        print(
            f"Clustered {validation_metrics.get('total_politicians', 0)} "
            f"politicians. Formed {validation_metrics.get('cluster_count', 0)}"
            "clusters"
        )

        # Step 2: Calculate portfolio metrics
        print("Step 2: Calculating portfolio metrics...")

        portfolio_metrics_dataframe = generate_portfolio_metrics(config=config)

        if portfolio_metrics_dataframe.empty:
            print("Portfolio metrics calculation failed")
            return {}

        # Calculate portfolio metrics summary
        metrics_summary = {
            'total_politicians_with_metrics': len(portfolio_metrics_dataframe),
            'average_herfindahl_index': round(
                portfolio_metrics_dataframe['herfindahl_index'].mean(), 4
            ),
            'average_diversification_score': round(
                portfolio_metrics_dataframe['diversification_score'].mean(), 1
            ),
        }

        print("Calculated metrics for portfolio")
        print(
            f"{metrics_summary['total_politicians_with_metrics']} politicians"
        )
        print(f"Average HHI: {metrics_summary['average_herfindahl_index']}")

        # Step 3: Analyze party-level patterns
        print("Step 3: Analyzing party risk patterns...")

        party_risk_analysis = analyze_party_patterns(
            portfolio_metrics_dataframe
        )

        print(f"Analyzed patterns for {len(party_risk_analysis)} parties")

        # Step 4: Compile comprehensive results
        comprehensive_results = {
            'clustering_results': clustering_results,
            'portfolio_metrics': {
                'metrics_dataframe': portfolio_metrics_dataframe,
                'summary_statistics': metrics_summary
            },
            'party_analysis': party_risk_analysis,
            'analysis_parameters': {
                'embedding_method': embedding_method,
                'som_grid_size': som_grid_size,
                'total_politicians_analyzed': len(clustered_politicians)
            },
            'analysis_summary': {
                'total_politicians': len(clustered_politicians),
                'clusters_formed': validation_metrics.get('cluster_count', 0),
                'average_cluster_size': (
                    validation_metrics.get('average_cluster_size', 0)
                ),
                'embedding_method_used': embedding_method,
                'som_grid_dimensions': f"{som_grid_size}x{som_grid_size}",
                'portfolio_metrics_calculated': (
                    len(portfolio_metrics_dataframe)
                ),
                'parties_analyzed': len(party_risk_analysis)
            }
        }

        # Step 6: Export results to CSV files
        if save_results_to_csv:
            print("Step 4: Exporting results to CSV files...")

            if output_directory is None:
                data_directory = getattr(config, 'data_dir', 'data')
                output_directory = Path(data_directory) / "analysis_results"
            else:
                output_directory = Path(output_directory)

            save_analysis_results_to_csv(
                comprehensive_results, output_directory
            )

        # Display final summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 50)

        summary = comprehensive_results['analysis_summary']
        print(f"Politicians analyzed: {summary['total_politicians']}")
        print(f"Clusters formed: {summary['clusters_formed']}")
        print(f"Average cluster size: {summary['average_cluster_size']}")
        print(f"Metrics calculated: {summary['portfolio_metrics_calculated']}")
        print(f"Political parties analyzed: {summary['parties_analyzed']}")

        if save_results_to_csv:
            print(f"Results saved to: {output_directory}")

        return comprehensive_results

    except Exception as e:
        print(f"Analysis pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    print("Running comprehensive portfolio analysis...")

    # Execute complete analysis pipeline
    analysis_results = run_comprehensive_portfolio_analysis(
        embedding_method="custom_financial_weighted",
        som_grid_size=6,
        save_results_to_csv=True
    )

    if analysis_results:
        print("Comprehensive portfolio analysis completed successfully")
    else:
        print("Analysis failed")


if __name__ == "__main__":
    try:
        # Run complete analysis with default parameters
        analysis_results = run_comprehensive_portfolio_analysis(
            embedding_method="custom_financial_weighted",
            som_grid_size=6,
            save_results_to_csv=True
        )

        if analysis_results:
            # Display test results
            summary = analysis_results['analysis_summary']

            print("\nTEST RESULTS:")
            print(f"Politicians analyzed: {summary['total_politicians']}")
            print(f"Clusters formed: {summary['clusters_formed']}")
            print(f"Metrics: {summary['portfolio_metrics_calculated']}")
            print(f"Parties analyzed: {summary['parties_analyzed']}")
            print(f"Embedding method: {summary['embedding_method_used']}")

            print("\nTest completed successfully")
            print("Results are available in: data/analysis_results/")
        else:
            print("Comprehensive portfolio analysis test failed")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
