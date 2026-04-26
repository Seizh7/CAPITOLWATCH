#!/usr/bin/env python
# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Main CLI for analysis pipeline.

This CLI coordinates the feature engineering, evaluation, cluster analysis,
and visualization steps, providing both individual commands and a full
workflow.
"""

from pathlib import Path

import typer


app = typer.Typer(
    help="Cluster investment profiles of US politicians",
    no_args_is_help=True,
    add_completion=False,
)

OUTPUT_DIR = Path("data/outputs")
FIGURES_DIR = Path("data/figures")


@app.command()
def features() -> None:
    """
    Build the feature store from the database.

    Computes freq_baseline, freq_weighted, and sector_baseline matrices,
    then saves them to data/outputs/.

    Example:
        python -m capitolwatch.analysis features
    """
    typer.secho("\nBuilding Feature Store", fg=typer.colors.CYAN, bold=True)

    from capitolwatch.analysis.feature_store import build_feature_store

    try:
        build_feature_store()
        typer.secho("Feature store built", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(
            f"Feature store build failed: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


@app.command()
def evaluate() -> None:
    """
    Evaluate all 6 clustering experiments (internal + external metrics).

    Fits K-Means, DBSCAN, and SOM and computes internal and external metrics.
    Results are exported to data/outputs/ (CSV) and data/figures/ (PNG).

    Example:
        python -m capitolwatch.analysis evaluate
    """
    from capitolwatch.analysis.run_evaluation import (
        run_all_evaluations,
        print_comparison_table,
        run_external_evaluations,
    )

    eval_csv = str(OUTPUT_DIR / "evaluation_results.csv")
    eval_ext_csv = str(OUTPUT_DIR / "evaluation_results_external.csv")

    typer.secho("\nInternal Evaluation", fg=typer.colors.CYAN, bold=True)
    try:
        df = run_all_evaluations(output_path=eval_csv)
        print_comparison_table(df)
        typer.secho(
            f"Results saved to: {eval_csv}",
            fg=typer.colors.GREEN,
        )
    except Exception as exc:
        typer.secho(f"Evaluation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("\nExternal Evaluation", fg=typer.colors.CYAN, bold=True)
    try:
        run_external_evaluations(
            output_path=eval_ext_csv,
            confusion_matrix_dir=str(FIGURES_DIR),
        )
        typer.secho(
            f"Results saved to: {eval_ext_csv}",
            fg=typer.colors.GREEN,
        )
    except Exception as exc:
        typer.secho(
            f"External evaluation failed: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


@app.command()
def analyze() -> None:
    """
    Generate per-cluster narrative reports (Markdown).

    For each of the 6 experiments, describes the investment profile of
    every cluster. Reports are saved to data/figures/cluster_profiles/.

    Example:
        python -m capitolwatch.analysis analyze
    """
    typer.secho("\nCluster Analysis", fg=typer.colors.CYAN, bold=True)

    from capitolwatch.analysis.run_cluster_analysis import run_all_analyses

    profiles_dir = str(FIGURES_DIR / "cluster_profiles")
    try:
        profiles = run_all_analyses(output_dir=profiles_dir)
        typer.secho(
            f"{len(profiles)} reports saved to: {profiles_dir}",
            fg=typer.colors.GREEN
        )
    except Exception as exc:
        typer.secho(
            f"Cluster analysis failed: {exc}",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(code=1)


@app.command()
def visualize() -> None:
    """
    Generate all static PNG plots for the 6 clustering experiments.

    Produces heatmaps, cluster-size barplots, metrics barplots, and PCA
    scatter plots.

    Example:
        python -m capitolwatch.analysis visualize
    """
    from capitolwatch.analysis.run_visualization import (
        run_simple_plots,
        run_metrics_barplots,
        run_pca_plots,
    )

    typer.secho("\nGenerating visualizations", fg=typer.colors.CYAN, bold=True)
    try:
        run_simple_plots(output_dir=FIGURES_DIR)
        run_metrics_barplots(output_dir=FIGURES_DIR)
        run_pca_plots(output_dir=FIGURES_DIR)
        typer.secho(
            f"All plots saved to: {FIGURES_DIR.resolve()}",
            fg=typer.colors.GREEN
        )
    except Exception as exc:
        typer.secho(
            f"Visualization failed: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


@app.command()
def full_pipeline() -> None:
    """
    Run the complete analysis pipeline:
        features → evaluate → analyze → visualize

    Example:
        python -m capitolwatch.analysis full-pipeline
    """
    typer.secho(
        "\nStarting CAPITOLWATCH Analysis Pipeline",
        fg=typer.colors.CYAN,
        bold=True
    )

    # 1 — feature store
    typer.secho(
        "\nStep 1/4: Building feature store...",
        fg=typer.colors.BLUE,
        bold=True
    )
    from capitolwatch.analysis.feature_store import build_feature_store
    try:
        build_feature_store()
        typer.secho("Feature store built", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(
            f"Feature store failed: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # 2 — evaluation
    typer.secho(
        "\nStep 2/4: Running evaluation...", fg=typer.colors.BLUE, bold=True
    )
    from capitolwatch.analysis.run_evaluation import (
        run_all_evaluations,
        run_external_evaluations,
    )
    eval_csv = str(OUTPUT_DIR / "evaluation_results.csv")
    eval_ext_csv = str(OUTPUT_DIR / "evaluation_results_external.csv")
    try:
        run_all_evaluations(output_path=eval_csv)
        run_external_evaluations(
            output_path=eval_ext_csv,
            confusion_matrix_dir=str(FIGURES_DIR),
        )
        typer.secho("Evaluation done", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(
            f"Evaluation failed: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # 3 — cluster analysis
    typer.secho(
        "\nStep 3/4: Running cluster analysis...",
        fg=typer.colors.BLUE,
        bold=True
    )
    from capitolwatch.analysis.run_cluster_analysis import run_all_analyses
    try:
        run_all_analyses(output_dir=str(FIGURES_DIR / "cluster_profiles"))
        typer.secho("Cluster analysis done", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(
            f"Cluster analysis failed: {exc}",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(code=1)

    # 4 — visualizations
    typer.secho(
        "\nStep 4/4: Generating visualizations...",
        fg=typer.colors.BLUE,
        bold=True
    )
    from capitolwatch.analysis.run_visualization import (
        run_simple_plots,
        run_metrics_barplots,
        run_pca_plots,
    )
    try:
        run_simple_plots(output_dir=FIGURES_DIR)
        run_metrics_barplots(output_dir=FIGURES_DIR)
        run_pca_plots(output_dir=FIGURES_DIR)
        typer.secho("Visualizations done", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(
            f"Visualization failed: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    typer.secho(
        "\nAnalysis Pipeline Completed Successfully",
        fg=typer.colors.GREEN,
        bold=True
    )
    typer.secho(f"Outputs: {OUTPUT_DIR.resolve()}", fg=typer.colors.CYAN)
