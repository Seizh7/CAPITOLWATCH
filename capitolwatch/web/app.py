# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Streamlit dashboard

Five tabs:
    1. Comparison     : internal metrics table + barplots
    2. Best result    : DBSCAN + freq_weighted scatter / heatmap / outliers
    3. SOM            : U-Matrix and political map
    4. External       : ARI / NMI / V-Measure vs party labels
    5. Sector analysis: DBSCAN on sector_baseline (economic sectors)

Usage (from project root):
    streamlit run capitolwatch/web/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from capitolwatch.analysis.feature_store import load_features
from capitolwatch.analysis.preprocessing import (
    normalize_features,
)
from capitolwatch.web.charts import (
    PARTY_COLOR_MAP,
    barplot_metrics_plotly,
    heatmap_centroids_plotly,
    scatter_pca_plotly,
    som_map_plotly,
)

# Allow imports when streamlit changes the working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


_INTERNAL_CSV = _PROJECT_ROOT / "data/visualizations/evaluation_results.csv"
_EXTERNAL_CSV = (
    _PROJECT_ROOT / "data/visualizations/evaluation_results_external.csv"
)


@st.cache_data
def _load_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-computed evaluation CSV

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (internal_df, external_df).
            internal_df has an added "experiment" column for display labels.
    """
    internal = pd.read_csv(_INTERNAL_CSV)
    external = pd.read_csv(_EXTERNAL_CSV)
    internal["experiment"] = (
        internal["algo_name"] + " / " + internal["feature_type"]
    )
    return internal, external


@st.cache_data
def _load_politician_metadata() -> pd.DataFrame:
    """
    Load the politician labels from the feature store.

    Returns:
        pd.DataFrame: Columns id, first_name, last_name, party (79 rows).
    """
    return load_features("politician_labels")


@st.cache_data
def _get_dbscan_results(feature_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run cosine-DBSCAN grid search and return the best (X, labels).

    Re-uses the exact same parameters search as run_evaluation.py so
    results are consistent with the pre-computed CSV.

    Args:
        feature_type (str): "freq_baseline" or "freq_weighted".

    Returns:
        tuple[np.ndarray, np.ndarray]: (feature_matrix, cluster_labels).
    """
    from capitolwatch.analysis.run_evaluation import _get_dbscan_labels

    return _get_dbscan_labels(feature_type)


@st.cache_data
def _get_kmeans_results(feature_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run K-Means with silhouette-optimal K and return (X, labels).

    Args:
        feature_type (str): "freq_baseline" or "freq_weighted".

    Returns:
        tuple[np.ndarray, np.ndarray]: (feature_matrix, cluster_labels).
    """
    from capitolwatch.analysis.run_evaluation import _get_kmeans_labels

    return _get_kmeans_labels(feature_type)


@st.cache_data
def _get_som_full_results(
    feature_type: str,
    m: int = 7,
    n: int = 7,
    n_clusters: int = 3,
    n_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Train SOM and return all outputs needed for the dashboard.

    Args:
        feature_type (str): "freq_baseline" or "freq_weighted".
        m (int): Grid rows (default: 7).
        n (int): Grid columns (default: 7).
        n_clusters (int): Clusters to extract via K-Means on neuron weights.
        n_iterations (int): SOM training iterations.

    Returns:
        tuple: (X, labels, umatrix, bmu_coords) where
            X          -- (79, n_features) MinMax-scaled matrix,
            labels     -- (79,) cluster label per politician,
            umatrix    -- (m, n) U-Matrix from SOMClusterer.compute_umatrix(),
            bmu_coords -- list of (row, col) tuples, one per politician.
    """
    from capitolwatch.analysis.clustering.som import SOMClusterer

    matrix = load_features(feature_type)
    scaled, _ = normalize_features(matrix, MinMaxScaler())
    X = scaled.to_numpy()

    som = SOMClusterer(
        m=m,
        n=n,
        sigma=1.0,
        learning_rate=0.5,
        n_iterations=n_iterations,
        random_seed=42,
    )
    som.fit(X)
    som.extract_clusters(n_clusters=n_clusters)
    umatrix = som.compute_umatrix()
    return X, som.labels_, umatrix, som.bmu_coords_


# --- Shared utility helpers ---


def _build_hover_texts(
    politician_metadata: pd.DataFrame,
    labels: np.ndarray,
) -> list[str]:
    """
    Build one tooltip string per politician for Plotly hover.

    Args:
        politician_metadata (pd.DataFrame): id, first_name, last_name, party.
        labels (np.ndarray): Cluster labels aligned with metadata rows.

    Returns:
        list[str]: HTML-formatted tooltip strings (usable in hovertemplate).
    """
    texts = []
    for i, row in politician_metadata.reset_index(drop=True).iterrows():
        cluster = labels[i]
        cluster_str = "Outlier" if cluster == -1 else f"Cluster {cluster}"
        texts.append(
            f"{row['first_name']} {row['last_name']}<br>"
            f"Party: {row['party']}<br>"
            f"{cluster_str}"
        )
    return texts


def _get_party_colors(politician_metadata: pd.DataFrame) -> list[str]:
    """
    Map each politician's party to a hex marker colour.

    Args:
        politician_metadata (pd.DataFrame): Must contain a "party" column.

    Returns:
        list[str]: One hex colour string per row.
    """
    fallback = "#888888"
    return [
        PARTY_COLOR_MAP.get(party, fallback)
        for party in politician_metadata["party"]
    ]


# --- Tab rendering functions ---


def _tab_comparison(internal_df: pd.DataFrame) -> None:
    """
    Render the "Comparison" tab.

    Displays the metrics table for all 6 experiments and one interactive
    barplot for a user-selected metric.

    Args:
        internal_df (pd.DataFrame): Internal metrics CSV with an "experiment"
            column added.
    """
    st.header("Comparison of 6 clustering experiments")

    st.subheader("Internal metrics table")

    display_cols = [
        "experiment",
        "n_clusters",
        "n_outliers",
        "silhouette",
    ]
    st.dataframe(
        internal_df[display_cols].style.format(
            {
                "silhouette": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.info(
        "**K-Means silhouette artefact** — K-Means scores (0.76 / 0.68) are "
        "artificially inflated: the algorithm always selects K=2 and isolates "
        "Rick Scott (455 assets, maximum in the dataset) as a singleton "
        "cluster. "
        "A single-point cluster has silhouette ≈ 1 by definition, which makes "
        "the global score misleadingly high. "
        "DBSCAN correctly assigns him label -1 (outlier) without "
        "polluting the "
        "other clusters."
    )

    st.divider()

    st.plotly_chart(
        barplot_metrics_plotly(
            internal_df,
            metric_col="silhouette",
            title="Silhouette — 6 experiments",
        ),
        use_container_width=True,
    )


def _tab_best_result(politician_metadata: pd.DataFrame) -> None:
    """
    Render the "Best result — DBSCAN" tab.

    Shows PCA / t-SNE scatters, a centroid heatmap, and the outlier table
    for DBSCAN + freq_weighted (best silhouette among meaningful results).

    Args:
        politician_metadata (pd.DataFrame): Politician names and parties.
    """
    st.header("Best result — DBSCAN + freq_weighted")
    st.caption(
        "Best configuration: cosine metric, eps=0.5, min_samples=3. "
        "3 clusters, 15 outliers (19% noise ratio). "
        "Silhouette: 0.4857."
    )

    with st.spinner("Running DBSCAN (cosine grid search)..."):
        X, labels = _get_dbscan_results("freq_weighted")

    raw_matrix = load_features("freq_weighted")
    feature_names = list(raw_matrix.columns)
    hover_texts = _build_hover_texts(politician_metadata, labels)

    st.subheader("PCA 2D")
    st.plotly_chart(
        scatter_pca_plotly(
            X, labels, hover_texts, title="DBSCAN + freq_weighted — PCA"
        ),
        use_container_width=True,
    )

    st.subheader("Centroid heatmap (mean investment value per cluster)")
    st.plotly_chart(
        heatmap_centroids_plotly(
            raw_matrix.to_numpy(),
            labels,
            feature_names,
            title="DBSCAN + freq_weighted — cluster profiles",
        ),
        use_container_width=True,
    )

    st.subheader("Outliers identified by DBSCAN")
    outlier_mask = labels == -1
    outlier_df = (
        politician_metadata[outlier_mask.tolist()]
        .copy()
        .reset_index(drop=True)[["first_name", "last_name", "party"]]
    )
    outlier_df.index += 1
    st.dataframe(outlier_df, use_container_width=True)

    # Party breakdown of the 15 outliers
    if not outlier_df.empty:
        party_counts = outlier_df["party"].value_counts()
        rep = party_counts.get("Republican", 0)
        dem = party_counts.get("Democratic", 0)
        ind = party_counts.get("Independent", 0)
        st.info(
            f"**Outlier profile** — {rep} Republicans, {dem} Democrats"
            + (f", {ind} Independents" if ind else "") + ". "
            "These 15 politicians are not outliers because of their party "
            "affiliation, but because of the **scale and composition** of "
            "their portfolios: they invest in niche asset classes "
            "(Municipal Securities, LLCs, alternative funds) that are "
            "statistically rare in the dataset. "
            "Rick Scott stands out with 455 assets — 6x the average — "
            "and is the only politician unanimously flagged by all four "
            "methods (K-Means, DBSCAN, SOM x2)."
        )


def _tab_som(politician_metadata: pd.DataFrame) -> None:
    """
    Render the "SOM" tab.

    Trains a 7x7 Self-Organizing Map (cached) and shows:
      - the political map (U-Matrix + politicians overlaid on the grid)

    Args:
        politician_metadata (pd.DataFrame): Politician names and parties.
    """
    st.header("Self-Organizing Map (SOM)")

    feature_type = st.selectbox(
        "Feature type",
        ["freq_weighted", "freq_baseline"],
        help=(
            "freq_weighted: dollar-weighted subtypes  "
            "freq_baseline: count-based subtypes"
        ),
    )

    with st.spinner(f"Training SOM 7x7 on {feature_type} (may take ~10 s)..."):
        X, labels, umatrix, bmu_coords = _get_som_full_results(feature_type)

    hover_texts = _build_hover_texts(politician_metadata, labels)
    party_colors = _get_party_colors(politician_metadata)

    st.subheader("Political map (party colours)")
    st.plotly_chart(
        som_map_plotly(
            umatrix,
            bmu_coords,
            hover_texts,
            party_colors,
            title=f"SOM map — {feature_type}",
        ),
        use_container_width=True,
    )

    st.caption(
        "Grid 7x7 = 49 neurons for 79 politicians. "
        "Rule of thumb: 5*sqrt(N) = 5*sqrt(79) ≈ 44 neurons. "
        "Dark cells = cluster cores (low distance). "
        "Light cells = cluster boundaries."
    )


def _tab_external(
    external_df: pd.DataFrame,
    internal_df: pd.DataFrame,
) -> None:
    """
    Render the optional "External metrics" tab.

    External metrics (ARI, NMI, V-Measure) are computed against party labels
    as ground truth. All values near 0 show that clusters are independent of
    political affiliation.

    Args:
        external_df (pd.DataFrame): External metrics CSV.
        internal_df (pd.DataFrame): Internal metrics CSV (kept for reference).
    """
    st.header("External metrics — party labels as ground truth")
    st.caption(
        "These metrics measure whether clusters align with political parties. "
        "Values near 0 indicate independence between clusters and party."
    )

    # --- Metrics table ---
    display_cols = ["algo_name", "feature_type", "ari", "nmi", "v_measure"]
    st.subheader("ARI / NMI / V-Measure table")
    st.dataframe(
        external_df[display_cols].style.format(
            {"ari": "{:.4f}", "nmi": "{:.4f}", "v_measure": "{:.4f}"}
        ),
        use_container_width=True,
    )

    st.divider()

    # --- ARI / NMI / V-Measure barplots ---
    import copy
    ext_plot = copy.copy(external_df)
    ext_plot["experiment"] = (
        ext_plot["algo_name"] + " / " + ext_plot["feature_type"]
    )

    st.subheader("Barplots — external metrics")
    ext_cols = st.columns(3)
    for col, metric in zip(ext_cols, ["ari", "nmi", "v_measure"]):
        col.plotly_chart(
            barplot_metrics_plotly(
                ext_plot,
                metric_col=metric,
                title=metric.replace("_", " ").upper(),
            ),
            use_container_width=True,
        )

    st.divider()

    # --- Confusion matrices (cluster x party) ---
    from capitolwatch.analysis.evaluation import (  # noqa: E402
        build_confusion_matrix,
    )
    from capitolwatch.web.charts import (  # noqa: E402
        heatmap_confusion_plotly,
    )

    st.subheader("Confusion matrices — cluster x party")

    # Map party strings to integer codes expected by build_confusion_matrix()
    _PARTY_ORDER = ["Republican", "Democratic", "Independent"]
    _PARTY_ENCODE = {p: i for i, p in enumerate(_PARTY_ORDER)}

    politician_metadata = _load_politician_metadata()
    party_int = np.array(
        [_PARTY_ENCODE.get(p, -1) for p in politician_metadata["party"]]
    )

    # Map (algo, feature_type) to a cached loader that returns (X, labels)
    _LOADERS = {
        ("kmeans", "freq_baseline"): lambda: _get_kmeans_results(
            "freq_baseline"
        ),
        ("kmeans", "freq_weighted"): lambda: _get_kmeans_results(
            "freq_weighted"
        ),
        ("dbscan", "freq_baseline"): lambda: _get_dbscan_results(
            "freq_baseline"
        ),
        ("dbscan", "freq_weighted"): lambda: _get_dbscan_results(
            "freq_weighted"
        ),
        ("som", "freq_baseline"): lambda: _get_som_full_results(
            "freq_baseline"
        )[:2],
        ("som", "freq_weighted"): lambda: _get_som_full_results(
            "freq_weighted"
        )[:2],
    }

    exp_order = [
        ("kmeans", "freq_baseline"),
        ("kmeans", "freq_weighted"),
        ("dbscan", "freq_baseline"),
        ("dbscan", "freq_weighted"),
        ("som", "freq_baseline"),
        ("som", "freq_weighted"),
    ]

    # Render 2 confusion matrices per row so they don't get too small
    for row_start in range(0, len(exp_order), 2):
        row_exps = exp_order[row_start: row_start + 2]
        rcols = st.columns(len(row_exps))
        for rcol, (algo, ft) in zip(rcols, row_exps):
            with st.spinner(f"Computing {algo} / {ft}..."):
                _, pred_labels = _LOADERS[(algo, ft)]()
            conf = build_confusion_matrix(
                party_int, pred_labels, _PARTY_ORDER
            )
            rcol.plotly_chart(
                heatmap_confusion_plotly(
                    conf, title=f"{algo} / {ft}"
                ),
                use_container_width=True,
            )

    st.divider()

    # --- Scientific interpretation ---
    st.subheader("Scientific interpretation")
    st.markdown(
        "**Clusters are independent of political party affiliation.** "
        "All ARI values are close to 0 (from -0.029 to +0.006), meaning "
        "cluster assignments are no better than random at predicting party "
        "membership. NMI and V-Measure confirm this: the highest NMI is "
        "0.061 (DBSCAN + freq_weighted), indicating less than 6% of shared "
        "information between clusters and parties.\n\n"
        "**This is a scientifically meaningful result, not a failure.** "
        "It shows that investment behaviour cuts across party lines: "
        "Republicans and Democrats alike hold mutual funds, stocks, and ETFs "
        "in similar proportions. The clusters instead reflect "
        "**portfolio scale and complexity** — how many assets a senator "
        "holds and how specialised their investments are — rather than "
        "ideological differences.\n\n"
        "**Outliers reinforce this conclusion.** "
        "The 15 DBSCAN outliers include both Republicans and Democrats, "
        "united by unusually large or atypical portfolios, not by party. "
        "Rick Scott (Republican, 455 assets) is the most extreme case, but "
        "Mark R. Warner (Democrat) and Ron Wyden (Democrat) also appear, "
        "confirming that the pattern is bipartisan."
    )


def _tab_sector(politician_metadata: pd.DataFrame) -> None:
    """
    Render the "Sector analysis" tab.

    Runs DBSCAN on sector_baseline (11 economic sectors + 3 numeric features)
    and displays:
      - Interactive PCA 2D scatter (hover = name + party + cluster)
      - Table of sector outliers (label == -1)
      - Explanatory note comparing sector vs subtype analysis

    Args:
        politician_metadata (pd.DataFrame): Politician names and parties.
    """
    st.header("Sector analysis — DBSCAN on economic sectors")
    st.caption(
        "Feature set: sector_baseline — 11 economic sectors (Technology, "
        "Finance, Healthcare…) + 3 numeric features. "
        "82.6% of assets have no sector (\"Uncategorized\"), which is "
        "excluded from the feature dimensions to preserve signal."
    )

    with st.spinner("Running DBSCAN on sector_baseline..."):
        X, labels = _get_dbscan_results("sector_baseline")

    hover_texts = _build_hover_texts(politician_metadata, labels)

    # --- PCA 2D scatter ---
    st.subheader("PCA 2D — sector clusters")
    st.plotly_chart(
        scatter_pca_plotly(
            X,
            labels,
            hover_texts,
            title="DBSCAN + sector_baseline — PCA",
        ),
        use_container_width=True,
    )

    # --- Outlier table ---
    st.subheader("Outliers identified by DBSCAN (sector view)")
    outlier_mask = labels == -1
    outlier_df = (
        politician_metadata[outlier_mask.tolist()]
        .copy()
        .reset_index(drop=True)[["first_name", "last_name", "party"]]
    )
    outlier_df.index += 1

    if outlier_df.empty:
        st.info("No outliers detected with the selected parameters.")
    else:
        st.dataframe(outlier_df, use_container_width=True)
        party_counts = outlier_df["party"].value_counts()
        rep = party_counts.get("Republican", 0)
        dem = party_counts.get("Democratic", 0)
        ind = party_counts.get("Independent", 0)
        st.info(
            f"**{len(outlier_df)} sector outlier(s)** — "
            f"{rep} Republicans, {dem} Democrats"
            + (f", {ind} Independents" if ind else "") + ". "
            "These politicians have sector exposure patterns that are too "
            "sparse or atypical to form dense neighbourhoods in cosine "
            "space."
        )

    st.divider()

    # --- Explanatory note ---
    st.subheader("Sector analysis vs subtype analysis")
    st.markdown(
        "**Granularity difference** — The main analysis uses `subtype` "
        "(instrument type: Mutual Fund, Stock, ETF, Bond…), which captures "
        "*how* politicians invest. "
        "This tab uses `sector` (economic sector: Technology, Finance, "
        "Healthcare…), which captures *where* they invest.\n\n"
        "**Coverage caveat** — Only 17.4% of assets have a sector tag "
        "(11 sectors, sourced from product metadata). "
        "The remaining 82.6% fall under \"Uncategorized\", which is excluded "
        "from the feature dimensions to avoid a dominant zero-signal column. "
        "This means sector vectors are significantly sparser than subtype "
        "vectors (35 subtypes, ~98% coverage), and cluster structure may be "
        "less stable.\n\n"
        "**Interpretation** — Clusters here reflect *sector specialisation*: "
        "politicians who concentrate their tagged assets in one or two "
        "sectors (e.g., Technology or Finance) separate from those with "
        "diversified or no tagged holdings. "
        "Use this tab as a complementary lens, not a replacement for the "
        "subtype-based analysis."
    )


# --- Application entry point ---


def main() -> None:
    # Configure the Streamlit page and render all four tabs.

    st.set_page_config(
        page_title="CapitolWatch — Clustering",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("CapitolWatch — Investment clustering of US politicians")
    st.markdown(
        "Unsupervised learning on 79 US senators' financial disclosures. "
        "Dataset: 5 268 assets across 35 investment subtypes.  \n"
        "Algorithms: K-Means (baseline) · DBSCAN (density) · SOM (topology)."
    )

    internal_df, external_df = _load_evaluation_data()
    politician_metadata = _load_politician_metadata()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Comparison",
            "Best result — DBSCAN",
            "SOM",
            "External metrics",
            "Sector analysis",
        ]
    )

    with tab1:
        _tab_comparison(internal_df)

    with tab2:
        _tab_best_result(politician_metadata)

    with tab3:
        _tab_som(politician_metadata)

    with tab4:
        _tab_external(external_df, internal_df)

    with tab5:
        _tab_sector(politician_metadata)


if __name__ == "__main__":
    main()
