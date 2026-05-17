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


_INTERNAL_CSV = _PROJECT_ROOT / "data/outputs/evaluation_results.csv"
_EXTERNAL_CSV = (
    _PROJECT_ROOT / "data/outputs/evaluation_results_external.csv"
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
        "**Reading the scores** : K-Means shows the highest silhouette (0.77 and 0.70) "
        "but this needs context. It found only 2 clusters for both feature sets and placed "
        "Rick Scott alone in one of them, as his 455 assets make him an extreme outlier in the data. "
        "A single-point cluster always gets silhouette = 1, which pulls the average up artificially. "
        "DBSCAN's score (0.58) is lower but more reliable: it found 3 real clusters and flagged "
        "13 politicians as outliers (visible in the n_outliers column), "
        "without letting them modify the overall score."
    )

    st.divider()

    st.plotly_chart(
        barplot_metrics_plotly(
            internal_df,
            metric_col="silhouette",
            title="Silhouette - 6 experiments",
        ),
        use_container_width=True,
    )


def _tab_best_result(politician_metadata: pd.DataFrame) -> None:
    """
    Render the "Best result - DBSCAN" tab.

    Shows PCA / t-SNE scatters, a centroid heatmap, and the outlier table
    for DBSCAN + freq_weighted (best silhouette among meaningful results).

    Args:
        politician_metadata (pd.DataFrame): Politician names and parties.
    """
    st.header("Best result - DBSCAN + freq_weighted")
    st.caption(
        "Best configuration: cosine metric, eps=0.5, min_samples=3. "
        "3 clusters, 13 outliers (16% noise ratio). "
        "Silhouette: 0.5795."
    )

    with st.spinner("Running DBSCAN (cosine grid search)..."):
        X, labels = _get_dbscan_results("freq_weighted")

    raw_matrix = load_features("freq_weighted")
    feature_names = list(raw_matrix.columns)
    hover_texts = _build_hover_texts(politician_metadata, labels)

    st.subheader("PCA 2D")
    st.plotly_chart(
        scatter_pca_plotly(
            X, labels, hover_texts, title="DBSCAN + freq_weighted - PCA"
        ),
        use_container_width=True,
    )

    st.subheader("Centroid heatmap (mean investment value per cluster)")
    st.plotly_chart(
        heatmap_centroids_plotly(
            raw_matrix.to_numpy(),
            labels,
            feature_names,
            title="DBSCAN + freq_weighted - cluster profiles",
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

    # Party profile of the 13 outliers
    if not outlier_df.empty:
        party_counts = outlier_df["party"].value_counts()
        rep = party_counts.get("Republican", 0)
        dem = party_counts.get("Democratic", 0)
        ind = party_counts.get("Independent", 0)
        st.info(
            f"**Outlier profile:** {rep} Republicans, {dem} Democrats"
            + (f", {ind} Independents" if ind else "") + ". "
            "These 13 politicians were flagged as outliers not because of their party, "
            "but because of the **size and composition** of their portfolios. "
            "They hold asset types that are rare in the dataset "
            "(Municipal Securities, LLCs, alternative funds). "
            "Rick Scott is the most extreme case with 455 assets (6x the average) "
            "and is the only politician flagged as an outlier by all three algorithms."
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
            title=f"SOM map - {feature_type}",
        ),
        use_container_width=True,
    )

    st.caption(
        "7x7 grid = 49 neurons for 79 politicians. "
        "Recommended size: 5xsqrt(N) = 5xsqrt(79) ≈ 44 neurons. "
        "Dark cells = dense zones (cluster cores). "
        "Light cells = transition zones between clusters."
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
    st.header("External metrics: party labels as ground truth")
    st.caption(
        "These metrics compare the clusters found by each algorithm against party labels. "
        "A value close to 0 means the clusters have nothing to do with political party membership."
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

    st.subheader("External metrics barplots")
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

    st.subheader("Confusion matrices: cluster vs party")

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

    # --- Interpretation ---
    st.subheader("Interpretation")
    st.markdown(
        "**The clusters do not follow party lines.** "
        "All ARI values are close to 0 (between -0.036 and +0.012), which means "
        "knowing which cluster a senator belongs to does not help predict their party. "
        "NMI and V-Measure confirm this: the highest NMI is 0.097 (DBSCAN + freq_weighted), "
        "meaning clusters and parties share less than 10% of their information.\n\n"
        "It shows that investment behaviour is not tied to party: Republicans and Democrats "
        "hold the same types of assets (mutual funds, stocks, ETFs) in similar proportions. "
        "The clusters reflect **portfolio size and asset type** rather than political ideology.\n\n"
        "**The outliers confirm this.** "
        "The 13 DBSCAN outliers include both Republicans and Democrats, "
        "grouped together because of unusually large or atypical portfolios, not because of party. "
        "Rick Scott (Republican, 455 assets) is the clearest case, but Mark R. Warner "
        "and Ron Wyden (both Democrats) also appear, showing the pattern cuts across party lines."
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
    st.header("Sector analysis - DBSCAN on economic sectors")
    st.caption(
        "Feature set: sector_baseline - 11 economic sectors (Technology, "
        "Finance, Healthcare…) + 3 numeric features. "
        "82.6% of assets have no sector tag (\"Uncategorized\"), so this column "
        "was removed to avoid it dominating the feature vectors."
    )

    with st.spinner("Running DBSCAN on sector_baseline..."):
        X, labels = _get_dbscan_results("sector_baseline")

    raw_sector_matrix = load_features("sector_baseline")
    sector_feature_names = list(raw_sector_matrix.columns)
    hover_texts = _build_hover_texts(politician_metadata, labels)

    # --- PCA 2D scatter ---
    st.subheader("PCA 2D: sector clusters")
    st.plotly_chart(
        scatter_pca_plotly(
            X,
            labels,
            hover_texts,
            title="DBSCAN + sector_baseline - PCA",
        ),
        use_container_width=True,
    )

    # --- Centroid heatmap ---
    st.subheader("Centroid heatmap (mean investment count per cluster)")
    st.plotly_chart(
        heatmap_centroids_plotly(
            raw_sector_matrix.to_numpy(),
            labels,
            sector_feature_names,
            title="DBSCAN + sector_baseline - cluster profiles",
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
            f"**{len(outlier_df)} sector outlier(s):** "
            f"{rep} Republicans, {dem} Democrats"
            + (f", {ind} Independents" if ind else "") + ". "
            "These politicians have sector investment patterns that are too unusual "
            "or sparse to be grouped with others."
        )

    st.divider()

    # --- Explanatory note ---
    st.subheader("Sector analysis vs subtype analysis")
    st.markdown(
        "**Two different views of the same data.** The main analysis uses `subtype` "
        "(the type of financial instrument: Mutual Fund, Stock, ETF, Bond…), which shows "
        "*how* politicians invest. "
        "This tab uses `sector` (the economic sector: Technology, Finance, Healthcare…), "
        "which shows *where* they invest.\n\n"
        "**Coverage issue.** Only 17.4% of assets have a sector tag (11 sectors). "
        "The remaining 82.6% are labelled \"Uncategorized\" and were excluded from the "
        "feature vectors to avoid introducing a column with almost no useful signal. "
        "As a result, sector vectors are much sparser than subtype vectors "
        "(34 subtypes, ~98% coverage), and cluster quality may be lower.\n\n"
        "**Reading this tab.** The clusters here show *sector concentration*: "
        "politicians who put most of their tagged assets into one or two sectors "
        "(e.g., Technology or Finance) are separated from those with more diverse "
        "or mostly untagged holdings. "
    )


# --- Application entry point ---


def main() -> None:
    # Configure the Streamlit page and render all four tabs.

    st.set_page_config(
        page_title="CapitolWatch - Clustering",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("CapitolWatch - Investment clustering of US politicians")
    st.markdown(
        "Unsupervised learning applied to 79 US senators' financial disclosures. "
        "The dataset contains 5,268 assets across 34 investment subtypes. "
        "Three algorithms are compared: K-Means, DBSCAN, and SOM (Self-Organizing Map)."
    )

    internal_df, external_df = _load_evaluation_data()
    politician_metadata = _load_politician_metadata()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Comparison",
            "Best result - DBSCAN",
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
