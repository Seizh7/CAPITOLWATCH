# CAPITOLWATCH

[![CI](https://github.com/Seizh7/CAPITOLWATCH/actions/workflows/ci.yml/badge.svg)](https://github.com/Seizh7/CAPITOLWATCH/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Unsupervised clustering of US senators' financial investment portfolios (2023 annual disclosures). Compares three algorithms — K-Means, DBSCAN, and SOM — across two feature representations (frequency counts and value-weighted vectors).

---

## Quick start — Docker

> The SQLite database and pre-computed feature store are required before starting the container. See [Development](#development) if you need to rebuild them from scratch.

```bash
# Build the image
docker compose build

# Start the dashboard
docker compose up

# Stop the dashboard
docker compose down
```

Dashboard available at **http://localhost:8501**.

---

## Dashboard

The Streamlit interface exposes five tabs:

| Tab | Content |
|-----|---------|
| **Comparison** | Internal metrics (Silhouette) across all 6 experiments — table + bar charts |
| **Best result** | DBSCAN + `freq_weighted` — PCA scatter plot, centroid heatmap, outlier list |
| **SOM** | U-Matrix and political map |
| **External** | ARI / NMI / V-Measure against party labels for each experiment |
| **Sector analysis** | DBSCAN clustering on economic sector vectors |

---

## Architecture

```
capitolwatch/
├── datapipeline/      # Scraping + database construction
├── analysis/          # Feature engineering, clustering, evaluation
├── web/               # Streamlit dashboard + Plotly charts
└── services/          # Database access layer
config/                # Settings (db path, feature store path…)
data/
├── capitolwatch.db    # SQLite database
└── feature_store/     # Pre-computed feature matrices (.pkl)
```

---

## Development

### Prerequisites

- Python 3.9
- Docker

### Local install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard locally (no Docker):

```bash
streamlit run capitolwatch/web/app.py
```

### Pipeline — rebuild from raw data

The data collection pipeline scrapes the US Senate website and populates a SQLite database. This step is only needed if you want to collect data for a different year or rebuild the database.

```bash
pip install -r requirements-pipeline.txt

# Full pipeline (scraping + database build)
python -m capitolwatch.datapipeline full-pipeline --year 2023
```

After rebuilding the database, run the full analysis pipeline:

```bash
# Full pipeline: features → evaluate → analyze → visualize
python -m capitolwatch.analysis full-pipeline
```

### Tests

```bash
pytest
```

---

## License

Copyright 2026 Seizh7 — Licensed under the [Apache License, Version 2.0](LICENSE).
