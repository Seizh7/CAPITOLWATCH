# CAPITOLWATCH

[![CI](https://github.com/Seizh7/CAPITOLWATCH/actions/workflows/ci.yml/badge.svg)](https://github.com/Seizh7/CAPITOLWATCH/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Unsupervised clustering of US senators' financial investment portfolios (2023 annual disclosures). Compares three algorithms (K-Means, DBSCAN, and SOM) across two feature representations (frequency counts and value-weighted vectors).

---

## Quick start - Docker

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
| **Comparison** | Internal metrics (Silhouette) across all 6 experiments (table + bar charts)|
| **Best result** | DBSCAN + `freq_weighted` (PCA scatter plot, centroid heatmap, outlier list) |
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

### Pipeline - rebuild from raw data

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

## Dependencies

All runtime dependencies are pinned in [`requirements.txt`](requirements.txt). The data pipeline additionally requires [`requirements-pipeline.txt`](requirements-pipeline.txt).

| Library | Version | Role |
|---------|---------|------|
| [scikit-learn](https://scikit-learn.org/) | 1.6.1 | K-Means, DBSCAN, metrics (Silhouette, ARI, NMI) |
| [MiniSom](https://github.com/JustGlowing/minisom) | 2.3.5 | Self-Organizing Map (SOM) |
| [numpy](https://numpy.org/) | 1.26.4 | Numerical arrays and matrix operations |
| [pandas](https://pandas.pydata.org/) | 2.3.2 | Tabular data manipulation |
| [matplotlib](https://matplotlib.org/) | 3.9.4 | Static figures (PCA, heatmaps, barplots) |
| [seaborn](https://seaborn.pydata.org/) | 0.13.2 | Statistical visualisation styling |
| [plotly](https://plotly.com/python/) | 6.3.0 | Interactive charts in the dashboard |
| [streamlit](https://streamlit.io/) | 1.50.0 | Web dashboard interface |
| [joblib](https://joblib.readthedocs.io/) | 1.5.1 | Feature store serialisation (`.pkl` files) |
| [python-dotenv](https://saurabh-kumar.com/python-dotenv/) | 1.1.1 | Environment variable loading |
| [selenium](https://www.selenium.dev/) | 4.34.2 | Headless browser scraping *(pipeline only)* |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | 4.13.4 | HTML report parsing *(pipeline only)* |
| [requests](https://requests.readthedocs.io/) | 2.32.5 | HTTP calls to Congress and OpenFIGI APIs *(pipeline only)* |
| [yfinance](https://ranaroussi.github.io/yfinance/) | 0.2.65 | Financial product enrichment *(pipeline only)* |
| [typer](https://typer.tiangolo.com/) | 0.21.1 | CLI interface for both modules *(pipeline only)* |

---

## License

Copyright 2026 Seizh7 - Licensed under the [Apache License, Version 2.0](LICENSE).
