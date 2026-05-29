# Manuel Technique — CapitolWatch

**Version :** 1.0  
**Licence :** Apache License 2.0  
**Auteur :** Seizh7  
**Date :** 18 mai 2026

---

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Structure des répertoires](#2-structure-des-répertoires)
3. [Base de données](#3-base-de-données)
4. [Module `services`](#4-module-services)
5. [Module `datapipeline`](#5-module-datapipeline)
6. [Module `analysis`](#6-module-analysis)
7. [Module `web`](#7-module-web)
8. [Configuration](#8-configuration)
9. [Interface en ligne de commande (CLI)](#9-interface-en-ligne-de-commande-cli)
10. [Déploiement Docker](#10-déploiement-docker)
11. [Tests](#11-tests)
12. [Dépendances](#12-dépendances)
13. [Interactions entre bibliothèques dans le pipeline](#13-interactions-entre-bibliothèques-dans-le-pipeline)

---

## 1. Vue d'ensemble de l'architecture

CapitolWatch est structuré autour de deux pipelines distincts et exposés via une interface web et des CLI.

L'architecture s'articule autour de quatre domaines principaux :
- **`datapipeline`** : Scraping des rapports d'investissement, enrichissement et insertion en base de données.
- **`services`** : Couche d'accès unique à la base de données (SQLite). Tous les autres modules doivent passer par ici pour y lire ou écrire des données.
- **`analysis`** : Pipeline de Machine Learning (Data Load, Feature Engineering, Clustering, Evaluation).
- **`web`** : Interface utilisateur (tableau de bord interactif) lisant les résultats de l'analyse.

Le principe architectural central est la **séparation des responsabilités** : chaque sous-module n'a accès aux données que via la couche `services/`, qui encapsule toutes les requêtes SQL. L'analyse ne connaît pas la structure interne de la base ; la base ne connaît pas les algorithmes de clustering.

---

## 2. Structure des répertoires

```
CAPITOLWATCH/
├── capitolwatch/                  # Paquet Python principal
│   ├── __init__.py
│   ├── db.py                      # Connexion SQLite partagée
│   ├── analysis/                  # Pipeline ML
│   │   ├── __init__.py
│   │   ├── __main__.py            # Point d'entrée : python -m capitolwatch.analysis
│   │   ├── cli.py                 # Commandes Typer (features, evaluate, analyze, visualize)
│   │   ├── data_loader.py         # Chargement données depuis la base
│   │   ├── feature_engineering.py # Construction des vecteurs de features
│   │   ├── preprocessing.py       # Normalisation (StandardScaler / MinMaxScaler)
│   │   ├── feature_store.py       # Sérialisation et rechargement des matrices
│   │   ├── evaluation.py          # Métriques internes et externes
│   │   ├── cluster_analysis.py    # Interprétation et description des clusters
│   │   ├── visualization.py       # Graphiques statiques (matplotlib/seaborn)
│   │   ├── run_evaluation.py      # Orchestration de toutes les expériences
│   │   ├── run_cluster_analysis.py
│   │   ├── run_visualization.py
│   │   └── clustering/            # Algorithmes de clustering
│   │       ├── __init__.py
│   │       ├── base.py            # Classe abstraite BaseClusterer
│   │       ├── kmeans.py          # KMeansClusterer
│   │       ├── dbscan.py          # DBSCANClusterer
│   │       ├── som.py             # SOMClusterer
│   │       ├── run_kmeans.py      # Script de test isolé K-Means
│   │       ├── run_dbscan.py      # Script de test isolé DBSCAN
│   │       └── run_som.py         # Script de test isolé SOM
│   ├── datapipeline/              # Pipeline de collecte de données
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── cli.py                 # Commandes Typer (full-pipeline, scraping, database)
│   │   ├── scraping/              # Téléchargement automatisé
│   │   │   ├── core.py
│   │   │   ├── scraper.py         # Extraction des liens depuis l'EFD
│   │   │   ├── downloader.py      # Téléchargement des HTML
│   │   │   └── driver.py          # Initialisation Selenium WebDriver
│   │   └── database/              # Ingestion et enrichissement
│   │       ├── core.py            # Logique pipeline séparée du CLI
│   │       ├── congress_api.py    # Client API Congress.gov
│   │       ├── import_reports.py  # Import des HTML en base
│   │       ├── matching_workflow.py # Association rapport ↔ politicien
│   │       ├── politician_matcher.py # Algorithme de correspondance de noms
│   │       ├── parse_report_assets.py # Extraction des actifs depuis HTML
│   │       ├── enrich_products.py # Enrichissement via OpenFIGI / yFinance
│   │       └── geographic_enrichment.py
│   ├── services/                  # Couche d'accès à la base de données
│   │   ├── __init__.py
│   │   ├── init_db.py             # Création du schéma SQLite
│   │   ├── politicians.py         # CRUD politiciens
│   │   ├── reports.py             # CRUD rapports
│   │   ├── assets.py              # CRUD actifs
│   │   ├── products.py            # CRUD produits financiers
│   │   └── analytics.py           # Requêtes analytiques multi-tables
│   └── web/                       # Interface Streamlit
│       ├── __init__.py
│       ├── app.py                 # Application principale (5 onglets)
│       └── charts.py              # Fabrique de graphiques Plotly
├── config/
│   ├── __init__.py
│   └── settings.py                # Classe Config (variables d'environnement)
├── data/
│   ├── capitolwatch.db            # Base SQLite (générée par datapipeline)
│   ├── manual_overrides.json      # Corrections manuelles de correspondance
│   └── outputs/                   # Feature store et résultats d'évaluation
│       ├── freq_baseline.pkl
│       ├── freq_weighted.pkl
│       ├── sector_baseline.pkl
│       ├── politician_labels.pkl
│       ├── evaluation_results.csv
│       └── evaluation_results_external.csv
├── docs/                          # Documentation technique et utilisateur
├── tests/                         # Tests unitaires pytest
├── Dockerfile
├── docker-compose.yml
├── requirements.txt               # Dépendances runtime (dashboard + analyse)
├── requirements-pipeline.txt      # Dépendances supplémentaires (scraping)
└── pytest.ini
```

---

## 3. Base de données

### 3.1 Technologie

CapitolWatch utilise **SQLite** comme base de données embarquée. Ce choix est adapté à un volume de données modéré (de l'ordre de N politiciens et M milliers d'actifs) et supprime toute dépendance externe à un serveur de base de données. Le fichier `capitolwatch.db` est monté en lecture seule dans le conteneur Docker.

### 3.2 Schéma relationnel


- **`politicians`** : La table centrale contenant chaque sénateur.
  - `id`, `last_name`, `first_name`, `party`

- **`reports`** : Chaque rapport financier est lié à un sénateur.
  - `id`, `politician_id`, `source_file`, `year`...

- **`assets`** : Chaque actif (action, obligation, etc.) déclaré dans un rapport.
  - `id`, `report_id` , `product_id` , `value`...

- **`products`** : Un catalogue de tous les produits financiers uniques pour éviter les doublons.
  - `id`, `name`, `type`, `subtype`, `ticker`...


**Points notables :**

- `assets.parent_asset_id` : auto-référence permettant de modéliser des actifs imbriqués (ex. : un fonds contenant d'autres titres).
- `assets.value` : stockée sous forme de texte brut (`"$1,001 - $15,000"`) tel que déclaré dans le rapport officiel. La conversion en valeur numérique est effectuée à la lecture par `data_loader.parse_value_range()`.
- `products.is_enriched` : booléen indiquant si le produit a été enrichi via les APIs externes (OpenFIGI, yFinance).
- Les clés étrangères sont activées à chaque connexion via `PRAGMA foreign_keys = ON;` (voir `capitolwatch/db.py`).

### 3.3 Point d'entrée de la connexion

Toute connexion à la base transite par la fonction `get_connection()` définie dans `capitolwatch/db.py` :

```python
conn = get_connection(config)
# Retourne une connexion SQLite avec :
#   - row_factory = sqlite3.Row  (accès aux colonnes par nom)
#   - PRAGMA foreign_keys = ON
```

---

## 4. Module `services`

Le module `services/` est la **seule** couche autorisée à émettre des requêtes SQL. Aucun autre module ne doit interagir directement avec la base de données. Il est donc présenté en premier car `datapipeline` et `analysis` en dépendent tous les deux.

| Fichier | Responsabilité |
|---------|---------------|
| `init_db.py` | Création des tables (`CREATE TABLE IF NOT EXISTS`) |
| `politicians.py` | Insertion et lecture des politiciens |
| `reports.py` | Insertion et lecture des rapports |
| `assets.py` | Insertion et lecture des actifs |
| `products.py` | Insertion, lecture et mise à jour des produits |
| `analytics.py` | Requêtes analytiques (jointures multi-tables) pour le clustering et la dashboard |

Toutes les fonctions suivent un pattern d'injection de dépendances : les paramètres `config` et `connection` sont passés en *keyword-only*. Cela permet à l'appelant soit d'injecter une connexion existante pour de meilleures performances, soit de laisser la fonction en créer une pour un usage autonome, facilitant ainsi les tests unitaires.

**Exemple :**

```python
def get_politician_id_by_name(
    first_name: str,
    last_name: str,
    *,
    config: Optional[object] = None,
    connection=None,
) -> Optional[str]: ...
```

---

## 5. Module `datapipeline`

### 5.1 Rôle

Le module `datapipeline` est responsable de la collecte, de l'ingestion et de l'enrichissement des données. Il est découpé en deux sous-modules principaux : `scraping` et `database`.

### 5.2 Sous-module `scraping`

Ce module gère le téléchargement des rapports financiers depuis le portail public du Sénat.

| Fichier | Rôle |
| :--- | :--- |
| `driver.py` | Initialisation du ChromeDriver Selenium en mode headless. |
| `scraper.py` | Navigation sur `efdsearch.senate.gov` et extraction des liens de rapports. |
| `downloader.py` | Téléchargement des pages HTML identifiées par le scraper. |
| `core.py` | Orchestration du pipeline de scraping : `driver` -> `scraper` -> `downloader`. |

> Le scraper utilise **Selenium** car le portail EFD charge les résultats de recherche dynamiquement en JavaScript, rendant une approche statique (ex: `requests`) inefficace.

### 5.3 Sous-module `database`

Ce module prend en charge l'insertion des données brutes en base, leur association et leur enrichissement.

| Fichier | Rôle |
| :--- | :--- |
| `congress_api.py` | Client pour l'API `Congress.gov` afin de récupérer la liste officielle des sénateurs. |
| `import_reports.py` | Lecture des HTML locaux, calcul de checksum et insertion dans la table `reports`. |
| `politician_matcher.py` | Algorithme de correspondance de noms pour lier un rapport à un politicien. |
| `parse_report_assets.py` | Parsing du HTML des rapports avec BeautifulSoup pour extraire les actifs financiers. |
| `enrich_products.py` | Enrichissement des produits via des API externes (OpenFIGI, yFinance) pour obtenir le secteur. |
| `core.py` | Logique métier qui orchestre les étapes d'ingestion et d'enrichissement. |

#### Algorithme de correspondance de noms (`politician_matcher.py`)

Les rapports HTML ne contiennent pas d'identifiant structuré lié à un politicien. Le nom est extrait du nom de fichier (ex. `annual_report_john_smith_2024.html`) et comparé aux noms en base. En cas d'ambiguïté, le fichier `data/manual_overrides.json` fournit des corrections manuelles. La bibliothèque `NAMEMATCHING` (dépôt externe) améliore la précision en cas d'accent, d'abréviation ou de variation orthographique.

---

## 6. Module `analysis`

### 6.1 Vue d'ensemble du pipeline d'analyse

| Étape / Script | Fonctions clés | Entrées / Traitements | Sorties / Artefacts |
| :--- | :--- | :--- | :--- |
| `data_loader.py` | `load_politicians()`, `load_assets_with_products()` | Données brutes depuis la base via `services`. | DataFrames Pandas chargés en mémoire. |
| `feature_engineering.py` | `create_*_vectors()`, `compute_numerical_features()` | DataFrames de `data_loader`. | Matrices de features (politiciens × features). |
| `feature_store.py` | `build_feature_store()`, `load_features()` | Matrices de features. | `data/outputs/*.pkl` (sérialisation `joblib`). |
| `preprocessing.py` | `normalize_features()` | `StandardScaler` (K-Means/DBSCAN) ou `MinMaxScaler` (SOM). | Matrices de features normalisées. |
| `clustering/` | `.fit()`, `extract_clusters()` | Entraînement des modèles (K-Means, DBSCAN, SOM). | Modèles entraînés et labels de clusters. |
| `evaluation.py` | `calculate_silhouette_score()`, `evaluate_external()` | Labels de clusters et vérité terrain (partis). | Scores de performance (Silhouette, ARI, NMI). |
| `run_evaluation.py` | Orchestration de l'évaluation. | Exécution des différentes expériences. | `data/outputs/evaluation_results*.csv`. |
| `visualization.py` | `plot_*()` | Données et résultats des clusters. | `data/figures/*.png`. |
| `web/charts.py` | `create_*_chart()` | Données et résultats pour l'interface web. | Figures interactives `plotly.graph_objects`. |


### 6.2 Chargement des données (`data_loader.py`)

Ce module sert de pont entre la base de données et le pipeline d'analyse. Il utilise les fonctions du module `services` pour charger les données brutes et les prépare sous forme de DataFrames Pandas.

**Fichier :** `capitolwatch/analysis/data_loader.py`  
**Fonctions clés :**

```python
load_politicians()           # -> DataFrame (N_politiciens × 4) : id, first_name, last_name, party
load_assets_with_products()  # -> DataFrame (N_actifs × 7) : politician_id, product_name,
                             #   subtype, sector, owner, value, value_numeric
parse_value_range(value_str) # "$1,001 - $15,000" -> 8000.50
```

La fonction `parse_value_range()` gère les cas particuliers :

| Entrée | Sortie |
|--------|--------|
| `"$1,001 - $15,000"` | `8000.50` (moyenne de la plage) |
| `"None (or less than $201)"` | `201.0` |
| `""` ou `NULL` | `0.0` |
| `"$50,000,001 +"` | `50 000 001.0` (borne inférieure) |

### 6.3 Ingénierie des features (`feature_engineering.py`)

Cette étape transforme les données brutes en un format numérique exploitable par les algorithmes de clustering. Trois stratégies de vectorisation sont implémentées :

| Matrice | Dimensions | Description |
|---------|-----------|-------------|
| `freq_baseline` | N_politiciens × N_features | Comptage d'actifs par subtype + `total_assets` + `diversity` |
| `freq_weighted` | N_politiciens × N_features | Somme des valeurs investies par subtype + `total_assets` + `diversity` |
| `sector_baseline` | N_politiciens × N_sector_features | Comptage par secteur économique (GICS) |

Les `N_features` sont principalement constituées des différents `subtypes` d'actifs (ex: Mutual Fund, Stock, ETF, Municipal Security, etc.).

Les deux caractéristiques additionnelles `total_assets` et `diversity` sont concaténées à droite de chaque matrice.

### 6.4 Normalisation (`preprocessing.py`)

Une fois les features construites, elles doivent être mises à l'échelle (normalisées) pour que les algorithmes de clustering fonctionnent correctement. En effet, des features avec des ordres de grandeur très différents (par exemple, un nombre d'actifs vs la valeur totale du portefeuille) peuvent biaiser les résultats. Deux approches sont utilisées en fonction de l'algorithme cible :

| Scaler | Formule | Utilisé par |
|--------|---------|-------------|
| `StandardScaler` | $(X - \mu) / \sigma$ | K-Means, DBSCAN |
| `MinMaxScaler` | $(X - X_{\min}) / (X_{\max} - X_{\min})$ | SOM |

Le `StandardScaler` est requis pour les algorithmes basés sur la distance euclidienne (K-Means et DBSCAN) afin que chaque dimension contribue équitablement au calcul de distance. Le `MinMaxScaler` est préféré pour le SOM car les poids initiaux de MiniSom sont dans `[0, 1]` par défaut.

**Fichier :** `capitolwatch/analysis/preprocessing.py`  
**Fonction :** `normalize_features()`

```python
matrix_scaled, fitted_scaler = normalize_features(matrix, StandardScaler())
# matrix_scaled  : pd.DataFrame, même index et colonnes que matrix
# fitted_scaler  : scaler ajusté, réutilisable pour transform() sur de nouveaux points
```

Le retour du scaler ajusté est intentionnel : il permet, lors d'une inférence ultérieure, d'appliquer la même transformation sans recalculer $\mu$ et $\sigma$ sur les nouvelles données.

### 6.5 Algorithmes de clustering (`clustering/`)

#### Classe abstraite `BaseClusterer`

Toutes les implémentations héritent de `BaseClusterer` et exposent l'interface commune :

**Fichier :** `capitolwatch/analysis/clustering/base.py`  
**Classe :** `BaseClusterer`

```python
class BaseClusterer(ABC):
    def fit(self, matrix: np.ndarray) -> "BaseClusterer": ...
    def predict(self, matrix: np.ndarray) -> np.ndarray: ...
    def get_params(self) -> dict: ...
```

Ce polymorphisme permet à `run_evaluation.py` de parcourir les trois algorithmes de manière uniforme sans connaître leurs détails d'implémentation.

#### `KMeansClusterer` (kmeans.py)

Wrapper autour de `sklearn.cluster.KMeans`. Fournit en plus :

- `find_optimal_k(matrix, k_range)` : calcule l'inertie (méthode du coude) et le score de Silhouette pour chaque K dans la plage donnée.
- `plot_elbow()` et `plot_silhouette_analysis()` : visualisations de la sélection de K.

Paramètres clés : `n_clusters`, `random_state=42`, `n_init=10`.

#### `DBSCANClusterer` (dbscan.py)

Wrapper autour de `sklearn.cluster.DBSCAN`. Attributs supplémentaires après `fit()` :

- `labels_` : tableau de labels ; `-1` indique un point de bruit (outlier).
- `n_clusters_` : nombre de clusters (bruit exclu).
- `n_outliers_` : nombre de points non assignés à un cluster.

Paramètres clés : `eps` (rayon de voisinage), `min_samples`.

#### `SOMClusterer` (som.py)

Wrapper autour de `minisom.MiniSom`. Le SOM est une étape en deux temps :

1. `fit(matrix)` : entraîne la carte et calcule les Best Matching Units (BMU) pour chaque politicien.
2. `extract_clusters(n_clusters)` : applique K-Means sur les poids des neurones activés pour extraire des labels de clusters discrets.

Paramètres clés : `m`, `n` (taille de grille), `sigma`, `learning_rate`, `n_iterations`.

#### Architecture des Classes 

**BaseClusterer (Classe Abstraite)**

*   **Attributs** : `matrix` (données), `labels` (résultats du clustering).
*   **Méthodes** : `fit(X)` (abstraite), `predict(X)`.

 **Héritée par** : `KMeansClusterer`, `DBSCANClusterer`, `SOMClusterer`

**KMeansClusterer (Hérite de BaseClusterer)**

*   **Attributs** : `n_clusters`, `max_iter`, `_model`.
*   **Méthodes** : `fit(X)`, `get_centroids()`.
*   **Association** : Enveloppe la classe externe `sklearn.cluster.KMeans`

**DBSCANClusterer (Hérite de BaseClusterer)**

*   **Attributs** : `eps`, `min_samples`, `_model`.
*   **Méthodes** : `fit(X)`, `get_core_samples()`.
*   **Association** : Enveloppe la classe externe `sklearn.cluster.DBSCAN`

**SOMClusterer (Hérite de BaseClusterer)**

*   **Attributs** : `x_dim`, `y_dim`, `learning_rate`, `_model`.
*   **Méthodes** : `fit(X)`, `extract_clusters()`, `plot_som_map()`.
*   **Association** : Enveloppe la classe externe `minisom.MiniSom`

### 6.6 Évaluation (`evaluation.py`)

#### Métriques internes

- **Score de Silhouette** : mesure la cohésion intra-cluster et la séparation inter-cluster. Valeur dans $[-1, 1]$ ; plus c'est proche de 1, mieux c'est.

#### Métriques externes (comparaison avec les partis politiques)

| Métrique | Description |
|----------|-------------|
| ARI (Adjusted Rand Index) | Mesure de similarité entre deux partitions, corrigée du hasard |
| NMI (Normalized Mutual Information) | Information mutuelle normalisée entre clusters et partis |
| V-Measure | Moyenne de l'homogénéité et de la complétude |

Les métriques externes ne sont utilisées qu'a posteriori pour mesurer si les clusters découverts correspondent aux partis politiques. Le parti n'est pas fourni comme feature au clustering.

### 6.7 Feature store (`feature_store.py`)

Les matrices sont sérialisées avec **joblib** dans `data/outputs/` :

```
data/outputs/
├── freq_baseline.pkl       # DataFrame (N_politiciens, N_features)
├── freq_weighted.pkl       # DataFrame (N_politiciens, N_features)
├── sector_baseline.pkl     # DataFrame (N_politiciens, N_sectors)
└── politician_labels.pkl   # DataFrame (N_politiciens, 4)
```

**Fichier :** `capitolwatch/analysis/feature_store.py`  
**Fonctions :** `build_feature_store()`, `load_features()`

```python
build_feature_store()                        # calcule et persiste les 4 .pkl
matrix = load_features("freq_baseline")      # -> pd.DataFrame (N_politiciens, N_features)
labels = load_features("politician_labels")  # -> pd.DataFrame (N_politiciens, 4)
```

### 6.8 Visualisation

#### `visualization.py` :  graphiques statiques (matplotlib/seaborn)

| Fonction | Sortie |
|----------|--------|
| `plot_dimensionality_reduction()` | Projection PCA 2D des clusters |
| `plot_centroid_heatmap()` | Heatmap des valeurs moyennes par cluster |
| `plot_metrics_barplot()` | Comparaison des scores de Silhouette entre expériences |
| `plot_cluster_sizes()` | Répartition du nombre de politiciens par cluster |

Toutes les figures sont sauvegardées en PNG dans `data/figures/`.

#### `charts.py` : graphiques interactifs (Plotly)

Chaque fonction retourne un objet `go.Figure` pour le tableau de bord Streamlit. Les données ne sont jamais chargées dans ce module ; la responsabilité de fournir les tableaux et labels appartient à `app.py`.

---

## 7. Module `web`

### 7.1 Architecture de l'application Streamlit

L'application est organisée en cinq onglets :

| Onglet | Contenu |
|--------|---------|
| **Comparison** | Tableau des métriques internes + barplots comparatifs |
| **Best result** | Scatter PCA, heatmap des centroïdes et liste des outliers pour DBSCAN + freq_weighted |
| **SOM** | U-Matrix et carte des politiciens sur la grille SOM |
| **External** | ARI / NMI / V-Measure vs labels des partis politiques |
| **Sector analysis** | DBSCAN appliqué à `sector_baseline` (analyse par secteur économique) |


### 7.2 Gestion du cache

Streamlit met en cache les données lourdes via le décorateur `@st.cache_data` pour éviter de recalculer les features et les résultats d'évaluation à chaque interaction :

**Fichier :** `capitolwatch/web/app.py`  
**Fonctions :** `_load_evaluation_data()`, `_load_politician_metadata()`, `_get_dbscan_results()`

```python
@st.cache_data
def _load_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame]: ...

@st.cache_data
def _load_politician_metadata() -> pd.DataFrame: ...

@st.cache_data
def _get_dbscan_results(feature_type: str) -> tuple[np.ndarray, np.ndarray]: ...
```

### 7.3 Séparation des responsabilités web

```
app.py          -> chargement des données, logique de mise en page, appel aux charts
charts.py       -> fabrication des figures Plotly (go.Figure) : aucun accès aux données
```

Cette séparation rend `charts.py` testable de manière indépendante et réutilisable.

---

## 8. Configuration

### 8.1 Classe `Config` (`config/settings.py`)

Toute la configuration est centralisée dans la classe `Config`. Elle est instanciée une fois à l'import de `config/__init__.py`.

**Fichier :** `config/settings.py`  
**Classe :** `Config`

```python
from config import CONFIG

CONFIG.db_path        # Path vers capitolwatch.db
CONFIG.congress_api_key  # Lue depuis la variable d'environnement CONGRESS_API_KEY
CONFIG.debug          # True en développement, False en production
```

### 8.2 Variables d'environnement

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `CONGRESS_API_KEY` | Oui (pipeline) | Clé API Congress.gov pour récupérer les sénateurs |
| `OPENFIGI_API_KEY` | Non | Clé API OpenFIGI pour l'enrichissement des produits |
| `APP_ENV` | Non | `development` (défaut) ou `production` (Docker) |

### 8.3 Fichier `.env`

```dotenv
CONGRESS_API_KEY=votre_clé
OPENFIGI_API_KEY=votre_clé     # optionnel
APP_ENV=development
```

En production Docker, `APP_ENV=production` est défini directement dans `docker-compose.yml`.

---

## 9. Interface en ligne de commande (CLI)

Les deux modules exposent une interface CLI basée sur **Typer**.

### 9.1 CLI `datapipeline`

Point d'entrée : `python -m capitolwatch.datapipeline`

```
capitolwatch.datapipeline
├── full-pipeline --year YEAR [--start DATE] [--end DATE]
│                             [--skip-scraping] [--skip-init]
├── scraping      --year YEAR [--start DATE] [--end DATE]
└── database
    ├── init
    ├── import    [--folder PATH]
    ├── match
    └── parse     [--folder PATH]
```

### 9.2 CLI `analysis`

Point d'entrée : `python -m capitolwatch.analysis`

```
capitolwatch.analysis
├── features      # Construit le feature store
├── evaluate      # Lance les 6 expériences et calcule les métriques
├── analyze       # Interprète et décrit les clusters
├── visualize     # Génère les figures PNG
└── full-pipeline # Enchaîne features -> evaluate -> analyze -> visualize
```


---

## 10. Déploiement Docker

### 10.1 Image multi-étapes (`Dockerfile`)

Le `Dockerfile` utilise un build en deux étapes pour minimiser la taille de l'image finale :

**Fichier :** `Dockerfile`

```
Stage 1 (builder) :
    FROM python:3.9-slim
    pip install -r requirements.txt --prefix=/install
    # Installe les dépendances dans un répertoire isolé

Stage 2 (runtime) :
    FROM python:3.9-slim
    COPY --from=builder /install /usr/local
    # Copie uniquement les packages installés, sans les outils de build
    COPY capitolwatch/ config/ pytest.ini
    USER appuser  # Exécution en utilisateur non-root
    CMD streamlit run capitolwatch/web/app.py ...
```

**Avantages du multi-stage :**
- L'image finale ne contient pas `pip` ni les caches de téléchargement.
- L'exécution en utilisateur non-root (`appuser`) respecte le principe du moindre privilège.

### 10.2 Volumes Docker

**Fichier :** `docker-compose.yml`

```yaml
volumes:
  - ./data/capitolwatch.db:/app/data/capitolwatch.db:ro
  - ./data/outputs:/app/data/outputs:ro
```

La base de données et les features pré-calculées sont montées en lecture seule (`:ro`). Le contenant ne modifie jamais les données source ; il les consulte uniquement.

### 10.3 Commandes Docker

```bash
# Construire et démarrer le tableau de bord
docker compose up

# Reconstruire l'image après modification du code
docker compose up --build

# Arrêter les conteneurs
docker compose down
```

L'image est accessible sur [http://localhost:8501](http://localhost:8501).

---

## 11. Tests

### 11.1 Configuration pytest


**Fichier :** `pytest.ini`

```ini
# pytest.ini
[pytest]
testpaths = tests
```

### 11.2 Exécution des tests

```bash
# Lancer tous les tests
pytest

# Avec rapport de couverture
pytest --cov=capitolwatch --cov-report=term-missing
```

### 11.3 Couverture des tests

| Fichier de test | Module testé |
|----------------|--------------|
| `test_data_loader.py` | `analysis/data_loader.py` | 
| `test_feature_engineering.py` | `analysis/feature_engineering.py` | 
| `test_preprocessing.py` | `analysis/preprocessing.py` | 
| `test_evaluation.py` | `analysis/evaluation.py` |
| `test_kmeans.py` | `clustering/kmeans.py` | 
| `test_dbscan.py` | `clustering/dbscan.py` | 
| `test_som.py` | `clustering/som.py` | 
| `test_scraper.py` | `datapipeline/scraping/scraper.py` | 
| `test_downloader.py` | `datapipeline/scraping/downloader.py` | 
| `test_extractor.py` | `datapipeline/database/` | 
| `test_pipeline.py` | Pipeline d'intégration | 

### 11.4 Injection de dépendances dans les tests

Les fonctions de service acceptent un paramètre `config` optionnel, ce qui permet de substituer une configuration de test pointant vers une base SQLite en mémoire (`:memory:`) sans modifier le code de production.

---

## 12. Dépendances

### 12.1 Dépendances runtime (`requirements.txt`)

Ces dépendances sont nécessaires pour faire tourner le tableau de bord et le module d'analyse. Elles sont installées dans le conteneur Docker.

| Bibliothèque | Version | Licence | Rôle | Justification du choix |
|--------------|---------|---------|------|------------------------|
| `python-dotenv` | 1.1.1 | BSD-3 | Chargement des variables d'environnement depuis `.env` | Standard de facto pour la gestion de configuration ; évite de coder les clés en dur |
| `pytest` | 8.3.5 | MIT | Framework de tests unitaires | Framework de test Python le plus adopté ; syntaxe concise, fixtures puissantes |
| `pytest-cov` | 6.1.0 | MIT | Mesure de couverture de code | Plugin officiel pytest pour l'intégration avec `coverage.py` |
| `joblib` | 1.5.1 | BSD-3 | Sérialisation des matrices NumPy | Optimisé pour les tableaux NumPy (compression mémoire partagée) ; alternative à `pickle` pour les données ML |
| `MiniSom` | 2.3.5 | MIT | Algorithme Self-Organizing Map | Seule implémentation Python légère et bien maintenue de Kohonen SOM ; interface proche de scikit-learn |
| `numpy` | 1.26.4 | BSD-3 | Calcul matriciel | Base numérique de l'écosystème Python scientifique ; requis par scikit-learn et MiniSom |
| `pandas` | 2.3.2 | BSD-3 | Manipulation de DataFrames | Standard pour le traitement tabulaire en Python ; intégration native avec SQLite via `read_sql` |
| `scikit-learn` | 1.6.1 | BSD-3 | K-Means, DBSCAN, normalisation, métriques | Bibliothèque ML de référence : API uniforme, algorithmes validés, documentation exhaustive |
| `matplotlib` | 3.9.4 | PSF | Graphiques statiques (PNG pour rapports) | Standard pour la génération de figures publication-quality en Python |
| `seaborn` | 0.13.2 | BSD-3 | Heatmaps et style avancé | Surcouche à matplotlib offrant des heatmaps annotées et des palettes perceptuellement uniformes |
| `plotly` | 6.3.0 | MIT | Graphiques interactifs dans Streamlit | Intégration native avec Streamlit via `st.plotly_chart()` ; zoom, survol et export PNG intégrés |
| `streamlit` | 1.50.0 | Apache-2.0 | Tableau de bord web | Transforme des scripts Python en applications web sans HTML/CSS/JS ; `@st.cache_data` pour la gestion du cache |

### 12.2 Dépendances pipeline (`requirements-pipeline.txt`)

Ces dépendances sont nécessaires uniquement pour exécuter le pipeline de collecte et d'ingestion. Elles ne sont pas installées dans le conteneur Docker.

| Bibliothèque | Version | Licence | Rôle | Justification du choix |
|--------------|---------|---------|------|------------------------|
| `beautifulsoup4` | 4.13.4 | MIT | Parsing HTML des rapports financiers | Bibliothèque de référence pour l'analyse HTML ; tolérance aux malformations des pages HTML du Congrès |
| `selenium` | 4.34.2 | Apache-2.0 | Automatisation du navigateur Chrome pour le scraping | Le portail EFD génère ses résultats via JavaScript ; `requests` seul ne peut pas accéder au DOM dynamique |
| `typer` | 0.21.1 | MIT | Construction des CLI | API déclarative basée sur les annotations de type Python ; génère automatiquement l'aide `--help` |
| `requests` | 2.32.5 | Apache-2.0 | Requêtes HTTP vers l'API Congress.gov et OpenFIGI | Standard HTTP Python ; gestion des sessions, timeouts et retry |
| `yfinance` | 0.2.65 | Apache-2.0 | Enrichissement des produits (secteur, industrie, ticker) | Accès gratuit aux données Yahoo Finance ; alternative légère aux APIs financières payantes |
| `boto3` | 1.40.24 | Apache-2.0 | Client AWS SDK (stockage optionnel) | Présent pour un éventuel archivage des rapports HTML dans S3 |
| `NAMEMATCHING` | HEAD | Apache-2.0 | Correspondance des noms de politiciens | Module interne (dépôt Seizh7/NAMEMATCHING) ; gère les accents, abréviations et variantes orthographiques. Le code se replie sur une similarité de chaînes basique si la bibliothèque est absente |

---

## 13. Interactions entre bibliothèques dans le pipeline

Cette section décrit comment les bibliothèques collaborent concrètement à chaque étape du système.

### Étape 1. Scraping : Selenium + BeautifulSoup4

```
Chrome (headless)
    ↑ contrôlé par
Selenium (driver.py)
    │ soumet le formulaire de recherche, navigue
    ↓ retourne le HTML de la page de résultats
BeautifulSoup4 (scraper.py)
    │ parse le DOM, extrait les liens href du tableau #filedReports
    ↓
Liste d'URLs -> downloader.py -> fichiers HTML locaux
```

Selenium gère le rendu JavaScript du portail EFD. BeautifulSoup4 analyse ensuite le HTML statique résultant. Les deux bibliothèques ne s'appellent pas directement : Selenium produit une chaîne HTML que BeautifulSoup4 consomme.

### Étape 2. Ingestion : BeautifulSoup4 + sqlite3

```
HTML local
    ↓ parsé par
BeautifulSoup4 (parse_report_assets.py)
    │ localise le tableau #grid_items, extrait (name, type, subtype, owner, value)
    ↓
sqlite3 (via services/assets.py, services/products.py)
    │ INSERT INTO assets / products
    ↓
capitolwatch.db
```

### Étape 3. Feature engineering : sqlite3 + pandas + numpy

```
capitolwatch.db
    ↓ requête SQL via sqlite3 + pandas.read_sql
pandas DataFrame (data_loader.py)
    │ groupby(), unstack(), fillna()
    ↓
numpy ndarray (via .values)
    │ opérations matricielles
    ↓
pandas DataFrame (matrices freq_baseline, freq_weighted)
    ↓ joblib.dump
data/outputs/*.pkl
```

pandas est utilisé pour la manipulation des données (regroupement par `politician_id` et `subtype`) ; numpy intervient pour les opérations numériques pures. joblib finalise la sérialisation.

### Étape 4. Normalisation : pandas + scikit-learn

```
pandas DataFrame (matrix)
    ↓ .values -> numpy ndarray
scikit-learn StandardScaler / MinMaxScaler
    │ fit_transform(matrix.values)
    ↓ numpy ndarray
pandas DataFrame (reconstruction avec index et colonnes préservés)
```

Le passage entre pandas et scikit-learn s'effectue via `.values` (extraction du tableau numpy sous-jacent). Le DataFrame est reconstruit après transformation pour conserver les noms de colonnes (subtypes) et l'index (identifiants des politiciens).

### Étape 5. Clustering : numpy + scikit-learn / MiniSom

```
numpy ndarray (matrix normalisée)
    ↓
sklearn.cluster.KMeans.fit()      -> labels_ (numpy ndarray)
sklearn.cluster.DBSCAN.fit()      -> labels_ (numpy ndarray, -1 = outlier)
minisom.MiniSom.train_random()    -> BMU 
    ↓
sklearn.cluster.KMeans (sur les poids des neurones)  -> labels_
```

MiniSom accepte directement des tableaux numpy. La post-extraction des clusters du SOM réutilise K-Means de scikit-learn appliqué aux poids des neurones (vecteurs de dimension 37), ce qui crée une dépendance indirecte entre MiniSom et scikit-learn.

### Étape 6. Évaluation : numpy + scikit-learn

```
numpy ndarray (matrix) + numpy ndarray (labels)
    ↓
sklearn.metrics.silhouette_score()
sklearn.metrics.adjusted_rand_score()
sklearn.metrics.normalized_mutual_info_score()
sklearn.metrics.homogeneity_completeness_v_measure()
    ↓
dict de métriques -> pandas DataFrame -> CSV (pandas.to_csv)
```

### Étape 7. Visualisation : numpy + sklearn + matplotlib/seaborn/plotly + streamlit

```
numpy ndarray (matrix) + labels
    ↓
sklearn.decomposition.PCA (réduction 37D -> 2D)
    ↓
matplotlib.pyplot / seaborn (figures statiques PNG)
plotly.graph_objects (figures interactives go.Figure)
    ↓
streamlit.plotly_chart() / st.pyplot() (affichage dans le navigateur)
```

PCA de scikit-learn est le point de jonction entre les données ML et les bibliothèques de visualisation. Matplotlib et Plotly reçoivent les mêmes coordonnées 2D mais produisent des sorties de nature différente : fichiers PNG (pour les rapports PDF) et objets interactifs (pour le dashboard).
