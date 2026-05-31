# Guide Utilisateur : CapitolWatch

**Version :** 1.0  
**Licence :** Apache License 2.0  
**Auteur :** Seizh7  
**Date :** 18 mai 2026

---

## Table des matières

1. [Présentation](#1-présentation)
2. [Prérequis système](#2-prérequis-système)
3. [Installation](#3-installation)
4. [Structure du projet](#4-structure-du-projet)
5. [Démarrage rapide](#5-démarrage-rapide)
6. [Pipeline de données (CLI)](#6-pipeline-de-données-cli)
7. [Pipeline d'analyse (CLI)](#7-pipeline-danalyse-cli)
8. [Tableau de bord web](#8-tableau-de-bord-web)
9. [Déploiement via Docker](#9-déploiement-via-docker)
10. [Configuration](#10-configuration)
11. [Données produites](#11-données-produites)
12. [Tests](#12-tests)
13. [Dépannage](#13-dépannage)
14. [Dépendances](#14-dépendances)

---

## 1. Présentation

CapitolWatch est un outil d'analyse des investissements financiers des élus américains. Il collecte automatiquement les déclarations financières annuelles publiées sur le portail officiel du Congrès (eFD, *Electronic Financial Disclosure*), les ingère dans une base de données, puis applique trois algorithmes de clustering non supervisé pour identifier des profils-types d'investissement.

Les résultats sont consultables via un tableau de bord web interactif (Streamlit) ou générés en fichiers (PNG, CSV, Markdown) en ligne de commande.

### Ce que fait CapitolWatch

| Étape | Description | Outil |
|-------|-------------|-------|
| Collecte | Téléchargement des rapports HTML depuis efdsearch.senate.gov | CLI `datapipeline` |
| Ingestion | Parsing HTML -> base SQLite | CLI `datapipeline` |
| Ingénierie des caractéristiques | Construction des matrices de features (vecteurs d'investissement) | CLI `analysis` |
| Clustering | K-Means, DBSCAN, SOM | CLI `analysis` |
| Évaluation | Métriques internes (Silhouette) et externes (ARI, NMI) | CLI `analysis` |
| Visualisation | Graphiques PCA, heatmaps, U-Matrix | CLI `analysis` ou tableau de bord |
| Dashboard | Interface web interactive | Docker ou Streamlit local |

---

## 2. Prérequis système

### Exécution du tableau de bord uniquement (Docker)

- [Docker](https://www.docker.com/) 20.10 ou supérieur
- [Docker Compose](https://docs.docker.com/compose/) v2
- 2 Go de RAM disponibles
- Port **8501** libre sur la machine hôte

### Exécution complète (pipeline + analyse)

- Python **3.9** ou supérieur
- Git
- Chrome ou Chromium (pour le scraping Selenium)
- ChromeDriver compatible avec la version de Chrome installée
- Clé API **Congress.gov** (gratuite, inscription sur [api.congress.gov](https://api.congress.gov/))
- Clé API **OpenFIGI** (optionnelle, pour l'enrichissement des produits financiers)
- 2 Go d'espace disque disponibles

---

## 3. Installation

### 3.1 Cloner le dépôt

```bash
git clone https://github.com/Seizh7/CAPITOLWATCH.git
cd CAPITOLWATCH
```

### 3.2 Créer un environnement virtuel

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3.3 Installer les dépendances

**Pour le tableau de bord et l'analyse uniquement :**

```bash
pip install -r requirements.txt
```

**Pour le pipeline de données complet (scraping inclus) :**

```bash
pip install -r requirements.txt
pip install -r requirements-pipeline.txt
```

> `requirements-pipeline.txt` inclut Selenium, BeautifulSoup4, les clients d'API financières et le module de correspondance de noms.

### 3.4 Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet :

```dotenv
CONGRESS_API_KEY=votre_clé_congress_api
OPENFIGI_API_KEY=votre_clé_openfigi      # optionnel
APP_ENV=development
```

> En production (Docker), `APP_ENV=production` est déjà défini dans `docker-compose.yml`.

---

## 4. Structure du projet

```
CAPITOLWATCH/
├── capitolwatch/
│   ├── analysis/          # Algorithmes de clustering et évaluation
│   │   └── clustering/    # K-Means, DBSCAN, SOM
│   ├── datapipeline/      # Collecte et ingestion des données
│   │   ├── scraping/      # Téléchargement des rapports HTML
│   │   └── database/      # Parsing et stockage en base
│   ├── services/          # Couche d'accès à la base de données
│   └── web/               # Tableau de bord Streamlit
├── config/                # Configuration centralisée
├── data/
│   ├── capitolwatch.db    # Base SQLite (générée par le pipeline)
│   └── outputs/           # Features et résultats d'évaluation
├── tests/                 # Tests unitaires
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-pipeline.txt
```


---

## 5. Démarrage rapide

### Scénario A : Lancement simplifié avec le script `run.sh` (Recommandé)

Un script `run.sh` est fourni pour automatiser le lancement de l'application via Docker. Il s'assure que Docker est en cours d'exécution, puis construit les images et lance les conteneurs.

```bash
# Rendre le script exécutable (une seule fois)
chmod +x run.sh

# Lancer l'application
./run.sh
```

Le script exécute `docker compose up --build -d` en arrière-plan. Le tableau de bord sera disponible sur [http://localhost:8501](http://localhost:8501).

Pour arrêter l'application :
```bash
docker compose down
```

> **Prérequis :** Ce script nécessite que les données aient déjà été calculées (voir [Scénario C](#scénario-c-je-veux-recalculer-lanalyse-complète-depuis-la-base-existante) et [D](#scénario-d-je-veux-tout-relancer-depuis-zéro-scraping-inclus)). Assurez-vous que `data/capitolwatch.db` et les fichiers dans `data/outputs/` sont présents.

### Scénario B : Je veux juste voir le tableau de bord (données pré-calculées)

Si vous ne souhaitez pas utiliser le script `run.sh`, vous pouvez lancer manuellement le conteneur Docker.

```bash
# Vérifier que data/capitolwatch.db et data/outputs/ sont présents
docker compose up
```

Ouvrir [http://localhost:8501](http://localhost:8501) dans un navigateur.

### Scénario C : Je veux recalculer l'analyse complète depuis la base existante

```bash
# Étape 1 : Construire les features
python -m capitolwatch.analysis features

# Étape 2 : Lancer tout le pipeline d'analyse
python -m capitolwatch.analysis full-pipeline
```

### Scénario D : Je veux tout relancer depuis zéro (scraping inclus)

```bash
# Étape 1 : Collecter et ingérer les données
python -m capitolwatch.datapipeline full-pipeline --year 2023

# Étape 2 : Analyser
python -m capitolwatch.analysis full-pipeline
```

---

## 6. Pipeline de données (CLI)

Le module `datapipeline` collecte les rapports financiers et construit la base SQLite. Il est accessible via :

```bash
python -m capitolwatch.datapipeline [COMMANDE] [OPTIONS]
```

### 6.1 Workflow complet en une commande

```bash
python -m capitolwatch.datapipeline full-pipeline --year 2023
```

**Options disponibles :**

| Option | Raccourci | Description | Exemple |
|--------|-----------|-------------|---------|
| `--year` | `-y` | Année cible des rapports *(obligatoire)* | `--year 2023` |
| `--start` | `-s` | Date de début de recherche (MM/DD/YYYY) | `--start 01/01/2024` |
| `--end` | `-e` | Date de fin de recherche (MM/DD/YYYY) | `--end 12/31/2024` |
| `--skip-scraping` | / | Passer l'étape de téléchargement | `--skip-scraping` |
| `--skip-init` | / | Passer l'initialisation de la base | `--skip-init` |

**Étapes exécutées :**

1. Téléchargement des rapports HTML depuis efdsearch.senate.gov
2. Initialisation de la base SQLite et import des sénateurs (Congress API)
3. Import des rapports dans la base
4. Correspondance des politiciens aux rapports
5. Extraction des actifs depuis les HTML

### 6.2 Commandes individuelles

#### Scraping

```bash
python -m capitolwatch.datapipeline.scraping --year 2023
python -m capitolwatch.datapipeline.scraping --year 2023 --start 01/01/2024 --end 06/30/2024
```

Télécharge les rapports HTML dans le dossier configuré (par défaut `data/annual_reports_<year>/`).

#### Gestion de la base de données

```bash
# Initialiser la base et importer les sénateurs depuis l'API Congress
python -m capitolwatch.datapipeline.database init

# Importer les rapports HTML dans la base
python -m capitolwatch.datapipeline.database import
python -m capitolwatch.datapipeline.database import --folder data/annual_reports_2023

# Associer chaque rapport à un politicien
python -m capitolwatch.datapipeline.database match

# Extraire les actifs financiers des rapports HTML
python -m capitolwatch.datapipeline.database parse
python -m capitolwatch.datapipeline.database parse --folder data/annual_reports_2023
```

Ces commandes peuvent aussi être invoquées via le CLI principal :

```bash
python -m capitolwatch.datapipeline scraping --year 2023
python -m capitolwatch.datapipeline database init
python -m capitolwatch.datapipeline database import
python -m capitolwatch.datapipeline database match
python -m capitolwatch.datapipeline database parse
```

---

## 7. Pipeline d'analyse (CLI)

Le module `analysis` applique les algorithmes de clustering et génère les résultats. Il nécessite une base `capitolwatch.db` peuplée (voir section 6).

```bash
python -m capitolwatch.analysis [COMMANDE]
```

### 7.1 Workflow complet en une commande

```bash
python -m capitolwatch.analysis full-pipeline
```

Exécute les quatre étapes dans l'ordre : `features` -> `evaluate` -> `analyze` -> `visualize`.

### 7.2 Commandes individuelles

#### `features` : Construire le feature store

```bash
python -m capitolwatch.analysis features
```

Calcule les matrices de features à partir de la base et les sauvegarde dans `data/outputs/` :

- `freq_baseline.pkl` : vecteurs de fréquences d'actifs par subtype (ex: N politiciens × M subtypes)
- `freq_weighted.pkl` : vecteurs pondérés par valeur investie (ex: N politiciens × M subtypes)
- `politician_labels.pkl` : identifiants et partis des politiciens analysés

> Cette étape est requise avant toutes les autres commandes d'analyse.


#### `evaluate` : Évaluer les 6 expériences de clustering

```bash
python -m capitolwatch.analysis evaluate
```

Entraîne K-Means, DBSCAN et SOM sur les deux vectorisations, puis calcule :

- **Métriques internes :** score de Silhouette, nombre de clusters, nombre d'outliers
- **Métriques externes :** ARI (Adjusted Rand Index), NMI (Normalized Mutual Information), V-Measure

Produit :
- `data/outputs/evaluation_results.csv` : métriques internes
- `data/outputs/evaluation_results_external.csv` : métriques externes
- `data/figures/confusion_matrix_*.png` : matrices de confusion clusters × partis

#### `analyze` : Générer les profils narratifs des clusters

```bash
python -m capitolwatch.analysis analyze
```

Pour chacune des 6 expériences, décrit le profil d'investissement de chaque cluster (subtypes dominants, valeur moyenne, politiciens représentatifs).

Produit des rapports Markdown dans `data/figures/cluster_profiles/`.

#### `visualize` : Générer les graphiques statiques

```bash
python -m capitolwatch.analysis visualize
```

Produit dans `data/figures/` :

- Heatmaps des centroïdes par cluster
- Barplots comparatifs des métriques
- Projections PCA 2D avec labels de partis

---

## 8. Tableau de bord web

Le tableau de bord est une application **Streamlit** interactive accessible depuis un navigateur. Il lit les données pré-calculées (base SQLite et fichiers `data/outputs/`); aucun recalcul à la volée n'est effectué lors de la navigation.

### Lancement local (sans Docker)

```bash
streamlit run capitolwatch/web/app.py
```

Ouvrir [http://localhost:8501](http://localhost:8501).

> Nécessite que `data/outputs/evaluation_results.csv`, `data/outputs/evaluation_results_external.csv` et les fichiers `.pkl` soient présents. Lancer `python -m capitolwatch.analysis full-pipeline` au préalable si ce n'est pas le cas.

### Navigation dans le tableau de bord

Le tableau de bord est organisé en **cinq onglets** :

#### Onglet 1 : Comparaison des expériences

Vue d'ensemble des 6 expériences (3 algorithmes × 2 vectorisations) :
- Tableau des métriques (Silhouette, nombre de clusters, outliers)
- Barplots comparatifs pour identifier la meilleure configuration

#### Onglet 2 : Meilleur résultat (DBSCAN)

Analyse détaillée du meilleur clustering (DBSCAN sur `freq_weighted`) :
- Projection PCA 2D : chaque point représente un politicien, colorié par cluster
- Heatmap des centroïdes : profil d'investissement moyen par cluster
- Liste des outliers détectés (politiciens aux investissements atypiques)

Utiliser les menus déroulants pour explorer `freq_baseline` ou `freq_weighted`.

#### Onglet 3 : SOM (Self-Organizing Map)

Visualisation topologique du SOM (grille 7×7) :
- **U-Matrix** : carte de chaleur des distances entre neurones voisins (zones sombres = frontières de clusters)
- **Carte politique** : projection des politiciens sur la grille, colorés par parti


#### Onglet 4 : Évaluation externe

Comparaison des clusters avec les partis politiques :
- Barplots ARI, NMI et V-Measure pour les 6 expériences
- Interprétation : un ARI > 0.3 indique une correspondance significative avec les partis


#### Onglet 5 : Analyse sectorielle

Résultats du clustering DBSCAN appliqué à la vectorisation `sector_baseline` (grands secteurs économiques au lieu des subtypes détaillés) :
- Permet d'identifier des profils sectoriels (technologie, immobilier, énergie…)


---

## 9. Déploiement via Docker

Docker embarque uniquement le tableau de bord web. Le pipeline de données et l'analyse doivent être exécutés localement au préalable pour produire les fichiers nécessaires.

### Prérequis

Les fichiers suivants doivent exister avant de démarrer le conteneur :

```
data/
├── capitolwatch.db              # base SQLite
└── outputs/
    ├── evaluation_results.csv
    ├── evaluation_results_external.csv
    ├── freq_baseline.pkl
    ├── freq_weighted.pkl
    └── politician_labels.pkl
```

### Démarrer le tableau de bord

La méthode recommandée est d'utiliser le script `run.sh` fourni à la racine du projet. Il gère la construction de l'image et le lancement du conteneur pour vous.

```bash
./run.sh
```

Alternativement, vous pouvez utiliser `docker-compose` manuellement :

```bash
docker compose up
```

L'image est construite automatiquement si elle n'existe pas encore. Le tableau de bord est accessible sur [http://localhost:8501](http://localhost:8501).


### Options utiles

```bash
# Construire l'image sans démarrer le conteneur
docker compose build

# Démarrer en arrière-plan
docker compose up -d

# Arrêter le conteneur
docker compose down

# Voir les logs en temps réel
docker compose logs -f
```

### Ce qui est monté en volume (lecture seule)

| Volume local | Chemin dans le conteneur | Accès |
|---|---|---|
| `./data/capitolwatch.db` | `/app/data/capitolwatch.db` | Lecture seule |
| `./data/outputs/` | `/app/data/outputs/` | Lecture seule |

Le code source est copié dans l'image lors du build. Modifier les fichiers Python nécessite de reconstruire l'image (`docker compose build`).

### Sécurité

- Le processus s'exécute sous l'utilisateur non-root `appuser`
- La base de données et les outputs sont montés en lecture seule
- Aucune donnée sensible n'est incluse dans l'image

---

## 10. Configuration

Toute la configuration est centralisée dans `config/settings.py` et pilotée par des variables d'environnement. Il n'y a aucune valeur codée en dur dans le code source.

### Variables d'environnement disponibles

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `CONGRESS_API_KEY` | Clé API Congress.gov *(obligatoire pour `database init`)* | / |
| `OPENFIGI_API_KEY` | Clé API OpenFIGI pour l'enrichissement produits *(optionnel)* | / |
| `APP_ENV` | Environnement (`development` / `production`) | `development` |
| `DB_PATH` | Chemin vers la base SQLite | `data/capitolwatch.db` |
| `OUTPUT_FOLDER` | Dossier des rapports HTML téléchargés | `data/annual_reports_<year>/` |

### Obtenir une clé API Congress.gov

1. Se rendre sur [https://api.congress.gov/](https://api.congress.gov/)
2. Cliquer sur *Sign Up*
3. Compléter le formulaire (usage académique ou personnel)
4. La clé est envoyée par e-mail immédiatement
5. L'ajouter dans le fichier `.env` : `CONGRESS_API_KEY=ma_clé`

---

## 11. Données produites

### Base de données (`data/capitolwatch.db`)

Base SQLite contenant les tables :

| Table | Contenu |
|-------|---------|
| `politicians` | Sénateurs actifs (id, prénom, nom, parti) |
| `reports` | Rapports financiers téléchargés (un par sénateur par an) |
| `assets` | Actifs financiers déclarés |
| `products` | Produits financiers enrichis (nom, secteur) |

### Feature store (`data/outputs/`)

| Fichier | Description | Dimensions |
|---------|-------------|------------|
| `freq_baseline.pkl` | Fréquences d'actifs par subtype | (N politiciens, M subtypes) |
| `freq_weighted.pkl` | Valeurs investies par subtype | (N politiciens, M subtypes) |
| `politician_labels.pkl` | Métadonnées politiciens | (N politiciens, 4 colonnes) |
| `evaluation_results.csv` | Métriques internes des 6 expériences | / |
| `evaluation_results_external.csv` | Métriques externes (ARI, NMI, V-Measure) | / |

### Figures (`data/figures/`)

| Fichier | Description |
|---------|-------------|
| `pca_*.png` | Projections PCA 2D par expérience |
| `heatmap_*.png` | Heatmaps des centroïdes |
| `metrics_*.png` | Barplots comparatifs des métriques |
| `confusion_matrix_*.png` | Matrices de confusion clusters × partis |
| `cluster_profiles/*.md` | Rapports narratifs par cluster |

---

## 12. Tests

Les tests unitaires couvrent les modules principaux (data loader, feature engineering, algorithmes de clustering, pipeline).

```bash
# Lancer tous les tests
pytest

# Avec rapport de couverture
pytest --cov=capitolwatch --cov-report=term-missing

# Lancer un fichier de test spécifique
pytest tests/test_kmeans.py
pytest tests/test_dbscan.py
pytest tests/test_som.py
```

### Suites de tests disponibles

| Fichier | Module testé |
|---------|-------------|
| `test_data_loader.py` | `analysis/data_loader.py` | 
| `test_feature_engineering.py` | `analysis/feature_engineering.py` |
| `test_preprocessing.py` | `analysis/preprocessing.py` |
| `test_evaluation.py` | `analysis/evaluation.py` |
| `test_kmeans.py` | `clustering/kmeans.py` |
| `test_dbscan.py` | `clustering/dbscan.py` | 
| `test_som.py` | `clustering/som.py` |
| `test_pipeline.py` | Pipeline complet |
| `test_scraper.py` | `datapipeline/scraping/` |
| `test_extractor.py` | `datapipeline/database/` |
| `test_downloader.py` | `datapipeline/scraping/` |

---

## 13. Dépannage

### Le tableau de bord Docker ne démarre pas

**Symptôme :** Le conteneur s'arrête immédiatement ou affiche une erreur au démarrage.

**Vérification :**
```bash
docker compose logs
```

**Cause fréquente :** Les fichiers `data/outputs/evaluation_results.csv` ou `data/outputs/evaluation_results_external.csv` sont absents.

**Solution :** Exécuter le pipeline d'analyse en local avant de démarrer Docker :
```bash
python -m capitolwatch.analysis full-pipeline
docker compose up
```

---

### Erreur `ModuleNotFoundError` lors du scraping

**Symptôme :**
```
ModuleNotFoundError: No module named 'selenium'
```

**Cause :** `requirements-pipeline.txt` n'a pas été installé.

**Solution :**
```bash
pip install -r requirements-pipeline.txt
```

---

### Erreur `CONGRESS_API_KEY not set`

**Symptôme :** La commande `database init` échoue avec une erreur d'authentification.

**Solution :** Créer ou compléter le fichier `.env` à la racine du projet :
```dotenv
CONGRESS_API_KEY=votre_clé_ici
```

---

### Erreur ChromeDriver introuvable

**Symptôme :**
```
WebDriverException: 'chromedriver' executable needs to be in PATH
```

**Solution :** Télécharger ChromeDriver depuis [https://chromedriver.chromium.org/](https://chromedriver.chromium.org/) en choisissant la version correspondant à Chrome installé, puis l'ajouter au `PATH` système.

---

### Le feature store est vide ou manquant

**Symptôme :** Les commandes `evaluate`, `analyze` ou `visualize` échouent avec une erreur de fichier introuvable.

**Solution :** Reconstruire le feature store :
```bash
python -m capitolwatch.analysis features
```

---

### Les tests échouent avec `FileNotFoundError`

**Symptôme :** Les tests de clustering échouent car la base de données est absente.

**Cause :** La base `data/capitolwatch.db` n'a pas été générée.

**Solution :** Utiliser la base de développement fournie, ou exécuter le pipeline complet :
```bash
python -m capitolwatch.datapipeline full-pipeline --year 2023
```

---

## 14. Dépendances

Toutes les dépendances sont épinglées à une version précise dans [`requirements.txt`](../requirements.txt) (tableau de bord + analyse) et [`requirements-pipeline.txt`](../requirements-pipeline.txt) (pipeline de données uniquement).

### Dépendances principales (`requirements.txt`)

| Bibliothèque | Version | Rôle |
|---|---|---|
| [scikit-learn](https://scikit-learn.org/) | 1.6.1 | K-Means, DBSCAN, métriques (Silhouette, ARI, NMI) |
| [MiniSom](https://github.com/JustGlowing/minisom) | 2.3.5 | Self-Organizing Map (SOM) |
| [numpy](https://numpy.org/) | 1.26.4 | Tableaux numériques et opérations matricielles |
| [pandas](https://pandas.pydata.org/) | 2.3.2 | Manipulation des données tabulaires |
| [matplotlib](https://matplotlib.org/) | 3.9.4 | Figures statiques (PCA, heatmaps, barplots) |
| [seaborn](https://seaborn.pydata.org/) | 0.13.2 | Mise en forme des visualisations statistiques |
| [plotly](https://plotly.com/python/) | 6.3.0 | Graphiques interactifs dans le tableau de bord |
| [streamlit](https://streamlit.io/) | 1.50.0 | Interface web du tableau de bord |
| [joblib](https://joblib.readthedocs.io/) | 1.5.1 | Sérialisation du feature store (fichiers `.pkl`) |
| [python-dotenv](https://saurabh-kumar.com/python-dotenv/) | 1.1.1 | Chargement des variables d'environnement |

### Dépendances du pipeline (`requirements-pipeline.txt`)

| Bibliothèque | Version | Rôle |
|---|---|---|
| [selenium](https://www.selenium.dev/) | 4.34.2 | Scraping via navigateur headless |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | 4.13.4 | Parsing des rapports HTML |
| [requests](https://requests.readthedocs.io/) | 2.32.5 | Appels HTTP vers les APIs Congress.gov et OpenFIGI |
| [yfinance](https://ranaroussi.github.io/yfinance/) | 0.2.65 | Enrichissement des produits financiers |
| [typer](https://typer.tiangolo.com/) | 0.21.1 | Interface CLI des deux modules |

### Dépendances internes

| Bibliothèque | Version | Rôle | Description |
|---|---|---|---|
| **NAMEMATCHING** | HEAD | Correspondance des noms de politiciens | Module interne (dépôt `Seizh7/NAMEMATCHING`) gérant les accents, abréviations et variantes orthographiques. Le code se replie sur une similarité de chaînes basique si la bibliothèque est absente. |

---

## Licence

Ce logiciel est distribué sous licence **Apache License 2.0**.  
Voir le fichier [LICENSE](../LICENSE) pour le texte complet.

Copyright © 2026 Seizh7
