"""Chemins, connexion PostgreSQL et constantes du projet.

Les artefacts bruts fournis (kit de départ) restent à la racine du projet.
La donnée est organisée en architecture *médaillon* : schéma ``bronze`` pour
la donnée brute fidèle à la source (silver / gold viendront aux US suivantes).

La base cible est l'instance PostgreSQL du conteneur Docker fourni par le
formateur (``docker/pgdocker``). Les identifiants ci-dessous reprennent ceux du
``docker-compose`` ; surcharge possible via les variables ``INDUSENSE_PG_*`` ou
une URL complète ``INDUSENSE_DB_URL``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL

load_dotenv()

# --- Arborescence -----------------------------------------------------------
# Racine du dépôt = deux niveaux au-dessus de ce fichier (src/indusense/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Architecture des données (convention Cookiecutter Data Science).
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"  # données brutes, immuables (read-only)
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

# Artefacts bruts fournis (cf. PROJECT.md § Artefacts), dans data/raw/.
RAW_TELEMETRY = DATA_RAW / "telemetry.csv"
RAW_INCIDENTS = DATA_RAW / "releves_incidents.csv"
RAW_MACHINE_SQL = DATA_RAW / "machine.sql"

# Gold Dataset final (US 1.5).
GOLD_DATASET = DATA_PROCESSED / "gold_dataset.parquet"

# Rapports & contrat de schéma (structure des données brutes, versionné).
REPORTS_DIR = PROJECT_ROOT / "reports"
SCHEMA_REFERENCE = REPORTS_DIR / "schema_reference.json"

# --- Artefacts d'ingestion (runs horodatés, versionnés) ---------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ANALYSE_INCIDENTS_DIR = ARTIFACTS_DIR / "analyses" / "incidents"
ANALYSE_TELEMETRY_DIR = ARTIFACTS_DIR / "analyses" / "telemetry"
ANALYSE_REFERENTIEL_DIR = ARTIFACTS_DIR / "analyses" / "referentiel"
SILVER_DIR = ARTIFACTS_DIR / "silver"
GOLD_DIR = ARTIFACTS_DIR / "gold"
ML_DIR = ARTIFACTS_DIR / "ml"
RUN_TS_FORMAT = "%Y%m%d%H%M"  # dossier de run : AAAAMMJJHHMM

# --- Entraînement ML (B5) ----------------------------------------------------
# Tracking MLflow local (backend SQLite, non versionné — cf. .gitignore).
MLFLOW_TRACKING_URI = f"sqlite:///{(ML_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "indusense-maintenance-ml"
ML_DEFAULT_HORIZON = 24  # horizon par défaut (h) — cf. LABEL_HORIZONS
ML_DECISION_THRESHOLD = 0.5  # seuil de décision par défaut (choix motivé -> B7)

# --- Optimisation d'hyperparamètres (B7) -------------------------------------
# Backend Optuna local (SQLite, non versionné — cf. .gitignore), séparé de mlflow.db.
TUNING_DIR = ARTIFACTS_DIR / "tuning"
OPTUNA_STORAGE = f"sqlite:///{(TUNING_DIR / 'optuna.db').as_posix()}"
TUNING_N_TRIALS = 50  # budget par défaut (décision utilisateur, pas de timeout)
TUNING_N_SPLITS = 5   # CV train-only (aligné sur la convention n_splits=5 du projet)

# --- Explicabilité SHAP (B8) --------------------------------------------------
EXPLAIN_DIR = ARTIFACTS_DIR / "explain"
EXPLAIN_TOP_N = 15  # nombre de features affichées dans les graphiques d'importance
# Sous-échantillon d'AFFICHAGE du nuage SHAP (beeswarm) — le calcul |SHAP| moyen reste sur
# le test complet, seul le nuage de points est sous-échantillonné pour la lisibilité/poids du SVG.
EXPLAIN_BEESWARM_SAMPLE = 2000

# Fenêtres glissantes (heures) des features de dynamique/dégradation du gold, et horizons
# (heures) des labels « incident à venir ». Alignés volontairement (cf. décision projet).
ROLLING_WINDOWS = (6, 12, 24, 48)
LABEL_HORIZONS = (6, 12, 24, 48)

# Mesures capteurs de la télémétrie (ordre stable pour les figures/corrélations).
# `pieces_produced` est traité à part comme variable de production (sortie machine).
TELEMETRY_SENSORS = (
    "temperature_c",
    "pressure_bar",
    "voltage_mean_v",
    "rotation_mean_rpm",
)
TELEMETRY_PRODUCTION = "pieces_produced"

# Unités des capteurs (dimension ``silver.sensor`` : un capteur = une ligne).
# Ajouter/retirer un capteur = modifier ce mapping, sans toucher au schéma silver.
SENSOR_UNITS = {
    "temperature_c": "°C",
    "pressure_bar": "bar",
    "voltage_mean_v": "V",
    "rotation_mean_rpm": "rpm",
}

# Les 9 signaux d'incident (flags type_*), ordre stable pour le dataset/figures.
INCIDENT_SIGNALS = (
    "type_surchauffe",
    "type_baisse_pression",
    "type_vibration",
    "type_bruit_mecanique",
    "type_surconsommation",
    "type_blocage_mecanique",
    "type_alarme_capteur",
    "type_arret_urgence",
    "type_defaut_qualite",
)

# --- PostgreSQL -------------------------------------------------------------
# Valeurs par défaut alignées sur docker/pgdocker/.env (conteneur du formateur).
PG_HOST = os.getenv("INDUSENSE_PG_HOST", "localhost")
PG_PORT = int(os.getenv("INDUSENSE_PG_PORT", "5432"))
PG_DATABASE = os.getenv("INDUSENSE_PG_DATABASE", "indusense_db")
PG_USER = os.getenv("INDUSENSE_PG_USER", "indusense_user")
# Pas de secret en dur : le mot de passe doit venir de .env (gitignoré) ou de l'env.
PG_PASSWORD = os.getenv("INDUSENSE_PG_PASSWORD", "")

# Schémas de l'architecture médaillon.
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD = "gold"

# URL SQLAlchemy (driver psycopg 3). ``URL.create`` gère l'échappement du mot de
# passe (caractères spéciaux comme « @ »). Surchargeable via INDUSENSE_DB_URL.
DB_URL: str | URL = os.getenv("INDUSENSE_DB_URL") or URL.create(
    "postgresql+psycopg",
    username=PG_USER,
    password=PG_PASSWORD,
    host=PG_HOST,
    port=PG_PORT,
    database=PG_DATABASE,
)
