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

# --- Artefacts d'ingestion (runs horodatés, versionnés) ---------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INGEST_INCIDENTS_DIR = ARTIFACTS_DIR / "ingestions" / "incidents"
RUN_TS_FORMAT = "%Y%m%d%H%M"  # dossier de run : AAAAMMJJHHMM

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
