"""Chemins, connexion SQL Server et constantes du projet.

Les artefacts bruts fournis (kit de départ) restent à la racine du projet.
La donnée est organisée en architecture *médaillon* : schéma ``bronze`` pour
la donnée brute fidèle à la source (silver / gold viendront aux US suivantes).
"""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv

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

# --- SQL Server -------------------------------------------------------------
SQL_SERVER = os.getenv("INDUSENSE_SQL_SERVER", "XANADU-PC03")
SQL_DATABASE = os.getenv("INDUSENSE_SQL_DATABASE", "AELION_SPRINT01")
ODBC_DRIVER = os.getenv("INDUSENSE_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")

# Schémas de l'architecture médaillon.
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD = "gold"


def _odbc_connect_string() -> str:
    return (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes"
    )


# URL SQLAlchemy (auth Windows intégrée). Surchargeable via INDUSENSE_DB_URL.
DB_URL = os.getenv(
    "INDUSENSE_DB_URL",
    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(_odbc_connect_string()),
)
