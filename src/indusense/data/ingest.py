"""Script d'ingestion réutilisable : CSV bruts -> schéma ``bronze`` (US 1.2).

Idempotent : chaque table cible est tronquée avant rechargement, de sorte que
relancer l'import produit toujours le même état.

Usage (une fois l'environnement uv prêt) :

    uv run indusense-ingest --migrate            # charge machine.sql (réf.) + tous les CSV
    uv run indusense-ingest --source telemetry   # recharge seulement la télémétrie
    uv run indusense-ingest --source incidents
    uv run python -m indusense.data.ingest --help
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from indusense import config
from indusense.data.db import get_engine


@dataclass(frozen=True)
class CsvSource:
    """Décrit une source CSV et sa table bronze cible."""

    name: str
    csv_path: Path
    table: str
    parse_dates: list[str] = field(default_factory=list)


SOURCES: dict[str, CsvSource] = {
    "telemetry": CsvSource(
        name="telemetry",
        csv_path=config.RAW_TELEMETRY,
        table="telemetry",
        parse_dates=["timestamp"],
    ),
    "incidents": CsvSource(
        name="incidents",
        csv_path=config.RAW_INCIDENTS,
        table="incident",
        parse_dates=["date"],
    ),
}

def load_reference_data(engine: Engine) -> None:
    """Charge les tables de référence (machine, maintenance) dans ``bronze``.

    Exécute le script PostgreSQL ``data/raw/machine.sql`` (CREATE TABLE + seed,
    idempotent via ``ON CONFLICT``) en positionnant ``search_path`` sur ``bronze``
    pour que les tables non qualifiées y soient créées. Le script gère sa propre
    transaction (``BEGIN``/``COMMIT``), d'où l'exécution en isolation AUTOCOMMIT.
    """
    script = config.RAW_MACHINE_SQL.read_text(encoding="utf-8")
    ident = config.SCHEMA_BRONZE.replace('"', '""')
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(f'SET search_path TO "{ident}", public')
        conn.exec_driver_sql(script)


def _truncate(engine: Engine, table: str) -> None:
    ident_schema = config.SCHEMA_BRONZE.replace('"', '""')
    ident_table = table.replace('"', '""')
    fq = f'"{ident_schema}"."{ident_table}"'
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {fq} RESTART IDENTITY"))


def load_csv_to_bronze(
    engine: Engine,
    source: CsvSource,
    chunksize: int = 10_000,
    truncate: bool = True,
) -> int:
    """Charge un CSV dans sa table bronze. Retourne le nombre de lignes insérées."""
    if not source.csv_path.exists():
        raise FileNotFoundError(source.csv_path)

    if truncate:
        _truncate(engine, source.table)

    total = 0
    reader = pd.read_csv(
        source.csv_path,
        parse_dates=source.parse_dates or None,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.to_sql(
            source.table,
            engine,
            schema=config.SCHEMA_BRONZE,
            if_exists="append",
            index=False,
        )
        total += len(chunk)
    return total


def setup_bronze(engine: Engine, migrate: bool = False) -> None:
    """Charge les données de référence dans bronze (le **schéma est géré par Alembic**).

    Prérequis : ``uv run alembic upgrade head`` (crée les schémas/tables typés). Ici on ne
    crée plus de DDL ; ``machine.sql`` (``load_reference_data``) ne fait qu'**insérer** les
    données (ses ``CREATE TABLE IF NOT EXISTS`` sont des no-op, tables déjà créées).
    """
    if migrate:
        load_reference_data(engine)
        print("Données de référence (machine, maintenance) chargées dans bronze.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion CSV -> schéma bronze (PostgreSQL)."
    )
    parser.add_argument(
        "--source",
        choices=[*SOURCES.keys(), "all"],
        default="all",
        help="Source à charger (défaut: all).",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Charger les données de référence (machine, maintenance) depuis machine.sql.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Ne pas vider la table avant insertion (append pur).",
    )
    args = parser.parse_args(argv)

    engine = get_engine()
    setup_bronze(engine, migrate=args.migrate)

    targets = SOURCES.values() if args.source == "all" else [SOURCES[args.source]]
    for src in targets:
        n = load_csv_to_bronze(engine, src, truncate=not args.no_truncate)
        print(f"{src.name}: {n} lignes -> {config.SCHEMA_BRONZE}.{src.table}")


if __name__ == "__main__":
    main()
