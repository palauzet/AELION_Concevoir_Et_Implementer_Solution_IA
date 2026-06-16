"""Script d'ingestion réutilisable : CSV bruts -> schéma ``bronze`` (US 1.2).

Idempotent : chaque table cible est tronquée avant rechargement, de sorte que
relancer l'import produit toujours le même état.

Usage (une fois l'environnement uv prêt) :

    uv run indusense-ingest --migrate            # transfère dbo->bronze + charge tout
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
from indusense.data.db import Base, ensure_schema, get_engine


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

# Tables créées hors ORM (par machine.sql) à rapatrier dans bronze.
DBO_TABLES_TO_MIGRATE = ("machine", "maintenance")


def migrate_dbo_to_bronze(
    engine: Engine, tables: tuple[str, ...] = DBO_TABLES_TO_MIGRATE
) -> list[str]:
    """Transfère les tables de ``dbo`` vers ``bronze`` (idempotent).

    Ne fait rien si la table est déjà dans bronze ou absente de dbo.
    """
    moved: list[str] = []
    bronze = config.SCHEMA_BRONZE
    bronze_ident = bronze.replace("]", "]]")
    check = text(
        "SELECT 1 FROM sys.tables t JOIN sys.schemas s ON t.schema_id = s.schema_id "
        "WHERE s.name = :b AND t.name = :t"
    )
    with engine.begin() as conn:
        for t in tables:
            # Noms issus de DBO_TABLES_TO_MIGRATE (constante) : insérés en littéral.
            t_ident = t.replace("]", "]]")
            t_lit = t.replace("'", "''")
            bronze_lit = bronze.replace("'", "''")
            stmt = text(
                "IF EXISTS (SELECT 1 FROM sys.tables tt "
                "JOIN sys.schemas s ON tt.schema_id = s.schema_id "
                f"          WHERE s.name = 'dbo' AND tt.name = N'{t_lit}') "
                "AND NOT EXISTS (SELECT 1 FROM sys.tables tt "
                "JOIN sys.schemas s ON tt.schema_id = s.schema_id "
                f"               WHERE s.name = N'{bronze_lit}' AND tt.name = N'{t_lit}') "
                f"ALTER SCHEMA [{bronze_ident}] TRANSFER [dbo].[{t_ident}]"
            )
            conn.execute(stmt)
            if conn.execute(check, {"t": t, "b": bronze}).first():
                moved.append(t)
    return moved


def _truncate(engine: Engine, table: str) -> None:
    fq = f"[{config.SCHEMA_BRONZE}].[{table}]"
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {fq}"))


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
    """Prépare le schéma bronze : schéma, migration dbo, puis tables ORM."""
    ensure_schema(engine, config.SCHEMA_BRONZE)
    if migrate:
        moved = migrate_dbo_to_bronze(engine)
        if moved:
            print(f"Migration dbo -> bronze : {', '.join(moved)}")
    # Importe les modèles pour les enregistrer sur Base.metadata, puis crée
    # les tables manquantes (telemetry, incident ; machine/maintenance déjà là).
    from indusense.data import models  # noqa: F401

    Base.metadata.create_all(engine)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion CSV -> schéma bronze (AELION_SPRINT01)."
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
        help="Transférer dbo.machine / dbo.maintenance vers bronze avant chargement.",
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
