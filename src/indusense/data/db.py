"""Connexion SQL Server + gestion des schémas médaillon (US 1.2, C1/C3).

Auth Windows intégrée vers l'instance locale (cf. ``config.DB_URL``).
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from indusense import config


class Base(DeclarativeBase):
    """Base déclarative commune à tous les modèles ORM du projet."""


def get_engine(echo: bool = False) -> Engine:
    """Engine SQLAlchemy vers AELION_SPRINT01.

    ``fast_executemany`` accélère fortement les inserts en masse (pyodbc).
    """
    return create_engine(config.DB_URL, echo=echo, fast_executemany=True)


SessionLocal = sessionmaker()


def ensure_schema(engine: Engine, schema: str) -> None:
    """Crée le schéma s'il n'existe pas (idempotent).

    Le nom de schéma provient de constantes internes ; il est inséré en
    littéral (CREATE SCHEMA doit être seul dans son batch, d'où l'EXEC) avec
    échappement défensif des crochets et apostrophes.
    """
    ident = schema.replace("]", "]]")
    literal = schema.replace("'", "''")
    stmt = text(
        f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{literal}') "
        f"EXEC('CREATE SCHEMA [{ident}]')"
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def create_bronze_schema_and_tables(engine: Engine) -> None:
    """Crée le schéma bronze puis les tables ORM manquantes."""
    # Import local pour enregistrer les modèles sur Base.metadata.
    from indusense.data import models  # noqa: F401

    ensure_schema(engine, config.SCHEMA_BRONZE)
    Base.metadata.create_all(engine)
