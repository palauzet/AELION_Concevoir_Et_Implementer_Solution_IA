"""Connexion PostgreSQL + gestion des schémas médaillon (US 1.2, C1/C3).

Cible l'instance du conteneur Docker fourni (cf. ``config.DB_URL``, driver psycopg 3).
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from indusense import config


class Base(DeclarativeBase):
    """Base déclarative commune à tous les modèles ORM du projet."""


def get_engine(echo: bool = False) -> Engine:
    """Engine SQLAlchemy vers la base PostgreSQL du projet."""
    return create_engine(config.DB_URL, echo=echo)


SessionLocal = sessionmaker()


def ensure_schema(engine: Engine, schema: str) -> None:
    """Crée le schéma s'il n'existe pas (idempotent).

    Le nom de schéma provient de constantes internes ; il est inséré en littéral
    (``CREATE SCHEMA`` n'accepte pas de paramètre lié) avec échappement défensif
    des guillemets doubles.
    """
    ident = schema.replace('"', '""')
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{ident}"'))


def create_bronze_schema_and_tables(engine: Engine) -> None:
    """Crée le schéma bronze puis les tables ORM manquantes."""
    # Import local pour enregistrer les modèles sur Base.metadata.
    from indusense.data import models  # noqa: F401

    ensure_schema(engine, config.SCHEMA_BRONZE)
    Base.metadata.create_all(engine)
