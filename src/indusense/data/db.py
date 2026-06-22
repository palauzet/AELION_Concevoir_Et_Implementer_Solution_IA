"""Connexion PostgreSQL + gestion des schémas médaillon (US 1.2, C1/C3).

Cible l'instance du conteneur Docker fourni (cf. ``config.DB_URL``, driver psycopg 3).
"""

from __future__ import annotations

import pandas as pd
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


def read_bronze(table: str, engine: Engine | None = None) -> pd.DataFrame:
    """Lit une table du schéma ``bronze`` dans un DataFrame.

    Point d'entrée **unique** des analyses en aval (médaillon) : la donnée brute n'a
    qu'un seul lecteur, l'ingestion ``raw -> bronze`` ; tout le reste lit ``bronze``.
    Le nom de table provient de constantes internes (inséré en littéral, échappé).
    """
    ident = table.replace('"', '""')
    with (engine or get_engine()).connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM "{config.SCHEMA_BRONZE}"."{ident}"'), conn)


def ensure_schema(engine: Engine, schema: str) -> None:
    """Crée le schéma s'il n'existe pas (idempotent).

    Le nom de schéma provient de constantes internes ; il est inséré en littéral
    (``CREATE SCHEMA`` n'accepte pas de paramètre lié) avec échappement défensif
    des guillemets doubles.
    """
    ident = schema.replace('"', '""')
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{ident}"'))
