"""Environnement Alembic — InduSense.

L'URL et la metadata viennent du projet (``indusense.config`` / ``indusense.data.models``),
pas de ``alembic.ini``. ``include_schemas=True`` pour gérer les schémas médaillon
(``bronze``, ``silver``).
"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL

from alembic import context
from indusense import config as app_config
from indusense.data import models  # noqa: F401 — enregistre toutes les tables sur Base.metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = models.Base.metadata

# config.DB_URL est un objet URL (ou une str si surcharge INDUSENSE_DB_URL).
# str(URL) masque le mot de passe -> rendre la chaîne en clair pour la connexion.
_DB = app_config.DB_URL
DB_URL = _DB.render_as_string(hide_password=False) if isinstance(_DB, URL) else _DB


def run_migrations_offline() -> None:
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="public",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(DB_URL, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema="public",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
