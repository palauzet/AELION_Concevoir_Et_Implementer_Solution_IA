"""silver measurement is_saturated

Revision ID: f1b64dacb88b
Revises: 0139f5eed4ed
Create Date: 2026-06-22 22:47:15.974600

Flag ``silver.measurement.is_saturated`` (par mesure) : valeur **écrêtée** sur une borne de
plage capteur (saturation), distincte de ``is_outlier``. Ajouté NOT NULL avec un défaut
serveur transitoire (pour les lignes existantes), puis défaut retiré pour coller au modèle ORM
(régénéré ensuite par ``indusense-silver``).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1b64dacb88b'
down_revision: Union[str, Sequence[str], None] = '0139f5eed4ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'measurement',
        sa.Column('is_saturated', sa.Boolean(), nullable=False, server_default=sa.false()),
        schema='silver',
    )
    op.alter_column('measurement', 'is_saturated', server_default=None, schema='silver')


def downgrade() -> None:
    op.drop_column('measurement', 'is_saturated', schema='silver')
