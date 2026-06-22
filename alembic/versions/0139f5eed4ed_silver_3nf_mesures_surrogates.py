"""silver 3NF mesures + surrogates

Revision ID: 0139f5eed4ed
Revises: 22c40158772a
Create Date: 2026-06-22 19:50:48.614025

Normalisation 3NF du silver (motivée par l'évolutivité capteurs) :
- dimensions à clés surrogates ``silver.model`` (type machine) et ``silver.sensor`` ;
- ``silver.machine`` : PK surrogate ``machine_pk`` (au lieu de ``machine_code``),
  ``machine_code`` clé métier unique, ``model`` (texte) -> ``model_id`` (FK) ;
- mesures normalisées en forme longue : en-tête ``silver.reading`` (FK ``machine_pk``) +
  détail ``silver.measurement`` (FK ``reading`` + ``sensor``) — remplace ``silver.telemetry`` ;
- faits ``maintenance`` / ``incident`` : FK machine via ``machine_pk`` (surrogate).

Le silver étant **régénérable** (TRUNCATE + ``indusense-silver``), on reconstruit les tables
impactées par le changement de PK plutôt que de jongler avec les FK en cascade.
``silver.component`` (lookup) est conservée.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0139f5eed4ed'
down_revision: Union[str, Sequence[str], None] = '22c40158772a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SILVER = "silver"

def _incident_enriched_columns() -> list[sa.Column]:
    """Colonnes communes des incidents enrichis (objets neufs à chaque appel)."""
    return [
        sa.Column('severity', sa.Integer(), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('shift', sa.String(length=16), nullable=True),
        sa.Column('type_surchauffe', sa.Integer(), nullable=False),
        sa.Column('type_baisse_pression', sa.Integer(), nullable=False),
        sa.Column('type_vibration', sa.Integer(), nullable=False),
        sa.Column('type_bruit_mecanique', sa.Integer(), nullable=False),
        sa.Column('type_surconsommation', sa.Integer(), nullable=False),
        sa.Column('type_blocage_mecanique', sa.Integer(), nullable=False),
        sa.Column('type_alarme_capteur', sa.Integer(), nullable=False),
        sa.Column('type_arret_urgence', sa.Integer(), nullable=False),
        sa.Column('type_defaut_qualite', sa.Integer(), nullable=False),
        sa.Column('n_signals', sa.Integer(), nullable=False),
        sa.Column('signal', sa.String(length=32), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('jour', sa.DateTime(), nullable=False),
        sa.Column('semaine_iso', sa.String(length=12), nullable=False),
        sa.Column('weekday', sa.String(length=12), nullable=False),
        sa.Column('coherence', sa.Float(), nullable=False),
        sa.Column('comment_present', sa.Float(), nullable=False),
        sa.Column('machine_valide', sa.Float(), nullable=False),
        sa.Column('severity_valide', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
    ]


def upgrade() -> None:
    """3NF : surrogates machine/model, mesures en forme longue (reading/measurement)."""
    # Tables référençant l'ancienne PK machine (machine_code) -> reconstruites.
    op.drop_table('incident', schema=SILVER)
    op.drop_table('telemetry', schema=SILVER)
    op.drop_table('maintenance', schema=SILVER)
    op.drop_table('machine', schema=SILVER)

    # Dimensions à clés surrogates.
    op.create_table(
        'model',
        sa.Column('model_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint('model_id'),
        sa.UniqueConstraint('name'),
        schema=SILVER,
    )
    op.create_table(
        'sensor',
        sa.Column('sensor_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=32), nullable=False),
        sa.Column('unit', sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint('sensor_id'),
        sa.UniqueConstraint('name'),
        schema=SILVER,
    )

    # Dimension machine : PK surrogate, machine_code clé métier, FK model.
    op.create_table(
        'machine',
        sa.Column('machine_pk', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('machine_code', sa.String(length=16), nullable=False),
        sa.Column('commissioning_date', sa.DateTime(), nullable=False),
        sa.Column('max_daily_capacity', sa.Integer(), nullable=False),
        sa.Column('max_hourly_capacity_pieces', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('production_line', sa.String(length=16), nullable=False),
        sa.Column('location', sa.String(length=16), nullable=False),
        sa.Column('criticality', sa.String(length=8), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('heures_equivalentes_jour', sa.Integer(), nullable=False),
        sa.Column('capacite_incoherente', sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "criticality IN ('LOW', 'MEDIUM', 'HIGH')", name='ck_silver_machine_criticality'
        ),
        sa.ForeignKeyConstraint(
            ['model_id'], [f'{SILVER}.model.model_id'], name='fk_silver_machine_model'
        ),
        sa.PrimaryKeyConstraint('machine_pk'),
        sa.UniqueConstraint('machine_code', name='uq_silver_machine_code'),
        schema=SILVER,
    )

    # Faits télémétrie normalisés : en-tête reading + détail measurement (forme longue).
    op.create_table(
        'reading',
        sa.Column('reading_id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('machine_pk', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('during_maintenance', sa.Boolean(), nullable=False),
        sa.Column('pieces_produced', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ['machine_pk'], [f'{SILVER}.machine.machine_pk'], name='fk_silver_reading_machine'
        ),
        sa.PrimaryKeyConstraint('reading_id'),
        schema=SILVER,
    )
    op.create_table(
        'measurement',
        sa.Column('measurement_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('reading_id', sa.Integer(), nullable=False),
        sa.Column('sensor_id', sa.Integer(), nullable=False),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('was_imputed', sa.Boolean(), nullable=False),
        sa.Column('is_outlier', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ['reading_id'], [f'{SILVER}.reading.reading_id'], name='fk_silver_measurement_reading'
        ),
        sa.ForeignKeyConstraint(
            ['sensor_id'], [f'{SILVER}.sensor.sensor_id'], name='fk_silver_measurement_sensor'
        ),
        sa.PrimaryKeyConstraint('measurement_id'),
        sa.UniqueConstraint('reading_id', 'sensor_id', name='uq_silver_measurement'),
        schema=SILVER,
    )

    # Maintenance : FK machine via surrogate machine_pk (+ component_id, CHECK conservés).
    op.create_table(
        'maintenance',
        sa.Column('maintenance_id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('machine_pk', sa.Integer(), nullable=False),
        sa.Column('maintenance_at', sa.DateTime(), nullable=False),
        sa.Column('maintenance_type', sa.String(length=16), nullable=False),
        sa.Column('component_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('related_incident_id', sa.String(length=16), nullable=True),
        sa.Column('duration_hours', sa.Float(), nullable=False),
        sa.CheckConstraint(
            "maintenance_type IN ('proactive', 'reactive')", name='ck_silver_maintenance_type'
        ),
        sa.ForeignKeyConstraint(
            ['machine_pk'], [f'{SILVER}.machine.machine_pk'], name='fk_silver_maintenance_machine'
        ),
        sa.ForeignKeyConstraint(
            ['component_id'], [f'{SILVER}.component.component_id'],
            name='fk_silver_maintenance_component',
        ),
        sa.PrimaryKeyConstraint('maintenance_id'),
        schema=SILVER,
    )

    # Incident : FK machine via surrogate machine_pk (nullable si code hors référentiel).
    op.create_table(
        'incident',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('incident_id', sa.String(length=16), nullable=True),
        sa.Column('machine_pk', sa.Integer(), nullable=True),
        *_incident_enriched_columns(),
        sa.ForeignKeyConstraint(
            ['machine_pk'], [f'{SILVER}.machine.machine_pk'], name='fk_silver_incident_machine'
        ),
        sa.PrimaryKeyConstraint('id'),
        schema=SILVER,
    )


def downgrade() -> None:
    """Retour à l'état post-normalisation mesurée (star-schema, télémétrie large)."""
    op.drop_table('incident', schema=SILVER)
    op.drop_table('measurement', schema=SILVER)
    op.drop_table('reading', schema=SILVER)
    op.drop_table('maintenance', schema=SILVER)
    op.drop_table('machine', schema=SILVER)
    op.drop_table('sensor', schema=SILVER)
    op.drop_table('model', schema=SILVER)

    # Dimension machine : PK = machine_code, model en texte (star-schema).
    op.create_table(
        'machine',
        sa.Column('machine_code', sa.String(length=16), nullable=False),
        sa.Column('commissioning_date', sa.DateTime(), nullable=False),
        sa.Column('max_daily_capacity', sa.Integer(), nullable=False),
        sa.Column('max_hourly_capacity_pieces', sa.Integer(), nullable=False),
        sa.Column('model', sa.String(length=32), nullable=False),
        sa.Column('production_line', sa.String(length=16), nullable=False),
        sa.Column('location', sa.String(length=16), nullable=False),
        sa.Column('criticality', sa.String(length=8), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('heures_equivalentes_jour', sa.Integer(), nullable=False),
        sa.Column('capacite_incoherente', sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "criticality IN ('LOW', 'MEDIUM', 'HIGH')", name='ck_silver_machine_criticality'
        ),
        sa.PrimaryKeyConstraint('machine_code'),
        schema=SILVER,
    )

    # Télémétrie large (FK machine_code), sans model/criticality.
    op.create_table(
        'telemetry',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('machine_id', sa.String(length=16), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('temperature_c', sa.Float(), nullable=True),
        sa.Column('pressure_bar', sa.Float(), nullable=True),
        sa.Column('voltage_mean_v', sa.Float(), nullable=True),
        sa.Column('rotation_mean_rpm', sa.Float(), nullable=True),
        sa.Column('pieces_produced', sa.Integer(), nullable=False),
        sa.Column('during_maintenance', sa.Boolean(), nullable=False),
        sa.Column('temperature_c_was_imputed', sa.Boolean(), nullable=False),
        sa.Column('pressure_bar_was_imputed', sa.Boolean(), nullable=False),
        sa.Column('voltage_mean_v_was_imputed', sa.Boolean(), nullable=False),
        sa.Column('rotation_mean_rpm_was_imputed', sa.Boolean(), nullable=False),
        sa.Column('temperature_c_is_outlier', sa.Boolean(), nullable=False),
        sa.Column('pressure_bar_is_outlier', sa.Boolean(), nullable=False),
        sa.Column('voltage_mean_v_is_outlier', sa.Boolean(), nullable=False),
        sa.Column('rotation_mean_rpm_is_outlier', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ['machine_id'], [f'{SILVER}.machine.machine_code'],
            name='fk_silver_telemetry_machine',
        ),
        sa.PrimaryKeyConstraint('id'),
        schema=SILVER,
    )

    # Maintenance (FK machine_code), component_id lookup conservé.
    op.create_table(
        'maintenance',
        sa.Column('maintenance_id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('machine_code', sa.String(length=16), nullable=False),
        sa.Column('maintenance_at', sa.DateTime(), nullable=False),
        sa.Column('maintenance_type', sa.String(length=16), nullable=False),
        sa.Column('component_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('related_incident_id', sa.String(length=16), nullable=True),
        sa.Column('duration_hours', sa.Float(), nullable=False),
        sa.CheckConstraint(
            "maintenance_type IN ('proactive', 'reactive')", name='ck_silver_maintenance_type'
        ),
        sa.ForeignKeyConstraint(
            ['machine_code'], [f'{SILVER}.machine.machine_code'],
            name='fk_silver_maintenance_machine',
        ),
        sa.ForeignKeyConstraint(
            ['component_id'], [f'{SILVER}.component.component_id'],
            name='fk_silver_maintenance_component',
        ),
        sa.PrimaryKeyConstraint('maintenance_id'),
        schema=SILVER,
    )

    # Incident (FK machine_id -> machine_code), sans model/criticality.
    op.create_table(
        'incident',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('incident_id', sa.String(length=16), nullable=True),
        sa.Column('machine_id', sa.String(length=16), nullable=True),
        *_incident_enriched_columns(),
        sa.ForeignKeyConstraint(
            ['machine_id'], [f'{SILVER}.machine.machine_code'],
            name='fk_silver_incident_machine',
        ),
        sa.PrimaryKeyConstraint('id'),
        schema=SILVER,
    )
