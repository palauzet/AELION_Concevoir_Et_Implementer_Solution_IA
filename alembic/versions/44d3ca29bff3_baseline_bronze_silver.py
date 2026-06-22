"""baseline bronze + silver

Revision ID: 44d3ca29bff3
Revises: 
Create Date: 2026-06-22 15:17:13.838542

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '44d3ca29bff3'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    op.execute("CREATE SCHEMA IF NOT EXISTS silver")
    op.create_table('incident',
    sa.Column('incident_pk', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('incident_id', sa.String(length=16), nullable=True),
    sa.Column('date', sa.Date(), nullable=True),
    sa.Column('time', sa.String(length=8), nullable=True),
    sa.Column('operator_name', sa.String(length=128), nullable=True),
    sa.Column('machine_id', sa.String(length=16), nullable=True),
    sa.Column('severity', sa.Integer(), nullable=True),
    sa.Column('operator_badge', sa.String(length=16), nullable=True),
    sa.Column('comment', sa.String(), nullable=True),
    sa.Column('shift', sa.String(length=16), nullable=True),
    sa.Column('type_surchauffe', sa.Integer(), nullable=True),
    sa.Column('type_baisse_pression', sa.Integer(), nullable=True),
    sa.Column('type_vibration', sa.Integer(), nullable=True),
    sa.Column('type_bruit_mecanique', sa.Integer(), nullable=True),
    sa.Column('type_surconsommation', sa.Integer(), nullable=True),
    sa.Column('type_blocage_mecanique', sa.Integer(), nullable=True),
    sa.Column('type_alarme_capteur', sa.Integer(), nullable=True),
    sa.Column('type_arret_urgence', sa.Integer(), nullable=True),
    sa.Column('type_defaut_qualite', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('incident_pk'),
    schema='bronze'
    )
    op.create_table('machine',
    sa.Column('machine_code', sa.String(length=16), nullable=False),
    sa.Column('commissioning_date', sa.Date(), nullable=False),
    sa.Column('max_daily_capacity', sa.Integer(), nullable=False),
    sa.Column('max_hourly_capacity_pieces', sa.Integer(), nullable=False),
    sa.Column('model', sa.String(length=32), nullable=False),
    sa.Column('production_line', sa.String(length=16), nullable=False),
    sa.Column('location', sa.String(length=16), nullable=False),
    sa.Column('criticality', sa.String(length=8), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("criticality IN ('LOW', 'MEDIUM', 'HIGH')", name='ck_machine_criticality'),
    sa.CheckConstraint('max_daily_capacity > 0', name='ck_machine_daily_cap'),
    sa.CheckConstraint('max_hourly_capacity_pieces > 0', name='ck_machine_hourly_cap'),
    sa.PrimaryKeyConstraint('machine_code'),
    schema='bronze'
    )
    op.create_index('idx_machine_line', 'machine', ['production_line'], unique=False, schema='bronze')
    op.create_index('idx_machine_location', 'machine', ['location'], unique=False, schema='bronze')
    op.create_table('telemetry',
    sa.Column('telemetry_id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('machine_id', sa.String(length=16), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('temperature_c', sa.Float(), nullable=True),
    sa.Column('pressure_bar', sa.Float(), nullable=True),
    sa.Column('voltage_mean_v', sa.Float(), nullable=True),
    sa.Column('rotation_mean_rpm', sa.Float(), nullable=True),
    sa.Column('pieces_produced', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('telemetry_id'),
    schema='bronze'
    )
    op.create_table('incident',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('incident_id', sa.String(length=16), nullable=True),
    sa.Column('machine_id', sa.String(length=16), nullable=True),
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
    sa.Column('model', sa.String(length=32), nullable=True),
    sa.Column('criticality', sa.String(length=8), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    schema='silver'
    )
    op.create_table('machine',
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
    sa.PrimaryKeyConstraint('machine_code'),
    schema='silver'
    )
    op.create_table('telemetry',
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
    sa.Column('model', sa.String(length=32), nullable=True),
    sa.Column('criticality', sa.String(length=8), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    schema='silver'
    )
    op.create_table('maintenance',
    sa.Column('maintenance_id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('machine_code', sa.String(length=16), nullable=False),
    sa.Column('maintenance_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('maintenance_type', sa.String(length=16), nullable=False),
    sa.Column('action_type', sa.String(length=32), nullable=False),
    sa.Column('component', sa.String(length=64), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('related_incident_id', sa.String(length=16), nullable=True),
    sa.Column('duration_hours', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("maintenance_type IN ('proactive', 'reactive')", name='ck_maintenance_type'),
    sa.CheckConstraint('duration_hours > 0', name='ck_maintenance_duration'),
    sa.ForeignKeyConstraint(['machine_code'], ['bronze.machine.machine_code'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('maintenance_id'),
    schema='bronze'
    )
    op.create_index('idx_maintenance_machine_time', 'maintenance', ['machine_code', 'maintenance_at'], unique=False, schema='bronze')
    op.create_index('idx_maintenance_type', 'maintenance', ['maintenance_type'], unique=False, schema='bronze')
    op.create_table('maintenance',
    sa.Column('maintenance_id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('machine_code', sa.String(length=16), nullable=False),
    sa.Column('maintenance_at', sa.DateTime(), nullable=False),
    sa.Column('maintenance_type', sa.String(length=16), nullable=False),
    sa.Column('component', sa.String(length=64), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('related_incident_id', sa.String(length=16), nullable=True),
    sa.Column('duration_hours', sa.Float(), nullable=False),
    sa.Column('model', sa.String(length=32), nullable=True),
    sa.Column('criticality', sa.String(length=8), nullable=True),
    sa.ForeignKeyConstraint(['machine_code'], ['silver.machine.machine_code'], ),
    sa.PrimaryKeyConstraint('maintenance_id'),
    schema='silver'
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema — rollback complet (schémas médaillon recréables à l'identique)."""
    op.execute("DROP SCHEMA IF EXISTS silver CASCADE")
    op.execute("DROP SCHEMA IF EXISTS bronze CASCADE")
