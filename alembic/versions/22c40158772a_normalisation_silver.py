"""normalisation silver

Revision ID: 22c40158772a
Revises: 44d3ca29bff3
Create Date: 2026-06-22 15:46:37.088109

Normalisation mesurée (star-schema) :
- lookup ``silver.component`` (FK depuis maintenance) ;
- retrait des copies redondantes ``model`` / ``criticality`` des faits (dimension machine) ;
- FK ``machine_id`` (telemetry, incident) vers ``silver.machine`` ;
- CHECK sur les enums ``criticality`` / ``maintenance_type`` (ajout manuel : non autogénéré).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '22c40158772a'
down_revision: Union[str, Sequence[str], None] = '44d3ca29bff3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'component',
        sa.Column('component_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('component_id'),
        sa.UniqueConstraint('name'),
        schema='silver',
    )
    op.create_check_constraint(
        'ck_silver_machine_criticality', 'machine',
        "criticality IN ('LOW', 'MEDIUM', 'HIGH')", schema='silver',
    )
    # incident : retrait des dims redondantes + FK machine
    op.drop_column('incident', 'model', schema='silver')
    op.drop_column('incident', 'criticality', schema='silver')
    op.create_foreign_key(
        'fk_silver_incident_machine', 'incident', 'machine',
        ['machine_id'], ['machine_code'], source_schema='silver', referent_schema='silver',
    )
    # maintenance : component -> component_id (lookup) + CHECK + retrait des dims
    op.add_column(
        'maintenance', sa.Column('component_id', sa.Integer(), nullable=False), schema='silver'
    )
    op.create_foreign_key(
        'fk_silver_maintenance_component', 'maintenance', 'component',
        ['component_id'], ['component_id'], source_schema='silver', referent_schema='silver',
    )
    op.create_check_constraint(
        'ck_silver_maintenance_type', 'maintenance',
        "maintenance_type IN ('proactive', 'reactive')", schema='silver',
    )
    op.drop_column('maintenance', 'model', schema='silver')
    op.drop_column('maintenance', 'component', schema='silver')
    op.drop_column('maintenance', 'criticality', schema='silver')
    # telemetry : FK machine + retrait des dims
    op.create_foreign_key(
        'fk_silver_telemetry_machine', 'telemetry', 'machine',
        ['machine_id'], ['machine_code'], source_schema='silver', referent_schema='silver',
    )
    op.drop_column('telemetry', 'model', schema='silver')
    op.drop_column('telemetry', 'criticality', schema='silver')


def downgrade() -> None:
    op.drop_constraint('fk_silver_telemetry_machine', 'telemetry', schema='silver', type_='foreignkey')
    op.add_column('telemetry', sa.Column('model', sa.VARCHAR(length=32), nullable=True), schema='silver')
    op.add_column('telemetry', sa.Column('criticality', sa.VARCHAR(length=8), nullable=True), schema='silver')

    op.drop_constraint('ck_silver_maintenance_type', 'maintenance', schema='silver', type_='check')
    op.drop_constraint('fk_silver_maintenance_component', 'maintenance', schema='silver', type_='foreignkey')
    op.add_column(
        'maintenance',
        sa.Column('component', sa.VARCHAR(length=64), nullable=False, server_default=''),
        schema='silver',
    )
    op.add_column('maintenance', sa.Column('model', sa.VARCHAR(length=32), nullable=True), schema='silver')
    op.add_column('maintenance', sa.Column('criticality', sa.VARCHAR(length=8), nullable=True), schema='silver')
    op.drop_column('maintenance', 'component_id', schema='silver')

    op.drop_constraint('fk_silver_incident_machine', 'incident', schema='silver', type_='foreignkey')
    op.add_column('incident', sa.Column('model', sa.VARCHAR(length=32), nullable=True), schema='silver')
    op.add_column('incident', sa.Column('criticality', sa.VARCHAR(length=8), nullable=True), schema='silver')

    op.drop_constraint('ck_silver_machine_criticality', 'machine', schema='silver', type_='check')
    op.drop_table('component', schema='silver')
