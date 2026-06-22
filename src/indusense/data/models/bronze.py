"""Modèles ORM du schéma ``bronze`` (US 1.2, C1/C3) — source de vérité du schéma.

Bronze = fidélité à la source. ``machine`` / ``maintenance`` reprennent ici les contraintes
de ``data/raw/machine.sql`` (CHECK, FK, index, ``DEFAULT NOW()``) pour que les modèles — et
non le dump — soient la source unique du schéma (créé par Alembic). ``telemetry`` / ``incident``
sont alimentées depuis les CSV (clé technique auto-incrémentée).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from indusense.config import SCHEMA_BRONZE
from indusense.data.db import Base


class Machine(Base):
    __tablename__ = "machine"
    __table_args__ = (
        CheckConstraint("max_daily_capacity > 0", name="ck_machine_daily_cap"),
        CheckConstraint("max_hourly_capacity_pieces > 0", name="ck_machine_hourly_cap"),
        CheckConstraint("criticality IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_machine_criticality"),
        Index("idx_machine_line", "production_line"),
        Index("idx_machine_location", "location"),
        {"schema": SCHEMA_BRONZE},
    )

    machine_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    commissioning_date: Mapped[object] = mapped_column(Date, nullable=False)
    max_daily_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    max_hourly_capacity_pieces: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(32), nullable=False)
    production_line: Mapped[str] = mapped_column(String(16), nullable=False)
    location: Mapped[str] = mapped_column(String(16), nullable=False)
    criticality: Mapped[str] = mapped_column(String(8), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Maintenance(Base):
    __tablename__ = "maintenance"
    __table_args__ = (
        CheckConstraint(
            "maintenance_type IN ('proactive', 'reactive')", name="ck_maintenance_type"
        ),
        CheckConstraint("duration_hours > 0", name="ck_maintenance_duration"),
        Index("idx_maintenance_machine_time", "machine_code", "maintenance_at"),
        Index("idx_maintenance_type", "maintenance_type"),
        {"schema": SCHEMA_BRONZE},
    )

    maintenance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    machine_code: Mapped[str] = mapped_column(
        String(16),
        ForeignKey(f"{SCHEMA_BRONZE}.machine.machine_code", ondelete="RESTRICT"),
        nullable=False,
    )
    maintenance_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    maintenance_type: Mapped[str] = mapped_column(String(16), nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    related_incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_hours: Mapped[object] = mapped_column(Numeric(6, 2), nullable=False)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Telemetry(Base):
    """Relevés capteurs horaires (telemetry.csv)."""

    __tablename__ = "telemetry"
    __table_args__ = {"schema": SCHEMA_BRONZE}

    telemetry_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    timestamp: Mapped[object] = mapped_column(DateTime, nullable=True)
    temperature_c: Mapped[float | None] = mapped_column(nullable=True)
    pressure_bar: Mapped[float | None] = mapped_column(nullable=True)
    voltage_mean_v: Mapped[float | None] = mapped_column(nullable=True)
    rotation_mean_rpm: Mapped[float | None] = mapped_column(nullable=True)
    pieces_produced: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Incident(Base):
    """Relevés manuels d'incidents (releves_incidents.csv) — DCP anonymisées en silver."""

    __tablename__ = "incident"
    __table_args__ = {"schema": SCHEMA_BRONZE}

    incident_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    date: Mapped[object] = mapped_column(Date, nullable=True)
    time: Mapped[str | None] = mapped_column(String(8), nullable=True)
    operator_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    severity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    operator_badge: Mapped[str | None] = mapped_column(String(16), nullable=True)
    comment: Mapped[str | None] = mapped_column(String, nullable=True)
    shift: Mapped[str | None] = mapped_column(String(16), nullable=True)
    type_surchauffe: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_baisse_pression: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_vibration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_bruit_mecanique: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_surconsommation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_blocage_mecanique: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_alarme_capteur: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_arret_urgence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_defaut_qualite: Mapped[int | None] = mapped_column(Integer, nullable=True)
