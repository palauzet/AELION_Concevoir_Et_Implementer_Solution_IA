"""Modèles ORM du schéma ``bronze`` (US 1.2, C1/C3).

Principe *bronze* = fidélité à la source, tolérance aux données « sales » :
colonnes majoritairement nullables, pas de contrainte d'intégrité forte
(le nettoyage et les FK arrivent en couche *silver*, US 1.3+).

- ``machine`` / ``maintenance`` : créées et alimentées dans ``bronze`` par
  ``machine.sql`` (cf. ``ingest.load_reference_data``). Les modèles ci-dessous
  les reflètent pour interrogation.
- ``telemetry`` / ``incident`` : alimentées depuis les CSV par le script d'import.

``DateTime(timezone=True)`` se traduit en ``TIMESTAMPTZ`` sous PostgreSQL, en
cohérence avec ``machine.sql`` ; ``DateTime`` (sans tz) en ``timestamp``.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Date, DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from indusense.config import SCHEMA_BRONZE
from indusense.data.db import Base


class Machine(Base):
    __tablename__ = "machine"
    __table_args__ = {"schema": SCHEMA_BRONZE}

    machine_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    commissioning_date: Mapped[object] = mapped_column(Date)
    max_daily_capacity: Mapped[int] = mapped_column(Integer)
    max_hourly_capacity_pieces: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(32))
    production_line: Mapped[str] = mapped_column(String(16))
    location: Mapped[str] = mapped_column(String(16))
    criticality: Mapped[str] = mapped_column(String(8))
    is_active: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True))


class Maintenance(Base):
    __tablename__ = "maintenance"
    __table_args__ = {"schema": SCHEMA_BRONZE}

    maintenance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    machine_code: Mapped[str] = mapped_column(String(16))
    maintenance_at: Mapped[object] = mapped_column(DateTime(timezone=True))
    maintenance_type: Mapped[str] = mapped_column(String(16))
    action_type: Mapped[str] = mapped_column(String(32))
    component: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(String)
    related_incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_hours: Mapped[object] = mapped_column(Numeric(6, 2))
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True))


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
    """Relevés manuels d'incidents (releves_incidents.csv).

    Contient des données personnelles (operator_name, operator_badge) à
    anonymiser en US 1.1 / C2 — conservées brutes ici, pseudonymisées en silver.
    """

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
