"""Modèles ORM du schéma ``silver`` (US 1.3) — schéma typé de la couche nettoyée/intégrée.

Reflètent **exactement** les DataFrames produits par ``indusense.data.silver`` (un test
d'alignement garde ces modèles synchronisés avec les transformations pandas). Dates en
``DateTime`` **sans fuseau** (naïf, conforme à l'uniformisation silver).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from indusense.config import SCHEMA_SILVER
from indusense.data.db import Base


class SilverMachine(Base):
    __tablename__ = "machine"
    __table_args__ = {"schema": SCHEMA_SILVER}

    machine_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    commissioning_date: Mapped[object] = mapped_column(DateTime)
    max_daily_capacity: Mapped[int] = mapped_column(Integer)
    max_hourly_capacity_pieces: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(32))
    production_line: Mapped[str] = mapped_column(String(16))
    location: Mapped[str] = mapped_column(String(16))
    criticality: Mapped[str] = mapped_column(String(8))
    is_active: Mapped[bool] = mapped_column(Boolean)
    heures_equivalentes_jour: Mapped[int] = mapped_column(Integer)
    capacite_incoherente: Mapped[bool] = mapped_column(Boolean)


class SilverMaintenance(Base):
    __tablename__ = "maintenance"
    __table_args__ = {"schema": SCHEMA_SILVER}

    maintenance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    machine_code: Mapped[str] = mapped_column(
        String(16), ForeignKey(f"{SCHEMA_SILVER}.machine.machine_code")
    )
    maintenance_at: Mapped[object] = mapped_column(DateTime)
    maintenance_type: Mapped[str] = mapped_column(String(16))
    component: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    related_incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_hours: Mapped[float] = mapped_column(Float)
    model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    criticality: Mapped[str | None] = mapped_column(String(8), nullable=True)


class SilverTelemetry(Base):
    """Télémétrie nettoyée : dédoublonnée, imputée (maintenance-aware), flaggée, enrichie."""

    __tablename__ = "telemetry"
    __table_args__ = {"schema": SCHEMA_SILVER}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(16))
    timestamp: Mapped[object] = mapped_column(DateTime)
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_bar: Mapped[float | None] = mapped_column(Float, nullable=True)
    voltage_mean_v: Mapped[float | None] = mapped_column(Float, nullable=True)
    rotation_mean_rpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pieces_produced: Mapped[int] = mapped_column(Integer)
    during_maintenance: Mapped[bool] = mapped_column(Boolean)
    temperature_c_was_imputed: Mapped[bool] = mapped_column(Boolean)
    pressure_bar_was_imputed: Mapped[bool] = mapped_column(Boolean)
    voltage_mean_v_was_imputed: Mapped[bool] = mapped_column(Boolean)
    rotation_mean_rpm_was_imputed: Mapped[bool] = mapped_column(Boolean)
    temperature_c_is_outlier: Mapped[bool] = mapped_column(Boolean)
    pressure_bar_is_outlier: Mapped[bool] = mapped_column(Boolean)
    voltage_mean_v_is_outlier: Mapped[bool] = mapped_column(Boolean)
    rotation_mean_rpm_is_outlier: Mapped[bool] = mapped_column(Boolean)
    model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    criticality: Mapped[str | None] = mapped_column(String(8), nullable=True)


class SilverIncident(Base):
    """Incidents anonymisés, enrichis (signal/confiance, axes temporels), dates uniformisées."""

    __tablename__ = "incident"
    __table_args__ = {"schema": SCHEMA_SILVER}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    severity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    shift: Mapped[str | None] = mapped_column(String(16), nullable=True)
    type_surchauffe: Mapped[int] = mapped_column(Integer)
    type_baisse_pression: Mapped[int] = mapped_column(Integer)
    type_vibration: Mapped[int] = mapped_column(Integer)
    type_bruit_mecanique: Mapped[int] = mapped_column(Integer)
    type_surconsommation: Mapped[int] = mapped_column(Integer)
    type_blocage_mecanique: Mapped[int] = mapped_column(Integer)
    type_alarme_capteur: Mapped[int] = mapped_column(Integer)
    type_arret_urgence: Mapped[int] = mapped_column(Integer)
    type_defaut_qualite: Mapped[int] = mapped_column(Integer)
    n_signals: Mapped[int] = mapped_column(Integer)
    signal: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[object] = mapped_column(DateTime)
    jour: Mapped[object] = mapped_column(DateTime)
    semaine_iso: Mapped[str] = mapped_column(String(12))
    weekday: Mapped[str] = mapped_column(String(12))
    coherence: Mapped[float] = mapped_column(Float)
    comment_present: Mapped[float] = mapped_column(Float)
    machine_valide: Mapped[float] = mapped_column(Float)
    severity_valide: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    criticality: Mapped[str | None] = mapped_column(String(8), nullable=True)
