"""Modèles ORM du schéma ``silver`` (US 1.3) — schéma typé, **normalisé (mesuré)**.

Normalisation star-schema : la dimension ``machine`` porte ``model``/``criticality`` (non
dupliqués dans les faits, récupérables par jointure) ; ``component`` est une table de lookup
(FK depuis ``maintenance``) ; les FK ``machine_id`` relient les faits à la dimension. Les
enums (``criticality``, ``maintenance_type``) sont contraints par CHECK, pas par table.

Reflètent **exactement** les DataFrames produits par ``indusense.data.silver`` (test
d'alignement). Dates en ``DateTime`` **sans fuseau** (naïf, uniformisation silver).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
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

_MACHINE_FK = f"{SCHEMA_SILVER}.machine.machine_code"


class SilverMachine(Base):
    """Dimension machine (porte ``model`` / ``criticality`` pour tout le star-schema)."""

    __tablename__ = "machine"
    __table_args__ = (
        CheckConstraint(
            "criticality IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_silver_machine_criticality"
        ),
        {"schema": SCHEMA_SILVER},
    )

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


class SilverComponent(Base):
    """Lookup des composants de maintenance (référencé par ``maintenance.component_id``)."""

    __tablename__ = "component"
    __table_args__ = {"schema": SCHEMA_SILVER}

    component_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)


class SilverMaintenance(Base):
    __tablename__ = "maintenance"
    __table_args__ = (
        CheckConstraint(
            "maintenance_type IN ('proactive', 'reactive')", name="ck_silver_maintenance_type"
        ),
        {"schema": SCHEMA_SILVER},
    )

    maintenance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    machine_code: Mapped[str] = mapped_column(String(16), ForeignKey(_MACHINE_FK))
    maintenance_at: Mapped[object] = mapped_column(DateTime)
    maintenance_type: Mapped[str] = mapped_column(String(16))
    component_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(f"{SCHEMA_SILVER}.component.component_id")
    )
    description: Mapped[str] = mapped_column(Text)
    related_incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_hours: Mapped[float] = mapped_column(Float)


class SilverTelemetry(Base):
    """Télémétrie nettoyée : dédoublonnée, imputée (maintenance-aware), flaggée (FK machine)."""

    __tablename__ = "telemetry"
    __table_args__ = {"schema": SCHEMA_SILVER}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(16), ForeignKey(_MACHINE_FK))
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


class SilverIncident(Base):
    """Incidents anonymisés, enrichis (signal/confiance, axes temporels), FK machine."""

    __tablename__ = "incident"
    __table_args__ = {"schema": SCHEMA_SILVER}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(
        String(16), ForeignKey(_MACHINE_FK), nullable=True
    )
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
