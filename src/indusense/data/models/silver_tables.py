"""Modèles ORM du schéma ``silver`` (US 1.3) — schéma typé, **normalisé 3NF**.

Normalisation 3NF, motivée par l'**évolutivité** : ajouter/retirer un capteur ne doit pas
modifier le schéma. Les **mesures de capteurs** sont donc normalisées (un capteur = une
**ligne**) via la dimension ``sensor`` et la décomposition en-tête/détail
``reading`` (1 relevé horodaté) / ``measurement`` (1 mesure capteur). Des **clés primaires
artificielles (surrogates)** identifient les machines (``machine_pk``) et les types de
machines (``model_id``) ; les faits référencent la machine par ``machine_pk``.

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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from indusense.config import SCHEMA_SILVER
from indusense.data.db import Base

_MACHINE_FK = f"{SCHEMA_SILVER}.machine.machine_pk"


class SilverModel(Base):
    """Dimension type de machine (clé surrogate ``model_id``, référencée par machine)."""

    __tablename__ = "model"
    __table_args__ = {"schema": SCHEMA_SILVER}

    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(32), unique=True)


class SilverSensor(Base):
    """Dimension capteur (clé surrogate ``sensor_id``). Un capteur = une ligne (évolutivité)."""

    __tablename__ = "sensor"
    __table_args__ = {"schema": SCHEMA_SILVER}

    sensor_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(32), unique=True)
    unit: Mapped[str] = mapped_column(String(16))


class SilverMachine(Base):
    """Dimension machine : PK surrogate ``machine_pk``, ``machine_code`` clé métier, FK model."""

    __tablename__ = "machine"
    __table_args__ = (
        CheckConstraint(
            "criticality IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_silver_machine_criticality"
        ),
        UniqueConstraint("machine_code", name="uq_silver_machine_code"),
        {"schema": SCHEMA_SILVER},
    )

    machine_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    machine_code: Mapped[str] = mapped_column(String(16))
    commissioning_date: Mapped[object] = mapped_column(DateTime)
    max_daily_capacity: Mapped[int] = mapped_column(Integer)
    max_hourly_capacity_pieces: Mapped[int] = mapped_column(Integer)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(f"{SCHEMA_SILVER}.model.model_id", name="fk_silver_machine_model")
    )
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
    machine_pk: Mapped[int] = mapped_column(
        Integer, ForeignKey(_MACHINE_FK, name="fk_silver_maintenance_machine")
    )
    maintenance_at: Mapped[object] = mapped_column(DateTime)
    maintenance_type: Mapped[str] = mapped_column(String(16))
    component_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            f"{SCHEMA_SILVER}.component.component_id", name="fk_silver_maintenance_component"
        ),
    )
    description: Mapped[str] = mapped_column(Text)
    related_incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_hours: Mapped[float] = mapped_column(Float)


class SilverReading(Base):
    """En-tête d'un relevé horodaté (1 ligne / (machine, timestamp)) — FK machine surrogate."""

    __tablename__ = "reading"
    __table_args__ = {"schema": SCHEMA_SILVER}

    reading_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    machine_pk: Mapped[int] = mapped_column(
        Integer, ForeignKey(_MACHINE_FK, name="fk_silver_reading_machine")
    )
    timestamp: Mapped[object] = mapped_column(DateTime)
    during_maintenance: Mapped[bool] = mapped_column(Boolean)
    pieces_produced: Mapped[int] = mapped_column(Integer)


class SilverMeasurement(Base):
    """Détail : 1 mesure capteur d'un relevé (forme longue, évolutive). FK reading + sensor."""

    __tablename__ = "measurement"
    __table_args__ = (
        UniqueConstraint("reading_id", "sensor_id", name="uq_silver_measurement"),
        {"schema": SCHEMA_SILVER},
    )

    measurement_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reading_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(f"{SCHEMA_SILVER}.reading.reading_id", name="fk_silver_measurement_reading"),
    )
    sensor_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(f"{SCHEMA_SILVER}.sensor.sensor_id", name="fk_silver_measurement_sensor"),
    )
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    was_imputed: Mapped[bool] = mapped_column(Boolean)
    is_outlier: Mapped[bool] = mapped_column(Boolean)
    is_saturated: Mapped[bool] = mapped_column(Boolean)


class SilverIncident(Base):
    """Incidents anonymisés, enrichis (signal/confiance, axes temporels), FK machine surrogate."""

    __tablename__ = "incident"
    __table_args__ = {"schema": SCHEMA_SILVER}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    machine_pk: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(_MACHINE_FK, name="fk_silver_incident_machine"), nullable=True
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
