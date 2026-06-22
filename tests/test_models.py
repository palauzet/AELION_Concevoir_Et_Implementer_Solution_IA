"""Tests des modèles ORM (schéma typé bronze/silver) et de leur alignement au silver."""

from __future__ import annotations

from indusense.data import models, silver
from indusense.data.models import silver_tables as silver_models
from tests.test_silver import _bronze_samples


def test_metadata_has_all_tables() -> None:
    assert set(models.Base.metadata.tables) == {
        "bronze.machine", "bronze.maintenance", "bronze.telemetry", "bronze.incident",
        "silver.model", "silver.sensor", "silver.machine", "silver.component",
        "silver.maintenance", "silver.reading", "silver.measurement", "silver.incident",
    }


def test_silver_models_match_builders(monkeypatch) -> None:
    """Garde-fou anti-drift : colonnes modèles silver == colonnes produites par build_*."""
    samples = _bronze_samples()
    monkeypatch.setattr(silver, "read_bronze", lambda table: samples[table].copy())

    mtb = samples["maintenance"]
    model = silver.build_model(samples["machine"])
    sensor = silver.build_sensor()
    machine = silver.build_machine(model)
    component = silver.build_component(mtb)
    clean = silver.build_clean_telemetry()[0]
    reading = silver.build_reading(clean, machine)
    pairs = [
        (silver_models.SilverModel, model),
        (silver_models.SilverSensor, sensor),
        (silver_models.SilverMachine, machine),
        (silver_models.SilverComponent, component),
        (silver_models.SilverMaintenance, silver.build_maintenance(mtb, component, machine)),
        (silver_models.SilverReading, reading),
        (silver_models.SilverMeasurement, silver.build_measurement(clean, reading, sensor)),
        (silver_models.SilverIncident, silver.build_incident(machine)),
    ]
    for orm, df in pairs:
        # Clés techniques auto-incrémentées par la DB (absentes des DataFrames).
        model_cols = set(orm.__table__.columns.keys()) - {"id", "measurement_id"}
        assert model_cols == set(df.columns), (
            f"{orm.__tablename__}: écart {model_cols.symmetric_difference(df.columns)}"
        )
