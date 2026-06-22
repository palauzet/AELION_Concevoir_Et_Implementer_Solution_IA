"""Tests des modèles ORM (schéma typé bronze/silver) et de leur alignement au silver."""

from __future__ import annotations

from indusense.data import models, silver
from indusense.data.models import silver as silver_models
from tests.test_silver import _bronze_samples


def test_metadata_has_all_tables() -> None:
    assert set(models.Base.metadata.tables) == {
        "bronze.machine", "bronze.maintenance", "bronze.telemetry", "bronze.incident",
        "silver.machine", "silver.component", "silver.maintenance", "silver.telemetry",
        "silver.incident",
    }


def test_silver_models_match_builders(monkeypatch) -> None:
    """Garde-fou anti-drift : colonnes modèles silver == colonnes produites par build_*."""
    samples = _bronze_samples()
    monkeypatch.setattr(silver, "read_bronze", lambda table: samples[table].copy())

    mtb = samples["maintenance"]
    component = silver.build_component(mtb)
    pairs = [
        (silver_models.SilverMachine, silver.build_machine()),
        (silver_models.SilverComponent, component),
        (silver_models.SilverMaintenance, silver.build_maintenance(mtb, component)),
        (silver_models.SilverIncident, silver.build_incident()),
        (silver_models.SilverTelemetry, silver.build_telemetry()[0]),
    ]
    for model, df in pairs:
        # `id` = clé technique auto-incrémentée (absente du DataFrame, remplie par la DB).
        model_cols = set(model.__table__.columns.keys()) - {"id"}
        assert model_cols == set(df.columns), (
            f"{model.__tablename__}: écart {model_cols.symmetric_difference(df.columns)}"
        )
