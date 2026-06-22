"""Tests de la construction de la couche silver (US 1.3 + intégration)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from indusense import config
from indusense.data import silver


def _telemetry_with_flags() -> pd.DataFrame:
    """6 relevés horaires : NaN opérationnel (idx 1) + NaN en maintenance (idx 4)."""
    return pd.DataFrame({
        "machine_id": "MACH-01",
        "timestamp": pd.date_range("2025-06-01", periods=6, freq="h"),
        "temperature_c": [40.0, np.nan, 42.0, 50.0, np.nan, 44.0],
        "pressure_bar": [190.0, 191.0, 192.0, 193.0, 194.0, 195.0],
        "voltage_mean_v": [227.0] * 6,
        "rotation_mean_rpm": [1500.0] * 6,
        "pieces_produced": [10, 11, 12, 0, 0, 13],
        "during_maintenance": [False, False, False, True, True, False],
        "_segment": [0, 0, 0, 1, 1, 1],
    })


def test_impute_maintenance_aware() -> None:
    out = silver.impute_telemetry(_telemetry_with_flags())
    # NaN opérationnel (idx 1) interpolé (40 -> 42 => 41) et marqué imputé.
    assert out.loc[1, "temperature_c"] == 41.0
    assert bool(out.loc[1, "temperature_c_was_imputed"]) is True
    # NaN en fenêtre de maintenance (idx 4) NON imputé (reste NaN, non opérationnel).
    assert pd.isna(out.loc[4, "temperature_c"])
    assert bool(out.loc[4, "temperature_c_was_imputed"]) is False


def test_build_component_and_mapping() -> None:
    mt = _bronze_samples()["maintenance"]
    comp = silver.build_component(mt)
    assert list(comp.columns) == ["component_id", "name"]
    assert comp["component_id"].is_unique and comp["name"].is_unique
    mapped = silver.build_maintenance(mt, comp)
    assert "component" not in mapped.columns and "component_id" in mapped.columns
    assert mapped["component_id"].isin(comp["component_id"]).all()  # FK valide


def test_flag_outliers() -> None:
    df = _telemetry_with_flags().copy()
    df.loc[0, "pressure_bar"] = 10_000.0  # valeur extrême
    out = silver.flag_outliers(df)
    assert bool(out.loc[0, "pressure_bar_is_outlier"]) is True
    assert "rotation_mean_rpm_is_outlier" in out.columns


def test_flag_and_segment() -> None:
    df = pd.DataFrame({
        "machine_id": "MACH-01",
        "timestamp": pd.date_range("2025-06-01", periods=6, freq="h"),
        **{c: 1.0 for c in config.TELEMETRY_SENSORS},
    })
    windows = pd.DataFrame({
        "machine_code": ["MACH-01"],
        "start": [pd.Timestamp("2025-06-01 03:00:00")],
        "end": [pd.Timestamp("2025-06-01 05:00:00")],
    })
    out = silver._flag_and_segment(df, windows)
    # 03:00 et 04:00 dans la fenêtre [03:00 ; 05:00).
    assert out["during_maintenance"].tolist() == [False, False, False, True, True, False]
    # Segment passe de 0 à 1 après le début de maintenance (respect du reset).
    assert out["_segment"].tolist() == [0, 0, 0, 1, 1, 1]


# --- Smoke : run() complet avec bronze mocké (sans base) --------------------
def _bronze_samples() -> dict[str, pd.DataFrame]:
    sig = list(config.INCIDENT_SIGNALS)
    flags = {s: [0, 0] for s in sig}
    flags[sig[0]] = [1, 0]
    flags[sig[1]] = [0, 1]
    telemetry = pd.DataFrame({
        "telemetry_id": range(1, 7),
        "machine_id": "MACH-01",
        "timestamp": list(pd.date_range("2025-06-01", periods=6, freq="h")),
        "temperature_c": [40.0, np.nan, 42.0, 50.0, np.nan, 44.0],
        "pressure_bar": [190.0, 191.0, 192.0, 193.0, 194.0, 195.0],
        "voltage_mean_v": [227.0] * 6,
        "rotation_mean_rpm": [1500.0] * 6,
        "pieces_produced": [10, 11, 12, 0, 0, 13],
    })
    telemetry = pd.concat([telemetry, telemetry.iloc[[0]]], ignore_index=True)  # doublon de clé
    return {
        "machine": pd.DataFrame({
            "machine_code": ["MACH-01"], "commissioning_date": ["2021-05-12"],
            "max_daily_capacity": [770], "max_hourly_capacity_pieces": [48],
            "model": ["InduPress-X2"], "production_line": ["Ligne-A"],
            "location": ["Atelier-2"], "criticality": ["MEDIUM"], "is_active": [True],
        }),
        "maintenance": pd.DataFrame({
            "maintenance_id": [1], "machine_code": ["MACH-01"],
            "maintenance_at": ["2025-06-01 03:00:00+00"], "maintenance_type": ["reactive"],
            "action_type": ["changement_suite_panne"], "component": ["capteur pression"],
            "description": ["Intervention corrective après INC-000001"],
            "related_incident_id": ["INC-000001"], "duration_hours": [2.0],
        }),
        "telemetry": telemetry,
        "incident": pd.DataFrame({
            "incident_pk": [1, 2], "incident_id": ["INC-000001", "INC-000002"],
            "date": ["2025-06-01", "2025-06-01"], "time": ["05:00", "09:00"],
            "operator_name": ["X", "Y"], "machine_id": ["MACH-01", "MACH-01"],
            "severity": [2, 3], "operator_badge": ["OP1", "OP2"],
            "comment": ["chauffe", "fuite"], "shift": ["nuit", "matin"], **flags,
        }),
    }


def test_run_smoke(tmp_path, monkeypatch) -> None:
    samples = _bronze_samples()
    monkeypatch.setattr(config, "SILVER_DIR", tmp_path)
    monkeypatch.setattr(silver, "read_bronze", lambda table: samples[table].copy())
    monkeypatch.setattr(silver, "write_silver_tables", lambda tables, engine: None)  # pas de DB
    monkeypatch.setattr(silver, "get_engine", lambda: None)

    meta = silver.run()
    run_dir = tmp_path / meta["run_id"]
    for name in ("machine", "component", "maintenance", "telemetry", "incident"):
        assert (run_dir / f"{name}.parquet").exists()
    tel = pd.read_parquet(run_dir / "telemetry.parquet")
    assert meta["telemetry"]["n_doublons_supprimes"] == 1  # doublon retiré
    assert {"during_maintenance", "temperature_c_was_imputed",
            "temperature_c_is_outlier"} <= set(tel.columns)
    assert {"model", "criticality"}.isdisjoint(tel.columns)  # dims non copiées (normalisé)
    incident = pd.read_parquet(run_dir / "incident.parquet")
    assert "operator_name" not in incident.columns  # anonymisé
    assert {"model", "criticality"}.isdisjoint(incident.columns)
    maintenance = pd.read_parquet(run_dir / "maintenance.parquet")
    assert "component_id" in maintenance.columns  # lookup normalisé
    assert {"component", "action_type", "model", "criticality", "created_at"}.isdisjoint(
        maintenance.columns
    )
    assert (run_dir / "figures").is_dir()

    # Dates uniformisées : datetime64 naïf (sans fuseau) partout ; incident fusionné.
    def _naive_dt(s: pd.Series) -> bool:
        return "datetime64" in str(s.dtype) and getattr(s.dtype, "tz", None) is None

    machine = pd.read_parquet(run_dir / "machine.parquet")
    assert _naive_dt(tel["timestamp"])
    assert _naive_dt(maintenance["maintenance_at"])
    assert _naive_dt(machine["commissioning_date"])
    assert {"timestamp"} <= set(incident.columns)
    assert {"date", "time"}.isdisjoint(incident.columns)  # fusionnées en timestamp
    assert _naive_dt(incident["timestamp"])
    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["run_id"] == meta["run_id"]
    assert (tmp_path / "METHODOLOGIE.md").exists()
