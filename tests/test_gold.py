"""Tests de la construction de la couche gold (feature engineering, anti-fuite)."""

from __future__ import annotations

import pandas as pd

from indusense import config
from indusense.data import gold


def test_add_labels_binaire_et_rul() -> None:
    """Labels (t, t+H] et RUL = heures jusqu'au prochain incident."""
    df = pd.DataFrame({
        "machine_pk": [1] * 5,
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="6h"),  # 00,06,12,18,24h
    })
    incident = pd.DataFrame({"machine_pk": [1], "timestamp": [pd.Timestamp("2025-01-01 13:00")]})
    out = gold.add_labels(df, incident)
    # incident à 13:00 -> seul t=12:00 a un incident dans (t, t+6h].
    assert out["label_incident_6h"].tolist() == [0, 0, 1, 0, 0]
    # RUL : 13h, 7h, 1h depuis 00/06/12 ; censuré après l'incident.
    assert out.loc[2, "rul_hours"] == 1.0
    assert out.loc[2, "rul_censored"] == 0
    assert out.loc[3, "rul_censored"] == 1 and pd.isna(out.loc[3, "rul_hours"])


def _wide_two_segments() -> pd.DataFrame:
    """Une machine, 6 relevés horaires, reset de maintenance entre idx 2 et 3 (segment 0/1)."""
    df = pd.DataFrame({
        "machine_pk": 1,
        "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
        "segment": [0, 0, 0, 1, 1, 1],
        "temperature_c": [10.0, 20.0, 30.0, 100.0, 110.0, 120.0],
    })
    for s in ("pressure_bar", "voltage_mean_v", "rotation_mean_rpm"):
        df[s] = 1.0
    return df


def test_dynamic_features_segment_aware_et_trailing() -> None:
    out = gold.add_dynamic_features(_wide_two_segments())
    # La fenêtre 48h NE franchit PAS la maintenance : 1re ligne du segment 1 = elle-même.
    assert out.loc[3, "temperature_c_mean_48h"] == 100.0
    assert out.loc[5, "temperature_c_mean_48h"] == 110.0  # moyenne 100/110/120
    # Trailing strict : la 1re ligne n'utilise aucune valeur future.
    assert out.loc[0, "temperature_c_mean_6h"] == 10.0
    # Pente en °C/heure (rampe +10/h sur le segment 1) — verrouille l'échelle (pas ×1000).
    assert abs(out.loc[5, "temperature_c_slope_48h"] - 10.0) < 1e-6
    # Cold-start : 1re ligne d'un segment -> std NaN, pente 0 (indéfinies à 1 point).
    assert pd.isna(out.loc[0, "temperature_c_std_6h"])
    assert out.loc[3, "temperature_c_slope_48h"] == 0.0
    assert {f"temperature_c_std_{w}h" for w in config.ROLLING_WINDOWS} <= set(out.columns)


def test_time_since_maintenance_depuis_reprise() -> None:
    """``time_since_last_maintenance_h`` se mesure depuis la FIN (reprise), pas le début."""
    df = pd.DataFrame({"machine_pk": [1], "timestamp": [pd.Timestamp("2025-01-01 06:00")]})
    maintenance = pd.DataFrame({"machine_pk": [1],
                                "maintenance_at": [pd.Timestamp("2025-01-01 02:00")],
                                "duration_hours": [2.0]})  # fin = 04:00
    incident = pd.DataFrame({"machine_pk": pd.Series([], dtype="int64"),
                             "timestamp": pd.Series([], dtype="datetime64[ns]")})
    out = gold.add_history_features(df, maintenance, incident)
    # 06:00 − fin(04:00) = 2 h (et non 4 h depuis le début 02:00).
    assert out.loc[0, "time_since_last_maintenance_h"] == 2.0


def test_assign_split_chronologique() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=100, freq="h")})
    out = gold.assign_split(df)
    assert set(out["split"]) == {"train", "val", "test"}
    # Disjoints et ordonnés dans le temps (pas de mélange aléatoire).
    assert out.loc[out["split"] == "train", "timestamp"].max() \
        <= out.loc[out["split"] == "val", "timestamp"].min()
    assert out.loc[out["split"] == "val", "timestamp"].max() \
        <= out.loc[out["split"] == "test", "timestamp"].min()


# --- Smoke : build_gold complet sur un silver synthétique (sans base) --------
def _silver_sample() -> dict[str, pd.DataFrame]:
    reading, meas, rid = [], [], 1
    for pk in (1, 2):
        ts = pd.date_range("2025-01-01", periods=40, freq="h")
        for k, t in enumerate(ts):
            dm = pk == 1 and 10 <= k < 13  # fenêtre de maintenance machine 1
            reading.append({"reading_id": rid, "machine_pk": pk, "timestamp": t,
                            "during_maintenance": dm, "pieces_produced": 20 + k % 5})
            for sid, name in enumerate(config.TELEMETRY_SENSORS, 1):
                meas.append({"reading_id": rid, "sensor_id": sid, "value": float(40 + k % 7 + sid),
                             "was_imputed": bool(k % 9 == 0), "is_outlier": False,
                             "is_saturated": bool(k == 5 and name == "temperature_c")})
            rid += 1
    sensor = pd.DataFrame({"sensor_id": range(1, 5), "name": list(config.TELEMETRY_SENSORS),
                           "unit": ["°C", "bar", "V", "rpm"]})
    machine = pd.DataFrame({
        "machine_pk": [1, 2], "machine_code": ["MACH-01", "MACH-02"],
        "model": ["X1", "X2"], "criticality": ["HIGH", "LOW"],
        "production_line": ["A", "B"], "location": ["L1", "L2"],
        "max_hourly_capacity_pieces": [48, 50],
        "commissioning_date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
    })
    maintenance = pd.DataFrame({"maintenance_id": [1], "machine_pk": [1],
                                "maintenance_at": [pd.Timestamp("2025-01-01 10:00")],
                                "duration_hours": [3.0]})
    incident = pd.DataFrame({"id": [1, 2], "machine_pk": [1, 2],
                             "timestamp": [pd.Timestamp("2025-01-01 20:00"),
                                           pd.Timestamp("2025-01-01 18:00")]})
    return {"reading": pd.DataFrame(reading), "measurement": pd.DataFrame(meas),
            "sensor": sensor, "machine": machine, "maintenance": maintenance, "incident": incident}


def test_build_gold_smoke() -> None:
    df, metrics = gold.build_gold(_silver_sample())

    # Points opérationnels uniquement (3 relevés en maintenance machine 1 retirés).
    assert "during_maintenance" not in df.columns
    assert len(df) == 80 - 3
    # Cibles présentes ; aucune colonne décrivant l'incident (anti-fuite).
    assert {f"label_incident_{h}h" for h in config.LABEL_HORIZONS} <= set(df.columns)
    assert {"rul_hours", "rul_censored", "split"} <= set(df.columns)
    assert {"severity", "type_surchauffe", "signal", "confidence", "incident_id"}.isdisjoint(
        df.columns
    )
    # Features attendues + métriques cohérentes.
    assert {"temperature_c_mean_24h", "n_saturated_6h", "time_since_last_maintenance_h",
            "utilisation", "age_days", "hour_sin", "shift"} <= set(df.columns)
    assert metrics["n_lignes"] == len(df) and metrics["n_features"] > 20
    assert set(metrics["taux_positifs"]) == {f"{h}h" for h in config.LABEL_HORIZONS}
    assert set(metrics["split"]) <= {"train", "val", "test"}
    # Garde-fou : aucune valeur imputée/outlier brute par capteur ne fuite dans le gold.
    assert not any(c.endswith("_was_imputed") for c in df.columns)
