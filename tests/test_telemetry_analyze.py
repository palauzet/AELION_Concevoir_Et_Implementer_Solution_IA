"""Tests du pipeline d'analyse de la télémétrie (US 1.x)."""

from __future__ import annotations

import json

import pandas as pd

from indusense import config
from indusense.data import telemetry_analyze as tel


def _sample(n: int = 48) -> pd.DataFrame:
    """Deux machines, n relevés horaires chacune (grille régulière)."""
    rows = []
    for machine in ("MACH-01", "MACH-02"):
        ts = pd.date_range("2025-06-01", periods=n, freq="h")
        for i, t in enumerate(ts):
            rows.append({
                "machine_id": machine,
                "timestamp": t,
                "temperature_c": 45.0 + i % 10,
                "pressure_bar": 195.0 + i % 5,
                "voltage_mean_v": 227.0 + (i % 3) * 0.5,
                "rotation_mean_rpm": 1500.0 + i % 20,
                "pieces_produced": i % 50,
            })
    return pd.DataFrame(rows)


def test_check_no_personal_data() -> None:
    assert tel.check_no_personal_data(_sample())["anonymisation_requise"] is False
    with_pii = _sample().assign(operator_name="X")
    report = tel.check_no_personal_data(with_pii)
    assert report["anonymisation_requise"] is True
    assert "operator_name" in report["colonnes_dcp"]


def test_deduplicate_removes_key_duplicates() -> None:
    df = _sample()
    dup = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 doublons de clé
    out, removed = tel.deduplicate(dup)
    assert removed == 2
    assert out.duplicated(subset=["machine_id", "timestamp"]).sum() == 0


def test_detect_outliers_bounds() -> None:
    df = _sample()
    out = tel.detect_outliers(df)
    assert list(out.index) == list(config.TELEMETRY_SENSORS)
    assert (out["borne_basse"] <= out["borne_haute"]).all()
    assert (out["n_outliers"] >= 0).all()
    assert (out["pct"].between(0, 100)).all()


def test_correlations_shape() -> None:
    corr = tel.correlations(_sample())
    cols = [*config.TELEMETRY_SENSORS, config.TELEMETRY_PRODUCTION]
    for method in ("pearson", "spearman"):
        mat = corr[method]
        assert mat.shape == (len(cols), len(cols))
        # Diagonale = 1 (corrélation d'une variable avec elle-même).
        assert all(abs(mat.loc[c, c] - 1.0) < 1e-9 for c in cols)


def test_run_analysis_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ANALYSE_TELEMETRY_DIR", tmp_path)
    # Échantillon (au lieu des 134k lignes réelles) : test rapide et isolé.
    monkeypatch.setattr(tel, "load_telemetry_raw", lambda: _sample(48))
    meta = tel.run_analysis()

    run_dir = tmp_path / meta["run_id"]
    assert (run_dir / "telemetry_clean.parquet").exists()
    df = pd.read_parquet(run_dir / "telemetry_clean.parquet")
    assert len(df) == meta["n_lignes"]
    assert {"jour", "heure", "weekday", "shift"} <= set(df.columns)
    assert (run_dir / "figures").is_dir()
    assert len(list((run_dir / "figures").glob("*.svg"))) == 9
    assert meta["anonymisation_requise"] is False

    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["run_id"] == meta["run_id"]
    assert (tmp_path / "runs.md").exists()
    assert (tmp_path / "METHODOLOGIE.md").exists()
