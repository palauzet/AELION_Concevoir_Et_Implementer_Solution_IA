"""Tests du pipeline d'analyse de la télémétrie (US 1.x)."""

from __future__ import annotations

import json

import pandas as pd

from indusense import config
from indusense.data import telemetry as tel


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


def test_check_unit_consistency() -> None:
    df = _sample()
    out = tel.check_unit_consistency(df)
    assert list(out.index) == list(config.TELEMETRY_SENSORS)
    assert not out["unite_suspecte"].any()  # même unité => amplitude inter-machine ~1
    # Une machine en Fahrenheit (≈ ×1,8 + 32) => amplitude suspecte sur la température.
    fahr = df.copy()
    m2 = fahr["machine_id"] == "MACH-02"
    fahr.loc[m2, "temperature_c"] = fahr.loc[m2, "temperature_c"] * 1.8 + 32
    out2 = tel.check_unit_consistency(fahr)
    assert bool(out2.loc["temperature_c", "unite_suspecte"]) is True


def test_detect_saturation() -> None:
    df = _sample()  # rampes régulières : aucun empilement sur une borne
    base = tel.detect_saturation(df)
    assert not base["sature_haut"].any() and not base["sature_bas"].any()
    # Force un plafond : 6 relevés écrêtés exactement sur 80.0 (empilement).
    sat = df.copy()
    sat.loc[sat.index[:6], "temperature_c"] = 80.0
    out = tel.detect_saturation(sat)
    assert bool(out.loc["temperature_c", "sature_haut"]) is True
    assert out.loc["temperature_c", "n_satures"] >= 6
    assert bool(out.loc["pressure_bar", "sature_haut"]) is False  # témoin : pas d'écrêtage


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
    # Échantillon (au lieu de lire bronze) : test rapide et isolé.
    monkeypatch.setattr(tel, "load_telemetry", lambda: _sample(48))
    meta = tel.run_analysis()

    run_dir = tmp_path / meta["run_id"]
    assert (run_dir / "telemetry_clean.parquet").exists()
    df = pd.read_parquet(run_dir / "telemetry_clean.parquet")
    assert len(df) == meta["n_lignes"]
    assert {"jour", "heure", "weekday", "shift"} <= set(df.columns)
    # Export CSV des données résultats (même contenu que le parquet).
    csv_path = run_dir / "telemetry_clean.csv"
    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert len(df_csv) == meta["n_lignes"]
    assert list(df_csv.columns) == list(df.columns)
    assert (run_dir / "figures").is_dir()
    assert len(list((run_dir / "figures").glob("*.svg"))) == 9
    assert meta["anonymisation_requise"] is False
    # Contrôles qualité capteurs (échantillon régulier : ni saturation ni unité suspecte).
    assert meta["n_satures_total"] == 0
    assert meta["capteurs_satures"] == [] and meta["capteurs_unite_suspecte"] == []

    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["run_id"] == meta["run_id"]
    assert (tmp_path / "runs.md").exists()
    assert (tmp_path / "METHODOLOGIE.md").exists()
