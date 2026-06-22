"""Tests du pipeline d'analyse du référentiel (US 1.x)."""

from __future__ import annotations

import json

import pandas as pd

from indusense import config
from indusense.data import referentiel as ref


def _machine() -> pd.DataFrame:
    return pd.DataFrame({
        "machine_code": ["MACH-01", "MACH-02", "MACH-03"],
        "commissioning_date": pd.to_datetime(["2021-05-12", "2024-09-07", "2019-07-23"]).date,
        "max_daily_capacity": [770, 800, 1405],
        "max_hourly_capacity_pieces": [48, 50, 88],
        "model": ["InduPress-X2", "InduPress-X2", "InduPress-X1"],
        "production_line": ["Ligne-A", "Ligne-A", "Ligne-B"],
        "location": ["Atelier-2", "Atelier-1", "Atelier-1"],
        "criticality": ["MEDIUM", "LOW", "HIGH"],
        "is_active": [True, True, True],
    })


def _maintenance() -> pd.DataFrame:
    return pd.DataFrame({
        "maintenance_id": [1, 2, 3, 4, 5, 6],
        "machine_code": ["MACH-01", "MACH-01", "MACH-02", "MACH-03", "MACH-02", "MACH-03"],
        "maintenance_at": pd.to_datetime([
            "2025-06-15", "2025-08-14", "2025-07-01", "2025-09-02", "2025-10-05", "2025-11-15",
        ]),
        "maintenance_type": ["proactive", "proactive", "proactive",
                             "reactive", "proactive", "reactive"],
        "component": ["capteur pression", "filtre hydraulique", "capteur pression",
                      "roulement axe principal", "courroie moteur", "capteur pression"],
        "related_incident_id": [None, None, None, "INC-000100", None, "INC-000200"],
        "duration_hours": [2.2, 2.8, 1.8, 5.5, 3.0, 6.1],
    })


def test_check_integrity_ok() -> None:
    rep = ref.check_integrity(_machine(), _maintenance())
    assert rep["machine_pk_unique"] and rep["maintenance_pk_unique"]
    assert rep["machines_orphelines"] == []
    assert rep["reactive_sans_incident"] == 0
    assert rep["anonymisation_requise"] is False


def test_check_integrity_detects_problems() -> None:
    maint = _maintenance()
    maint.loc[len(maint)] = [7, "MACH-99", pd.Timestamp("2025-12-01"),
                             "reactive", "x", None, 4.0]  # orpheline + réactif sans incident
    rep = ref.check_integrity(_machine().assign(operator_name="X"), maint)
    assert "MACH-99" in rep["machines_orphelines"]
    assert rep["reactive_sans_incident"] == 1
    assert rep["anonymisation_requise"] is True


def test_maintenance_summary() -> None:
    summ = ref.maintenance_summary(_maintenance())
    assert summ["par_type"].loc["reactive", "count"] == 2
    assert summ["par_type"].loc["proactive", "count"] == 4
    assert summ["composants"].idxmax() == "capteur pression"
    assert summ["n_lie_incident"] == 2


def test_capacity_by_model() -> None:
    cap = ref.capacity_by_model(_machine())
    # X1 (1405) > X2 (moyenne 785) en capacité journalière.
    assert cap.index[0] == "InduPress-X1"


def test_run_analysis_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ANALYSE_REFERENTIEL_DIR", tmp_path)
    monkeypatch.setattr(ref, "load_referentiel", lambda: (_machine(), _maintenance()))
    meta = ref.run_analysis()

    run_dir = tmp_path / meta["run_id"]
    assert meta["n_machines"] == 3 and meta["n_maintenances"] == 6
    assert meta["n_reactive"] == 2 and meta["n_proactive"] == 4
    assert (run_dir / "figures").is_dir()
    assert len(list((run_dir / "figures").glob("*.svg"))) == 9
    # Export CSV des deux tables analysées.
    for nom, n in (("machine", 3), ("maintenance", 6)):
        assert (run_dir / f"{nom}.csv").exists()
        assert len(pd.read_csv(run_dir / f"{nom}.csv", encoding="utf-8-sig")) == n

    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["run_id"] == meta["run_id"]
    assert (tmp_path / "runs.md").exists()
    assert (tmp_path / "METHODOLOGIE.md").exists()
