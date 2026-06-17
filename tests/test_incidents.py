"""Tests du pipeline d'ingestion des incidents (US 1.1)."""

from __future__ import annotations

import json

import pandas as pd

from indusense import config
from indusense.data import incidents as ing
from indusense.data.anonymize import anonymize_operators


def _sample() -> pd.DataFrame:
    base = {s: [0, 0] for s in config.INCIDENT_SIGNALS}
    base["type_surchauffe"] = [1, 1]
    base["type_vibration"] = [0, 1]  # 2e ligne = 2 signaux actifs
    return pd.DataFrame(
        {
            "incident_id": ["INC-1", "INC-2"],
            "date": ["2025-06-01", "2025-06-02"],
            "time": ["05:00", "22:00"],
            "operator_name": ["Jean Dupont", "Marie Curie"],
            "machine_id": ["MACH-01", "MACH-02"],
            "severity": [1, 3],
            "operator_badge": ["OP1001", "OP1002"],
            "comment": ["chauffe anormale", None],
            "shift": ["matin", "nuit"],
            **base,
        }
    )


def test_anonymize_drops_operator_cols() -> None:
    out = anonymize_operators(_sample())
    assert "operator_name" not in out.columns
    assert "operator_badge" not in out.columns
    assert "comment" in out.columns  # conservé (non-DCP)


def test_confidence_bounds_and_columns() -> None:
    df = ing.compute_confidence(ing.enrich(anonymize_operators(_sample())))
    for col in ("signal", "n_signals", "coherence", "confidence"):
        assert col in df.columns
    assert df["confidence"].between(0, 1).all()
    # Ligne 1 : 1 signal + comment + machine + severity valides -> confiance maximale.
    assert df.loc[0, "confidence"] == 1.0
    # Ligne 2 : 2 signaux (coherence 0.5) et comment absent -> confiance moindre.
    assert df.loc[1, "confidence"] < 1.0


def test_severity_type_association() -> None:
    df = ing.enrich(anonymize_operators(_sample()))
    assoc = ing.severity_type_association(df)
    # 2 incidents -> 3 couples (INC-2 porte 2 signaux : surchauffe + vibration).
    assert assoc["n_couples"] == 3
    assert set(assoc["sev_par_type"].index) == {"surchauffe", "vibration"}
    # Statistiques bien bornées.
    assert 0.0 <= assoc["cramers_v"] <= 1.0
    assert 0.0 <= assoc["p_chi2"] <= 1.0
    assert 0.0 <= assoc["p_kruskal"] <= 1.0


def test_machine_association() -> None:
    df = ing.enrich(anonymize_operators(_sample()))
    assoc = ing.machine_association(df)
    assert assoc["n_machines"] == 2  # MACH-01, MACH-02
    assert set(assoc["sev_par_machine"].index) == {"MACH-01", "MACH-02"}
    # Statistiques bien bornées.
    assert 0.0 <= assoc["cramers_v"] <= 1.0
    assert 0.0 <= assoc["p_chi2"] <= 1.0
    assert 0.0 <= assoc["p_kruskal"] <= 1.0


def test_run_ingestion_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ANALYSE_INCIDENTS_DIR", tmp_path)
    # Injecte le CSV brut comme stand-in de bronze (même donnée, sans dépendance DB).
    monkeypatch.setattr(ing, "load_incidents", lambda: pd.read_csv(config.RAW_INCIDENTS))
    meta = ing.run_ingestion()

    run_dir = tmp_path / meta["run_id"]
    assert (run_dir / "incidents_anonymized.parquet").exists()
    df = pd.read_parquet(run_dir / "incidents_anonymized.parquet")
    assert "operator_name" not in df.columns
    assert len(df) == meta["n_lignes"]

    # Export CSV des données retenues : même contenu que le parquet.
    csv_path = run_dir / "incidents_anonymized.csv"
    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert "operator_name" not in df_csv.columns
    assert len(df_csv) == meta["n_lignes"]
    assert list(df_csv.columns) == list(df.columns)
    # Anti-régression encodage : accents propres (« mécanique »), pas de mojibake.
    comments = df_csv["comment"].dropna()
    assert comments.str.contains("mécanique").any()
    assert not comments.str.contains("Ã").any()
    assert (run_dir / "figures").is_dir()
    assert len(list((run_dir / "figures").glob("*.svg"))) == 12

    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["run_id"] == meta["run_id"]
    for key in ("n_lignes", "n_colonnes", "machines_uniques", "n_nan_total"):
        assert key in runs[-1]
    assert (tmp_path / "runs.md").exists()
    assert (tmp_path / "METHODOLOGIE.md").exists()
