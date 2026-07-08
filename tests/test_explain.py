"""Tests du module d'explicabilité SHAP (TP B8) — fixtures synthétiques, pas de Postgres,
pas de vrai backend MLflow/Optuna en dehors de ``tmp_path``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from indusense import config
from indusense.ml import explain
from tests.test_ml import _gold_sample

VALID_BEST_PARAMS = {
    "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0.0,
    "reg_alpha": 0.01, "reg_lambda": 0.01,
}


def test_latest_tuning_run_selectionne_le_plus_recent(tmp_path) -> None:
    older, newer = tmp_path / "202601010000", tmp_path / "202602020000"
    for d in (older, newer):
        d.mkdir()
        (d / "best_params.json").write_text(json.dumps(VALID_BEST_PARAMS), encoding="utf-8")
    assert explain.latest_tuning_run(tmp_path) == newer


def test_latest_tuning_run_leve_si_aucun_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        explain.latest_tuning_run(tmp_path / "vide")


def test_load_best_params(tmp_path) -> None:
    run_dir = tmp_path / "202601010000"
    run_dir.mkdir()
    (run_dir / "best_params.json").write_text(json.dumps(VALID_BEST_PARAMS), encoding="utf-8")
    assert explain.load_best_params(run_dir) == VALID_BEST_PARAMS


def test_clean_feature_names() -> None:
    names = ["num__temperature_c_mean_6h", "cat__model_X1", "age_days"]
    assert explain._clean_feature_names(names) == [
        "temperature_c_mean_6h", "model_X1", "age_days",
    ]


def test_mean_abs_shap_table_part_normalisee_et_triee() -> None:
    shap_values = np.array([
        [1.0, -2.0, 0.0],
        [3.0, 2.0, 0.0],
    ])
    table = explain.mean_abs_shap_table(shap_values, ["f1", "f2", "f3"])
    # |SHAP| moyens : f1=2.0, f2=2.0, f3=0.0 -> triés décroissant, f3 en dernier.
    assert table["feature"].tolist()[-1] == "f3"
    assert table["mean_abs_shap"].iloc[-1] == 0.0
    assert table["importance_share"].sum() == pytest.approx(1.0)


def test_mean_abs_shap_table_gere_la_variance_nulle() -> None:
    shap_values = np.zeros((4, 2))
    table = explain.mean_abs_shap_table(shap_values, ["f1", "f2"])
    assert (table["importance_share"] == 0.0).all()


def test_compare_models_importance_rang_et_delta() -> None:
    table_b5 = pd.DataFrame({
        "feature": ["f1", "f2", "f3", "f4", "f5"],
        "mean_abs_shap": [0.5, 0.3, 0.15, 0.04, 0.01],
        "importance_share": [0.5, 0.3, 0.15, 0.04, 0.01],
    })
    # f5 volontairement absente de table_b7 (jamais censé arriver en production, cf.
    # l'assertion de run() sur les noms de features -- ici on vérifie juste que le merge
    # ne plante pas et retombe sur un rang/valeur par défaut).
    table_b7 = pd.DataFrame({
        "feature": ["f2", "f1", "f3", "f4"],
        "mean_abs_shap": [0.06, 0.025, 0.01, 0.005],
        "importance_share": [0.6, 0.25, 0.1, 0.05],
    })

    comparison = explain.compare_models_importance(table_b5, table_b7, top_n=10)

    row = comparison.set_index("feature")
    assert row.loc["f1", "rank_b5"] == 1 and row.loc["f1", "rank_b7"] == 2
    assert row.loc["f1", "delta_rank"] == -1
    assert row.loc["f2", "rank_b5"] == 2 and row.loc["f2", "rank_b7"] == 1
    assert row.loc["f2", "delta_rank"] == 1
    assert row.loc["f3", "delta_rank"] == 0
    # f5 absente de b7 -> pire rang (= max(5,4)+1 = 6) et valeurs à 0, pas de crash.
    assert row.loc["f5", "rank_b7"] == 6
    assert row.loc["f5", "share_b7"] == 0.0
    assert row.loc["f5", "mean_abs_shap_b7"] == 0.0


def test_compare_models_importance_filtre_sur_top_n() -> None:
    table_b5 = pd.DataFrame({
        "feature": ["f1", "f2", "f3"],
        "mean_abs_shap": [0.6, 0.3, 0.1],
        "importance_share": [0.6, 0.3, 0.1],
    })
    table_b7 = pd.DataFrame({
        "feature": ["f1", "f2", "f3"],
        "mean_abs_shap": [0.5, 0.3, 0.2],
        "importance_share": [0.5, 0.3, 0.2],
    })
    comparison = explain.compare_models_importance(table_b5, table_b7, top_n=1)
    # top-1 de chaque modèle est "f1" dans les deux -> union = {f1} uniquement.
    assert comparison["feature"].tolist() == ["f1"]


def test_build_pipelines_deux_cles_non_fittees() -> None:
    pipelines = explain.build_pipelines(
        ["temperature_c_mean_6h"], ["model"], scale_pos_weight=3.0,
        best_params=VALID_BEST_PARAMS,
    )
    assert set(pipelines) == {"b5_sobre", "b7_tune"}

    b5_model = pipelines["b5_sobre"].named_steps["model"]
    assert b5_model.n_estimators == 300 and b5_model.max_depth == 6  # défauts "sobres" B5

    b7_model = pipelines["b7_tune"].named_steps["model"]
    assert b7_model.n_estimators == VALID_BEST_PARAMS["n_estimators"]
    assert b7_model.max_depth == VALID_BEST_PARAMS["max_depth"]
    assert b7_model.scale_pos_weight == 3.0


def test_run_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "EXPLAIN_DIR", tmp_path / "explain")
    monkeypatch.setattr(config, "ML_DIR", tmp_path / "ml")  # confirme l'absence de MLflow

    tuning_run_dir = tmp_path / "tuning_run"
    tuning_run_dir.mkdir()
    (tuning_run_dir / "best_params.json").write_text(
        json.dumps(VALID_BEST_PARAMS), encoding="utf-8"
    )

    meta = explain.run(
        df=_gold_sample(), horizon=24, tuning_run_dir=tuning_run_dir, top_n=5,
    )

    run_dir = tmp_path / "explain" / meta["run_id"]
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "importance_b5_sobre.csv").exists()
    assert (run_dir / "importance_b7_tune.csv").exists()
    assert (run_dir / "comparaison_importance.csv").exists()
    assert len(list((run_dir / "figures").glob("*.svg"))) == 4
    assert (tmp_path / "explain" / "runs.json").exists()
    assert (tmp_path / "explain" / "runs.md").exists()
    assert (tmp_path / "explain" / "METHODOLOGIE.md").exists()
    assert isinstance(meta["top_feature_b5"], str)
    assert isinstance(meta["top_feature_b7"], str)
    assert "note_horizon" not in meta  # horizon par défaut (24h) -> pas de note

    # Aucun tracking MLflow dans ce module -- confirme qu'aucun backend n'a été créé.
    assert not (tmp_path / "ml" / "mlflow.db").exists()


def test_run_smoke_note_horizon_si_different_de_24h(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "EXPLAIN_DIR", tmp_path / "explain")

    tuning_run_dir = tmp_path / "tuning_run"
    tuning_run_dir.mkdir()
    (tuning_run_dir / "best_params.json").write_text(
        json.dumps(VALID_BEST_PARAMS), encoding="utf-8"
    )

    meta = explain.run(
        df=_gold_sample(), horizon=6, tuning_run_dir=tuning_run_dir, top_n=5,
    )
    assert "note_horizon" in meta
