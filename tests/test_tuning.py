"""Tests du module d'optimisation d'hyperparamètres (TP B7) — fixtures synthétiques, pas de
Postgres, pas de vrai backend MLflow/Optuna en dehors de `tmp_path`."""

from __future__ import annotations

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from indusense import config
from indusense.ml import training, tuning
from tests.test_ml import _gold_sample


def test_smd_zero_sur_identiques() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert tuning._smd(a, a.copy()) == 0.0
    # variance nulle des deux côtés (valeurs constantes, mais différentes) -> forcé à 0.0.
    assert tuning._smd(np.array([1.0, 1.0]), np.array([5.0, 5.0])) == 0.0


def test_smd_detecte_decalage() -> None:
    a = np.array([10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0])
    b = np.array([20.0, 20.5, 19.5, 20.2, 19.8, 20.1, 19.9, 20.3, 19.7, 20.0])
    smd = tuning._smd(a, b)
    assert smd > 1.0  # décalage net (~10) devant l'écart-type (~0.3)


def test_make_train_folds_chronologique() -> None:
    ts = pd.Series(pd.date_range("2025-01-01", periods=24, freq="h"))
    order, folds = tuning.make_train_folds(ts, n_splits=3)
    sorted_ts = ts.iloc[order].reset_index(drop=True)
    for tr_idx, va_idx in folds:
        assert tr_idx.max() < va_idx.min()
        assert sorted_ts.iloc[tr_idx].max() < sorted_ts.iloc[va_idx].min()

    # Invariant à l'ordre d'entrée : même chronologie et mêmes tailles de folds retrouvées.
    shuffled = ts.sample(frac=1, random_state=1).reset_index(drop=True)
    order2, folds2 = tuning.make_train_folds(shuffled, n_splits=3)
    sorted_ts2 = shuffled.iloc[order2].reset_index(drop=True)
    assert sorted_ts2.equals(sorted_ts)
    assert [(len(a), len(b)) for a, b in folds] == [(len(a), len(b)) for a, b in folds2]


def test_suggest_params_dans_les_bornes() -> None:
    fixed = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 1.0,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
    }
    trial = optuna.trial.FixedTrial(fixed)
    params = tuning.suggest_params(trial)
    assert set(params) == set(fixed)
    assert params == fixed  # FixedTrial lève si une des bornes déclarées exclut cette valeur


def test_compare_fold_partitions_forme_et_smd_nul() -> None:
    # Même distribution répétée des deux côtés du fold -> SMD strictement nul.
    X_sorted = pd.DataFrame({"feat": [1.0, 1.1, 0.9, 1.05, 1.0, 1.1, 0.9, 1.05]})
    y_sorted = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    folds = [(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]))]

    out = tuning.compare_fold_partitions(X_sorted, y_sorted, folds, ["feat"])

    assert list(out.columns) == [
        "fold", "n_train", "n_val", "pos_rate_train", "pos_rate_val", "pos_rate_abs_diff",
        "smd_abs_max", "smd_max_feature", "n_features_smd_notable",
    ]
    assert out.loc[0, "n_train"] == 4 and out.loc[0, "n_val"] == 4
    assert out.loc[0, "smd_abs_max"] < 0.1
    assert out.loc[0, "n_features_smd_notable"] == 0


def test_build_xgb_pipeline_applique_params() -> None:
    params = {
        "n_estimators": 111, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.7,
        "colsample_bytree": 0.7, "min_child_weight": 2, "gamma": 0.5,
        "reg_alpha": 0.01, "reg_lambda": 0.01,
    }
    pipeline = tuning.build_xgb_pipeline(["num1"], ["cat1"], scale_pos_weight=3.0, params=params)

    model = pipeline.named_steps["model"]
    assert model.n_estimators == 111
    assert model.max_depth == 4
    assert model.scale_pos_weight == 3.0

    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = {name: trans for name, trans, _cols in preprocessor.transformers}["num"]
    assert "scaler" not in numeric_pipeline.named_steps  # pas de scaling (modèle arbre)


def test_log_run_to_mlflow_nested(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ML_DIR", tmp_path / "ml")
    training.ensure_mlflow_experiment()

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0, 1])
    pipeline = Pipeline([("model", LogisticRegression())])
    pipeline.fit(X, y)

    with mlflow.start_run(run_name="parent_test"):
        # Ne doit pas lever "Run with UUID ... is already active" : c'est tout l'objet du test.
        training.log_run_to_mlflow("child_test", pipeline, {"p": 1}, {"m": 0.5}, [], nested=True)


def test_run_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ML_DIR", tmp_path / "ml")
    monkeypatch.setattr(config, "TUNING_DIR", tmp_path / "tuning")

    meta = tuning.run(df=_gold_sample(), horizon=24, n_trials=2, n_splits=3)

    assert isinstance(meta["best_params"], dict) and len(meta["best_params"]) == 9
    assert isinstance(meta["pr_auc_test"], float)
    assert isinstance(meta["pr_auc_test_baseline"], float)

    run_dir = tmp_path / "tuning" / meta["run_id"]
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "best_params.json").exists()
    assert (run_dir / "split_fold_relevance.csv").exists()
    # 4 figures normalement ; l'importance fANOVA (fig. 2) peut être ignorée si les essais
    # (n_trials=2 ici) n'ont pas assez de variance pour être évaluée (cf. make_figures).
    assert len(list((run_dir / "figures").glob("*.svg"))) in (3, 4)
    assert (tmp_path / "tuning" / "runs.json").exists()
    assert (tmp_path / "tuning" / "runs.md").exists()
    assert (tmp_path / "tuning" / "optuna.db").exists()
    assert (tmp_path / "ml" / "mlflow.db").exists()
