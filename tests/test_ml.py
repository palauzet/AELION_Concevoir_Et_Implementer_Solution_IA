"""Tests du module d'entraînement ML (TP B5) — fixtures synthétiques, pas de Postgres."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from indusense import config
from indusense.ml import training


def _gold_sample() -> pd.DataFrame:
    """~40 lignes, 2 machines (horodatages disjoints, pas de collision), gold-like.

    NaN volontaire sur ``temperature_c_mean_6h`` (1re heure de chaque machine) pour exercer
    l'imputeur médiane. ``location`` constante = "L1" (pour tester une valeur inconnue en
    dehors de la fixture, cf. ``test_build_preprocessor_onehot_handles_unknown``).
    """
    n = 20
    rows = []
    machines = (
        (1, "MACH-01", "X1", "HIGH", "A", pd.Timestamp("2025-01-01 00:00")),
        (2, "MACH-02", "X2", "LOW", "B", pd.Timestamp("2025-01-01 00:30")),
    )
    for pk, code, model, crit, line, start in machines:
        ts = pd.date_range(start, periods=n, freq="h")
        for k, t in enumerate(ts):
            rows.append({
                "machine_pk": pk,
                "machine_code": code,
                "timestamp": t,
                "model": model,
                "criticality": crit,
                "production_line": line,
                "location": "L1",
                "shift": "nuit" if t.hour < 6 else "jour",
                "temperature_c_mean_6h": np.nan if k == 0 else 40.0 + k * 0.5,
                "pressure_bar_mean_24h": 190.0 + k,
                "age_days": 100.0 + pk,
                "hour_sin": np.sin(2 * np.pi * t.hour / 24),
                "rul_hours": float(n - k),
                "rul_censored": 0,
                "label_incident_6h": int(pk == 1 and k % 6 == 5),
                "label_incident_12h": int(pk == 1 and k % 6 >= 3),
                "label_incident_24h": int(k % 4 == 3),
                "label_incident_48h": int(k % 3 == 2),
            })
    df = pd.DataFrame(rows).sort_values(["timestamp", "machine_pk"]).reset_index(drop=True)
    n_total = len(df)
    df["split"] = np.select(
        [df.index < int(n_total * 0.6), df.index < int(n_total * 0.8)],
        ["train", "val"], default="test",
    )
    return df


def test_split_features_target_excludes_leakage() -> None:
    df = _gold_sample()
    X, y, numeric_cols, categorical_cols = training.split_features_target(df, horizon=24)

    assert "rul_hours" not in X.columns and "rul_censored" not in X.columns
    for h in config.LABEL_HORIZONS:
        assert f"label_incident_{h}h" not in X.columns
    for c in training.IDENTIFIER_COLS:
        assert c not in X.columns
    assert y.equals(df["label_incident_24h"])
    assert set(categorical_cols) == set(training.CATEGORICAL_COLS)
    assert "temperature_c_mean_6h" in numeric_cols


def test_build_preprocessor_median_imputation_train_only() -> None:
    df = _gold_sample()
    X, _y, numeric_cols, categorical_cols = training.split_features_target(df)
    masks = training.train_val_test_masks(df)
    X_train = X[masks["train"]]

    pre = training.build_preprocessor(numeric_cols, categorical_cols, scale=False)
    pre.fit(X_train)
    out = pre.transform(X_train)

    train_median = X_train["temperature_c_mean_6h"].median()
    col_idx = numeric_cols.index("temperature_c_mean_6h")
    was_nan = X_train["temperature_c_mean_6h"].isna().to_numpy()
    assert was_nan.any(), "la fixture doit contenir au moins un NaN train à imputer"
    assert np.allclose(out[was_nan, col_idx], train_median)


def test_build_preprocessor_onehot_handles_unknown() -> None:
    X_train = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "model": ["X1", "X1", "X2"]})
    pre = training.build_preprocessor(["num1"], ["model"], scale=False)
    pre.fit(X_train)

    X_new = pd.DataFrame({"num1": [4.0], "model": ["INCONNU"]})
    out = pre.transform(X_new)  # ne doit pas lever d'exception

    cat_block = out[:, 1:]  # après la seule colonne numérique
    assert np.allclose(cat_block, 0.0)


def test_compute_scale_pos_weight() -> None:
    y_train = pd.Series([0, 0, 0, 1])
    assert training.compute_scale_pos_weight(y_train) == 3.0


def test_time_series_cv_chronological_order() -> None:
    df = _gold_sample()
    X, y, numeric_cols, categorical_cols = training.split_features_target(df)

    models_a = training.build_models(numeric_cols, categorical_cols, scale_pos_weight=1.0)
    sorted_result = training.time_series_cv(
        models_a["logreg"], X, y, df["timestamp"], n_splits=3
    )

    shuffled_idx = df.sample(frac=1, random_state=1).index
    X_shuf, y_shuf = X.loc[shuffled_idx], y.loc[shuffled_idx]
    ts_shuf = df["timestamp"].loc[shuffled_idx]
    models_b = training.build_models(numeric_cols, categorical_cols, scale_pos_weight=1.0)
    shuffled_result = training.time_series_cv(
        models_b["logreg"], X_shuf, y_shuf, ts_shuf, n_splits=3
    )

    # Même résultat quel que soit l'ordre d'entrée -> confirme le tri interne par timestamp seul.
    assert sorted_result == shuffled_result


def test_build_comparison_table_sorted_by_pr_auc() -> None:
    def _metrics(pr_auc: float) -> dict:
        return {"pr_auc_test": pr_auc, "roc_auc_test": 0.6,
                "precision": 0.5, "recall": 0.5, "f1": 0.5,
                "tn": 1, "fp": 2, "fn": 3, "tp": 4}

    def _cv() -> dict:
        return {"pr_auc_cv_mean": 0.4, "pr_auc_cv_std": 0.01,
                "roc_auc_cv_mean": 0.55, "roc_auc_cv_std": 0.02}

    results = {
        "logreg": {"metrics": _metrics(0.30), "cv": _cv()},
        "random_forest": {"metrics": _metrics(0.55), "cv": _cv()},
        "xgboost": {"metrics": _metrics(0.42), "cv": _cv()},
    }
    table = training.build_comparison_table(results)

    assert list(table.columns) == training.COMPARISON_COLUMNS
    assert table.iloc[0]["model"] == "random_forest"
    assert table["pr_auc_test"].is_monotonic_decreasing


def test_confusion_grid_matches_course_convention() -> None:
    """La figure doit suivre la convention du cours : Panne (positif) en premier sur les
    deux axes -> TP en haut à gauche, FN en haut à droite, FP en bas à gauche, TN en bas
    à droite (et non l'ordre 0/1 de sklearn, négatif d'abord, qui inverserait la lecture)."""
    row = pd.Series({"tn": 1, "fp": 2, "fn": 3, "tp": 4})
    grid = training._confusion_grid(row)
    assert grid[0, 0] == 4  # TP
    assert grid[0, 1] == 3  # FN
    assert grid[1, 0] == 2  # FP
    assert grid[1, 1] == 1  # TN


def test_precision_recall_curve_data_matches_sklearn_at_threshold() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1, 0, 1, 0])
    proba = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.55, 0.9])

    pr_df = training.precision_recall_curve_data(y_true, proba)

    for t in pr_df["threshold"].sample(min(3, len(pr_df)), random_state=0):
        y_pred = (proba >= t).astype(int)
        row = pr_df.loc[np.isclose(pr_df["threshold"], t)].iloc[0]
        assert np.isclose(row["precision"], precision_score(y_true, y_pred, zero_division=0))
        assert np.isclose(row["recall"], recall_score(y_true, y_pred, zero_division=0))
        assert np.isclose(row["f1"], f1_score(y_true, y_pred, zero_division=0))


def test_best_threshold_picks_max_metric() -> None:
    pr_df = pd.DataFrame({
        "threshold": [0.2, 0.5, 0.8],
        "precision": [0.5, 0.6, 0.9],
        "recall": [0.9, 0.6, 0.2],
        "f1": [0.64, 0.60, 0.33],
        "f2": [0.78, 0.60, 0.24],
    })
    assert training.best_threshold(pr_df, "f1") == 0.2
    assert training.best_threshold(pr_df, "f2") == 0.2


def test_fit_winner_pipeline_has_no_persistence_side_effects(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ML_DIR", tmp_path / "ml")
    df = _gold_sample()

    pipeline, X_test, y_test = training.fit_winner_pipeline(24, "logreg", df=df)

    masks = training.train_val_test_masks(df)
    assert len(X_test) == int(masks["test"].sum())
    assert len(y_test) == int(masks["test"].sum())
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert not (tmp_path / "ml").exists()  # pas de run journalisé, contrairement à run()


def test_run_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ML_DIR", tmp_path / "ml")

    meta = training.run(df=_gold_sample(), horizon=24)

    assert meta["winner"] in {"logreg", "random_forest", "xgboost"}
    assert len(meta["comparison"]) == 3
    assert (tmp_path / "ml" / meta["run_id"] / "run_metadata.json").exists()
    figures_dir = tmp_path / "ml" / meta["run_id"] / "figures"
    assert len(list(figures_dir.glob("*.svg"))) == 3
    assert (tmp_path / "ml" / "runs.json").exists()
    assert (tmp_path / "ml" / "runs.md").exists()
    assert (tmp_path / "ml" / "mlflow.db").exists()
