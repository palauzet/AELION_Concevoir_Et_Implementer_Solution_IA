"""Optimisation d'hyperparamètres — XGBoost @ 24h (TP B7).

Repart du gagnant du POC B5 (`src/indusense/ml/training.py`) : **XGBoost, horizon 24h**
(PR-AUC test 0.450, hyperparamètres "sobres" non optimisés). Ce module cherche de meilleurs
hyperparamètres avec **Optuna** (TPE, 50 essais par défaut), en validation croisée
**5 folds strictement train-only** (jamais `training.time_series_cv`, qui mélange
train+val+test — acceptable pour une estimation de robustesse en B5, inadapté à une
sélection de modèle qui fuiterait sinon val/test dans le tuning).

Le découpage `TimeSeriesSplit` ne dépend que des données (timestamps du train), pas des
hyperparamètres testés : la **pertinence statistique de chaque fold est calculée une seule
fois**, avant `study.optimize(...)`, jamais recalculée à chaque essai.

Suivi : un run MLflow imbriqué par essai (params + PR-AUC CV, sans artefact modèle) sous un
run parent "étude", + un run distinct pour le modèle final réentraîné (train uniquement,
avec artefact). Étude Optuna elle-même journalisée dans un backend SQLite séparé
(`artifacts/tuning/optuna.db`).

Sorties : run versionné `artifacts/tuning/<run_id>/` (figures + `best_params.json` +
`split_fold_relevance.csv`), journal (`runs.json`/`runs.md`/`METHODOLOGIE.md`, même
convention que les autres modules).

Prérequis : `indusense-gold` a peuplé `gold.dataset` / `data/processed/gold_dataset.parquet`.

Usage :

    uv run indusense-tune
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from indusense import config  # noqa: E402
from indusense.ml import training  # noqa: E402

# Optuna logue chaque essai en INFO par défaut — bruyant sur 50 essais, aligné sur la
# discipline mlflow (training.py) de ne garder que les erreurs réelles.
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.ERROR)


# --- Folds fixes (calculés une fois, jamais recalculés par essai) -------------------------
def make_train_folds(
    timestamp_train: pd.Series, n_splits: int = config.TUNING_N_SPLITS
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Tri chronologique (timestamp seul, même logique que ``training.time_series_cv``) +
    ``TimeSeriesSplit``. ``order``/``folds`` sont calculés une seule fois et réutilisés par
    tous les essais Optuna ET par le contrôle de pertinence des folds.
    """
    order = np.argsort(timestamp_train.to_numpy(), kind="stable")
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(order))
    return order, folds


# --- Pertinence statistique des folds (une fois, avant l'étude) --------------------------
def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Écart standardisé (Austin 2009) : ``(b.mean()-a.mean())/pooled_std`` ; 0.0 si
    ``pooled_std`` nul (même formule que ``gold.compare_splits_numeric``)."""
    pooled_std = float(np.sqrt((np.var(a) + np.var(b)) / 2))
    return float((np.mean(b) - np.mean(a)) / pooled_std) if pooled_std > 0 else 0.0


def compare_fold_partitions(
    X_sorted: pd.DataFrame, y_sorted: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]], numeric_cols: list[str],
) -> pd.DataFrame:
    """Pertinence statistique de chaque fold — calculée **une fois**, avant l'étude Optuna.

    Une ligne par fold : taux de positifs inner-train vs inner-val (+ écart absolu), écart
    standardisé (SMD) maximal sur les features numériques (+ feature concernée). Les
    catégorielles sont volontairement exclues : leur stabilité globale a déjà été validée
    dans ``gold.py`` (session précédente) — la revérifier sur des sous-échantillons
    chronologiques de train n'apporterait pas de signal nouveau. Diagnostic, pas un filtre :
    un écart notable n'exclut aucun fold, il est journalisé pour information.
    """
    rows = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        y_tr, y_va = y_sorted.iloc[tr_idx], y_sorted.iloc[va_idx]
        smds = {
            col: _smd(X_sorted[col].iloc[tr_idx].to_numpy(), X_sorted[col].iloc[va_idx].to_numpy())
            for col in numeric_cols
        }
        max_feature = max(smds, key=lambda c: abs(smds[c])) if smds else None
        rows.append({
            "fold": i, "n_train": len(tr_idx), "n_val": len(va_idx),
            "pos_rate_train": round(float(y_tr.mean()), 4),
            "pos_rate_val": round(float(y_va.mean()), 4),
            "pos_rate_abs_diff": round(abs(float(y_va.mean()) - float(y_tr.mean())), 4),
            "smd_abs_max": round(abs(smds[max_feature]), 4) if max_feature else 0.0,
            "smd_max_feature": max_feature,
            "n_features_smd_notable": sum(1 for v in smds.values() if abs(v) >= 0.1),
        })
    return pd.DataFrame(rows)


# --- Espace de recherche + pipeline XGBoost paramétré -------------------------------------
def suggest_params(trial: optuna.Trial) -> dict:
    """Espace de recherche XGBoost (9 hyperparamètres). ``scale_pos_weight``, `eval_metric`,
    `n_jobs`, `random_state` restent fixes (cf. ``build_xgb_pipeline``) — les défauts B5
    sobres (300/6/0.1/0.8/0.8) sont dans ces bornes, l'étude peut donc retrouver le baseline.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_xgb_pipeline(
    numeric_cols: list[str], categorical_cols: list[str], scale_pos_weight: float,
    params: dict, random_state: int = 42,
) -> Pipeline:
    """Pipeline XGBoost paramétré par ``params`` (essai Optuna) — analogue de la branche
    ``"xgboost"`` de ``training.build_models``, même prétraitement (pas de scaling, modèle
    arbre) et mêmes constantes fixes."""
    return Pipeline([
        ("preprocessor", training.build_preprocessor(numeric_cols, categorical_cols, scale=False)),
        ("model", XGBClassifier(
            **params, scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=random_state,
        )),
    ])


# --- Étude Optuna (CV train-only, MLflow imbriqué par essai) ------------------------------
def run_study(
    X_sorted: pd.DataFrame, y_sorted: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]],
    numeric_cols: list[str], categorical_cols: list[str], scale_pos_weight: float,
    *, horizon: int, run_id: str, n_trials: int = config.TUNING_N_TRIALS,
) -> optuna.Study:
    """Crée et lance l'étude Optuna (TPE, sans pruning — les folds chronologiques ont des
    tailles de train croissantes, pas une vraie courbe d'apprentissage ; un pruning médian
    écarterait à tort un essai difficile sur un seul fold).

    Un run MLflow imbriqué par essai (params + PR-AUC CV moyen/écart-type, **sans** artefact
    modèle — sérialiser un XGBoost par essai serait un gâchis) est ouvert à l'intérieur de
    l'objectif ; suppose qu'un run MLflow parent "étude" est déjà actif (cf. ``run()``).
    """
    training.ensure_mlflow_experiment()
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    # Recalculé depuis config.TUNING_DIR à chaque appel (pas la constante figée
    # config.OPTUNA_STORAGE) pour rester redirigeable en test, même raison que
    # training.ensure_mlflow_experiment vs config.MLFLOW_TRACKING_URI.
    storage = f"sqlite:///{(config.TUNING_DIR / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_{horizon}h_{run_id}",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )

    def _objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        pipeline = build_xgb_pipeline(numeric_cols, categorical_cols, scale_pos_weight, params)
        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_sorted.iloc[tr_idx], y_sorted.iloc[tr_idx])
            proba = fold_pipeline.predict_proba(X_sorted.iloc[va_idx])[:, 1]
            score = float(average_precision_score(y_sorted.iloc[va_idx], proba))
            fold_scores.append(score)
            trial.report(score, fold_idx)  # no-op sans pruner ; prêt pour en activer un plus tard
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        trial.set_user_attr("pr_auc_cv_std", std_score)
        with mlflow.start_run(
            run_name=f"trial{trial.number:03d}_xgboost_{horizon}h_{run_id}", nested=True
        ):
            mlflow.log_params(params)
            mlflow.log_metrics({"pr_auc_cv_mean": mean_score, "pr_auc_cv_std": std_score})
        return mean_score

    study.optimize(_objective, n_trials=n_trials)
    return study


def refit_best(
    X_train: pd.DataFrame, y_train: pd.Series, numeric_cols: list[str], categorical_cols: list[str],
    scale_pos_weight: float, best_params: dict, random_state: int = 42,
) -> Pipeline:
    """Réentraîne XGBoost aux meilleurs hyperparamètres trouvés, sur le **train complet**
    (comparable au PR-AUC test B5 déjà committé — val/test restent non vus par toute la
    recherche)."""
    pipeline = build_xgb_pipeline(
        numeric_cols, categorical_cols, scale_pos_weight, best_params, random_state
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# --- Figures (SVG, même style que training.make_figures) ---------------------------------
def make_figures(
    study: optuna.Study, fold_report: pd.DataFrame,
    metrics_tuned: dict, metrics_baseline: dict, out_dir: Path,
) -> list[str]:
    """4 figures : historique d'optimisation, importance des hyperparamètres, comparaison
    tuné vs baseline B5 (test), pertinence statistique des folds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["svg.fonttype"] = "none"
    files: list[str] = []

    def _save(fig: plt.Figure, name: str, caption: str) -> None:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5,
                 style="italic", color="#444444", wrap=True)
        fig.savefig(out_dir / name, format="svg", bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    # 1. Historique d'optimisation
    trials_df = study.trials_dataframe()
    running_best = trials_df["value"].cummax()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials_df["number"], trials_df["value"], "o", ms=3, alpha=0.4,
             color="#4C72B0", label="Essai")
    ax.plot(trials_df["number"], running_best, color="#C44E52", label="Meilleur courant")
    ax.set(title="1. Historique d'optimisation", xlabel="Essai",
           ylabel="PR-AUC CV (moyenne 5 folds)")
    ax.legend()
    _save(fig, "01_historique_optimisation.svg",
          "Mesure : PR-AUC moyen sur les folds train-only (jamais val/test) par essai Optuna ; "
          "la courbe rouge est le meilleur score trouvé jusqu'ici.")

    # 2. Importance des hyperparamètres (fANOVA) — nécessite une variance non nulle entre
    # essais, peut échouer avec trop peu d'essais (ex. tests, ou tous les essais à égalité) ;
    # ce n'est qu'un diagnostic, on ne bloque pas la génération des autres figures pour ça.
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())[::-1]
        values = list(importances.values())[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names, values, color="#55A868")
        ax.set(title="2. Importance des hyperparamètres (fANOVA)", xlabel="Importance relative")
        _save(fig, "02_importance_hyperparametres.svg",
              "Mesure : contribution de chaque hyperparamètre à la variance du PR-AUC observé "
              "sur l'ensemble des essais (fANOVA, scikit-learn).")
    except (RuntimeError, ValueError):
        pass

    # 3. Baseline B5 vs tuné (test)
    x = np.arange(1)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - width / 2, [metrics_baseline["pr_auc_test"]], width,
           label="B5 (sobre)", color="#4C72B0")
    ax.bar(x + width / 2, [metrics_tuned["pr_auc_test"]], width,
           label="Tuné (Optuna)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(["PR-AUC test"])
    ax.legend()
    ax.set(title="3. Baseline B5 vs tuné (test)", ylabel="PR-AUC")
    _save(fig, "03_baseline_vs_tune.svg",
          "Mesure : PR-AUC sur le split test (jamais vu pendant la recherche), XGBoost 24h — "
          "hyperparamètres sobres (B5) vs optimisés (Optuna, ce module).")

    # 4. Pertinence statistique des folds (train-only, calculée une fois)
    x4 = np.arange(len(fold_report))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x4 - width / 2, fold_report["pos_rate_train"], width,
           label="Inner-train", color="#4C72B0")
    ax.bar(x4 + width / 2, fold_report["pos_rate_val"], width,
           label="Inner-val", color="#DD8452")
    for _, row in fold_report.iterrows():
        ax.text(row["fold"], max(row["pos_rate_train"], row["pos_rate_val"]),
                f"|SMD|={row['smd_abs_max']:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x4)
    ax.set_xticklabels([f"fold {i}" for i in fold_report["fold"]])
    ax.legend()
    ax.set(title="4. Pertinence statistique des folds (train-only)",
           ylabel="Taux de positifs")
    _save(fig, "04_pertinence_folds.svg",
          "Mesure : comparabilité de chaque fold, calculée une seule fois (pas par essai) — "
          "diagnostic, n'exclut aucun fold. |SMD|>=0.1 = écart notable sur au moins une feature.")
    return files


# --- Journal + méthodologie (même convention que training.py/gold.py) --------------------
def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — optimisation hyperparamètres (B7)\n\n"
        "| Run | Horizon | Essais | PR-AUC B5 (test) | PR-AUC tuné (test) | Δ | "
        "PR-AUC CV (moy±σ) | Folds \\|SMD\\| max |\n"
        "|---|---:|---:|---:|---:|---:|---|---:|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['horizon']}h | {r['n_trials']} | {r['pr_auc_test_baseline']} | "
        f"{r['pr_auc_test']} | {r['delta']:+.4f} | {r['pr_auc_cv_mean']}±{r['pr_auc_cv_std']} | "
        f"{r['smd_abs_max']} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TUNING_DIR / "runs.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({k: meta[k] for k in
                 ("run_id", "horizon", "n_trials", "pr_auc_test_baseline", "pr_auc_test",
                  "delta", "pr_auc_cv_mean", "pr_auc_cv_std", "smd_abs_max")})
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.TUNING_DIR / "runs.md").write_text(_render_runs_md(runs), encoding="utf-8")


def write_methodologie() -> None:
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — optimisation d'hyperparamètres (B7)

## Rôle
Optimise les hyperparamètres du gagnant du POC B5 — **XGBoost, horizon 24h** — avec
**Optuna** (TPE), plutôt que les valeurs "sobres" non optimisées de `training.build_models`.

## Validation croisée — strictement train-only (écart volontaire vs B5)
`training.time_series_cv` calcule sa CV sur **tout** le dataset (train+val+test réunis) :
acceptable pour une estimation de robustesse informative en B5, mais fuiterait val/test
dans la sélection de modèle si réutilisé ici. Ce module construit donc ses folds
(`TimeSeriesSplit`, tri par `timestamp` seul) **uniquement sur le train** — val et test
restent totalement non vus par la recherche d'hyperparamètres.

## Pertinence statistique des folds — calculée une fois, jamais par essai
Le découpage des folds ne dépend que des données (timestamps), pas des hyperparamètres
testés : `compare_fold_partitions` (taux de positifs inner-train vs inner-val, écart
standardisé — SMD, Austin 2009 — maximal sur les features numériques) est calculé **une
seule fois avant `study.optimize(...)`**, jamais recalculé à chaque essai. Diagnostic (écrit
en CSV + figure), pas un filtre : un écart notable n'exclut aucun fold.

## Espace de recherche (9 hyperparamètres XGBoost)
`n_estimators` [100-800], `max_depth` [3-10], `learning_rate` [0.01-0.3, log], `subsample`
[0.5-1.0], `colsample_bytree` [0.5-1.0], `min_child_weight` [1-10], `gamma` [0.0-5.0],
`reg_alpha`/`reg_lambda` [1e-3-10, log]. Fixés : `scale_pos_weight` (train), `eval_metric`,
`n_jobs`, `random_state`.

## Objectif et budget
PR-AUC moyen sur les folds (maximiser), 50 essais par défaut, sampler TPE (graine fixe),
**sans pruning** (les folds chronologiques ont des tailles de train croissantes, pas une
vraie courbe d'apprentissage — un pruning médian écarterait à tort un essai difficile sur un
seul fold).

## Réentraînement final et comparaison
Le pipeline aux meilleurs hyperparamètres est réentraîné sur le **train complet**, évalué
une fois sur **test** (jamais vu pendant la recherche), et comparé au XGBoost "sobre" B5
(même train/test) pour un delta honnête.

## Suivi
Un run MLflow imbriqué par essai (params + PR-AUC CV, sans artefact modèle) sous un run
parent "étude", + un run distinct pour le modèle final (avec artefact). Étude Optuna
journalisée dans un backend SQLite séparé (`artifacts/tuning/optuna.db`). Les deux backends
(mlflow.db, optuna.db) sont hors versionnement.
"""
    (config.TUNING_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration -------------------------------------------------------------------
def run(
    source: str = "parquet",
    horizon: int = config.ML_DEFAULT_HORIZON,
    n_trials: int = config.TUNING_N_TRIALS,
    n_splits: int = config.TUNING_N_SPLITS,
    df: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> dict:
    """Optimise les hyperparamètres XGBoost (horizon donné) via Optuna, CV train-only à
    ``n_splits`` folds, journalise le run, retourne les métadonnées.

    ``df`` optionnel : injecte un DataFrame gold déjà chargé (tests / notebook), même esprit
    que ``training.run(df=...)``.
    """
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.TUNING_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    gold_df = df if df is not None else training.load_gold(source=source)
    X, y, numeric_cols, categorical_cols = training.split_features_target(gold_df, horizon=horizon)
    masks = training.train_val_test_masks(gold_df)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    timestamp_train = gold_df.loc[masks["train"], "timestamp"]
    scale_pos_weight = training.compute_scale_pos_weight(y_train)

    order, folds = make_train_folds(timestamp_train, n_splits=n_splits)
    X_sorted = X_train.iloc[order].reset_index(drop=True)
    y_sorted = y_train.iloc[order].reset_index(drop=True)

    fold_report = compare_fold_partitions(X_sorted, y_sorted, folds, numeric_cols)
    fold_report.to_csv(run_dir / "split_fold_relevance.csv", index=False)

    training.ensure_mlflow_experiment()
    with mlflow.start_run(run_name=f"study_xgboost_{horizon}h_{run_id}"):
        mlflow.log_params({
            "horizon": horizon, "n_trials": n_trials, "n_splits": n_splits,
            "sampler": "TPE", "pruner": "none",
        })
        study = run_study(
            X_sorted, y_sorted, folds, numeric_cols, categorical_cols, scale_pos_weight,
            horizon=horizon, run_id=run_id, n_trials=n_trials,
        )
        best_params = study.best_params
        pr_auc_cv_mean = float(study.best_value)
        pr_auc_cv_std = float(study.best_trial.user_attrs.get("pr_auc_cv_std", 0.0))

        tuned_pipeline = refit_best(
            X_train, y_train, numeric_cols, categorical_cols, scale_pos_weight, best_params
        )
        metrics_tuned = training.evaluate(tuned_pipeline, X_test, y_test)

        baseline_pipeline = training.build_models(
            numeric_cols, categorical_cols, scale_pos_weight
        )["xgboost"]
        baseline_pipeline.fit(X_train, y_train)
        metrics_baseline = training.evaluate(baseline_pipeline, X_test, y_test)

        figures = make_figures(study, fold_report, metrics_tuned, metrics_baseline, fig_dir)

        params_log = {**best_params, "horizon": horizon, "n_trials": n_trials, "n_splits": n_splits}
        flat_metrics = {
            **metrics_tuned, "pr_auc_cv_mean": pr_auc_cv_mean, "pr_auc_cv_std": pr_auc_cv_std,
            "pr_auc_test_baseline": metrics_baseline["pr_auc_test"],
        }
        training.log_run_to_mlflow(
            f"xgboost_tuned_{horizon}h_{run_id}", tuned_pipeline, params_log, flat_metrics,
            [fig_dir / f for f in figures], nested=True,
        )

    delta = round(metrics_tuned["pr_auc_test"] - metrics_baseline["pr_auc_test"], 4)
    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "horizon": horizon,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "best_params": best_params,
        "pr_auc_test": round(metrics_tuned["pr_auc_test"], 4),
        "pr_auc_test_baseline": round(metrics_baseline["pr_auc_test"], 4),
        "delta": delta,
        "pr_auc_cv_mean": round(pr_auc_cv_mean, 4),
        "pr_auc_cv_std": round(pr_auc_cv_std, 4),
        "smd_abs_max": (
            round(float(fold_report["smd_abs_max"].max()), 4) if len(fold_report) else 0.0
        ),
        "figures": figures,
        "chemin": str(run_dir),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_methodologie()
    append_run(meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Optimisation d'hyperparamètres XGBoost via Optuna (CV 5 folds train-only)."
    )
    parser.add_argument("--source", choices=["parquet", "sql"], default="parquet")
    parser.add_argument("--horizon", type=int, choices=list(config.LABEL_HORIZONS),
                        default=config.ML_DEFAULT_HORIZON)
    parser.add_argument("--n-trials", type=int, default=config.TUNING_N_TRIALS)
    parser.add_argument("--n-splits", type=int, default=config.TUNING_N_SPLITS)
    args = parser.parse_args(argv)

    meta = run(source=args.source, horizon=args.horizon,
               n_trials=args.n_trials, n_splits=args.n_splits)
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(f"  horizon={meta['horizon']}h essais={meta['n_trials']}")
    print(f"  PR-AUC test : baseline B5={meta['pr_auc_test_baseline']} -> "
          f"tuné={meta['pr_auc_test']} (delta={meta['delta']:+.4f})")
    print(f"  PR-AUC CV (train-only) : {meta['pr_auc_cv_mean']} ± {meta['pr_auc_cv_std']}")
    print(f"  pertinence folds : |SMD| max = {meta['smd_abs_max']}")


if __name__ == "__main__":
    main()
