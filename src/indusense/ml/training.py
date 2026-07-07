"""Entraînement & comparaison de modèles — maintenance prédictive (TP B5).

Repart du **gold** InduSense (``src/indusense/data/gold.py``, pas du dataset générique du
brief) : classification binaire ``label_incident_{6,12,24,48}h``, avec deux pièges à éviter
— la **fuite temporelle** et le **déséquilibre de classes**.

Le prétraitement (imputation médiane, standardisation, encodage) est fait dans des
``Pipeline`` scikit-learn **fittées sur le train uniquement** : c'est ici que se referme le
correctif anti-fuite du silver (retrait de la médiane de secours globale) — l'imputeur
différé promis à cette occasion.

Sorties : run versionné ``artifacts/ml/<run_id>/`` (figures + tableau comparatif +
run_metadata.json), journal (``runs.json``/``runs.md``/``METHODOLOGIE.md``, même convention
que les autres modules) + tracking MLflow (backend SQLite local, hors versionnement).

Prérequis : ``indusense-gold`` a peuplé ``gold.dataset`` / ``data/processed/gold_dataset.parquet``.

Usage :

    uv run indusense-train
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
import mlflow.sklearn  # noqa: E402

# MLflow est bavard en INFO/WARNING (export d'environnement, dépréciations internes) —
# ne garder que les erreurs réelles pour ne pas noyer la sortie du notebook/CLI.
logging.getLogger("mlflow").setLevel(logging.ERROR)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from indusense import config  # noqa: E402
from indusense.data.db import get_engine  # noqa: E402

# --- Rôle des colonnes du gold (fidèle au schéma réel, pas au brief générique) ----
IDENTIFIER_COLS = ["machine_pk", "machine_code", "timestamp", "split"]
CATEGORICAL_COLS = ["model", "criticality", "production_line", "location", "shift"]
# `rul_hours`/`rul_censored` encodent directement la cible (heures jusqu'au prochain
# incident) : ce sont des alternatives de cible (régression de survie), jamais des features.
LEAKAGE_COLS = ["rul_hours", "rul_censored"]
LABEL_COLS = [f"label_incident_{h}h" for h in config.LABEL_HORIZONS]

COMPARISON_COLUMNS = [
    "model", "pr_auc_test", "roc_auc_test", "pr_auc_cv_mean", "pr_auc_cv_std",
    "precision", "recall", "f1", "tn", "fp", "fn", "tp",
]


# --- Chargement ---------------------------------------------------------------
def load_gold(source: str = "parquet", engine=None) -> pd.DataFrame:
    """Charge NOTRE gold (parquet canonique par défaut ; ``source="sql"`` = ``gold.dataset``).

    Trié par ``(machine_pk, timestamp)`` (US B5 étape 1). Le parquet fonctionne sans base
    (pas besoin de Docker) ; le mode SQL relit la table matérialisée par ``indusense-gold``.
    """
    if source == "parquet":
        if not config.GOLD_DATASET.exists():
            raise FileNotFoundError(
                f"Gold introuvable ({config.GOLD_DATASET}). Reconstruire : "
                "`docker compose -f docker/pgdocker/docker-compose.yml up -d` puis "
                "`uv run indusense-silver` et `uv run indusense-gold`."
            )
        df = pd.read_parquet(config.GOLD_DATASET)
    elif source == "sql":
        with (engine or get_engine()).connect() as conn:
            df = pd.read_sql(f'SELECT * FROM "{config.SCHEMA_GOLD}"."dataset"', conn)
    else:
        raise ValueError(f"source inconnue : {source!r} (attendu 'parquet' ou 'sql')")
    return df.sort_values(["machine_pk", "timestamp"]).reset_index(drop=True)


# --- Features / cible / splits -------------------------------------------------
def split_features_target(
    df: pd.DataFrame, horizon: int = config.ML_DEFAULT_HORIZON
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Sépare ``X``/``y`` : exclut identifiants, RUL (fuite) et **les 4 labels** de X."""
    target_col = f"label_incident_{horizon}h"
    non_feature = set(IDENTIFIER_COLS) | set(LEAKAGE_COLS) | set(LABEL_COLS)
    X = df.drop(columns=[c for c in non_feature if c in df.columns])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    return X, y, numeric_cols, categorical_cols


def train_val_test_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Masques booléens depuis la colonne ``split`` (chronologique) — aucun split aléatoire."""
    return {name: (df["split"] == name) for name in ("train", "val", "test")}


# --- Prétraitement (fit train uniquement) --------------------------------------
def build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str], *, scale: bool
) -> ColumnTransformer:
    """Imputation médiane (numérique) + encodage (catégoriel), le tout fit-train dans la Pipeline.

    L'imputeur médiane numérique referme le correctif anti-fuite du silver (la médiane de
    secours globale, fuyante, avait été retirée en attendant ce pipeline modèle).
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop",
    )


# --- Déséquilibre ---------------------------------------------------------------
def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """``negatifs / positifs`` sur le TRAIN uniquement (XGBoost ``scale_pos_weight``)."""
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    return (neg / pos) if pos else 1.0


# --- Modèles ---------------------------------------------------------------------
def build_models(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_pos_weight: float,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """3 pipelines (prétraitement + modèle), du plus simple au plus expressif.

    LogReg/RF : ``class_weight="balanced"``. XGBoost : ``scale_pos_weight`` (train). HP
    XGBoost sobres — l'optimisation est renvoyée au module B7.
    """
    return {
        "logreg": Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, scale=True)),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, scale=False)),
            ("model", RandomForestClassifier(
                class_weight="balanced", n_estimators=300, n_jobs=-1, random_state=random_state
            )),
        ]),
        "xgboost": Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, scale=False)),
            ("model", XGBClassifier(
                scale_pos_weight=scale_pos_weight, n_estimators=300, max_depth=6,
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                eval_metric="aucpr", n_jobs=-1, random_state=random_state,
            )),
        ]),
    }


# --- Évaluation --------------------------------------------------------------------
def evaluate(
    pipeline: Pipeline, X: pd.DataFrame, y_true: pd.Series,
    *, threshold: float = config.ML_DECISION_THRESHOLD,
) -> dict:
    """PR-AUC (métrique primaire, déséquilibre) + ROC-AUC + matrice de confusion au seuil.

    Clés suffixées ``_test`` : ``evaluate`` n'est appelée que sur le split test dans ce
    module (cf. ``run()``), en cohérence avec ``COMPARISON_COLUMNS``.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc_test": float(average_precision_score(y_true, proba)),
        "roc_auc_test": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def time_series_cv(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, timestamp: pd.Series, n_splits: int = 5
) -> dict:
    """Validation croisée **temporelle** (estimation robuste), triée par horodatage seul.

    Tri par ``timestamp`` seul (pas ``machine_pk`` puis ``timestamp``) pour des folds
    globalement chronologiques : les features sont déjà calculées par machine (fenêtres
    segment-aware), donc mélanger les machines dans le temps est sûr ici.
    """
    order = np.argsort(timestamp.to_numpy(), kind="stable")
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    cv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_validate(
        pipeline, X_sorted, y_sorted, cv=cv,
        scoring=["average_precision", "roc_auc"],
    )
    return {
        "pr_auc_cv_mean": float(scores["test_average_precision"].mean()),
        "pr_auc_cv_std": float(scores["test_average_precision"].std()),
        "roc_auc_cv_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_cv_std": float(scores["test_roc_auc"].std()),
    }


# --- Tableau comparatif ------------------------------------------------------------
def build_comparison_table(results: dict) -> pd.DataFrame:
    """Une ligne par modèle, triée par ``pr_auc_test`` décroissant (ligne 0 = gagnant)."""
    rows = [{"model": name, **r["metrics"], **r["cv"]} for name, r in results.items()]
    return pd.DataFrame(rows)[COMPARISON_COLUMNS].sort_values(
        "pr_auc_test", ascending=False
    ).reset_index(drop=True)


def _confusion_grid(row: pd.Series) -> np.ndarray:
    """Grille 2x2 dans la convention pédagogique du cours (schéma de référence) : la classe
    positive « Panne » est mise **en premier** sur les deux axes -> TP en haut à gauche,
    FN en haut à droite, FP en bas à gauche, TN en bas à droite. Sans ça, l'ordre 0/1 de
    sklearn (négatif d'abord) inverse visuellement la lecture par rapport au schéma du cours.
    """
    return np.array([[row["tp"], row["fn"]], [row["fp"], row["tn"]]])


# --- Figures (SVG, même style que gold.make_figures) -------------------------------
def make_figures(results: dict, comparison: pd.DataFrame, out_dir: Path) -> list[str]:
    """3 figures : comparaison PR/ROC-AUC, matrice de confusion du gagnant, PR-AUC CV."""
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

    # 1. Comparaison PR-AUC / ROC-AUC (test)
    x = np.arange(len(comparison))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, comparison["pr_auc_test"], width, label="PR-AUC", color="#C44E52")
    ax.bar(x + width / 2, comparison["roc_auc_test"], width, label="ROC-AUC", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["model"])
    ax.legend()
    ax.set(title="1. Comparaison PR-AUC / ROC-AUC (test)", ylabel="Score")
    _save(fig, "01_comparaison_pr_roc_auc.svg",
          "Mesure : score sur le split test (jamais utilisé pour l'entraînement/tuning). "
          "PR-AUC = métrique primaire (classes déséquilibrées) ; ne pas lire l'accuracy.")

    # 2. Matrice de confusion du gagnant (couleurs de statut : vert=TP, rouge=FN — danger,
    # orange=FP, bleu=TN — cf. _confusion_grid pour la convention d'axes).
    winner = comparison.iloc[0]
    cm = _confusion_grid(winner)
    status = np.array([[0, 1], [2, 3]])  # TP, FN / FP, TN -> indices de couleur
    status_cmap = ListedColormap(["#55A868", "#C44E52", "#DD8452", "#4C72B0"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(status, annot=cm.astype(str), fmt="", cmap=status_cmap, cbar=False,
                vmin=0, vmax=3, linewidths=1.5, linecolor="white",
                xticklabels=["Prédit : Panne", "Prédit : OK"],
                yticklabels=["Réel : Panne", "Réel : OK"],
                annot_kws={"fontsize": 13, "fontweight": "bold", "color": "white"}, ax=ax)
    ax.set(title=f"2. Matrice de confusion — {winner['model']} "
                 f"(seuil {config.ML_DECISION_THRESHOLD})")
    _save(fig, "02_matrice_confusion.svg",
          "Mesure : erreurs au seuil fixe sur le test. Faux négatifs (rouge) = pannes "
          "manquées, coût métier prioritaire ; faux positifs (orange) = fausses alertes.")

    # 3. PR-AUC — validation croisée temporelle (moyenne ± écart-type)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(comparison["model"], comparison["pr_auc_cv_mean"],
           yerr=comparison["pr_auc_cv_std"], color="#55A868", capsize=4)
    ax.set(title="3. PR-AUC — validation croisée temporelle (TimeSeriesSplit)",
           ylabel="PR-AUC (moyenne ± écart-type, 5 folds)")
    _save(fig, "03_cv_pr_auc.svg",
          "Mesure : robustesse de l'estimation PR-AUC sur des folds strictement "
          "chronologiques (pas de mélange aléatoire).")
    return files


# --- MLflow --------------------------------------------------------------------------
def log_run_to_mlflow(
    name: str, pipeline: Pipeline, params: dict, metrics: dict, fig_paths: list[Path]
) -> None:
    """Log params/métriques/figures/modèle sur un run MLflow (backend SQLite local).

    URI de tracking + emplacement des artefacts recalculés depuis ``config.ML_DIR`` à chaque
    appel (pas une constante figée) pour rester redirigeable en test (``config.ML_DIR``
    monkeypatché -> DB + artefacts isolés). Sans ``artifact_location`` explicite à la création
    de l'expérience, MLflow écrit les artefacts (figures, modèle) dans un ``mlruns/`` à la
    racine du process (CWD), hors de ``artifacts/ml/`` — d'où la création manuelle ici.
    Les métriques NaN (ex. ROC-AUC indéfini sur un fold CV dégénéré, trop peu de positifs)
    sont exclues : le backend SQLite MLflow rejette certains doublons NaN au même timestamp.
    """
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(config.ML_DIR / 'mlflow.db').as_posix()}")
    artifact_dir = config.ML_DIR / "mlflow_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is None:
        client.create_experiment(
            config.MLFLOW_EXPERIMENT, artifact_location=artifact_dir.resolve().as_uri()
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    clean_metrics = {k: v for k, v in metrics.items() if not (isinstance(v, float) and np.isnan(v))}
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(clean_metrics)
        for fig_path in fig_paths:
            mlflow.log_artifact(str(fig_path))
        # cloudpickle : le format par défaut (skops) refuse les types XGBoost non whitelistés.
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", serialization_format="cloudpickle"
        )


# --- Journal + méthodologie (même convention que gold.py) ----------------------------
def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — entraînement ML\n\n"
        "| Run | Horizon | Gagnant | PR-AUC | ROC-AUC | Seuil | Train/Val/Test |\n"
        "|---|---:|---|---:|---:|---:|---|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['horizon']}h | {r['winner']} | {r['pr_auc_test']} | "
        f"{r['roc_auc_test']} | {r['threshold']} | "
        f"{r['split'].get('train', 0)}/{r['split'].get('val', 0)}/{r['split'].get('test', 0)} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ML_DIR / "runs.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({k: meta[k] for k in
                 ("run_id", "horizon", "winner", "pr_auc_test", "roc_auc_test",
                  "threshold", "split")})
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.ML_DIR / "runs.md").write_text(_render_runs_md(runs), encoding="utf-8")


def write_methodologie() -> None:
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — entraînement ML (TP B5)

## Rôle
Premier POC de classification binaire (« panne dans les prochaines Hh ? ») à partir du
**gold** InduSense (`gold.dataset` / `data/processed/gold_dataset.parquet`) — pas du dataset
générique décrit dans le brief du TP.

## Colonnes exclues des features (anti-fuite)
- Identifiants : `machine_pk`, `machine_code`, `timestamp`, `split`.
- RUL (`rul_hours`, `rul_censored`) : encode directement la cible (heures jusqu'au prochain
  incident) — cible alternative (survie), jamais une feature.
- Les 4 `label_incident_{6,12,24,48}h` : un seul horizon devient `y`, les 3 autres sont
  retirés de `X` (sinon fuite directe de la cible).

## Prétraitement (fit train uniquement, dans la Pipeline)
- Numérique : imputation **médiane** — referme le correctif anti-fuite du silver (la
  médiane de secours globale, fuyante, avait été retirée du silver au profit de cet
  imputeur, fitté sur le train seul). Standardisation additionnelle pour la régression
  logistique (modèle linéaire).
- Catégoriel (`model`, `criticality`, `production_line`, `location`, `shift`) : imputation
  la plus fréquente + one-hot (`handle_unknown="ignore"`).

## Déséquilibre de classes
`class_weight="balanced"` (régression logistique, Random Forest) ; `scale_pos_weight`
calculé sur le **train uniquement** (XGBoost).

## Split & validation
Split **chronologique** existant (`split` : train/val/test, 70/15/15) — pas de split
aléatoire. Modèles fittés sur train, évalués une fois sur test. Estimation robuste
complémentaire par `TimeSeriesSplit` (5 folds, tri par horodatage seul).

## Métriques
**PR-AUC** en primaire (déséquilibre de classes — ne jamais lire l'accuracy seule).
ROC-AUC en secondaire. Précision/rappel/F1 + matrice de confusion à seuil fixe
(0.5 — le choix motivé du seuil est renvoyé au module B7).

## Suivi
Tracking MLflow (backend SQLite local, `artifacts/ml/mlflow.db`, hors versionnement) +
run versionné (`artifacts/ml/<run_id>/`, figures + tableau comparatif).
"""
    (config.ML_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration -------------------------------------------------------------------
def run(
    source: str = "parquet",
    horizon: int = config.ML_DEFAULT_HORIZON,
    threshold: float = config.ML_DECISION_THRESHOLD,
    df: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> dict:
    """Entraîne et compare les 3 modèles, journalise le run, retourne les métadonnées.

    ``df`` optionnel : injecte un DataFrame gold déjà chargé (tests / notebook) au lieu de
    lire le parquet/SQL — même esprit que ``gold.build_gold(silver_dict)`` testable
    indépendamment de ``gold.run()``.
    """
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.ML_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    gold_df = df if df is not None else load_gold(source=source)
    X, y, numeric_cols, categorical_cols = split_features_target(gold_df, horizon=horizon)
    masks = train_val_test_masks(gold_df)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    scale_pos_weight = compute_scale_pos_weight(y_train)

    models = build_models(numeric_cols, categorical_cols, scale_pos_weight)
    results: dict[str, dict] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test, threshold=threshold)
        cv = time_series_cv(pipeline, X, y, gold_df["timestamp"])
        results[name] = {"pipeline": pipeline, "metrics": metrics, "cv": cv}

    comparison = build_comparison_table(results)
    figures = make_figures(results, comparison, fig_dir)
    winner_row = comparison.iloc[0]

    for name, r in results.items():
        params = {
            "horizon": horizon, "threshold": threshold, "n_splits": 5,
            "model_class": type(r["pipeline"].named_steps["model"]).__name__,
        }
        flat_metrics = {**r["metrics"], **r["cv"]}
        log_run_to_mlflow(name, r["pipeline"], params, flat_metrics,
                          [fig_dir / f for f in figures])

    comparison.to_csv(run_dir / "comparaison.csv", index=False)

    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "horizon": horizon,
        "threshold": threshold,
        "winner": str(winner_row["model"]),
        "pr_auc_test": round(float(winner_row["pr_auc_test"]), 4),
        "roc_auc_test": round(float(winner_row["roc_auc_test"]), 4),
        "split": {k: int(v.sum()) for k, v in masks.items()},
        "comparison": comparison.to_dict(orient="records"),
        "figures": figures,
        "chemin": str(run_dir),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_methodologie()
    append_run(meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement & comparaison de modèles (LogReg / Random Forest / XGBoost)."
    )
    parser.add_argument("--source", choices=["parquet", "sql"], default="parquet")
    parser.add_argument("--horizon", type=int, choices=list(config.LABEL_HORIZONS),
                        default=config.ML_DEFAULT_HORIZON)
    parser.add_argument("--threshold", type=float, default=config.ML_DECISION_THRESHOLD)
    args = parser.parse_args(argv)

    meta = run(source=args.source, horizon=args.horizon, threshold=args.threshold)
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(f"  horizon={meta['horizon']}h seuil={meta['threshold']}")
    for row in meta["comparison"]:
        print(f"  {row['model']:<14} PR-AUC={row['pr_auc_test']:.4f} "
              f"ROC-AUC={row['roc_auc_test']:.4f}")
    print(f"  gagnant : {meta['winner']} (PR-AUC={meta['pr_auc_test']}) -> B7")


if __name__ == "__main__":
    main()
