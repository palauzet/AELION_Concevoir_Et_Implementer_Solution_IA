"""Explicabilité SHAP — XGBoost sobre (B5) vs tuné (B7), horizon 24h (TP B8).

Compare l'importance globale des features (SHAP, ``TreeExplainer``) entre le baseline B5
(hyperparamètres "sobres", ``training.build_models``) et le modèle B7 tuné par Optuna
(``tuning.build_xgb_pipeline`` + meilleurs hyperparamètres du dernier run de
``artifacts/tuning/``) — répond à « le tuning a-t-il changé ce que le modèle regarde ? »,
pas seulement « le score a-t-il changé ? » (déjà répondu par B7 : PR-AUC test 0,4528 ->
0,5218).

Portée volontairement limitée aux explications **globales** (résumé sur tout le test) :
pas d'explication locale (waterfall par observation) ni de dependence plots, hors périmètre
de cette première itération.

**Biais d'échelle et fix retenu** : comparer les valeurs *brutes* de |SHAP| moyen entre B5
et B7 serait trompeur — B7 est un modèle plus conservateur (learning_rate, profondeur et
nombre d'arbres plus faibles), il produit donc mécaniquement des valeurs SHAP (marge,
log-odds) plus petites sur *toutes* les features, sans que ça signifie qu'il "regarde
moins" les features. La comparaison utilise donc une **part normalisée**
(``importance_share``, somme = 1 par modèle, invariante à l'échelle) et le **rang**
(``rank_b5``/``rank_b7``/``delta_rank``) comme signaux robustes ; les valeurs brutes restent
disponibles dans les CSV pour qui veut les consulter.

Les valeurs SHAP sont calculées sur les features **après prétraitement** (imputation +
one-hot), car ``TreeExplainer`` explique le modèle arbre lui-même, pas le pipeline complet.
Espace de sortie : marge (log-odds), pas probabilité — comportement par défaut de
``TreeExplainer`` pour XGBoost binaire, suffisant pour une lecture d'importance relative.
Le calcul (|SHAP| moyen) porte sur le split **test complet** ; seul le nuage de points des
figures de résumé (beeswarm) est sous-échantillonné (``config.EXPLAIN_BEESWARM_SAMPLE``,
graine fixe) pour rester lisible et léger en SVG.

Pas de tracking MLflow ici : ce module ne produit ni n'enregistre de nouveau modèle (les
deux pipelines réentraînés ne font que reproduire des modèles déjà suivis par
``training``/``tuning``) — seulement des artefacts d'analyse (figures + tableaux). Sorties :
run versionné ``artifacts/explain/<run_id>/`` (4 figures + 3 CSV d'importance), journal
(``runs.json``/``runs.md``/``METHODOLOGIE.md``, même convention que les autres modules).

Prérequis : ``indusense-gold`` (gold peuplé) et **au moins un run** ``indusense-tune``
(pour les meilleurs hyperparamètres B7 — le plus récent est pris par défaut ; ces
hyperparamètres viennent d'un tuning à l'horizon 24h, cf. ``run()``).

Usage :

    uv run indusense-explain
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from indusense import config  # noqa: E402
from indusense.ml import training, tuning  # noqa: E402

# SHAP logue en INFO par défaut (barres de progression, notes internes) — même discipline
# que mlflow/optuna : ne garder que les erreurs réelles.
logging.getLogger("shap").setLevel(logging.ERROR)

MODEL_LABELS = {"b5_sobre": "XGBoost sobre (B5)", "b7_tune": "XGBoost tuné (B7)"}
_FEATURE_PREFIXES = ("num__", "cat__")


# --- Sélection du run de tuning (meilleurs hyperparamètres B7) -----------------------
def latest_tuning_run(tuning_dir: Path | None = None) -> Path:
    """Dernier run ``indusense-tune`` (tri lexicographique = chronologique, ``RUN_TS_FORMAT``)."""
    tuning_dir = tuning_dir or config.TUNING_DIR
    candidates = sorted(
        p for p in tuning_dir.iterdir() if p.is_dir() and (p / "best_params.json").exists()
    ) if tuning_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"Aucun run de tuning trouvé dans {tuning_dir} "
            "(lancer `uv run indusense-tune` d'abord)."
        )
    return candidates[-1]


def load_best_params(run_dir: Path | None = None) -> dict:
    """Meilleurs hyperparamètres XGBoost B7 (``best_params.json`` du run donné, sinon dernier)."""
    run_dir = run_dir or latest_tuning_run()
    return json.loads((run_dir / "best_params.json").read_text(encoding="utf-8"))


# --- Modèles à comparer ---------------------------------------------------------------
def build_pipelines(
    numeric_cols: list[str], categorical_cols: list[str],
    scale_pos_weight: float, best_params: dict,
) -> dict[str, Pipeline]:
    """Les 2 pipelines XGBoost comparés : sobre (B5) et tuné (B7, meilleurs hyperparamètres)."""
    return {
        "b5_sobre": training.build_models(
            numeric_cols, categorical_cols, scale_pos_weight
        )["xgboost"],
        "b7_tune": tuning.build_xgb_pipeline(
            numeric_cols, categorical_cols, scale_pos_weight, best_params
        ),
    }


# --- SHAP ------------------------------------------------------------------------------
def _clean_feature_names(names: list[str]) -> list[str]:
    """Retire les préfixes ``num__``/``cat__`` de ``ColumnTransformer.get_feature_names_out()``
    — cosmétique, pour des figures/CSV lisibles par un public métier."""
    cleaned = []
    for name in names:
        for prefix in _FEATURE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned.append(name)
    return cleaned


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """SHAP (``TreeExplainer``, marge/log-odds) sur les features **transformées** (fit-train
    déjà fait sur ``pipeline``). Retourne ``(shap_values, X_transformed, feature_names)``."""
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = np.asarray(explainer.shap_values(X_transformed))
    if shap_values.ndim == 3:
        # Certaines versions/config shap renvoient un tableau par classe -> classe positive.
        shap_values = shap_values[..., 1]
    return shap_values, X_transformed, feature_names


def mean_abs_shap_table(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Importance globale : |SHAP| moyen par feature (brut, trié décroissant) + sa part
    normalisée ``importance_share`` (somme = 1 par modèle, invariante à l'échelle du modèle
    — cf. biais d'échelle documenté en tête de module)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    total = table["mean_abs_shap"].sum()
    table["importance_share"] = table["mean_abs_shap"] / total if total > 0 else 0.0
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def compare_models_importance(
    table_b5: pd.DataFrame, table_b7: pd.DataFrame, top_n: int = config.EXPLAIN_TOP_N
) -> pd.DataFrame:
    """Fusionne les 2 tableaux d'importance : part normalisée + rang (calculé sur la
    totalité des features de chaque modèle, pas seulement l'union filtrée ci-dessous) +
    ``delta_rank`` (= rang B5 - rang B7 ; positif = la feature est montée en priorité dans le
    modèle tuné). Filtré ensuite à l'union des top ``top_n`` de chaque modèle, trié par part
    B7 décroissante."""
    ranked_b5 = table_b5.assign(rank_b5=range(1, len(table_b5) + 1))
    ranked_b7 = table_b7.assign(rank_b7=range(1, len(table_b7) + 1))
    top_features = list(
        dict.fromkeys([*ranked_b5["feature"].head(top_n), *ranked_b7["feature"].head(top_n)])
    )

    merged = ranked_b5.rename(
        columns={"mean_abs_shap": "mean_abs_shap_b5", "importance_share": "share_b5"}
    ).merge(
        ranked_b7.rename(
            columns={"mean_abs_shap": "mean_abs_shap_b7", "importance_share": "share_b7"}
        ),
        on="feature", how="outer",
    )
    # Rang/valeur par défaut si une feature manquait d'un côté (ne devrait pas arriver : les
    # deux modèles partagent le même espace de features transformées) -> pire rang + 1.
    worst_rank = max(len(table_b5), len(table_b7)) + 1
    merged["rank_b5"] = merged["rank_b5"].fillna(worst_rank).astype(int)
    merged["rank_b7"] = merged["rank_b7"].fillna(worst_rank).astype(int)
    value_cols = ["mean_abs_shap_b5", "mean_abs_shap_b7", "share_b5", "share_b7"]
    merged[value_cols] = merged[value_cols].fillna(0.0)
    merged["delta_rank"] = merged["rank_b5"] - merged["rank_b7"]

    merged = merged[merged["feature"].isin(top_features)]
    cols = ["feature", *value_cols, "rank_b5", "rank_b7", "delta_rank"]
    return merged[cols].sort_values("share_b7", ascending=False).reset_index(drop=True)


# --- Figures (SVG, même style que training.py/tuning.py) ------------------------------
def make_figures(
    shap_by_model: dict[str, np.ndarray], X_by_model: dict[str, np.ndarray],
    feature_names: list[str], comparison: pd.DataFrame, out_dir: Path,
    top_n: int = config.EXPLAIN_TOP_N,
) -> list[str]:
    """4 figures : résumé SHAP (beeswarm échantillonné) B5, résumé SHAP B7, comparaison
    d'importance normalisée, changement de rang (delta_rank)."""
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

    # 1-2. Résumé SHAP (beeswarm) par modèle — échantillon d'AFFICHAGE seulement (le |SHAP|
    # moyen, lui, est calculé sur le test complet en amont dans run()).
    n_rows = len(next(iter(X_by_model.values())))
    sample_size = min(config.EXPLAIN_BEESWARM_SAMPLE, n_rows)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_rows, size=sample_size, replace=False)

    for i, key in enumerate(("b5_sobre", "b7_tune"), start=1):
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_by_model[key][sample_idx], X_by_model[key][sample_idx],
            feature_names=feature_names, max_display=top_n, show=False,
        )
        fig = plt.gcf()
        fig.suptitle(f"{i}. Résumé SHAP — {MODEL_LABELS[key]}", y=1.02)
        _save(fig, f"0{i}_summary_{key}.svg",
              f"Mesure : contribution SHAP (marge/log-odds), échantillon aléatoire de "
              f"{sample_size} obs. test (graine 42, sur {n_rows} au total) pour la "
              "lisibilité — l'importance |SHAP| moyenne est calculée sur les "
              f"{n_rows} observations. Rouge = valeur de feature élevée, bleu = faible ; à "
              "droite de 0 = pousse vers le risque.")

    # 3. Comparaison d'importance NORMALISÉE (part relative, jamais les valeurs brutes : un
    # modèle plus conservateur a mécaniquement des |SHAP| bruts plus petits partout, sans que
    # ça signifie qu'il "regarde moins" les features — cf. biais d'échelle en tête de module).
    x = np.arange(len(comparison))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(x - width / 2, comparison["share_b5"], width,
            label=MODEL_LABELS["b5_sobre"], color="#4C72B0")
    ax.barh(x + width / 2, comparison["share_b7"], width,
            label=MODEL_LABELS["b7_tune"], color="#C44E52")
    ax.set_yticks(x)
    ax.set_yticklabels(comparison["feature"])
    ax.invert_yaxis()
    ax.legend()
    ax.set(title="3. Importance globale (part normalisée) — sobre vs tuné",
           xlabel="Part de |SHAP| moyen (somme = 1 par modèle)")
    _save(fig, "03_comparaison_importance.svg",
          "Mesure : part normalisée de |SHAP| moyen (invariante à l'échelle du modèle), "
          "union des features les plus importantes des deux modèles — comparable entre "
          "modèles, contrairement aux valeurs brutes (disponibles dans les CSV).")

    # 4. Changement de rang (delta_rank = rang B5 - rang B7 ; positif = monte en priorité
    # dans le modèle tuné) — la lecture la plus directe de « le tuning a-t-il changé ce que
    # le modèle regarde ? ».
    ranked = comparison.sort_values("delta_rank", key=lambda s: s.abs(), ascending=False)
    colors = ["#55A868" if d > 0 else "#C44E52" if d < 0 else "#8C8C8C"
              for d in ranked["delta_rank"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ranked["feature"], ranked["delta_rank"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set(title="4. Changement de rang d'importance (B5 -> B7)",
           xlabel="Δ rang = rang B5 - rang B7 (positif = monte en priorité dans B7)")
    _save(fig, "04_delta_rang.svg",
          "Mesure : différence de rang d'importance (|SHAP| moyen) entre les deux modèles, "
          "triée par ampleur de changement. Vert = la feature est montée en priorité avec le "
          "tuning ; rouge = elle a descendu ; gris = rang inchangé.")
    return files


# --- Journal + méthodologie (même convention que training.py/gold.py/tuning.py) -------
def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — explicabilité SHAP (B8)\n\n"
        "| Run | Horizon | Run de tuning | Top feature B5 | Top feature B7 |\n"
        "|---|---:|---|---|---|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['horizon']}h | {r['tuning_run_id']} | "
        f"{r['top_feature_b5']} | {r['top_feature_b7']} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    config.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EXPLAIN_DIR / "runs.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({k: meta[k] for k in
                 ("run_id", "horizon", "tuning_run_id", "top_feature_b5", "top_feature_b7")})
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.EXPLAIN_DIR / "runs.md").write_text(_render_runs_md(runs), encoding="utf-8")


def write_methodologie() -> None:
    config.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — explicabilité SHAP (B8)

## Rôle
Compare l'importance globale des features entre le XGBoost "sobre" (B5,
`training.build_models`) et le XGBoost tuné par Optuna (B7, meilleurs hyperparamètres du
dernier run `indusense-tune`) — répond à « le tuning a-t-il changé ce que le modèle
regarde ? », pas seulement « le score a-t-il changé ? » (déjà répondu par B7).

## Portée : explications globales uniquement
Résumé SHAP (beeswarm) + importance (|SHAP| moyen, part normalisée, rang) par feature, sur
l'ensemble du split **test**. Pas d'explication locale (une observation précise) ni de
dependence plot dans cette première itération — hors périmètre, pourrait être ajouté ensuite
sans changer cette base.

## Biais d'échelle entre modèles — pourquoi une part normalisée, pas les valeurs brutes
Les valeurs SHAP sont en marge (log-odds). B7 est un modèle plus conservateur que B5
(learning_rate, profondeur et nombre d'arbres plus faibles) : il produit donc mécaniquement
des |SHAP| bruts plus petits sur **toutes** les features, sans que ça signifie qu'il
"regarde moins" les features — un artefact d'échelle, pas un changement d'attention. La
comparaison utilise donc une **part normalisée** (`importance_share`, somme = 1 par modèle,
invariante à l'échelle) et le **rang** (`rank_b5`/`rank_b7`/`delta_rank`) comme signaux
robustes de comparaison inter-modèles. Les valeurs brutes restent dans les CSV
(`importance_b5_sobre.csv`/`importance_b7_tune.csv`) pour qui veut les consulter, mais ni la
figure de comparaison ni la lecture notebook ne s'appuient dessus pour comparer les modèles.

## Calcul sur le test complet, affichage sur un échantillon
Le |SHAP| moyen (et donc part/rang) est calculé sur les **19653 lignes du test complet**
(`TreeExplainer` est rapide et exact pour les modèles arbres, aucun échantillon de fond
nécessaire). Seul le nuage de points des figures de résumé (beeswarm) est sous-échantillonné
(`config.EXPLAIN_BEESWARM_SAMPLE`, graine 42, indices communs aux deux modèles) pour rester
lisible et léger en SVG — indiqué explicitement dans la légende de chaque figure.

## SHAP sur les features transformées, pas le pipeline complet
`shap.TreeExplainer` explique le modèle arbre (`XGBClassifier`) lui-même : les deux
pipelines sont donc décomposés (`preprocessor.transform(X_test)` puis `TreeExplainer` sur
`named_steps["model"]`), avec les noms de features post-imputation/one-hot
(`preprocessor.get_feature_names_out()`, préfixes `num__`/`cat__` retirés pour la lisibilité).
Valeurs en marge (log-odds), pas en probabilité — comportement par défaut de
`TreeExplainer` pour XGBoost binaire, exact, suffisant pour une lecture d'importance
relative. Les deux pipelines partagent le même `build_preprocessor` sur le même `X_train` :
`run()` vérifie explicitement que les noms de features transformées sont identiques entre
les deux modèles avant de construire les figures de comparaison.

## Modèles réentraînés, pas rechargés depuis MLflow
Les deux pipelines (B5 sobre, B7 tuné) sont reconstruits et réentraînés sur le **train**
directement dans ce module (mêmes fonctions que `training.py`/`tuning.py`), plutôt que
rechargés depuis leurs artefacts MLflow respectifs. Avec `random_state=42` fixe et les mêmes
données train, ce réentraînement est déterministe bit-à-bit (à versions de librairies
identiques) — ce n'est donc pas "expliquer un autre modèle", mais bien celui suivi par
`training`/`tuning`. Plus simple, et évite une dépendance à la sérialisation MLflow pour un
module d'analyse.

## Pas de tracking MLflow
Ce module ne produit ni n'enregistre de nouveau modèle : seulement des artefacts d'analyse
(figures + tableaux d'importance). Le suivi MLflow reste celui de `training.py`/`tuning.py`.

## Sélection du run de tuning et couplage à l'horizon
Par défaut, le run `indusense-tune` le plus récent (`artifacts/tuning/<run_id>/`, tri
chronologique par `run_id`) — surchargeable via `--tuning-run-id`. Les meilleurs
hyperparamètres B7 proviennent d'un tuning réalisé à l'horizon **24h** (seul horizon tuné à
ce jour) : les appliquer à un autre horizon (`--horizon`) reste possible (cohérence de
l'interface avec les autres modules) mais est exploratoire — `run_metadata.json` porte alors
un champ `note_horizon` signalant ce couplage.
"""
    (config.EXPLAIN_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration -------------------------------------------------------------------
def run(
    source: str = "parquet",
    horizon: int = config.ML_DEFAULT_HORIZON,
    tuning_run_dir: Path | None = None,
    top_n: int = config.EXPLAIN_TOP_N,
    df: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> dict:
    """Compare l'importance SHAP globale XGBoost sobre (B5) vs tuné (B7), journalise le run.

    ``df`` optionnel : injecte un DataFrame gold déjà chargé (tests / notebook), même esprit
    que ``training.run(df=...)``/``tuning.run(df=...)``.
    """
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.EXPLAIN_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    tuning_run_dir = tuning_run_dir or latest_tuning_run()
    best_params = load_best_params(tuning_run_dir)

    gold_df = df if df is not None else training.load_gold(source=source)
    X, y, numeric_cols, categorical_cols = training.split_features_target(gold_df, horizon=horizon)
    masks = training.train_val_test_masks(gold_df)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_test = X[masks["test"]]
    scale_pos_weight = training.compute_scale_pos_weight(y_train)

    pipelines = build_pipelines(numeric_cols, categorical_cols, scale_pos_weight, best_params)
    for pipeline in pipelines.values():
        pipeline.fit(X_train, y_train)

    shap_by_model: dict[str, np.ndarray] = {}
    X_by_model: dict[str, np.ndarray] = {}
    tables: dict[str, pd.DataFrame] = {}
    names_by_model: dict[str, list[str]] = {}
    for key, pipeline in pipelines.items():
        shap_values, X_transformed, names = compute_shap_values(pipeline, X_test)
        shap_by_model[key] = shap_values
        X_by_model[key] = X_transformed
        names_by_model[key] = names
        tables[key] = mean_abs_shap_table(shap_values, _clean_feature_names(names))

    # Les deux pipelines partagent le même preprocessor (mêmes numeric_cols/categorical_cols,
    # même X_train) : les noms de features transformées doivent être identiques -- vérifié
    # explicitement plutôt que supposé (cf. METHODOLOGIE.md).
    raw_names_list = list(names_by_model.values())
    if any(names != raw_names_list[0] for names in raw_names_list[1:]):
        raise AssertionError(
            "Les pipelines B5/B7 ne produisent pas les mêmes noms de features transformées "
            "-- vérifier build_preprocessor/numeric_cols/categorical_cols."
        )
    feature_names = _clean_feature_names(raw_names_list[0])

    for key in tables:
        tables[key].to_csv(run_dir / f"importance_{key}.csv", index=False)

    comparison = compare_models_importance(tables["b5_sobre"], tables["b7_tune"], top_n=top_n)
    comparison.to_csv(run_dir / "comparaison_importance.csv", index=False)

    figures = make_figures(
        shap_by_model, X_by_model, feature_names, comparison, fig_dir, top_n=top_n
    )

    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "horizon": horizon,
        "tuning_run_id": tuning_run_dir.name,
        "top_feature_b5": str(tables["b5_sobre"].iloc[0]["feature"]),
        "top_feature_b7": str(tables["b7_tune"].iloc[0]["feature"]),
        "top_n": top_n,
        "figures": figures,
        "chemin": str(run_dir),
    }
    if horizon != config.ML_DEFAULT_HORIZON:
        meta["note_horizon"] = (
            f"Hyperparamètres B7 issus d'un tuning à l'horizon {config.ML_DEFAULT_HORIZON}h "
            f"(run {tuning_run_dir.name}) — appliqués ici à l'horizon {horizon}h à titre "
            "exploratoire, non re-validés à cet horizon."
        )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_methodologie()
    append_run(meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Explicabilité SHAP — XGBoost sobre (B5) vs tuné (B7)."
    )
    parser.add_argument("--source", choices=["parquet", "sql"], default="parquet")
    parser.add_argument("--horizon", type=int, choices=list(config.LABEL_HORIZONS),
                        default=config.ML_DEFAULT_HORIZON)
    parser.add_argument("--tuning-run-id", default=None,
                        help="Run indusense-tune à utiliser (défaut : le plus récent).")
    parser.add_argument("--top-n", type=int, default=config.EXPLAIN_TOP_N)
    args = parser.parse_args(argv)

    tuning_run_dir = (config.TUNING_DIR / args.tuning_run_id) if args.tuning_run_id else None
    meta = run(source=args.source, horizon=args.horizon,
               tuning_run_dir=tuning_run_dir, top_n=args.top_n)
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(f"  horizon={meta['horizon']}h  tuning_run={meta['tuning_run_id']}")
    print(f"  top feature B5 (sobre) : {meta['top_feature_b5']}")
    print(f"  top feature B7 (tuné)  : {meta['top_feature_b7']}")
    if "note_horizon" in meta:
        print(f"  NOTE : {meta['note_horizon']}")


if __name__ == "__main__":
    main()
