"""Ingestion & analyse des relevés d'incidents (US 1.1, C1/C2).

Pipeline reproductible et tracé : à partir du CSV brut des incidents, produit un
dataset **anonymisé + enrichi** versionné (dossier horodaté), les graphes d'analyse
et un journal de suivi des runs.

Sorties (sous ``config.INGEST_INCIDENTS_DIR``) :

- ``runs.json`` / ``runs.md`` : journal des runs (lignes, colonnes, machines, NaN).
- ``METHODOLOGIE.md`` : justification anonymisation + définition de l'indice de confiance.
- ``AAAAMMJJHHMM/`` : un dossier par run avec le parquet, les métadonnées et ``figures/``.

Usage :

    uv run indusense-incidents
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interactif (génération de fichiers)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from indusense import config  # noqa: E402
from indusense.data.anonymize import anonymize_operators  # noqa: E402

SHIFT_ORDER = ["matin", "apres-midi", "nuit"]

# Pondérations de l'indice de confiance (somme = 1.0).
CONFIDENCE_WEIGHTS = {
    "coherence": 0.4,
    "comment_present": 0.2,
    "machine_valide": 0.2,
    "severity_valide": 0.2,
}


# --- Chargement -------------------------------------------------------------
def load_incidents_raw() -> pd.DataFrame:
    """Charge les incidents bruts (CSV en encodage cp1252 — accents français)."""
    return pd.read_csv(config.RAW_INCIDENTS, encoding="cp1252")


# --- Enrichissement ---------------------------------------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le signal dominant, le nombre de signaux et les axes temporels."""
    df = df.copy()
    signals = list(config.INCIDENT_SIGNALS)
    df["n_signals"] = df[signals].sum(axis=1)

    def _primary(row: pd.Series) -> str:
        actifs = [s for s in signals if row[s] == 1]
        if len(actifs) == 1:
            return actifs[0]
        return "multiple" if len(actifs) > 1 else "aucun"

    df["signal"] = df.apply(_primary, axis=1)

    df["dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
    )
    df["jour"] = df["dt"].dt.date
    df["semaine_iso"] = df["dt"].dt.strftime("%G-S%V")
    df["weekday"] = df["dt"].dt.day_name()
    return df


# --- Indice de confiance ----------------------------------------------------
def compute_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule l'indice de confiance *par signalement* (qualité du relevé).

    Composantes (chacune dans [0, 1]) :

    - ``coherence`` : 1.0 si exactement un signal actif, 0.5 si plusieurs, 0.0 si aucun
      (un signalement clairement catégorisé est plus fiable) ;
    - ``comment_present`` : 1.0 si un commentaire (saisie guidée) est présent ;
    - ``machine_valide`` : 1.0 si ``machine_id`` respecte le format du référentiel ;
    - ``severity_valide`` : 1.0 si ``severity`` ∈ [1, 5].

    ``confidence`` = somme pondérée (cf. ``CONFIDENCE_WEIGHTS``).
    """
    df = df.copy()
    df["coherence"] = df["n_signals"].map(lambda n: 1.0 if n == 1 else (0.5 if n > 1 else 0.0))
    df["comment_present"] = df["comment"].notna().astype(float)
    df["machine_valide"] = (
        df["machine_id"].astype(str).str.fullmatch(r"MACH-\d+").fillna(False).astype(float)
    )
    df["severity_valide"] = df["severity"].between(1, 5).astype(float)
    df["confidence"] = sum(
        df[comp] * weight for comp, weight in CONFIDENCE_WEIGHTS.items()
    ).round(3)
    return df


# --- Métriques --------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> dict:
    """Métriques de suivi du dataset produit."""
    nan_par_colonne = df.isna().sum()
    return {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1]),
        "machines_uniques": int(df["machine_id"].nunique()),
        "n_nan_total": int(nan_par_colonne.sum()),
        "n_nan_par_colonne": {k: int(v) for k, v in nan_par_colonne.items() if v > 0},
        "confidence_moyenne": round(float(df["confidence"].mean()), 3),
    }


# --- Figures ----------------------------------------------------------------
def _legend_twin(ax: plt.Axes, ax2: plt.Axes) -> None:
    """Fusionne les légendes de deux axes jumeaux (barres + courbe)."""
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def make_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Génère les 7 graphes d'analyse (SVG) dans ``out_dir``. Retourne les noms de fichiers.

    Chaque figure embarque une **légende explicative** (ce que le graphe mesure).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["svg.fonttype"] = "none"  # texte SVG réel (sélectionnable), pas des tracés
    signals = list(config.INCIDENT_SIGNALS)
    files: list[str] = []

    def _save(fig: plt.Figure, name: str, caption: str) -> None:
        fig.tight_layout()
        # Légende explicative sous la figure (incluse via bbox_inches="tight").
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5,
                 style="italic", color="#444444", wrap=True)
        fig.savefig(out_dir / name, format="svg", bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    # 1. Distribution par jour
    par_jour = df.groupby("jour").size()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(par_jour.index, par_jour.values, lw=0.9, color="#4C72B0")
    ax.set(title="Incidents par jour", xlabel="Jour", ylabel="Nombre d'incidents")
    _save(fig, "dist_incidents_par_jour.svg",
          "Mesure : nombre d'incidents signalés par jour sur la période. "
          "Révèle les pics ponctuels et la tendance temporelle.")

    # 2. Distribution par semaine ISO
    par_sem = df.groupby("semaine_iso").size()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(par_sem)), par_sem.values, color="#4C72B0")
    step = max(1, len(par_sem) // 20)
    ax.set_xticks(range(0, len(par_sem), step))
    ax.set_xticklabels(par_sem.index[::step], rotation=90, fontsize=7)
    ax.set(title="Incidents par semaine (ISO)", xlabel="Semaine", ylabel="Nombre d'incidents")
    _save(fig, "dist_incidents_par_semaine.svg",
          "Mesure : volume d'incidents par semaine ISO. "
          "Lisse le bruit journalier pour visualiser la charge hebdomadaire.")

    # 3. Distribution par shift
    par_shift = df["shift"].value_counts().reindex(SHIFT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=par_shift.index, y=par_shift.values, ax=ax, hue=par_shift.index, legend=False)
    ax.set(title="Incidents par shift", xlabel="Shift", ylabel="Nombre d'incidents")
    _save(fig, "dist_incidents_par_shift.svg",
          "Mesure : répartition des incidents par équipe (matin / après-midi / nuit). "
          "Détecte un éventuel effet d'équipe.")

    # 4. Histogramme par signal (+ confiance moyenne)
    counts = pd.Series({s: int(df[s].sum()) for s in signals}).sort_values(ascending=False)
    conf_sig = {s: df.loc[df[s] == 1, "confidence"].mean() for s in counts.index}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(counts)), counts.values, color="#55A868", label="Nombre d'incidents")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=8)
    ax.set(title="Incidents par signal + confiance moyenne", ylabel="Nombre d'incidents")
    ax2 = ax.twinx()
    ax2.plot(range(len(counts)), [conf_sig[s] for s in counts.index], "o-",
             color="#C44E52", label="Confiance moyenne")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Confiance moyenne (0–1)", color="#C44E52")
    _legend_twin(ax, ax2)
    _save(fig, "hist_par_signal.svg",
          "Mesure : nombre d'incidents par type de signal (barres) et indice de confiance "
          "moyen du signalement (courbe, 0–1). Pannes dominantes et fiabilité de saisie.")

    # 5. Histogramme par machine (+ confiance moyenne)
    cnt_mach = df["machine_id"].value_counts().sort_index()
    conf_mach = df.groupby("machine_id")["confidence"].mean().reindex(cnt_mach.index)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(cnt_mach)), cnt_mach.values, color="#4C72B0", label="Nombre d'incidents")
    ax.set_xticks(range(len(cnt_mach)))
    ax.set_xticklabels(cnt_mach.index, rotation=45, ha="right", fontsize=8)
    ax.set(title="Incidents par machine + confiance moyenne", ylabel="Nombre d'incidents")
    ax2 = ax.twinx()
    ax2.plot(range(len(cnt_mach)), conf_mach.values, "o-",
             color="#C44E52", label="Confiance moyenne")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Confiance moyenne (0–1)", color="#C44E52")
    _legend_twin(ax, ax2)
    _save(fig, "hist_par_machine.svg",
          "Mesure : nombre d'incidents par machine (barres) et indice de confiance moyen "
          "du signalement (courbe, 0–1). Repère les machines les plus touchées.")

    # 6. Distribution de l'indice de confiance
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["confidence"], bins=20, ax=ax, color="#8172B3")
    ax.set(title="Distribution de l'indice de confiance", xlabel="Confiance (0–1)",
           ylabel="Effectif")
    _save(fig, "hist_confiance.svg",
          "Mesure : distribution de l'indice de confiance des signalements = cohérence des "
          "flags + présence du commentaire + validité machine/sévérité.")

    # 7. Corrélation signaux + severity
    corr = df[[*signals, "severity"]].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 6})
    ax.set_title("Corrélation entre signaux (et severity)")
    _save(fig, "correlation_signaux.svg",
          "Mesure : corrélation de Pearson entre les 9 signaux et la sévérité. "
          "≈0 = signaux quasi exclusifs ; >0 = co-occurrence ; <0 = exclusion mutuelle.")

    return files


# --- Journal des runs -------------------------------------------------------
def _runs_json_path() -> Path:
    return config.INGEST_INCIDENTS_DIR / "runs.json"


def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — ingestion incidents\n\n"
        "| Run (AAAAMMJJHHMM) | Lignes | Colonnes | Machines | NaN | Confiance moy. |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['n_lignes']} | {r['n_colonnes']} | "
        f"{r['machines_uniques']} | {r['n_nan_total']} | {r['confidence_moyenne']} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    """Ajoute le run au journal JSON et régénère la vue Markdown."""
    config.INGEST_INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _runs_json_path()
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entry = {
        "run_id": meta["run_id"],
        "timestamp": meta["timestamp"],
        "source": meta["source"],
        "n_lignes": meta["n_lignes"],
        "n_colonnes": meta["n_colonnes"],
        "machines_uniques": meta["machines_uniques"],
        "n_nan_total": meta["n_nan_total"],
        "confidence_moyenne": meta["confidence_moyenne"],
        "chemin": meta["chemin"],
    }
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.INGEST_INCIDENTS_DIR / "runs.md").write_text(_render_runs_md(runs), encoding="utf-8")


def write_methodologie() -> None:
    """Écrit (ou rafraîchit) la note de méthodologie au niveau du dossier incidents."""
    config.INGEST_INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    weights = " + ".join(f"{w}·{c}" for c, w in CONFIDENCE_WEIGHTS.items())
    content = f"""# Méthodologie — ingestion des incidents

## Anonymisation des opérateurs (RGPD)
`operator_name` et `operator_badge` sont des **identifiants directs** (DCP). Ils ne
sont pas nécessaires aux analyses et sont donc **supprimés** (minimisation). Comme
aucune table de correspondance n'est conservée, l'anonymisation est **irréversible**
et le dataset sort du périmètre RGPD. `comment` est **conservé** (saisie guidée
décrivant le type de panne, sans donnée personnelle).

## Indice de confiance du signalement (par incident)
Score dans [0, 1] mesurant la **qualité du relevé** :

| Composante | Règle |
|---|---|
| `coherence` | 1.0 si 1 signal actif · 0.5 si plusieurs · 0.0 si aucun |
| `comment_present` | 1.0 si commentaire présent, sinon 0.0 |
| `machine_valide` | 1.0 si `machine_id` au format référentiel (`MACH-\\d+`) |
| `severity_valide` | 1.0 si `severity` ∈ [1, 5] |

`confidence = {weights}`
"""
    (config.INGEST_INCIDENTS_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration ----------------------------------------------------------
def run_ingestion(now: datetime | None = None) -> dict:
    """Exécute le pipeline complet et retourne les métadonnées du run."""
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.INGEST_INCIDENTS_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = load_incidents_raw()
    df = compute_confidence(enrich(anonymize_operators(raw)))

    parquet_path = run_dir / "incidents_anonymized.parquet"
    df.to_parquet(parquet_path, index=False)
    figures = make_figures(df, fig_dir)

    metrics = compute_metrics(df)
    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": str(config.RAW_INCIDENTS),
        "dataset": str(parquet_path),
        "figures": figures,
        "colonnes": list(df.columns),
        **metrics,
    }
    meta["chemin"] = str(run_dir)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_methodologie()
    append_run(meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion & analyse des relevés d'incidents (dataset anonymisé + graphes)."
    )
    parser.add_argument(
        "--source",
        choices=["csv"],
        default="csv",
        help="Source des incidents (défaut: csv brut data/raw).",
    )
    parser.parse_args(argv)

    meta = run_ingestion()
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(
        f"  lignes={meta['n_lignes']} colonnes={meta['n_colonnes']} "
        f"machines={meta['machines_uniques']} NaN={meta['n_nan_total']} "
        f"confiance_moy={meta['confidence_moyenne']}"
    )
    print(f"  {len(meta['figures'])} figures générées.")


if __name__ == "__main__":
    main()
