"""Analyse statistique de la télémétrie (US 1.x, C1/C2).

Pipeline reproductible et tracé : à partir de la table ``bronze.telemetry`` (médaillon ;
le CSV brut y est chargé par ``indusense-ingest``), vérifie l'absence de données
personnelles, dédoublonne (garde idempotente), enrichit d'axes temporels, puis produit un
dataset validé (parquet), les graphes d'analyse et un journal de runs.

Prérequis : ``indusense-ingest`` a alimenté ``bronze.telemetry``.

Sorties (sous ``config.ANALYSE_TELEMETRY_DIR``) :

- ``runs.json`` / ``runs.md`` : journal des runs.
- ``METHODOLOGIE.md`` : justification anonymisation (N/A) + dédoublonnage + outliers.
- ``AAAAMMJJHHMM/`` : un dossier par run avec le dataset validé (parquet **et** CSV),
  les métadonnées et ``figures/`` (9 graphes SVG numérotés/ordonnés).

Usage :

    uv run indusense-telemetry
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interactif (génération de fichiers)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from indusense import config  # noqa: E402
from indusense.data.db import read_bronze  # noqa: E402

SHIFT_ORDER = ["matin", "apres-midi", "nuit"]

# Motifs de noms de colonnes susceptibles de porter des données personnelles (DCP).
PII_PATTERNS = ("operator", "name", "nom", "badge", "email", "mail", "user", "agent")


# --- Chargement -------------------------------------------------------------
def load_telemetry() -> pd.DataFrame:
    """Charge la télémétrie depuis ``bronze.telemetry`` (médaillon).

    Prérequis : ``indusense-ingest`` a chargé le CSV brut dans bronze. ``timestamp`` est
    déjà typé en base ; la clé technique ``telemetry_id`` est écartée.
    """
    return read_bronze("telemetry").drop(columns=["telemetry_id"], errors="ignore")


# --- Étape 1 : anonymisation (vérification) ---------------------------------
def check_no_personal_data(df: pd.DataFrame) -> dict:
    """Vérifie l'absence de colonnes de données personnelles (traçabilité RGPD).

    La télémétrie ne contient que des mesures capteurs et un identifiant
    **d'équipement** (`machine_id`) : aucune anonymisation n'est requise. Cette
    fonction matérialise et prouve cette décision (détecte tout champ DCP éventuel).
    """
    cols_dcp = [
        c for c in df.columns if any(p in c.lower() for p in PII_PATTERNS)
    ]
    return {"colonnes_dcp": cols_dcp, "anonymisation_requise": bool(cols_dcp)}


# --- Étape 2 : dédoublonnage ------------------------------------------------
def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les doublons de clé ``(machine_id, timestamp)`` (idempotent).

    Un relevé est unique par machine et par heure ; on conserve la première
    occurrence. Retourne le DataFrame nettoyé et le nombre de lignes supprimées.
    """
    avant = len(df)
    out = df.drop_duplicates(subset=["machine_id", "timestamp"]).reset_index(drop=True)
    return out, avant - len(out)


# --- Enrichissement ---------------------------------------------------------
def _shift_from_hour(hour: int) -> str:
    """Mappe l'heure (0–23) sur l'équipe : matin 06–14 · après-midi 14–22 · nuit sinon."""
    if 6 <= hour < 14:
        return "matin"
    if 14 <= hour < 22:
        return "apres-midi"
    return "nuit"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les axes temporels : ``jour``, ``heure``, ``weekday``, ``shift``."""
    df = df.copy()
    df["jour"] = df["timestamp"].dt.date
    df["heure"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["shift"] = df["heure"].map(_shift_from_hour)
    return df


# --- Étape 3 : analyse statistique ------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> dict:
    """Statistiques descriptives globales et par machine (capteurs + production)."""
    cols = [*config.TELEMETRY_SENSORS, config.TELEMETRY_PRODUCTION]
    return {
        "global": df[cols].describe().T,
        "par_machine": df.groupby("machine_id")[cols].mean(),
    }


def detect_outliers(df: pd.DataFrame, cols: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Détecte les outliers par la méthode **IQR** (Tukey, 1,5·IQR) par variable.

    Retourne un tableau par variable : bornes, nombre et taux (%) d'outliers.
    """
    cols = cols or config.TELEMETRY_SENSORS
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < low) | (df[c] > high)).sum())
        rows.append({
            "borne_basse": round(float(low), 2),
            "borne_haute": round(float(high), 2),
            "n_outliers": n_out,
            "pct": round(100 * n_out / len(df), 2),
        })
    return pd.DataFrame(rows, index=list(cols))


def correlations(df: pd.DataFrame) -> dict:
    """Matrices de corrélation Pearson (linéaire) et Spearman (rangs).

    Sur les 4 capteurs + la production : capte les liens physiques entre mesures.
    """
    cols = [*config.TELEMETRY_SENSORS, config.TELEMETRY_PRODUCTION]
    return {
        "pearson": df[cols].corr(method="pearson"),
        "spearman": df[cols].corr(method="spearman"),
    }


# --- Métriques --------------------------------------------------------------
def compute_metrics(df: pd.DataFrame, n_doublons: int) -> dict:
    """Métriques de suivi du dataset produit."""
    outliers = detect_outliers(df)
    return {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1]),
        "machines_uniques": int(df["machine_id"].nunique()),
        "n_nan_total": int(df.isna().sum().sum()),
        "n_doublons_supprimes": int(n_doublons),
        "periode_debut": str(df["timestamp"].min()),
        "periode_fin": str(df["timestamp"].max()),
        "n_outliers_total": int(outliers["n_outliers"].sum()),
    }


# --- Figures ----------------------------------------------------------------
def make_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Génère les 9 graphes d'analyse (SVG) dans ``out_dir``, numérotés et ordonnés.

    Chaque figure embarque une **légende explicative**. Toutes les figures reposent
    sur des **agrégats** (moyennes, quantiles, corrélations) pour rester légères.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["svg.fonttype"] = "none"  # texte SVG réel (sélectionnable)
    sensors = list(config.TELEMETRY_SENSORS)
    prod = config.TELEMETRY_PRODUCTION
    mesures = [*sensors, prod]
    files: list[str] = []

    def _save(fig: plt.Figure, name: str, caption: str) -> None:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5,
                 style="italic", color="#444444", wrap=True)
        fig.savefig(out_dir / name, format="svg", bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    # === Bloc 1 — Vue d'ensemble ===========================================
    # 1. Séries temporelles (moyenne journalière par mesure)
    par_jour = df.groupby("jour")[mesures].mean()
    fig, axes = plt.subplots(len(mesures), 1, figsize=(11, 12), sharex=True)
    for ax, col in zip(axes, mesures, strict=True):
        ax.plot(par_jour.index, par_jour[col].to_numpy(), lw=0.8, color="#4C72B0")
        ax.set_ylabel(col, fontsize=8)
    axes[-1].set_xlabel("Jour")
    fig.suptitle("1. Séries temporelles — moyenne journalière par mesure", y=0.995)
    fig.tight_layout()
    _save(fig, "01_series_temporelles.svg",
          "Mesure : moyenne journalière de chaque variable sur la période. "
          "Révèle tendances, saisonnalité et dérives lentes des capteurs.")

    # 2. Distributions des mesures (histogrammes)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, mesures, strict=False):
        sns.histplot(df[col], bins=50, ax=ax, color="#55A868")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    axes.flat[-1].axis("off")  # 6e case inutilisée (5 mesures)
    fig.suptitle("2. Distributions des mesures", y=0.995)
    fig.tight_layout()
    _save(fig, "02_distributions_capteurs.svg",
          "Mesure : distribution de chaque variable (50 classes). "
          "Forme (normale/asymétrique), étalement et modes éventuels.")

    # === Bloc 2 — Comparaison par machine ==================================
    # 3. Boxplots par machine (un capteur par sous-graphe)
    order = sorted(df["machine_id"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.flat, sensors, strict=True):
        sns.boxplot(data=df, x="machine_id", y=col, order=order, ax=ax,
                    showfliers=False, color="#4C72B0")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    fig.suptitle("3. Distribution des capteurs par machine (boxplots)", y=0.995)
    fig.tight_layout()
    _save(fig, "03_boxplots_par_machine.svg",
          "Mesure : dispersion (médiane, quartiles) de chaque capteur par machine. "
          "Repère une machine au comportement atypique (outliers masqués, cf. fig. 4).")

    # 4. Outliers par capteur (méthode IQR)
    outliers = detect_outliers(df, tuple(sensors))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(outliers)), outliers["n_outliers"].to_numpy(), color="#C44E52")
    ax.set_xticks(range(len(outliers)))
    ax.set_xticklabels(outliers.index, rotation=20, ha="right", fontsize=8)
    for i, (n, pct) in enumerate(zip(outliers["n_outliers"], outliers["pct"], strict=True)):
        ax.text(i, n, f"{int(n)}\n({pct}%)", ha="center", va="bottom", fontsize=7.5)
    ax.set(title="4. Outliers par capteur (méthode IQR)", ylabel="Nombre d'outliers")
    ax.set_ylim(0, max(outliers["n_outliers"].max() * 1.2, 1))
    _save(fig, "04_outliers_par_capteur.svg",
          "Mesure : nombre (et %) de valeurs hors [Q1−1,5·IQR ; Q3+1,5·IQR] par capteur. "
          "Quantifie les valeurs extrêmes à instruire avant modélisation.")

    # === Bloc 3 — Corrélations entre mesures ===============================
    corr = correlations(df)
    # 5. Corrélation de Pearson
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr["pearson"], cmap="coolwarm", center=0, annot=True, fmt=".2f",
                square=True, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("5. Corrélation de Pearson (linéaire) — mesures")
    _save(fig, "05_correlation_pearson.svg",
          "Mesure : corrélation linéaire de Pearson entre capteurs et production. "
          "Liens physiques directs entre grandeurs.")

    # 6. Corrélation de Spearman
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr["spearman"], cmap="PuOr", center=0, annot=True, fmt=".2f",
                square=True, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("6. Corrélation de Spearman (rangs) — mesures")
    _save(fig, "06_correlation_spearman.svg",
          "Mesure : corrélation de Spearman (sur les rangs) entre mesures. "
          "Capte les liens monotones non linéaires ; robuste aux outliers.")

    # === Bloc 4 — Profils temporels ========================================
    def _zscore_means(grouped: pd.DataFrame) -> pd.DataFrame:
        """Centre-réduit chaque colonne pour comparer les profils sur une échelle commune."""
        return (grouped - grouped.mean()) / grouped.std(ddof=0)

    # 7. Profil horaire (moyenne par heure du jour, mesures normalisées)
    par_heure = _zscore_means(df.groupby("heure")[mesures].mean())
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in mesures:
        ax.plot(par_heure.index, par_heure[col].to_numpy(), "o-", ms=3, lw=1, label=col)
    ax.set(title="7. Profil horaire (mesures centrées-réduites)",
           xlabel="Heure du jour", ylabel="Écart à la moyenne (z-score)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    _save(fig, "07_profil_horaire.svg",
          "Mesure : moyenne par heure du jour, centrée-réduite par variable (échelle "
          "commune). Révèle un éventuel cycle journalier (production, échauffement…).")

    # 8. Profil par shift (mesures normalisées, barres groupées)
    par_shift = _zscore_means(df.groupby("shift")[mesures].mean()).reindex(SHIFT_ORDER)
    x = np.arange(len(SHIFT_ORDER))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, col in enumerate(mesures):
        ax.bar(x + (j - 2) * width, par_shift[col].to_numpy(), width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(SHIFT_ORDER)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set(title="8. Profil par shift (mesures centrées-réduites)",
           xlabel="Équipe", ylabel="Écart à la moyenne (z-score)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    _save(fig, "08_profil_par_shift.svg",
          "Mesure : moyenne par équipe (matin/après-midi/nuit), centrée-réduite par "
          "variable. Détecte un effet d'équipe sur les mesures ou la production.")

    # === Bloc 5 — Lien capteurs ↔ production ===============================
    # 9. Corrélation de chaque capteur avec la production
    pear = corr["pearson"][prod].reindex(sensors)
    spear = corr["spearman"][prod].reindex(sensors)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    xs = np.arange(len(sensors))
    ax.bar(xs - 0.2, pear.to_numpy(), 0.4, label="Pearson", color="#4C72B0")
    ax.bar(xs + 0.2, spear.to_numpy(), 0.4, label="Spearman", color="#DD8452")
    ax.set_xticks(xs)
    ax.set_xticklabels(sensors, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set(title=f"9. Corrélation des capteurs avec « {prod} »",
           ylabel="Coefficient de corrélation")
    ax.legend(fontsize=8)
    _save(fig, "09_capteurs_vs_production.svg",
          f"Mesure : corrélation (Pearson/Spearman) de chaque capteur avec « {prod} ». "
          "Identifie les grandeurs liées au niveau de production.")

    return files


# --- Journal des runs -------------------------------------------------------
def _runs_json_path() -> Path:
    return config.ANALYSE_TELEMETRY_DIR / "runs.json"


def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — analyse télémétrie\n\n"
        "| Run (AAAAMMJJHHMM) | Lignes | Colonnes | Machines | Doublons retirés | "
        "NaN | Outliers |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['n_lignes']} | {r['n_colonnes']} | {r['machines_uniques']} | "
        f"{r['n_doublons_supprimes']} | {r['n_nan_total']} | {r['n_outliers_total']} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    """Ajoute le run au journal JSON et régénère la vue Markdown."""
    config.ANALYSE_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path = _runs_json_path()
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({
        "run_id": meta["run_id"],
        "timestamp": meta["timestamp"],
        "source": meta["source"],
        "n_lignes": meta["n_lignes"],
        "n_colonnes": meta["n_colonnes"],
        "machines_uniques": meta["machines_uniques"],
        "n_doublons_supprimes": meta["n_doublons_supprimes"],
        "n_nan_total": meta["n_nan_total"],
        "n_outliers_total": meta["n_outliers_total"],
        "chemin": meta["chemin"],
    })
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.ANALYSE_TELEMETRY_DIR / "runs.md").write_text(
        _render_runs_md(runs), encoding="utf-8"
    )


def write_methodologie() -> None:
    """Écrit (ou rafraîchit) la note de méthodologie au niveau du dossier télémétrie."""
    config.ANALYSE_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — analyse de la télémétrie

## Anonymisation (RGPD) : non requise
La télémétrie ne comporte **aucune donnée à caractère personnel** : uniquement des
mesures capteurs (`temperature_c`, `pressure_bar`, `voltage_mean_v`,
`rotation_mean_rpm`, `pieces_produced`), un horodatage et un identifiant
**d'équipement** (`machine_id`, non rattaché à une personne). La fonction
`check_no_personal_data` matérialise et prouve cette décision (détection de tout champ
DCP éventuel). Aucune transformation d'anonymisation n'est donc appliquée.

## Dédoublonnage
Clé métier d'unicité : **`(machine_id, timestamp)`** (un relevé par machine et par
heure). `deduplicate` retire les doublons de clé (conservation de la 1re occurrence) et
reporte le nombre supprimé — opération **idempotente**. Le compte effectif (doublons
retirés, NaN restants) figure dans le journal des runs et les métadonnées du run, pour
rester juste quelle que soit la version des données.

## Détection des outliers
Méthode **IQR (Tukey)** par variable : sont marquées extrêmes les valeurs hors de
l'intervalle [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]. Choix robuste (basé sur les quantiles, peu
sensible aux valeurs extrêmes elles-mêmes) et lisible. Les outliers sont **comptés et
signalés**, non supprimés à ce stade : leur traitement (capping, imputation) relèvera de
la couche *silver*, une fois leur cause instruite.

## Corrélations
- **Pearson** : liens **linéaires** entre mesures (grandeurs continues).
- **Spearman** : liens **monotones** (sur les rangs), robustes aux outliers et aux
  non-linéarités. Les comparer distingue une vraie relation d'un artefact.

## Profils temporels
Axes dérivés du `timestamp` : `jour`, `heure`, `weekday`, `shift` (matin 06–14 ·
après-midi 14–22 · nuit sinon). Les profils horaire et par équipe sont **centrés-réduits
par variable** (z-score) pour comparer des grandeurs d'échelles différentes sur un même
graphe.
"""
    (config.ANALYSE_TELEMETRY_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration ----------------------------------------------------------
def run_analysis(now: datetime | None = None) -> dict:
    """Exécute le pipeline complet d'analyse et retourne les métadonnées du run."""
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.ANALYSE_TELEMETRY_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = load_telemetry()
    pii = check_no_personal_data(raw)
    deduped, n_doublons = deduplicate(raw)
    df = enrich(deduped)

    parquet_path = run_dir / "telemetry_clean.parquet"
    df.to_parquet(parquet_path, index=False)
    # Export CSV des données validées (utf-8-sig : lisible sous Excel Windows).
    csv_path = run_dir / "telemetry_clean.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    figures = make_figures(df, fig_dir)

    metrics = compute_metrics(df, n_doublons)
    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": "bronze.telemetry",
        "dataset": str(parquet_path),
        "dataset_csv": str(csv_path),
        "anonymisation_requise": pii["anonymisation_requise"],
        "colonnes_dcp": pii["colonnes_dcp"],
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
        description="Analyse statistique de la télémétrie (dataset validé + graphes)."
    )
    parser.add_argument(
        "--source",
        choices=["csv"],
        default="csv",
        help="Source de la télémétrie (défaut: csv brut data/raw).",
    )
    parser.parse_args(argv)

    meta = run_analysis()
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(
        f"  lignes={meta['n_lignes']} colonnes={meta['n_colonnes']} "
        f"machines={meta['machines_uniques']} doublons_retires={meta['n_doublons_supprimes']} "
        f"NaN={meta['n_nan_total']} outliers={meta['n_outliers_total']}"
    )
    print(f"  anonymisation requise : {meta['anonymisation_requise']}")
    print(f"  {len(meta['figures'])} figures générées.")


if __name__ == "__main__":
    main()
