"""Analyse du référentiel machines & maintenance (US 1.x, C1/C2).

Le référentiel provient du **dump SQL fourni** (``data/raw/machine.sql``), chargé dans
``bronze.machine`` / ``bronze.maintenance`` par ``indusense-ingest --migrate``. Ce pipeline
lit la **couche bronze** (médaillon, source unique de vérité), vérifie son intégrité, puis
produit une analyse statistique (parc machines + activité de maintenance), des graphes
versionnés et un journal de runs.

Prérequis : ``indusense-ingest --migrate`` a alimenté ``bronze.machine`` / ``maintenance``.

Sorties (sous ``config.ANALYSE_REFERENTIEL_DIR``) :

- ``runs.json`` / ``runs.md`` : journal des runs.
- ``METHODOLOGIE.md`` : source, intégrité, choix d'analyse.
- ``AAAAMMJJHHMM/`` : un dossier par run avec les tables analysées (``machine`` &
  ``maintenance`` en parquet **et** CSV), les métadonnées et ``figures/`` (9 SVG).

Usage :

    uv run indusense-referentiel
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
from indusense.data.db import read_bronze  # noqa: E402

CRITICALITY_ORDER = ["LOW", "MEDIUM", "HIGH"]
PII_PATTERNS = ("operator", "name", "nom", "badge", "email", "mail", "user", "agent")


# --- Chargement (depuis bronze, alimenté par le dump via indusense-ingest) --
def load_referentiel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge ``bronze.machine`` et ``bronze.maintenance`` (médaillon).

    Le dump ``data/raw/machine.sql`` est chargé dans bronze par ``indusense-ingest
    --migrate`` ; l'analyse lit la couche gouvernée (source unique de vérité), pas le
    dump. ``duration_hours`` (Numeric) est ramené en float pour les calculs.
    """
    machine = read_bronze("machine")
    maintenance = read_bronze("maintenance")
    maintenance["duration_hours"] = maintenance["duration_hours"].astype(float)
    return machine, maintenance


# --- Intégrité (anonymisation N/A + cohérence référentielle) ----------------
def check_integrity(machine: pd.DataFrame, maintenance: pd.DataFrame) -> dict:
    """Vérifie l'intégrité du référentiel et l'absence de données personnelles.

    - unicité des clés primaires (`machine_code`, `maintenance_id`) ;
    - intégrité référentielle : tout `machine_code` de maintenance existe dans machine ;
    - cohérence réactif ↔ incident : une maintenance réactive référence un incident ;
    - absence de colonnes DCP (anonymisation non requise).
    """
    codes = set(machine["machine_code"])
    orphelins = sorted(set(maintenance["machine_code"]) - codes)
    reactive = maintenance["maintenance_type"] == "reactive"
    dcp = [c for c in (*machine.columns, *maintenance.columns)
           if any(p in c.lower() for p in PII_PATTERNS)]
    return {
        "machine_pk_unique": bool(machine["machine_code"].is_unique),
        "maintenance_pk_unique": bool(maintenance["maintenance_id"].is_unique),
        "machines_orphelines": orphelins,
        "reactive_sans_incident": int((reactive & maintenance["related_incident_id"].isna()).sum()),
        "colonnes_dcp": dcp,
        "anonymisation_requise": bool(dcp),
    }


# --- Analyse : parc machines ------------------------------------------------
def machine_distributions(machine: pd.DataFrame) -> dict:
    """Répartition du parc par modèle, criticité, ligne de production et atelier."""
    return {
        "model": machine["model"].value_counts(),
        "criticality": machine["criticality"].value_counts().reindex(CRITICALITY_ORDER),
        "production_line": machine["production_line"].value_counts().sort_index(),
        "location": machine["location"].value_counts().sort_index(),
    }


def capacity_by_model(machine: pd.DataFrame) -> pd.DataFrame:
    """Capacités moyennes (journalière / horaire) par modèle."""
    return machine.groupby("model")[
        ["max_daily_capacity", "max_hourly_capacity_pieces"]
    ].mean().sort_values("max_daily_capacity", ascending=False)


# --- Analyse : maintenance --------------------------------------------------
def maintenance_summary(maintenance: pd.DataFrame) -> dict:
    """Synthèse de l'activité de maintenance (type, durée, composants, lien incident)."""
    par_type = maintenance.groupby("maintenance_type")["duration_hours"].agg(
        ["count", "mean", "min", "max"]
    )
    return {
        "par_type": par_type,
        "composants": maintenance["component"].value_counts(),
        "par_machine": pd.crosstab(maintenance["machine_code"], maintenance["maintenance_type"]),
        "n_lie_incident": int(maintenance["related_incident_id"].notna().sum()),
    }


# --- Métriques --------------------------------------------------------------
def compute_metrics(machine: pd.DataFrame, maintenance: pd.DataFrame) -> dict:
    """Métriques de suivi du référentiel."""
    types = maintenance["maintenance_type"].value_counts()
    return {
        "n_machines": int(len(machine)),
        "n_maintenances": int(len(maintenance)),
        "n_modeles": int(machine["model"].nunique()),
        "n_proactive": int(types.get("proactive", 0)),
        "n_reactive": int(types.get("reactive", 0)),
        "duree_moyenne_h": round(float(maintenance["duration_hours"].mean()), 2),
    }


# --- Figures ----------------------------------------------------------------
def make_figures(machine: pd.DataFrame, maintenance: pd.DataFrame, out_dir: Path) -> list[str]:
    """Génère les 9 graphes d'analyse (SVG) dans ``out_dir``, numérotés et ordonnés."""
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

    dist = machine_distributions(machine)

    # === Bloc 1 — Parc machines ============================================
    # 1. Répartition par modèle
    s = dist["model"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=s.index, y=s.to_numpy(), hue=s.index, legend=False, ax=ax, palette="Blues_d")
    for i, v in enumerate(s.to_numpy()):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set(title="1. Machines par modèle", xlabel="Modèle", ylabel="Nombre de machines")
    _save(fig, "01_machines_par_modele.svg",
          "Mesure : nombre de machines par modèle d'InduPress. Composition du parc.")

    # 2. Répartition par criticité / ligne / atelier
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, titre in zip(
        axes, ["criticality", "production_line", "location"],
        ["Criticité", "Ligne de production", "Atelier"], strict=True,
    ):
        d = dist[key]
        sns.barplot(x=d.index, y=d.to_numpy(), hue=d.index, legend=False, ax=ax, palette="crest")
        ax.set(title=titre, xlabel="", ylabel="Nombre de machines")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("2. Répartition du parc par attribut", y=1.0)
    fig.tight_layout()
    _save(fig, "02_machines_par_attribut.svg",
          "Mesure : répartition des machines par criticité, ligne de production et atelier. "
          "Cartographie organisationnelle du parc.")

    # 3. Capacités par modèle
    cap = capacity_by_model(machine)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, titre in zip(
        axes, ["max_daily_capacity", "max_hourly_capacity_pieces"],
        ["Capacité journalière", "Capacité horaire"], strict=True,
    ):
        sns.barplot(x=cap.index, y=cap[col].to_numpy(), hue=cap.index, legend=False,
                    ax=ax, palette="flare")
        ax.set(title=titre, xlabel="", ylabel="Pièces")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("3. Capacité de production par modèle", y=1.0)
    fig.tight_layout()
    _save(fig, "03_capacite_par_modele.svg",
          "Mesure : capacité de production moyenne (journalière / horaire) par modèle. "
          "L'InduPress-X1 se distingue par une capacité supérieure.")

    # 4. Ancienneté du parc (année de mise en service)
    annee = pd.to_datetime(machine["commissioning_date"]).dt.year
    par_an = annee.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(par_an.index.astype(str), par_an.to_numpy(), color="#4C72B0")
    ax.set(title="4. Ancienneté du parc (année de mise en service)",
           xlabel="Année", ylabel="Nombre de machines")
    _save(fig, "04_anciennete_parc.svg",
          "Mesure : nombre de machines par année de mise en service. "
          "Vieillissement du parc (facteur de risque de panne).")

    # === Bloc 2 — Activité de maintenance ==================================
    summ = maintenance_summary(maintenance)
    par_type = summ["par_type"]

    # 5. Maintenance proactive vs réactive
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=par_type.index, y=par_type["count"].to_numpy(), hue=par_type.index,
                legend=False, ax=ax, palette={"proactive": "#55A868", "reactive": "#C44E52"})
    for i, (n, d) in enumerate(zip(par_type["count"], par_type["mean"], strict=True)):
        ax.text(i, n, f"{int(n)}\n({d:.1f} h)", ha="center", va="bottom", fontsize=8.5)
    ax.set(title="5. Maintenance : proactive vs réactive",
           xlabel="Type", ylabel="Nombre d'interventions")
    ax.set_ylim(0, par_type["count"].max() * 1.2)
    _save(fig, "05_maintenance_type.svg",
          "Mesure : volume d'interventions par type et durée moyenne (h). La réactive "
          "(suite à panne) est plus rare mais plus longue.")

    # 6. Maintenance par machine (empilé proactive / réactive)
    pm = summ["par_machine"].reindex(columns=["proactive", "reactive"]).fillna(0)
    pm = pm.loc[pm.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(pm.index, pm["proactive"], label="proactive", color="#55A868")
    ax.bar(pm.index, pm["reactive"], bottom=pm["proactive"], label="reactive", color="#C44E52")
    ax.set(title="6. Maintenances par machine", xlabel="Machine", ylabel="Nombre d'interventions")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=8)
    _save(fig, "06_maintenance_par_machine.svg",
          "Mesure : nombre d'interventions par machine, ventilé proactive / réactive. "
          "Repère les machines les plus sollicitées (réactif = pannes subies).")

    # 7. Composants les plus remplacés
    comp = summ["composants"].head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(comp.index, comp.to_numpy(), color="#8172B3")
    for i, v in enumerate(comp.to_numpy()):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.set(title="7. Composants les plus remplacés (top 10)", xlabel="Nombre de remplacements")
    _save(fig, "07_composants_remplaces.svg",
          "Mesure : composants les plus fréquemment remplacés. Cible les pièces critiques "
          "pour le stock de pièces détachées.")

    # 8. Durée d'intervention par type
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(data=maintenance, x="maintenance_type", y="duration_hours", hue="maintenance_type",
                legend=False, ax=ax, palette={"proactive": "#55A868", "reactive": "#C44E52"})
    ax.set(title="8. Durée d'intervention par type", xlabel="Type", ylabel="Durée (h)")
    _save(fig, "08_duree_par_type.svg",
          "Mesure : distribution de la durée d'intervention par type. Les interventions "
          "réactives (correctives) sont plus longues et plus dispersées.")

    # === Bloc 3 — Maintenance dans le temps ================================
    # 9. Interventions par mois (proactive vs réactive)
    m = maintenance.copy()
    m["mois"] = pd.to_datetime(m["maintenance_at"]).dt.strftime("%Y-%m")
    piv = m.groupby(["mois", "maintenance_type"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col, color in (("proactive", "#55A868"), ("reactive", "#C44E52")):
        if col in piv:
            ax.plot(piv.index, piv[col].to_numpy(), "o-", ms=4, label=col, color=color)
    ax.set(title="9. Interventions de maintenance par mois", xlabel="Mois",
           ylabel="Nombre d'interventions")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(fontsize=8)
    _save(fig, "09_maintenance_dans_le_temps.svg",
          "Mesure : volume mensuel d'interventions par type. Saisonnalité de la "
          "maintenance programmée et survenue des pannes (réactif).")

    return files


# --- Journal des runs -------------------------------------------------------
def _runs_json_path() -> Path:
    return config.ANALYSE_REFERENTIEL_DIR / "runs.json"


def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — analyse référentiel\n\n"
        "| Run (AAAAMMJJHHMM) | Machines | Maintenances | Modèles | Proactif | Réactif |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['n_machines']} | {r['n_maintenances']} | {r['n_modeles']} | "
        f"{r['n_proactive']} | {r['n_reactive']} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    """Ajoute le run au journal JSON et régénère la vue Markdown."""
    config.ANALYSE_REFERENTIEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _runs_json_path()
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({
        "run_id": meta["run_id"],
        "timestamp": meta["timestamp"],
        "source": meta["source"],
        "n_machines": meta["n_machines"],
        "n_maintenances": meta["n_maintenances"],
        "n_modeles": meta["n_modeles"],
        "n_proactive": meta["n_proactive"],
        "n_reactive": meta["n_reactive"],
        "chemin": meta["chemin"],
    })
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.ANALYSE_REFERENTIEL_DIR / "runs.md").write_text(
        _render_runs_md(runs), encoding="utf-8"
    )


def write_methodologie() -> None:
    """Écrit (ou rafraîchit) la note de méthodologie du dossier référentiel."""
    config.ANALYSE_REFERENTIEL_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — analyse du référentiel

## Source : la couche bronze (médaillon)
Le référentiel provient du dump `data/raw/machine.sql` (tables `machine` et
`maintenance`), chargé dans `bronze` par `indusense-ingest --migrate`. L'analyse lit la
**couche bronze** (`load_referentiel` → `SELECT … FROM bronze.*`), pas le dump : en
architecture médaillon, le brut n'a qu'un seul lecteur (l'ingestion) et tout l'aval
s'appuie sur la couche gouvernée — **source unique de vérité**, typage/encodage faits une
seule fois, lignage centralisé.

## Anonymisation (RGPD) : non requise
Le référentiel ne comporte **aucune donnée personnelle** : caractéristiques d'équipement
(modèle, capacité, ligne, atelier, criticité) et journal de maintenance (type, composant,
durée, incident lié). `check_integrity` détecte tout champ DCP éventuel.

## Contrôle d'intégrité référentielle
Avant analyse : unicité des clés primaires (`machine_code`, `maintenance_id`), intégrité
référentielle (tout `machine_code` de maintenance existe dans `machine`), et cohérence
réactif ↔ incident (une maintenance `reactive` référence un `related_incident_id`).

## Périmètre d'analyse
- **Parc machines** : répartition par modèle / criticité / ligne / atelier, capacités par
  modèle, ancienneté (année de mise en service).
- **Maintenance** : volume proactif vs réactif, durée par type, composants les plus
  remplacés, interventions par machine, saisonnalité mensuelle.

Note : `action_type` est **redondant** avec `maintenance_type` (`changement_programme` ↔
proactive, `changement_suite_panne` ↔ reactive) ; une seule des deux colonnes suffit en
feature engineering.
"""
    (config.ANALYSE_REFERENTIEL_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration ----------------------------------------------------------
def run_analysis(now: datetime | None = None) -> dict:
    """Exécute le pipeline complet d'analyse du référentiel et retourne les métadonnées."""
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.ANALYSE_REFERENTIEL_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    machine, maintenance = load_referentiel()
    integrity = check_integrity(machine, maintenance)

    # Export des deux tables analysées (parquet + CSV utf-8-sig : lisible sous Excel).
    datasets = {}
    for nom, d in (("machine", machine), ("maintenance", maintenance)):
        d.to_parquet(run_dir / f"{nom}.parquet", index=False)
        csv_path = run_dir / f"{nom}.csv"
        d.to_csv(csv_path, index=False, encoding="utf-8-sig")
        datasets[nom] = str(csv_path)

    figures = make_figures(machine, maintenance, fig_dir)

    metrics = compute_metrics(machine, maintenance)
    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": "bronze.machine + bronze.maintenance",
        "datasets_csv": datasets,
        "integrite": integrity,
        "figures": figures,
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
        description="Analyse du référentiel machines & maintenance (lu en base bronze)."
    )
    parser.parse_args(argv)

    meta = run_analysis()
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(
        f"  machines={meta['n_machines']} maintenances={meta['n_maintenances']} "
        f"modeles={meta['n_modeles']} proactive={meta['n_proactive']} "
        f"reactive={meta['n_reactive']}"
    )
    print(f"  anonymisation requise : {meta['integrite']['anonymisation_requise']}")
    print(f"  {len(meta['figures'])} figures générées.")


if __name__ == "__main__":
    main()
