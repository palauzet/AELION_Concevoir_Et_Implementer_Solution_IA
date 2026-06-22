"""Construction de la couche SILVER (Sprint 01, US 1.3 + intégration).

Lit la couche ``bronze`` (médaillon), produit des tables **nettoyées, conformées,
intégrées** et les écrit dans le schéma ``silver`` PostgreSQL, avec artefacts versionnés
(parquet par table, boxplots, journal, méthodologie).

Cœur US 1.3 sur la télémétrie (l'ordre est critique) :
1. dédoublonnage de clé ``(machine_id, timestamp)`` ;
2. flag ``during_maintenance`` (jointure temporelle avec ``bronze.maintenance``) ;
3. **imputation des NaN *maintenance-aware*** : interpolation linéaire par *segment
   inter-maintenance* (ne franchit pas une fenêtre — respecte le « reset » machine) et hors
   fenêtres ; les NaN en fenêtre de maintenance restent NaN (non opérationnels) ;
4. flag des outliers (IQR) — valeurs **conservées** (un outlier peut être l'anomalie) ;
5. enrichissement par la dimension machine (``model``, ``criticality``).

Prérequis : ``indusense-ingest --migrate`` a alimenté ``bronze``.

Usage :

    uv run indusense-silver
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sqlalchemy import text  # noqa: E402

from indusense import config  # noqa: E402
from indusense.data import incidents as inc  # noqa: E402
from indusense.data import telemetry as tel  # noqa: E402
from indusense.data.anonymize import anonymize_operators  # noqa: E402
from indusense.data.db import get_engine, read_bronze  # noqa: E402

SENSORS = list(config.TELEMETRY_SENSORS)


# --- Dates : format uniforme (datetime64 naïf, interprété UTC) --------------
AUDIT_COLS = ["created_at", "updated_at"]


def _naive_utc(s: pd.Series) -> pd.Series:
    """Conforme une série date/datetime (tz-aware, naïve ou date) en datetime64 **naïf** UTC."""
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


# --- Dimension machine (réutilisée pour l'enrichissement) -------------------
def _machine_dim() -> pd.DataFrame:
    return read_bronze("machine")[["machine_code", "model", "criticality"]]


def build_machine() -> pd.DataFrame:
    """Dimension machine conformée : dates uniformes, sans audit, + capacité dérivée."""
    m = read_bronze("machine").drop(columns=AUDIT_COLS, errors="ignore").copy()
    m["commissioning_date"] = _naive_utc(m["commissioning_date"])
    ratio = m["max_daily_capacity"] / m["max_hourly_capacity_pieces"]
    m["heures_equivalentes_jour"] = ratio.round().astype(int)
    m["capacite_incoherente"] = (ratio - ratio.median()).abs() > 1.0
    return m


def build_maintenance() -> pd.DataFrame:
    """Maintenance conformée : retire `action_type`/audit, date uniforme, enrichie."""
    mt = read_bronze("maintenance").drop(
        columns=["action_type", *AUDIT_COLS], errors="ignore"
    ).copy()
    mt["duration_hours"] = mt["duration_hours"].astype(float)
    mt["maintenance_at"] = _naive_utc(mt["maintenance_at"])
    return mt.merge(_machine_dim(), on="machine_code", how="left")


# --- Télémétrie : fenêtres de maintenance + segments ------------------------
def _maintenance_windows() -> pd.DataFrame:
    """Fenêtres de maintenance ``[start, end)`` par machine (tz alignée sur la télémétrie)."""
    mt = read_bronze("maintenance")[["machine_code", "maintenance_at", "duration_hours"]].copy()
    start = pd.to_datetime(mt["maintenance_at"], utc=True).dt.tz_localize(None)
    end = start + pd.to_timedelta(mt["duration_hours"].astype(float), unit="h")
    return pd.DataFrame({"machine_code": mt["machine_code"], "start": start, "end": end})


def _flag_and_segment(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ``during_maintenance`` (dans une fenêtre) et ``_segment`` (n° inter-maintenance).

    ``_segment`` = nombre de maintenances débutées avant le relevé (par machine) : interpoler
    *au sein* d'un segment ne franchit donc jamais une maintenance (respect du reset).
    """
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    during = np.zeros(len(df), dtype=bool)
    segment = np.zeros(len(df), dtype=int)
    starts_by_m = {m: np.sort(g["start"].to_numpy()) for m, g in windows.groupby("machine_code")}
    for machine, idx in df.groupby("machine_id").groups.items():
        ts = df.loc[idx, "timestamp"].to_numpy()
        w = windows[windows["machine_code"] == machine]
        inwin = np.zeros(len(ts), dtype=bool)
        for s, e in zip(w["start"].to_numpy(), w["end"].to_numpy(), strict=True):
            inwin |= (ts >= s) & (ts < e)
        starts = starts_by_m.get(machine, np.array([], dtype="datetime64[ns]"))
        seg = np.searchsorted(starts, ts, side="right")
        pos = df.index.get_indexer(idx)
        during[pos] = inwin
        segment[pos] = seg
    df["during_maintenance"] = during
    df["_segment"] = segment
    return df


def impute_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les NaN capteurs **hors fenêtres de maintenance**, par segment inter-maintenance.

    Interpolation linéaire dans le temps au sein de ``(machine_id, _segment)`` ; médiane en
    secours. Les NaN *en* fenêtre de maintenance restent NaN (non opérationnels → gold).
    ``<capteur>_was_imputed`` marque les valeurs réellement imputées.
    """
    df = df.copy()
    op_mask = ~df["during_maintenance"]
    for col in SENSORS:
        was_nan = df[col].isna()
        op = df.loc[op_mask, ["machine_id", "_segment", col]].copy()
        op[col] = op.groupby(["machine_id", "_segment"])[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df.loc[op_mask, col] = op[col]
        median = df.loc[op_mask, col].median()
        df.loc[op_mask & df[col].isna(), col] = median
        df[f"{col}_was_imputed"] = was_nan & df[col].notna()
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ``<capteur>_is_outlier`` (méthode IQR) — valeurs conservées."""
    df = df.copy()
    bounds = tel.detect_outliers(df, tuple(SENSORS))
    for col in SENSORS:
        low, high = bounds.loc[col, "borne_basse"], bounds.loc[col, "borne_haute"]
        df[f"{col}_is_outlier"] = ((df[col] < low) | (df[col] > high)).fillna(False)
    return df


def build_telemetry() -> tuple[pd.DataFrame, dict]:
    """Construit ``silver.telemetry`` et retourne (df, métriques)."""
    df = read_bronze("telemetry").drop(columns=["telemetry_id"], errors="ignore")
    df, n_doublons = tel.deduplicate(df)
    df["timestamp"] = _naive_utc(df["timestamp"])  # format date uniforme
    windows = _maintenance_windows()
    df = _flag_and_segment(df, windows)
    df = impute_telemetry(df)
    df = flag_outliers(df)
    df = df.merge(_machine_dim(), left_on="machine_id", right_on="machine_code", how="left")
    df = df.drop(columns=["machine_code", "_segment"])
    metrics = {
        "n_lignes": int(len(df)),
        "n_doublons_supprimes": int(n_doublons),
        "n_relevés_maintenance": int(df["during_maintenance"].sum()),
        "n_valeurs_imputees": {c: int(df[f"{c}_was_imputed"].sum()) for c in SENSORS},
        "n_outliers_flagges": {c: int(df[f"{c}_is_outlier"].sum()) for c in SENSORS},
        "n_nan_residuels": int(df[SENSORS].isna().sum().sum()),
    }
    return df, metrics


def build_incident() -> pd.DataFrame:
    """Construit ``silver.incident`` : anonymisé, enrichi, **dates uniformisées**.

    La date et l'heure (colonnes séparées `date` + chaîne `time`) sont fusionnées en un
    **`timestamp`** canonique (datetime64 naïf), cohérent avec la télémétrie et la
    maintenance ; les colonnes source redondantes sont retirées.
    """
    df = read_bronze("incident").drop(columns=["incident_pk"], errors="ignore")
    df = inc.compute_confidence(inc.enrich(anonymize_operators(df)))
    df = df.rename(columns={"dt": "timestamp"})
    df["timestamp"] = _naive_utc(df["timestamp"])
    df["jour"] = _naive_utc(df["jour"])
    df = df.drop(columns=["date", "time"], errors="ignore")
    return df.merge(_machine_dim(), left_on="machine_id", right_on="machine_code", how="left").drop(
        columns=["machine_code"]
    )


# --- Figures (viewers US 1.3) -----------------------------------------------
def make_figures(telemetry: pd.DataFrame, metrics: dict, out_dir: Path) -> list[str]:
    """Boxplots des capteurs + bilans NaN imputés / outliers flaggés. Retourne les noms."""
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

    # 1. Boxplots des capteurs (post-traitement) avec outliers
    fig, axes = plt.subplots(1, len(SENSORS), figsize=(13, 4))
    for ax, col in zip(axes, SENSORS, strict=True):
        sns.boxplot(y=telemetry[col], ax=ax, color="#4C72B0")
        ax.set(title=col, xlabel="", ylabel="")
    fig.suptitle("1. Boxplots des capteurs (silver) — détection des outliers (IQR)", y=1.0)
    fig.tight_layout()
    _save(fig, "01_boxplots_capteurs.svg",
          "Mesure : distribution (médiane, quartiles, moustaches IQR) de chaque capteur après "
          "dédoublonnage/imputation. Les points hors moustaches sont flaggés (non supprimés).")

    # 2. Bilan du nettoyage : NaN imputés & outliers flaggés par capteur
    imp = pd.Series(metrics["n_valeurs_imputees"])
    out = pd.Series(metrics["n_outliers_flagges"])
    x = np.arange(len(SENSORS))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, imp.reindex(SENSORS).to_numpy(), 0.4, label="NaN imputés", color="#55A868")
    ax.bar(x + 0.2, out.reindex(SENSORS).to_numpy(), 0.4, label="outliers flaggés", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(SENSORS, rotation=20, ha="right", fontsize=8)
    ax.set(title="2. Nettoyage silver par capteur", ylabel="Nombre de valeurs")
    ax.legend(fontsize=8)
    _save(fig, "02_bilan_nettoyage.svg",
          "Mesure : par capteur, nombre de NaN imputés (interpolation hors maintenance) et "
          "d'outliers flaggés (IQR, conservés). Quantifie le traitement appliqué.")

    return files


# --- Persistance + journal + méthodologie -----------------------------------
def write_silver_tables(tables: dict[str, pd.DataFrame], engine) -> None:
    """Recharge les tables silver (**schéma géré par Alembic**) : TRUNCATE puis append typé.

    Insertion en append dans les tables migrées (types/contraintes garantis). Ordre du dict
    (machine d'abord) pour respecter les FK ; TRUNCATE … CASCADE pour repartir propre.
    """
    qualified = ", ".join(f'"{config.SCHEMA_SILVER}"."{name}"' for name in tables)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {qualified} RESTART IDENTITY CASCADE"))
    for name, df in tables.items():
        df.to_sql(name, engine, schema=config.SCHEMA_SILVER, if_exists="append", index=False)


def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — construction silver\n\n"
        "| Run | Télémétrie | Doublons | Imputés | NaN rés. | Incidents |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['n_telemetry']} | {r['n_doublons_supprimes']} | "
        f"{r['n_imputes_total']} | {r['n_nan_residuels']} | {r['n_incidents']} |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    config.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SILVER_DIR / "runs.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({
        "run_id": meta["run_id"],
        "timestamp": meta["timestamp"],
        "n_telemetry": meta["telemetry"]["n_lignes"],
        "n_doublons_supprimes": meta["telemetry"]["n_doublons_supprimes"],
        "n_imputes_total": meta["n_imputes_total"],
        "n_nan_residuels": meta["telemetry"]["n_nan_residuels"],
        "n_incidents": meta["n_incidents"],
        "chemin": meta["chemin"],
    })
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.SILVER_DIR / "runs.md").write_text(_render_runs_md(runs), encoding="utf-8")


def write_methodologie() -> None:
    config.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — couche silver

## Rôle
Couche **nettoyée, conformée, intégrée** entre bronze (fidèle à la source) et gold
(dataset ML). Lue depuis bronze, écrite dans le schéma `silver` (PostgreSQL).

## Télémétrie — l'ordre des étapes est critique
1. **Dédoublonnage** de la clé `(machine_id, timestamp)`.
2. **Flag `during_maintenance`** : jointure temporelle avec la maintenance (relevé dans
   `[maintenance_at ; maintenance_at + duration_hours]`). On **flague, on ne supprime pas**.
3. **Imputation *maintenance-aware*** : interpolation linéaire dans le temps par **segment
   inter-maintenance** (ne franchit pas une fenêtre — respecte le « reset » de la machine
   après maintenance) et **hors fenêtres** ; les NaN *en* fenêtre restent NaN (régime non
   opérationnel). Médiane en secours. `*_was_imputed` trace les valeurs imputées.
   → Cohérence : on n'invente jamais de valeur « opérationnelle » pour une machine à l'arrêt,
   et on ne ponte pas le reset (sinon contamination du signal de dégradation, cible ML).
4. **Flag outliers (IQR)** : `*_is_outlier`, valeurs **conservées** — en maintenance
   prédictive un outlier peut être l'anomalie annonciatrice ; le tri se fera au gold.
5. **Enrichissement** : jointure dimension machine → `model`, `criticality`.

## Dates uniformisées (conformance)
Le bronze est hétérogène (incidents `date` + chaîne `time` ; télémétrie naïve ; maintenance
tz-aware). En silver, **toutes les dates/datetimes sont conformées en `datetime64` naïf
(interprété UTC)** : on retire les fuseaux (`maintenance_at`), on fusionne incident
`date`+`time` en un **`timestamp`** canonique, et on supprime les horodatages d'audit DB
(`created_at`/`updated_at`, non analytiques). Format unique aval : `YYYY-MM-DD HH:MM:SS`.

## Autres tables
- **incident** : anonymisé (`operator_*` supprimés), signal/confiance + axes temporels,
  enrichi `model`/`criticality`.
- **machine** : dimension conformée (+ `heures_equivalentes_jour` ≈ 16, flag
  `capacite_incoherente`).
- **maintenance** : `action_type` retiré (redondant avec `maintenance_type`), enrichie.

## Reporté au gold
Usage des fenêtres de maintenance (exclusion de l'entraînement et/ou features
`time_since_last_maintenance`, reset post-maintenance) ; choix de traitement des outliers.
"""
    (config.SILVER_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration ----------------------------------------------------------
def run(now: datetime | None = None) -> dict:
    """Construit la couche silver complète et retourne les métadonnées du run."""
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.SILVER_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    machine = build_machine()
    maintenance = build_maintenance()
    telemetry, tel_metrics = build_telemetry()
    incident = build_incident()
    tables = {
        "machine": machine,
        "maintenance": maintenance,
        "telemetry": telemetry,
        "incident": incident,
    }

    for name, df in tables.items():
        df.to_parquet(run_dir / f"{name}.parquet", index=False)
    figures = make_figures(telemetry, tel_metrics, fig_dir)
    write_silver_tables(tables, get_engine())

    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": "bronze.*",
        "telemetry": tel_metrics,
        "n_imputes_total": int(sum(tel_metrics["n_valeurs_imputees"].values())),
        "n_incidents": int(len(incident)),
        "n_machines": int(len(machine)),
        "n_maintenances": int(len(maintenance)),
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
    argparse.ArgumentParser(
        description="Construction de la couche silver (bronze -> silver, nettoyage/enrichissement)."
    ).parse_args(argv)

    meta = run()
    t = meta["telemetry"]
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(
        f"  telemetry: {t['n_lignes']} lignes (doublons retirés={t['n_doublons_supprimes']}, "
        f"maintenance={t['n_relevés_maintenance']}, imputés={meta['n_imputes_total']}, "
        f"NaN résiduels={t['n_nan_residuels']})"
    )
    print(f"  incident: {meta['n_incidents']} | machine: {meta['n_machines']} | "
          f"maintenance: {meta['n_maintenances']}")
    print(f"  {len(meta['figures'])} figures ; tables écrites dans le schéma "
          f"'{config.SCHEMA_SILVER}'.")


if __name__ == "__main__":
    main()
