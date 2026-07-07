"""Construction de la couche GOLD (Sprint 01, feature engineering — dataset ML).

Lit la couche ``silver`` (médaillon) et produit le **dataset de maintenance prédictive** :
une matrice **features + labels**, une ligne par ``(machine, heure)`` opérationnelle,
construite **sans fuite** (trailing strict, bornée au segment inter-maintenance).

Cibles (double, choix au modèle) :
- **labels binaires** ``label_incident_{6,12,24,48}h`` = 1 si un incident survient sur la
  machine dans ``(t, t+H]`` ;
- **RUL** ``rul_hours`` = heures jusqu'au prochain incident (``rul_censored`` = 1 si aucun).

Sorties : parquet canonique ``config.GOLD_DATASET`` + run versionné ``artifacts/gold/`` +
table SQL ``gold.dataset`` (matérialisée pour inspection, hors Alembic).

Prérequis : ``indusense-silver`` a peuplé le schéma ``silver``.

Usage :

    uv run indusense-gold
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
from indusense.data import telemetry as tel  # noqa: E402
from indusense.data.db import ensure_schema, get_engine  # noqa: E402

SENSORS = list(config.TELEMETRY_SENSORS)
WINDOWS = config.ROLLING_WINDOWS
HORIZONS = config.LABEL_HORIZONS
GROUP = ["machine_pk", "segment"]


# --- Lecture silver ---------------------------------------------------------
def _read_silver(table: str, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM "{config.SCHEMA_SILVER}"."{table}"'), conn)


def load_silver(engine) -> dict[str, pd.DataFrame]:
    """Charge les tables silver utiles et joint la dimension machine (+ modèle)."""
    machine = _read_silver("machine", engine).merge(
        _read_silver("model", engine).rename(columns={"name": "model"}), on="model_id"
    )
    return {
        "reading": _read_silver("reading", engine),
        "measurement": _read_silver("measurement", engine),
        "sensor": _read_silver("sensor", engine),
        "machine": machine,
        "maintenance": _read_silver("maintenance", engine),
        "incident": _read_silver("incident", engine),
    }


# --- Base : repivot des mesures (forme longue -> large) ---------------------
def build_base(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Repivot ``measurement`` (long) en colonnes capteurs + flags, joint à ``reading``."""
    m = silver["measurement"].merge(silver["sensor"][["sensor_id", "name"]], on="sensor_id")
    parts = {
        "": "value",
        "_was_imputed": "was_imputed",
        "_is_outlier": "is_outlier",
        "_is_saturated": "is_saturated",
    }
    base = silver["reading"].copy()
    for suffix, val in parts.items():
        wide = m.pivot(index="reading_id", columns="name", values=val)
        wide.columns = [f"{c}{suffix}" for c in wide.columns]
        base = base.merge(wide, on="reading_id", how="left")
    return base


def _assign_segment(df: pd.DataFrame, maintenance: pd.DataFrame) -> np.ndarray:
    """Segment inter-maintenance par machine = nb de maintenances débutées avant ``t``."""
    starts = {
        pk: np.sort(g["maintenance_at"].to_numpy())
        for pk, g in maintenance.groupby("machine_pk")
    }
    seg = np.zeros(len(df), dtype=int)
    for pk, idx in df.groupby("machine_pk").groups.items():
        s = starts.get(pk, np.array([], dtype="datetime64[ns]"))
        pos = df.index.get_indexer(idx)
        seg[pos] = np.searchsorted(s, df.loc[idx, "timestamp"].to_numpy(), side="right")
    return seg


# --- Features : dynamique / dégradation (rolling, segment-aware) -------------
def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrégats glissants **trailing** par capteur et fenêtre, **bornés au segment**.

    Pour chaque capteur × fenêtre {6,12,24,48}h : moyenne, écart-type, min, max et **pente**
    (régression linéaire via sommes glissantes, vectorisée). Le groupage ``(machine, segment)``
    garantit qu'aucune fenêtre ne franchit une maintenance (respect du reset).
    """
    df = df.sort_values([*GROUP, "timestamp"]).reset_index(drop=True)
    # Heures écoulées depuis le début du segment : robuste à la résolution datetime
    # (us/ns) et bien conditionné pour la pente (petits ``t``, pas d'annulation).
    df["__t_h"] = df.groupby(GROUP)["timestamp"].transform(
        lambda s: (s - s.min()) / np.timedelta64(1, "h")
    )
    df["__t2"] = df["__t_h"] ** 2
    for s in SENSORS:
        df[f"__ty_{s}"] = df["__t_h"] * df[s]
    g = df.groupby(GROUP, sort=False, group_keys=False)

    def _roll(col: str, win: str, how: str) -> np.ndarray:
        r = g.rolling(win, on="timestamp")[col]
        return getattr(r, how)().to_numpy()

    for w in WINDOWS:
        win = f"{w}h"
        st = _roll("__t_h", win, "sum")
        st2 = _roll("__t2", win, "sum")
        n = _roll("__t_h", win, "count")
        denom = n * st2 - st * st
        for s in SENSORS:
            df[f"{s}_mean_{win}"] = _roll(s, win, "mean")
            df[f"{s}_std_{win}"] = _roll(s, win, "std")
            df[f"{s}_min_{win}"] = _roll(s, win, "min")
            df[f"{s}_max_{win}"] = _roll(s, win, "max")
            sy = _roll(s, win, "sum")
            sty = _roll(f"__ty_{s}", win, "sum")
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.where(denom != 0, (n * sty - st * sy) / denom, 0.0)
            df[f"{s}_slope_{win}"] = slope
    return df.drop(columns=["__t_h", "__t2", *[f"__ty_{s}" for s in SENSORS]])


def add_quality_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compteurs glissants des flags qualité (imputé / outlier / saturé) toutes mesures."""
    df = df.copy()
    df["__imp"] = df[[f"{s}_was_imputed" for s in SENSORS]].sum(axis=1)
    df["__out"] = df[[f"{s}_is_outlier" for s in SENSORS]].sum(axis=1)
    df["__sat"] = df[[f"{s}_is_saturated" for s in SENSORS]].sum(axis=1)
    g = df.groupby(GROUP, sort=False, group_keys=False)
    for w in WINDOWS:
        win = f"{w}h"
        df[f"n_imputed_{win}"] = g.rolling(win, on="timestamp")["__imp"].sum().to_numpy()
        df[f"n_outlier_{win}"] = g.rolling(win, on="timestamp")["__out"].sum().to_numpy()
        df[f"n_saturated_{win}"] = g.rolling(win, on="timestamp")["__sat"].sum().to_numpy()
    return df.drop(columns=["__imp", "__out", "__sat"])


# --- Features : production / charge ------------------------------------------
def add_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """Charge & production : rolling 24h, cumul depuis la maintenance, taux d'utilisation."""
    df = df.copy()
    g = df.groupby(GROUP, sort=False, group_keys=False)
    df["prod_sum_24h"] = g.rolling("24h", on="timestamp")["pieces_produced"].sum().to_numpy()
    df["prod_mean_24h"] = g.rolling("24h", on="timestamp")["pieces_produced"].mean().to_numpy()
    df["prod_cumul_segment"] = (
        df.groupby(GROUP, sort=False)["pieces_produced"].cumsum().to_numpy()
    )
    df["utilisation"] = df["pieces_produced"] / df["max_hourly_capacity_pieces"]
    return df


# --- Features : historique maintenance / incident (passé strict) ------------
def add_history_features(
    df: pd.DataFrame, maintenance: pd.DataFrame, incident: pd.DataFrame
) -> pd.DataFrame:
    """``time_since_last_*`` et comptages glissants d'événements **strictement passés**.

    ``time_since_last_maintenance_h`` est mesuré depuis la **reprise** (fin de maintenance =
    ``maintenance_at + duration_hours``), donc ≈ temps de fonctionnement depuis la remise en
    service (n'inclut pas la durée d'immobilisation).
    """
    df = df.copy()
    for col in ("time_since_last_maintenance_h", "time_since_last_incident_h"):
        df[col] = np.nan
    for col in ("n_incidents_7d", "n_incidents_30d", "n_maintenance_30d"):
        df[col] = 0
    maint = {}  # par machine : (débuts triés, fins = début + durée)
    for pk, g in maintenance.groupby("machine_pk"):
        g = g.sort_values("maintenance_at")
        starts = g["maintenance_at"].to_numpy()
        ends = starts + pd.to_timedelta(g["duration_hours"].to_numpy(), unit="h").to_numpy()
        maint[pk] = (starts, ends)
    inc = {
        pk: np.sort(g["timestamp"].to_numpy())
        for pk, g in incident.dropna(subset=["machine_pk"]).groupby("machine_pk")
    }
    H7, H30 = np.timedelta64(7, "D"), np.timedelta64(30, "D")
    cols = {c: df.columns.get_loc(c) for c in (
        "time_since_last_maintenance_h", "time_since_last_incident_h",
        "n_incidents_7d", "n_incidents_30d", "n_maintenance_30d",
    )}
    for pk, idx in df.groupby("machine_pk").groups.items():
        t = df.loc[idx, "timestamp"].to_numpy()
        rows = df.index.get_indexer(idx)
        m = maint.get(pk)
        if m is not None:
            starts, ends = m
            prev = np.searchsorted(starts, t, side="right") - 1  # dernière maintenance débutée <= t
            ok = prev >= 0
            tsl = np.full(len(t), np.nan)
            # Mesuré depuis la REPRISE (fin de maintenance = début + durée), pas le début.
            # max(0, .) : sécurité contre un écart négatif au bord d'une fenêtre.
            tsl[ok] = np.maximum(0.0, (t[ok] - ends[prev[ok]]) / np.timedelta64(1, "h"))
            df.iloc[rows, cols["time_since_last_maintenance_h"]] = tsl
            df.iloc[rows, cols["n_maintenance_30d"]] = (
                np.searchsorted(starts, t, side="left")
                - np.searchsorted(starts, t - H30, side="left")
            )
        i = inc.get(pk, np.array([], dtype="datetime64[ns]"))
        if len(i):
            prev = np.searchsorted(i, t, side="left") - 1  # dernier incident < t
            ok = prev >= 0
            tsi = np.full(len(t), np.nan)
            tsi[ok] = (t[ok] - i[prev[ok]]) / np.timedelta64(1, "h")
            df.iloc[rows, cols["time_since_last_incident_h"]] = tsi
            below_t = np.searchsorted(i, t, side="left")
            df.iloc[rows, cols["n_incidents_7d"]] = (
                below_t - np.searchsorted(i, t - H7, side="left")
            )
            df.iloc[rows, cols["n_incidents_30d"]] = (
                below_t - np.searchsorted(i, t - H30, side="left")
            )
    return df


# --- Features : machine (statique) + temporel cyclique ----------------------
def add_static_and_cyclical(df: pd.DataFrame, machine: pd.DataFrame) -> pd.DataFrame:
    """Attributs machine (dimension) + encodage cyclique du temps."""
    cols = ["machine_pk", "machine_code", "model", "criticality", "production_line",
            "location", "max_hourly_capacity_pieces", "commissioning_date"]
    df = df.merge(machine[cols], on="machine_pk", how="left")
    df["age_days"] = (df["timestamp"] - df["commissioning_date"]).dt.days
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["shift"] = hour.map(tel._shift_from_hour)
    return df.drop(columns=["commissioning_date"])


# --- Labels (cibles) --------------------------------------------------------
def add_labels(df: pd.DataFrame, incident: pd.DataFrame) -> pd.DataFrame:
    """Labels binaires ``incident dans (t, t+H]`` + RUL (heures jusqu'au prochain incident)."""
    df = df.copy()
    for h in HORIZONS:
        df[f"label_incident_{h}h"] = 0
    df["rul_hours"] = np.nan
    df["rul_censored"] = 1
    inc = {
        pk: np.sort(g["timestamp"].to_numpy())
        for pk, g in incident.dropna(subset=["machine_pk"]).groupby("machine_pk")
    }
    loc = {c: df.columns.get_loc(c) for c in
           [*(f"label_incident_{h}h" for h in HORIZONS), "rul_hours", "rul_censored"]}
    for pk, idx in df.groupby("machine_pk").groups.items():
        i = inc.get(pk)
        if i is None or len(i) == 0:
            continue
        t = df.loc[idx, "timestamp"].to_numpy()
        rows = df.index.get_indexer(idx)
        le_t = np.searchsorted(i, t, side="right")          # incidents <= t
        nxt = le_t                                          # 1er incident > t
        has = nxt < len(i)
        rul = np.full(len(t), np.nan)
        rul[has] = (i[nxt[has]] - t[has]) / np.timedelta64(1, "h")
        df.iloc[rows, loc["rul_hours"]] = rul
        df.iloc[rows, loc["rul_censored"]] = (~has).astype(int)
        for h in HORIZONS:
            upper = t + np.timedelta64(h, "h")
            cnt = np.searchsorted(i, upper, side="right") - le_t  # incidents dans (t, t+H]
            df.iloc[rows, loc[f"label_incident_{h}h"]] = (cnt > 0).astype(int)
    return df


def assign_split(df: pd.DataFrame, ratios: tuple[float, float] = (0.70, 0.85)) -> pd.DataFrame:
    """Split **chronologique** train/val/test (découpe temporelle globale, non aléatoire)."""
    df = df.copy()
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    span = tmax - tmin
    c1, c2 = tmin + span * ratios[0], tmin + span * ratios[1]
    df["split"] = np.where(df["timestamp"] < c1, "train",
                           np.where(df["timestamp"] < c2, "val", "test"))
    return df


# --- Comparaison des splits (le split chronologique n'est pas aléatoire : on vérifie qu'il
# reste statistiquement raisonnable plutôt que de le supposer) -----------------------------
def compare_splits_label_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Taux de positifs par horizon et par split — la cible doit rester comparable d'un split
    à l'autre malgré le découpage chronologique (ex. pas de saisonnalité des pannes qui
    concentrerait les incidents sur le test)."""
    rows = []
    for h in HORIZONS:
        col = f"label_incident_{h}h"
        row: dict = {"horizon_h": h}
        for split in ("train", "val", "test"):
            sub = df.loc[df["split"] == split, col]
            row[split] = round(float(sub.mean()), 4) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compare_splits_numeric(
    df: pd.DataFrame, numeric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Moyenne par split + écart standardisé (SMD) train vs test, par feature numérique.

    ``smd = (moyenne_test - moyenne_train) / écart-type poolé`` (Austin 2009) : |SMD| ≥ 0.1
    signale un déséquilibre notable entre train et test à surveiller. Trié par |SMD|
    décroissant (les features les plus décalées en tête).
    """
    if numeric_cols is None:
        exclude = {"machine_pk", "rul_hours", "rul_censored"} | {
            f"label_incident_{h}h" for h in HORIZONS
        }
        numeric_cols = [c for c in df.select_dtypes("number").columns if c not in exclude]
    rows = []
    for col in numeric_cols:
        train = df.loc[df["split"] == "train", col]
        val = df.loc[df["split"] == "val", col]
        test = df.loc[df["split"] == "test", col]
        pooled_std = float(np.sqrt((train.var() + test.var()) / 2))
        smd = (test.mean() - train.mean()) / pooled_std if pooled_std > 0 else 0.0
        rows.append({
            "feature": col,
            "train_mean": round(float(train.mean()), 4),
            "val_mean": round(float(val.mean()), 4),
            "test_mean": round(float(test.mean()), 4),
            "smd_train_test": round(float(smd), 4),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("smd_train_test", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )


def compare_splits_categorical(
    df: pd.DataFrame, categorical_cols: list[str] | None = None
) -> pd.DataFrame:
    """Répartition (%) de chaque modalité catégorielle par split — détecte une dérive de
    composition du parc (ex. un modèle de machine concentré sur une période donnée)."""
    if categorical_cols is None:
        categorical_cols = ["model", "criticality", "production_line", "location", "shift"]
    rows = []
    for col in categorical_cols:
        shares = df.groupby("split")[col].value_counts(normalize=True).unstack("split").fillna(0.0)
        for level, r in shares.iterrows():
            rows.append({
                "feature": col, "modalite": level,
                "train": round(float(r.get("train", 0.0)), 4),
                "val": round(float(r.get("val", 0.0)), 4),
                "test": round(float(r.get("test", 0.0)), 4),
            })
    return pd.DataFrame(rows)


# --- Assemblage -------------------------------------------------------------
def build_gold(silver: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Construit la matrice gold complète (features + labels) et ses métriques."""
    base = build_base(silver)
    base["segment"] = _assign_segment(base, silver["maintenance"])
    base = base[~base["during_maintenance"]].reset_index(drop=True)  # points opérationnels

    df = add_dynamic_features(base)
    df = add_quality_counts(df)
    df = add_static_and_cyclical(df, silver["machine"])  # capacité requise par la production
    df = add_production_features(df)
    df = add_history_features(df, silver["maintenance"], silver["incident"])
    df = add_labels(df, silver["incident"])
    df = assign_split(df)

    # Colonnes techniques / brutes non features retirées du dataset final.
    drop = ["reading_id", "during_maintenance", "segment",
            *[f"{s}_was_imputed" for s in SENSORS],
            *[f"{s}_is_outlier" for s in SENSORS],
            *[f"{s}_is_saturated" for s in SENSORS]]
    df = df.drop(columns=drop).sort_values(["machine_pk", "timestamp"]).reset_index(drop=True)

    label_cols = [f"label_incident_{h}h" for h in HORIZONS]
    metrics = {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1]),
        "n_features": int(df.shape[1] - len(label_cols) - 5),  # - labels - rul(2) - keys(3)
        "taux_positifs": {f"{h}h": round(float(df[f"label_incident_{h}h"].mean()), 4)
                          for h in HORIZONS},
        "rul_censored": int(df["rul_censored"].sum()),
        "split": {k: int(v) for k, v in df["split"].value_counts().items()},
    }
    return df, metrics


# --- Figures ----------------------------------------------------------------
def make_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """3 figures : équilibre des labels, distribution du RUL, corrélations features↔label."""
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

    # 1. Taux de positifs par horizon
    rates = [df[f"label_incident_{h}h"].mean() * 100 for h in HORIZONS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"{h}h" for h in HORIZONS], rates, color="#C44E52")
    for x, r in enumerate(rates):
        ax.text(x, r, f"{r:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set(title="1. Équilibre des labels — taux de positifs par horizon",
           xlabel="Horizon", ylabel="% relevés positifs")
    _save(fig, "01_equilibre_labels.svg",
          "Mesure : part de relevés suivis d'un incident dans l'horizon. Quantifie le "
          "déséquilibre de classe (modéré) à gérer au modèle.")

    # 2. Distribution du RUL (non censuré, borné à 30 jours pour la lisibilité)
    rul = df.loc[df["rul_censored"] == 0, "rul_hours"].clip(upper=720)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(rul, bins=50, color="#4C72B0", ax=ax)
    ax.set(title="2. Distribution du RUL (heures jusqu'au prochain incident, ≤ 720 h)",
           xlabel="RUL (h)", ylabel="Nombre de relevés")
    _save(fig, "02_distribution_rul.svg",
          "Mesure : temps restant avant le prochain incident (cible de régression), relevés "
          "non censurés. Forme décroissante = incidents fréquents.")

    # 3. Corrélation des features numériques avec label_24h (top 15 |r|)
    num = df.select_dtypes("number").drop(
        columns=[c for c in df.columns if c.startswith("label_incident_") or c == "rul_censored"]
    )
    corr = num.corrwith(df["label_incident_24h"]).dropna().sort_values(key=abs, ascending=False)
    top = corr.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.to_numpy(), color=np.where(top > 0, "#C44E52", "#4C72B0"))
    ax.axvline(0, color="#444", lw=0.8)
    ax.set(title="3. Top features corrélées à label_incident_24h", xlabel="Corrélation (Pearson)")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, "03_correlations_label24h.svg",
          "Mesure : 15 features de plus forte corrélation (|r|) avec le label à 24 h. "
          "Premier tri du pouvoir prédictif (linéaire) avant modélisation.")

    # 4. Comparaison train/val/test : taux de positifs par horizon (le split est chronologique,
    # pas aléatoire — on vérifie l'équivalence statistique plutôt que de la supposer).
    split_rates = compare_splits_label_rates(df)
    x4 = np.arange(len(split_rates))
    width = 0.25
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, split in enumerate(("train", "val", "test")):
        ax.bar(x4 + (i - 1) * width, split_rates[split] * 100, width,
               label=split, color=colors[split])
    ax.set_xticks(x4)
    ax.set_xticklabels([f"{h}h" for h in split_rates["horizon_h"]])
    ax.legend()
    ax.set(title="4. Comparaison train/val/test — taux de positifs par horizon",
           xlabel="Horizon", ylabel="% relevés positifs")
    _save(fig, "04_comparaison_splits.svg",
          "Mesure : équivalence statistique du split chronologique sur la cible. Un taux très "
          "différent entre splits signalerait une dérive (ex. saisonnalité des pannes) au-delà "
          "du simple découpage temporel — cf. aussi compare_splits_numeric/categorical.")
    return files


# --- Persistance + journal + méthodologie -----------------------------------
def write_gold(df: pd.DataFrame, engine) -> None:
    """Matérialise le dataset gold en table SQL ``gold.dataset`` (remplacée à chaque run)."""
    ensure_schema(engine, config.SCHEMA_GOLD)
    df.to_sql("dataset", engine, schema=config.SCHEMA_GOLD, if_exists="replace",
              index=False, chunksize=10_000)


def _render_runs_md(runs: list[dict]) -> str:
    header = (
        "# Journal des runs — construction gold\n\n"
        "| Run | Lignes | Features | Pos. 24h | Pos. 48h | RUL censurés | Train/Val/Test | "
        "SMD max (feature) |\n"
        "|---|---:|---:|---:|---:|---:|---|---|\n"
    )
    rows = "".join(
        f"| {r['run_id']} | {r['n_lignes']} | {r['n_features']} | "
        f"{r['taux_positifs'].get('24h', '-')} | {r['taux_positifs'].get('48h', '-')} | "
        f"{r['rul_censored']} | "
        f"{r['split'].get('train', 0)}/{r['split'].get('val', 0)}/{r['split'].get('test', 0)} | "
        f"{r.get('smd_max', '-')} ({r.get('smd_max_feature', '-')}) |\n"
        for r in sorted(runs, key=lambda r: r["run_id"])
    )
    return header + rows


def append_run(meta: dict) -> None:
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GOLD_DIR / "runs.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append({k: meta[k] for k in
                 ("run_id", "n_lignes", "n_features", "taux_positifs", "rul_censored", "split",
                  "n_features_smd_notable", "smd_max", "smd_max_feature")})
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.GOLD_DIR / "runs.md").write_text(_render_runs_md(runs), encoding="utf-8")


def write_methodologie() -> None:
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Méthodologie — couche gold (feature engineering)

## Rôle
Dataset **ML de maintenance prédictive** : matrice features + labels, 1 ligne par
`(machine, heure)` **opérationnelle** (hors fenêtres de maintenance), depuis le silver.

## Cibles (double — choix au modèle)
- **Classification** : `label_incident_{6,12,24,48}h` = 1 si un incident survient sur la
  machine dans `(t, t+H]`.
- **RUL** (régression) : `rul_hours` = heures jusqu'au prochain incident ; `rul_censored` = 1
  s'il n'y a pas d'incident futur (fin de série) — `rul_hours` laissé NULL.

## Features (brutes — pas de scaling/PCA ici)
- **Niveaux** capteurs + production (repivot de `silver.measurement`).
- **Dynamique** : moyenne/écart-type/min/max/**pente** par capteur, fenêtres glissantes
  **6/12/24/48 h**, *trailing strict* et **bornées au segment inter-maintenance** (jamais de
  franchissement d'une maintenance → on ne ponte pas le reset machine).
- **Qualité** : compteurs glissants `n_imputed/outlier/saturated` (flags silver).
- **Historique** : `time_since_last_maintenance_h` (depuis la **reprise** = fin de
  maintenance), `time_since_last_incident_h`, `n_incidents_7d/30d`, `n_maintenance_30d`
  (événements **strictement passés**).
- **Production/charge** : rolling 24h, cumul depuis maintenance, `utilisation` (vs capacité).
- **Machine** (dimension) : modèle, criticité, âge, ligne, atelier, capacité.
- **Temporel cyclique** : `hour/dow` en sin/cos, `shift`.

## Anti-fuite (cœur de la phase)
- Toutes les fenêtres sont **trailing** (incluent `t`, jamais le futur).
- Les colonnes décrivant l'incident (`severity`, `type_*`, `signal`, `confidence`, `comment`)
  et la maintenance qui y répond sont **exclues** : ce sont la cible, pas des features.
- **Scaling / PCA / rééquilibrage : NON faits ici** → au modèle, ajustés sur le **train**
  uniquement (sinon fuite par statistiques globales).
- **Split chronologique** `train/val/test` (70/15/15 du temps) matérialisé en colonne
  `split` : protocole reproductible et non aléatoire (TimeSeriesSplit au modèle).

## Équivalence statistique des splits
Un split chronologique n'est pas aléatoire : rien ne garantit *a priori* que train/val/test
restent comparables (dérive de composition du parc, saisonnalité des pannes...). Vérifié à
chaque run plutôt que supposé (`compare_splits_label_rates/numeric/categorical`, écrits en
CSV dans `artifacts/gold/<run_id>/`) :
- Taux de positifs par horizon et par split (figure 4).
- Écart standardisé (SMD, Austin 2009) train vs test par feature numérique — |SMD| ≥ 0.1
  signale un déséquilibre notable (`n_features_smd_notable`/`smd_max` dans `runs.md`).
- Répartition des catégorielles (`model`, `criticality`...) par split.

## Sorties
Parquet canonique (`data/processed/gold_dataset.parquet`) + run versionné
(`artifacts/gold/<run_id>/`) + table SQL `gold.dataset` (matérialisée pour inspection, **hors
Alembic** : matrice large régénérée en bloc, sans schéma incrémental ni FK).
"""
    (config.GOLD_DIR / "METHODOLOGIE.md").write_text(content, encoding="utf-8")


# --- Orchestration ----------------------------------------------------------
def run(now: datetime | None = None) -> dict:
    """Construit le dataset gold complet et retourne les métadonnées du run."""
    now = now or datetime.now()
    run_id = now.strftime(config.RUN_TS_FORMAT)
    run_dir = config.GOLD_DIR / run_id
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df, metrics = build_gold(load_silver(engine))

    df.to_parquet(run_dir / "gold_dataset.parquet", index=False)
    config.GOLD_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.GOLD_DATASET, index=False)  # parquet canonique
    figures = make_figures(df, fig_dir)
    if engine is not None:
        write_gold(df, engine)

    numeric_cmp = compare_splits_numeric(df)
    compare_splits_label_rates(df).to_csv(run_dir / "split_label_rates.csv", index=False)
    numeric_cmp.to_csv(run_dir / "split_numeric_stats.csv", index=False)
    compare_splits_categorical(df).to_csv(run_dir / "split_categorical_shares.csv", index=False)
    smd_notable = numeric_cmp.loc[numeric_cmp["smd_train_test"].abs() >= 0.1]

    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": "silver.*",
        "figures": figures,
        "chemin": str(run_dir),
        **metrics,
        "n_features_smd_notable": int(len(smd_notable)),
        "smd_max": round(float(numeric_cmp["smd_train_test"].abs().max()), 4)
        if len(numeric_cmp) else 0.0,
        "smd_max_feature": str(numeric_cmp.iloc[0]["feature"]) if len(numeric_cmp) else None,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_methodologie()
    append_run(meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(
        description="Construction de la couche gold (silver -> dataset ML features + labels)."
    ).parse_args(argv)

    meta = run()
    print(f"Run {meta['run_id']} -> {meta['chemin']}")
    print(f"  lignes={meta['n_lignes']} features={meta['n_features']} "
          f"colonnes={meta['n_colonnes']}")
    print(f"  taux positifs : {meta['taux_positifs']}")
    print(f"  RUL censurés : {meta['rul_censored']} | split : {meta['split']}")
    print(f"  équivalence splits : {meta['n_features_smd_notable']} feature(s) |SMD|>=0.1 "
          f"(max={meta['smd_max']} sur {meta['smd_max_feature']})")
    print(f"  {len(meta['figures'])} figures ; table "
          f"'{config.SCHEMA_GOLD}.dataset' + parquet écrits.")


if __name__ == "__main__":
    main()
