"""Analyse et comparatif de la **structure** des données brutes (US 1.x, C1).

Outil de gouvernance du schéma : profile chaque source brute (colonnes, types, comptes)
et compare la structure courante à un **contrat versionné** (``reports/schema_reference.json``)
afin de détecter toute **dérive** lors de l'arrivée d'une nouvelle version des données.

Important : le schéma de la couche ``bronze`` est figé par les modèles ORM ; pour détecter
une dérive *du brut*, on profile donc les **sources brutes** :
- CSV (``telemetry``, ``incidents``) via pandas ;
- DDL ``CREATE TABLE`` du dump ``machine.sql`` (``machine``, ``maintenance``).

Usage :

    uv run indusense-schema            # rapport de dérive vs le contrat
    uv run indusense-schema --update   # (re)génère le contrat de référence
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from indusense import config

CSV_SOURCES: dict[str, Path] = {
    "telemetry": config.RAW_TELEMETRY,
    "incidents": config.RAW_INCIDENTS,
}
SQL_TABLES: tuple[str, ...] = ("machine", "maintenance")


# --- Profilage --------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profil structurel d'un DataFrame : nb de lignes + type/non-nuls/uniques par colonne."""
    return {
        "n_rows": int(len(df)),
        "columns": {
            c: {
                "type": str(df[c].dtype),
                "non_null": int(df[c].notna().sum()),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        },
    }


def profile_csv(path: Path) -> dict:
    """Profil structurel d'un fichier CSV."""
    return profile_dataframe(pd.read_csv(path))


def parse_sql_columns(sql: str, table: str) -> dict[str, str]:
    """Extrait les colonnes/types de la **DDL** ``CREATE TABLE {table} (...)`` d'un dump.

    Ne lit que la déclaration (pas les ``INSERT``). Ignore les lignes de contraintes.
    """
    m = re.search(
        rf"CREATE TABLE (?:IF NOT EXISTS )?{table}\s*\((.*?)\);",
        sql, re.IGNORECASE | re.DOTALL,
    )
    if m is None:
        raise ValueError(f"DDL introuvable pour la table « {table} »")
    skip = ("PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK")
    cols: dict[str, str] = {}
    for raw_line in m.group(1).splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or line.upper().startswith(skip):
            continue
        parts = line.split()
        if len(parts) >= 2:
            cols[parts[0].strip('"')] = parts[1].upper()
    return cols


def profile_sql_table(sql: str, table: str) -> dict:
    """Profil structurel d'une table du dump (depuis sa DDL ; pas de comptage de lignes)."""
    return {
        "n_rows": None,
        "columns": {c: {"type": t} for c, t in parse_sql_columns(sql, table).items()},
    }


def build_reference() -> dict:
    """Profile les 4 sources brutes (telemetry, incidents, machine, maintenance)."""
    ref: dict[str, dict] = {}
    for name, path in CSV_SOURCES.items():
        ref[name] = {"source": str(path), **profile_csv(path)}
    sql = config.RAW_MACHINE_SQL.read_text(encoding="utf-8")
    for table in SQL_TABLES:
        ref[table] = {"source": str(config.RAW_MACHINE_SQL), **profile_sql_table(sql, table)}
    return ref


# --- Contrat versionné ------------------------------------------------------
def save_reference(ref: dict, path: Path | None = None) -> Path:
    """Écrit le contrat de schéma (JSON) ; retourne le chemin."""
    path = path or config.SCHEMA_REFERENCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_reference(path: Path | None = None) -> dict:
    """Charge le contrat de schéma de référence."""
    path = path or config.SCHEMA_REFERENCE
    return json.loads(path.read_text(encoding="utf-8"))


# --- Comparaison ------------------------------------------------------------
def _types(profile: dict) -> dict[str, str]:
    """Mapping colonne -> type (partie structurelle d'un profil)."""
    return {c: meta["type"] for c, meta in profile["columns"].items()}


def compare(ref_types: dict[str, str], new_types: dict[str, str]) -> dict:
    """Compare deux mappings colonne->type : colonnes ajoutées / supprimées / types modifiés."""
    ref_cols, new_cols = set(ref_types), set(new_types)
    added = sorted(new_cols - ref_cols)
    removed = sorted(ref_cols - new_cols)
    changed = {
        c: (ref_types[c], new_types[c])
        for c in sorted(ref_cols & new_cols)
        if ref_types[c] != new_types[c]
    }
    return {
        "identique": not (added or removed or changed),
        "ajoutees": added,
        "supprimees": removed,
        "types_modifies": changed,
    }


def check_drift(reference_path: Path | None = None) -> dict:
    """Compare la structure courante des sources au contrat de référence."""
    ref = load_reference(reference_path)
    current = build_reference()
    drift: dict[str, dict] = {}
    for name, prof in current.items():
        if name not in ref:
            drift[name] = {"identique": False, "nouvelle_source": True}
        else:
            drift[name] = compare(_types(ref[name]), _types(prof))
    for name in ref:
        if name not in current:
            drift[name] = {"identique": False, "source_disparue": True}
    return {"current": current, "drift": drift}


# --- CLI --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse & comparatif de la structure des données brutes (contrat de schéma)."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="(Re)génère le contrat de référence reports/schema_reference.json.",
    )
    args = parser.parse_args(argv)

    if args.update:
        ref = build_reference()
        path = save_reference(ref)
        print(f"Contrat de schéma écrit -> {path}")
        for name, prof in ref.items():
            extra = f", {prof['n_rows']} lignes" if prof.get("n_rows") is not None else " (DDL)"
            print(f"  {name}: {len(prof['columns'])} colonnes{extra}")
        return

    result = check_drift()
    print("Structure des sources brutes :")
    for name, prof in result["current"].items():
        extra = f", {prof['n_rows']} lignes" if prof.get("n_rows") is not None else " (DDL)"
        print(f"  {name}: {len(prof['columns'])} colonnes{extra}")

    print("\nComparatif vs contrat de référence :")
    drift_found = False
    for name, d in result["drift"].items():
        if d.get("identique"):
            print(f"  {name}: conforme")
        else:
            drift_found = True
            print(f"  {name}: DÉRIVE -> {d}")

    if drift_found:
        print("\n⚠️ Dérive de schéma détectée. Si elle est légitime : `indusense-schema --update`.")
        raise SystemExit(1)
    print("\nAucune dérive de schéma.")


if __name__ == "__main__":
    main()
