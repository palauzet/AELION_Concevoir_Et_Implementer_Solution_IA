"""Tests de l'outil d'analyse/comparatif de structure (contrat de schéma)."""

from __future__ import annotations

import pandas as pd

from indusense.data import schema


def test_profile_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 2], "b": ["x", None, "y"]})
    prof = schema.profile_dataframe(df)
    assert prof["n_rows"] == 3
    assert prof["columns"]["a"]["type"] == "int64"
    assert prof["columns"]["a"]["n_unique"] == 2
    assert prof["columns"]["b"]["non_null"] == 2


def test_parse_sql_columns() -> None:
    sql = (
        "CREATE TABLE IF NOT EXISTS machine (\n"
        "    machine_code   VARCHAR(16) PRIMARY KEY,\n"
        "    capacity       INTEGER NOT NULL CHECK (capacity > 0),\n"
        "    is_active      BOOLEAN NOT NULL DEFAULT TRUE\n"
        ");\n"
        "CREATE INDEX idx ON machine(machine_code);\n"
    )
    cols = schema.parse_sql_columns(sql, "machine")
    assert cols == {"machine_code": "VARCHAR(16)", "capacity": "INTEGER", "is_active": "BOOLEAN"}


def test_compare_detects_changes() -> None:
    ref = {"a": "int64", "b": "object", "c": "float64"}
    # b retiré, d ajouté, c change de type
    new = {"a": "int64", "c": "int64", "d": "object"}
    diff = schema.compare(ref, new)
    assert diff["identique"] is False
    assert diff["ajoutees"] == ["d"]
    assert diff["supprimees"] == ["b"]
    assert diff["types_modifies"] == {"c": ("float64", "int64")}


def test_compare_identical() -> None:
    cols = {"a": "int64", "b": "object"}
    assert schema.compare(cols, dict(cols))["identique"] is True


def test_check_drift_no_drift(tmp_path, monkeypatch) -> None:
    # Contrat = structure courante -> aucune dérive attendue.
    monkeypatch.setattr(schema.config, "SCHEMA_REFERENCE", tmp_path / "ref.json")
    schema.save_reference(schema.build_reference())
    result = schema.check_drift()
    assert all(d.get("identique") for d in result["drift"].values())
