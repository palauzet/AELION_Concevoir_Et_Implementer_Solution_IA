"""Tests de fumée : le package s'importe et les artefacts bruts sont présents."""

from indusense import __version__, config


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_raw_artifacts_present() -> None:
    assert config.RAW_TELEMETRY.exists()
    assert config.RAW_INCIDENTS.exists()
    assert config.RAW_MACHINE_SQL.exists()
