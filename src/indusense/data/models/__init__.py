"""Modèles ORM par couche médaillon. Importer ce package enregistre toutes les tables
sur ``Base.metadata`` (source de vérité du schéma, pilotée par Alembic).
"""

from __future__ import annotations

from indusense.data.db import Base
from indusense.data.models.bronze import Incident, Machine, Maintenance, Telemetry
from indusense.data.models.silver import (
    SilverIncident,
    SilverMachine,
    SilverMaintenance,
    SilverTelemetry,
)

__all__ = [
    "Base",
    "Incident",
    "Machine",
    "Maintenance",
    "Telemetry",
    "SilverIncident",
    "SilverMachine",
    "SilverMaintenance",
    "SilverTelemetry",
]
