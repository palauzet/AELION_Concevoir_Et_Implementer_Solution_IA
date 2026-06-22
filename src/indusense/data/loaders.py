"""Chargement des artefacts bruts (US 1.1, C1).

Point d'entrée pour lire les CSV bruts (aperçu/acquisition, cf.
``notebooks/00_acquisition``). L'analyse exploratoire se fait sur ``bronze`` dans
``notebooks/01_analyse_exploratoire``.
"""

from __future__ import annotations

import pandas as pd

from indusense import config


def load_telemetry() -> pd.DataFrame:
    """Charge les relevés capteurs (température / pression hydraulique)."""
    return pd.read_csv(config.RAW_TELEMETRY)


def load_incidents() -> pd.DataFrame:
    """Charge les relevés manuels d'incidents."""
    return pd.read_csv(config.RAW_INCIDENTS)
