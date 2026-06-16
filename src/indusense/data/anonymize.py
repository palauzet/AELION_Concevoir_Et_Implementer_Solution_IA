"""Anonymisation / minimisation des données (US 1.1, C2).

Objectifs : garantir l'anonymat des opérateurs, éviter la ré-identification
par combinaison de features, limiter les biais (machines sur-représentées,
pannes minimisées).
"""

from __future__ import annotations

import pandas as pd


def anonymize_operators(df: pd.DataFrame) -> pd.DataFrame:
    """Pseudonymise/retire les identifiants opérateurs. À implémenter (US 1.1)."""
    raise NotImplementedError
