"""Détection d'outliers et imputation (US 1.3, C3/C4).

Distinguer bruit vs anomalie. Choix à documenter/argumenter :
- détection : Z-Score, IQR, ...
- imputation : Mean, Median, KNN, ...
- viewers : boxplots pour visualiser détection et traitement.
"""

from __future__ import annotations

import pandas as pd


def detect_outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Masque booléen des outliers via la règle IQR (Tukey). À valider (US 1.3)."""
    raise NotImplementedError


def impute_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute les valeurs manquantes selon la stratégie retenue (US 1.3)."""
    raise NotImplementedError
