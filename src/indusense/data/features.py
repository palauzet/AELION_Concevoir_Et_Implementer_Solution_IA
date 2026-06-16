"""Réduction dimensionnelle & Gold Dataset (US 1.5, C3).

Simplifier pour réussir : PCA / feature selection, puis production du
Gold Dataset prêt pour l'entraînement (parquet, cf. ``config.GOLD_DATASET``).
"""

from __future__ import annotations

import pandas as pd

from indusense import config


def build_gold_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le Gold Dataset final (US 1.5). À implémenter."""
    raise NotImplementedError


def save_gold_dataset(df: pd.DataFrame) -> None:
    """Sauvegarde le Gold Dataset au format parquet."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.GOLD_DATASET, index=False)
