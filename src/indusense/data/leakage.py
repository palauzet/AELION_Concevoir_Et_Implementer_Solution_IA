"""Anti data-leakage & séries temporelles (US 1.4, C3/C4).

Éviter que le futur influence le passé :
- comprendre pourquoi un ``train_test_split`` aléatoire est incorrect ici,
- feature engineering temporel (lags, rolling windows),
- ``TimeSeriesSplit`` + jeu de validation « futur ».
"""

from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    """Ajoute lags et fenêtres glissantes sans fuite temporelle (US 1.4)."""
    raise NotImplementedError
