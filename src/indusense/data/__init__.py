"""Sous-package data : chargement, anonymisation, intégration et nettoyage.

Un module par User Story du Sprint 01 :

- ``loaders``    : chargement des CSV/SQL bruts (US 1.1, C1)
- ``anonymize``  : anonymisation / minimisation (US 1.1, C2)
- ``db``         : modèles SQLAlchemy + migration/rollback (US 1.2, C1/C3)
- ``cleaning``   : outliers (Z-Score/IQR) + imputation (US 1.3, C3/C4)
- ``leakage``    : feature engineering temporel + TimeSeriesSplit (US 1.4, C3/C4)
- ``features``   : PCA / feature selection -> Gold Dataset (US 1.5, C3)
"""
