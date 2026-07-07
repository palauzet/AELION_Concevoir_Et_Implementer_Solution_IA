# Méthodologie — couche gold (feature engineering)

## Rôle
Dataset **ML de maintenance prédictive** : matrice features + labels, 1 ligne par
`(machine, heure)` **opérationnelle** (hors fenêtres de maintenance), depuis le silver.

## Cibles (double — choix au modèle)
- **Classification** : `label_incident_{6,12,24,48}h` = 1 si un incident survient sur la
  machine dans `(t, t+H]`.
- **RUL** (régression) : `rul_hours` = heures jusqu'au prochain incident ; `rul_censored` = 1
  s'il n'y a pas d'incident futur (fin de série) — `rul_hours` laissé NULL.

## Features (brutes — pas de scaling/PCA ici)
- **Niveaux** capteurs + production (repivot de `silver.measurement`).
- **Dynamique** : moyenne/écart-type/min/max/**pente** par capteur, fenêtres glissantes
  **6/12/24/48 h**, *trailing strict* et **bornées au segment inter-maintenance** (jamais de
  franchissement d'une maintenance → on ne ponte pas le reset machine).
- **Qualité** : compteurs glissants `n_imputed/outlier/saturated` (flags silver).
- **Historique** : `time_since_last_maintenance_h` (depuis la **reprise** = fin de
  maintenance), `time_since_last_incident_h`, `n_incidents_7d/30d`, `n_maintenance_30d`
  (événements **strictement passés**).
- **Production/charge** : rolling 24h, cumul depuis maintenance, `utilisation` (vs capacité).
- **Machine** (dimension) : modèle, criticité, âge, ligne, atelier, capacité.
- **Temporel cyclique** : `hour/dow` en sin/cos, `shift`.

## Anti-fuite (cœur de la phase)
- Toutes les fenêtres sont **trailing** (incluent `t`, jamais le futur).
- Les colonnes décrivant l'incident (`severity`, `type_*`, `signal`, `confidence`, `comment`)
  et la maintenance qui y répond sont **exclues** : ce sont la cible, pas des features.
- **Scaling / PCA / rééquilibrage : NON faits ici** → au modèle, ajustés sur le **train**
  uniquement (sinon fuite par statistiques globales).
- **Split chronologique** `train/val/test` (70/15/15 du temps) matérialisé en colonne
  `split` : protocole reproductible et non aléatoire (TimeSeriesSplit au modèle).

## Équivalence statistique des splits
Un split chronologique n'est pas aléatoire : rien ne garantit *a priori* que train/val/test
restent comparables (dérive de composition du parc, saisonnalité des pannes...). Vérifié à
chaque run plutôt que supposé (`compare_splits_label_rates/numeric/categorical`, écrits en
CSV dans `artifacts/gold/<run_id>/`) :
- Taux de positifs par horizon et par split (figure 4).
- Écart standardisé (SMD, Austin 2009) train vs test par feature numérique — |SMD| ≥ 0.1
  signale un déséquilibre notable (`n_features_smd_notable`/`smd_max` dans `runs.md`).
- Répartition des catégorielles (`model`, `criticality`...) par split.

## Sorties
Parquet canonique (`data/processed/gold_dataset.parquet`) + run versionné
(`artifacts/gold/<run_id>/`) + table SQL `gold.dataset` (matérialisée pour inspection, **hors
Alembic** : matrice large régénérée en bloc, sans schéma incrémental ni FK).
