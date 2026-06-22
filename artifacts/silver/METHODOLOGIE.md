# Méthodologie — couche silver

## Rôle
Couche **nettoyée, conformée, intégrée** entre bronze (fidèle à la source) et gold
(dataset ML). Lue depuis bronze, écrite dans le schéma `silver` (PostgreSQL).

## Télémétrie — l'ordre des étapes est critique
1. **Dédoublonnage** de la clé `(machine_id, timestamp)`.
2. **Flag `during_maintenance`** : jointure temporelle avec la maintenance (relevé dans
   `[maintenance_at ; maintenance_at + duration_hours]`). On **flague, on ne supprime pas**.
3. **Imputation *maintenance-aware*** : interpolation linéaire dans le temps par **segment
   inter-maintenance** (ne franchit pas une fenêtre — respecte le « reset » de la machine
   après maintenance) et **hors fenêtres** ; les NaN *en* fenêtre restent NaN (régime non
   opérationnel). Médiane en secours. `*_was_imputed` trace les valeurs imputées.
   → Cohérence : on n'invente jamais de valeur « opérationnelle » pour une machine à l'arrêt,
   et on ne ponte pas le reset (sinon contamination du signal de dégradation, cible ML).
4. **Flag outliers (IQR)** : `*_is_outlier`, valeurs **conservées** — en maintenance
   prédictive un outlier peut être l'anomalie annonciatrice ; le tri se fera au gold.
5. **Enrichissement** : jointure dimension machine → `model`, `criticality`.

## Dates uniformisées (conformance)
Le bronze est hétérogène (incidents `date` + chaîne `time` ; télémétrie naïve ; maintenance
tz-aware). En silver, **toutes les dates/datetimes sont conformées en `datetime64` naïf
(interprété UTC)** : on retire les fuseaux (`maintenance_at`), on fusionne incident
`date`+`time` en un **`timestamp`** canonique, et on supprime les horodatages d'audit DB
(`created_at`/`updated_at`, non analytiques). Format unique aval : `YYYY-MM-DD HH:MM:SS`.

## Autres tables
- **incident** : anonymisé (`operator_*` supprimés), signal/confiance + axes temporels,
  enrichi `model`/`criticality`.
- **machine** : dimension conformée (+ `heures_equivalentes_jour` ≈ 16, flag
  `capacite_incoherente`).
- **maintenance** : `action_type` retiré (redondant avec `maintenance_type`), enrichie.

## Reporté au gold
Usage des fenêtres de maintenance (exclusion de l'entraînement et/ou features
`time_since_last_maintenance`, reset post-maintenance) ; choix de traitement des outliers.
