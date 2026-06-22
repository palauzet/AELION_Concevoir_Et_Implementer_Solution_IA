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
5. **FK machine** : `machine_id` référence `silver.machine` (model/criticality récupérés par
   jointure, non copiés).

## Normalisation (mesurée, star-schema)
`model` et `criticality` ne sont **pas dupliqués** dans les faits : ils vivent dans la
dimension `silver.machine` et se récupèrent par **jointure** (`machine_id`/`machine_code`).
`component` est une **table de lookup** `silver.component` (FK `maintenance.component_id`).
Les enums (`criticality`, `maintenance_type`) sont contraints par **CHECK** (pas de table).
Choix *mesuré* : on retire la redondance et on pose les FK, sans 3NF stricte (jointures
minimisées pour le ML) ; les capteurs restent en **format large**.

## Dates uniformisées (conformance)
Le bronze est hétérogène (incidents `date` + chaîne `time` ; télémétrie naïve ; maintenance
tz-aware). En silver, **toutes les dates/datetimes sont conformées en `datetime64` naïf
(interprété UTC)** : on retire les fuseaux (`maintenance_at`), on fusionne incident
`date`+`time` en un **`timestamp`** canonique, et on supprime les horodatages d'audit DB
(`created_at`/`updated_at`, non analytiques). Format unique aval : `YYYY-MM-DD HH:MM:SS`.

## Autres tables
- **incident** : anonymisé (`operator_*` supprimés), signal/confiance + axes temporels ;
  `machine_id` = FK vers la dimension machine.
- **machine** : dimension (porte `model`/`criticality`) conformée (+ `heures_equivalentes_jour`
  ≈ 16, flag `capacite_incoherente`, CHECK criticité).
- **component** : lookup des composants (`component_id`, `name`).
- **maintenance** : `action_type`/`model`/`criticality` retirés ; `component` → `component_id`
  (FK lookup) ; CHECK `maintenance_type`.

## Reporté au gold
Usage des fenêtres de maintenance (exclusion de l'entraînement et/ou features
`time_since_last_maintenance`, reset post-maintenance) ; choix de traitement des outliers.
