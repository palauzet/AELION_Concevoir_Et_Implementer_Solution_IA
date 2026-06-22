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
5. **Décomposition 3NF (en-tête/détail)** : `reading` = un relevé horodaté (`reading_id`,
   `machine_pk`, `timestamp`, `during_maintenance`, `pieces_produced`) ; `measurement` = une
   mesure capteur (`reading_id`, `sensor_id`, `value`, `was_imputed`, `is_outlier`).

## Normalisation 3NF — motivée par l'évolutivité
Les **mesures de capteurs** sont normalisées en **forme longue** : un capteur = une **ligne**
de `measurement` référençant la dimension `silver.sensor`. Objectif : **ajouter/retirer un
capteur sans modifier le schéma** (data change, pas DDL) — il suffit d'éditer
`config.SENSOR_UNITS`. Des **clés surrogates** identifient les machines (`machine_pk`) et les
types de machines (`silver.model.model_id`) ; tous les faits référencent la machine par
`machine_pk`. `component` reste une table de lookup (`silver.component`). Les enums
(`criticality`, `maintenance_type`) sont contraints par **CHECK**.
Compromis assumé : ~4× plus de lignes de mesures et un **repivot** nécessaire au gold.

## Dates uniformisées (conformance)
Le bronze est hétérogène (incidents `date` + chaîne `time` ; télémétrie naïve ; maintenance
tz-aware). En silver, **toutes les dates/datetimes sont conformées en `datetime64` naïf
(interprété UTC)** : on retire les fuseaux (`maintenance_at`), on fusionne incident
`date`+`time` en un **`timestamp`** canonique, et on supprime les horodatages d'audit DB
(`created_at`/`updated_at`, non analytiques). Format unique aval : `YYYY-MM-DD HH:MM:SS`.

## Autres tables
- **model** : dimension type de machine (`model_id` surrogate, `name`).
- **sensor** : dimension capteur (`sensor_id` surrogate, `name`, `unit`).
- **machine** : dimension à PK surrogate (`machine_pk`), `machine_code` clé métier unique,
  `model` → `model_id` (FK) ; conformée (`heures_equivalentes_jour` ≈ 16, flag
  `capacite_incoherente`, CHECK criticité).
- **incident** : anonymisé (`operator_*` supprimés), signal/confiance + axes temporels ;
  `machine_id` → `machine_pk` (FK surrogate, nullable si code hors référentiel).
- **component** : lookup des composants (`component_id`, `name`).
- **maintenance** : `action_type`/`model`/`criticality` retirés ; `component` → `component_id`
  (FK lookup) ; `machine_code` → `machine_pk` (FK) ; CHECK `maintenance_type`.

## Reporté au gold
Usage des fenêtres de maintenance (exclusion de l'entraînement et/ou features
`time_since_last_maintenance`, reset post-maintenance) ; choix de traitement des outliers.
