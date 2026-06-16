# Dictionnaire des données — `AELION_SPRINT01`

> Extrait le 2026-06-16 depuis le serveur SQL Server local.
> Base : **`AELION_SPRINT01`** · Schéma : **`bronze`** (couche de données brutes / ingestion).
> Source : `sys.tables`, `sys.columns`, `sys.foreign_keys`, `sys.indexes`.

## Vue d'ensemble

| Table | Rôle | Lignes | PK | FK |
|-------|------|-------:|----|----|
| `bronze.machine` | Référentiel des machines (15 équipements) | 15 | `machine_code` | — |
| `bronze.telemetry` | Relevés capteurs (température, pression, tension, rotation, production) | 134 280 | `telemetry_id` (identity) | — *(machine_id non contraint)* |
| `bronze.incident` | Relevés manuels d'incidents + flags de typologie | 900 | `incident_pk` (identity) | — *(machine_id non contraint)* |
| `bronze.maintenance` | Interventions de maintenance | 115 | `maintenance_id` | `machine_code → machine` |

**Relation déclarée :** `FK_maintenance_machine` : `maintenance.machine_code → machine.machine_code`.

---

## `bronze.machine` — référentiel machines

| # | Colonne | Type | Null | Clé | Description |
|---|---------|------|------|-----|-------------|
| 1 | `machine_code` | nvarchar(16) | NOT NULL | **PK** | Code unique de la machine |
| 2 | `commissioning_date` | date | NOT NULL | | Date de mise en service |
| 3 | `max_daily_capacity` | int | NOT NULL | | Capacité journalière max (pièces) |
| 4 | `max_hourly_capacity_pieces` | int | NOT NULL | | Capacité horaire max (pièces) |
| 5 | `model` | nvarchar(32) | NOT NULL | | Modèle de la machine |
| 6 | `production_line` | nvarchar(16) | NOT NULL | idx | Ligne de production *(idx_machine_line)* |
| 7 | `location` | nvarchar(16) | NOT NULL | idx | Emplacement *(idx_machine_location)* |
| 8 | `criticality` | nvarchar(8) | NOT NULL | | Criticité |
| 9 | `is_active` | bit | NOT NULL | | Machine active (0/1) |
| 10 | `created_at` | datetimeoffset(7) | NOT NULL | | Horodatage de création |
| 11 | `updated_at` | datetimeoffset(7) | NOT NULL | | Horodatage de mise à jour |

## `bronze.telemetry` — relevés capteurs

| # | Colonne | Type | Null | Clé | Description |
|---|---------|------|------|-----|-------------|
| 1 | `telemetry_id` | int **IDENTITY** | NOT NULL | **PK** | Identifiant technique |
| 2 | `machine_id` | varchar(16) | NULL | | Machine concernée *(non contraint → `machine.machine_code`)* |
| 3 | `timestamp` | datetime2(7) | NULL | | Horodatage du relevé |
| 4 | `temperature_c` | float | NULL | | Température (°C) |
| 5 | `pressure_bar` | float | NULL | | Pression hydraulique (bar) |
| 6 | `voltage_mean_v` | float | NULL | | Tension moyenne (V) |
| 7 | `rotation_mean_rpm` | float | NULL | | Rotation moyenne (tr/min) |
| 8 | `pieces_produced` | int | NULL | | Pièces produites |

## `bronze.incident` — incidents (relevés manuels)

| # | Colonne | Type | Null | Clé | Description |
|---|---------|------|------|-----|-------------|
| 1 | `incident_pk` | int **IDENTITY** | NOT NULL | **PK** | Identifiant technique |
| 2 | `incident_id` | varchar(16) | NULL | | Référence métier de l'incident |
| 3 | `date` | date | NULL | | Date de l'incident |
| 4 | `time` | varchar(8) | NULL | | Heure (texte) |
| 5 | `operator_name` | varchar(128) | NULL | | ⚠️ **PII** — nom de l'opérateur |
| 6 | `machine_id` | varchar(16) | NULL | | Machine concernée *(non contraint)* |
| 7 | `severity` | int | NULL | | Gravité |
| 8 | `operator_badge` | varchar(16) | NULL | | ⚠️ **PII** — badge opérateur |
| 9 | `comment` | varchar(MAX) | NULL | | Commentaire libre *(⚠️ PII potentielle)* |
| 10 | `shift` | varchar(16) | NULL | | Équipe / poste |
| 11 | `type_surchauffe` | int | NULL | | Flag — surchauffe |
| 12 | `type_baisse_pression` | int | NULL | | Flag — baisse de pression |
| 13 | `type_vibration` | int | NULL | | Flag — vibration |
| 14 | `type_bruit_mecanique` | int | NULL | | Flag — bruit mécanique |
| 15 | `type_surconsommation` | int | NULL | | Flag — surconsommation |
| 16 | `type_blocage_mecanique` | int | NULL | | Flag — blocage mécanique |
| 17 | `type_alarme_capteur` | int | NULL | | Flag — alarme capteur |
| 18 | `type_arret_urgence` | int | NULL | | Flag — arrêt d'urgence |
| 19 | `type_defaut_qualite` | int | NULL | | Flag — défaut qualité |

## `bronze.maintenance` — interventions

| # | Colonne | Type | Null | Clé | Description |
|---|---------|------|------|-----|-------------|
| 1 | `maintenance_id` | int | NOT NULL | **PK** | Identifiant de l'intervention |
| 2 | `machine_code` | nvarchar(16) | NOT NULL | **FK** | Machine → `machine.machine_code` |
| 3 | `maintenance_at` | datetimeoffset(7) | NOT NULL | | Date/heure de l'intervention |
| 4 | `maintenance_type` | nvarchar(16) | NOT NULL | idx | Type de maintenance *(idx_maintenance_type)* |
| 5 | `action_type` | nvarchar(32) | NOT NULL | | Type d'action |
| 6 | `component` | nvarchar(64) | NOT NULL | | Composant concerné |
| 7 | `description` | nvarchar(MAX) | NOT NULL | | Description de l'intervention |
| 8 | `related_incident_id` | nvarchar(16) | NULL | | Incident lié (réf. métier `incident.incident_id`) |
| 9 | `duration_hours` | decimal(6,2) | NOT NULL | | Durée (heures) |
| 10 | `created_at` | datetimeoffset(7) | NOT NULL | | Horodatage de création |
| 11 | `updated_at` | datetimeoffset(7) | NOT NULL | | Horodatage de mise à jour |

> Index : `idx_maintenance_machine_time (machine_code, maintenance_at)`.

---

## Observations (pertinentes pour le Sprint 1)

- **PII à anonymiser (US 1.1 / C2)** : `incident.operator_name`, `incident.operator_badge`, et
  potentiellement `incident.comment` (texte libre). Risque de ré-identification par croisement
  `operator` × `shift` × `machine_id` × `date`.
- **Clés de jointure hétérogènes (US 1.2 / C3)** : la machine est désignée par `machine_id`
  (`varchar(16)`) dans `telemetry`/`incident` mais par `machine_code` (`nvarchar`) dans
  `machine`/`maintenance`. Aucune FK ne contraint `telemetry`/`incident` → vérifier l'intégrité
  référentielle (valeurs orphelines) avant amalgamation.
- **Nullabilité large** dans `telemetry` et `incident` : toutes les colonnes métier sont
  nullables → travail d'imputation / traitement des manquants à prévoir (US 1.3).
- **Typologie d'incident** : 9 colonnes `type_*` (flags entiers) ; à confirmer si binaire (0/1)
  ou comptage — candidat à un profilage de valeurs.
- **`incident.time`** stocké en `varchar(8)` (texte) alors que `date` est typé `date` → à
  uniformiser en horodatage lors du nettoyage.
