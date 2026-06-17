# Méthodologie — analyse du référentiel

## Source : le dump SQL fourni, parsé directement
Le référentiel provient du dump `data/raw/machine.sql` (tables `machine` et
`maintenance`). L'analyse **parse directement les `INSERT` du dump** (`load_referentiel`)
— donnée lue *à la source*, **sans dépendance à une base**. Le pipeline est ainsi
autonome et reproductible, comme ceux de la télémétrie et des incidents (lecture du CSV
brut). Le même dump alimente par ailleurs le schéma `bronze` via `indusense-ingest
--migrate` (chargement verbatim) : les deux vues portent la même donnée.

## Anonymisation (RGPD) : non requise
Le référentiel ne comporte **aucune donnée personnelle** : caractéristiques d'équipement
(modèle, capacité, ligne, atelier, criticité) et journal de maintenance (type, composant,
durée, incident lié). `check_integrity` détecte tout champ DCP éventuel.

## Contrôle d'intégrité référentielle
Avant analyse : unicité des clés primaires (`machine_code`, `maintenance_id`), intégrité
référentielle (tout `machine_code` de maintenance existe dans `machine`), et cohérence
réactif ↔ incident (une maintenance `reactive` référence un `related_incident_id`).

## Périmètre d'analyse
- **Parc machines** : répartition par modèle / criticité / ligne / atelier, capacités par
  modèle, ancienneté (année de mise en service).
- **Maintenance** : volume proactif vs réactif, durée par type, composants les plus
  remplacés, interventions par machine, saisonnalité mensuelle.

Note : `action_type` est **redondant** avec `maintenance_type` (`changement_programme` ↔
proactive, `changement_suite_panne` ↔ reactive) ; une seule des deux colonnes suffit en
feature engineering.
