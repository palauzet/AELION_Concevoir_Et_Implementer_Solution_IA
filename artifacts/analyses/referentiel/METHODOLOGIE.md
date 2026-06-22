# Méthodologie — analyse du référentiel

## Source : la couche bronze (médaillon)
Le référentiel provient du dump `data/raw/machine.sql` (tables `machine` et
`maintenance`), chargé dans `bronze` par `indusense-ingest --migrate`. L'analyse lit la
**couche bronze** (`load_referentiel` → `SELECT … FROM bronze.*`), pas le dump : en
architecture médaillon, le brut n'a qu'un seul lecteur (l'ingestion) et tout l'aval
s'appuie sur la couche gouvernée — **source unique de vérité**, typage/encodage faits une
seule fois, lignage centralisé.

## Anonymisation (RGPD) : non requise
Le référentiel ne comporte **aucune donnée personnelle** : caractéristiques d'équipement
(modèle, capacité, ligne, atelier, criticité) et journal de maintenance (type, composant,
durée, incident lié). `check_integrity` détecte tout champ DCP éventuel.

## Contrôle d'intégrité référentielle
Avant analyse : unicité des clés primaires (`machine_code`, `maintenance_id`), intégrité
référentielle (tout `machine_code` de maintenance existe dans `machine`), et cohérence
réactif ↔ incident (une maintenance `reactive` référence un `related_incident_id`).

### Cohérence des capacités
`max_daily_capacity ≈ max_hourly_capacity_pieces × k`, où *k* est le nombre d'heures de
production équivalentes par jour. Sur le parc, *k* ≈ **16** (≈ 16 h/j) et très homogène —
**l'hypothèse intuitive ×24 est donc fausse** (les machines ne produisent pas 24 h/24). Le
contrôle valide l'homogénéité du facteur (`capacite_ratio_median`) et **signale** les
machines déviantes (`capacite_machines_incoherentes`) — saisie suspecte. C'est un
**diagnostic bronze** (lecture seule, intra-source) ; dériver une capacité canonique ou
corriger une valeur relèverait du **silver**.

## Périmètre d'analyse
- **Parc machines** : répartition par modèle / criticité / ligne / atelier, capacités par
  modèle, ancienneté (année de mise en service).
- **Maintenance** : volume proactif vs réactif, durée par type, composants les plus
  remplacés, interventions par machine, saisonnalité mensuelle.

Note : `action_type` est **redondant** avec `maintenance_type` (`changement_programme` ↔
proactive, `changement_suite_panne` ↔ reactive) ; une seule des deux colonnes suffit en
feature engineering.
