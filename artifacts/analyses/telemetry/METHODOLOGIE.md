# Méthodologie — analyse de la télémétrie

## Anonymisation (RGPD) : non requise
La télémétrie ne comporte **aucune donnée à caractère personnel** : uniquement des
mesures capteurs (`temperature_c`, `pressure_bar`, `voltage_mean_v`,
`rotation_mean_rpm`, `pieces_produced`), un horodatage et un identifiant
**d'équipement** (`machine_id`, non rattaché à une personne). La fonction
`check_no_personal_data` matérialise et prouve cette décision (détection de tout champ
DCP éventuel). Aucune transformation d'anonymisation n'est donc appliquée.

## Dédoublonnage
Clé métier d'unicité : **`(machine_id, timestamp)`** (un relevé par machine et par
heure). `deduplicate` retire les doublons de clé (conservation de la 1re occurrence)
et reporte le nombre supprimé — opération **idempotente**. Sur les données fournies :
**0 doublon**, grille horaire complète (15 machines × 8 952 h), 0 NaN.

## Détection des outliers
Méthode **IQR (Tukey)** par variable : sont marquées extrêmes les valeurs hors de
l'intervalle [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]. Choix robuste (basé sur les quantiles, peu
sensible aux valeurs extrêmes elles-mêmes) et lisible. Les outliers sont **comptés et
signalés**, non supprimés à ce stade : leur traitement (capping, imputation) relèvera de
la couche *silver*, une fois leur cause instruite.

## Corrélations
- **Pearson** : liens **linéaires** entre mesures (grandeurs continues).
- **Spearman** : liens **monotones** (sur les rangs), robustes aux outliers et aux
  non-linéarités. Les comparer distingue une vraie relation d'un artefact.

## Profils temporels
Axes dérivés du `timestamp` : `jour`, `heure`, `weekday`, `shift` (matin 06–14 ·
après-midi 14–22 · nuit sinon). Les profils horaire et par équipe sont **centrés-réduits
par variable** (z-score) pour comparer des grandeurs d'échelles différentes sur un même
graphe.
