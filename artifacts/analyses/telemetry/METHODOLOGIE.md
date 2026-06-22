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
heure). `deduplicate` retire les doublons de clé (conservation de la 1re occurrence) et
reporte le nombre supprimé — opération **idempotente**. Le compte effectif (doublons
retirés, NaN restants) figure dans le journal des runs et les métadonnées du run, pour
rester juste quelle que soit la version des données.

## Détection des outliers
Méthode **IQR (Tukey)** par variable : sont marquées extrêmes les valeurs hors de
l'intervalle [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]. Choix robuste (basé sur les quantiles, peu
sensible aux valeurs extrêmes elles-mêmes) et lisible. Les outliers sont **comptés et
signalés**, non supprimés à ce stade : leur traitement (capping, imputation) relèvera de
la couche *silver*, une fois leur cause instruite.

## Contrôle des unités (cohérence inter-machine)
`check_unit_consistency` vérifie que toutes les machines mesurent un capteur dans la **même
unité** : si c'est le cas, le rapport **amplitude des moyennes inter-machine** (`max/min`)
reste proche de 1. Au-delà de **1,5×**, on **alerte** sur un possible mélange d'unités (ex.
°C vs °F donnerait ~×1,8 + 32). Heuristique de **dépistage**, pas de preuve formelle.

## Saturation capteur (écrêtage sur bornes de plage)
`detect_saturation` repère les valeurs **censurées** sur les limites de l'instrument : un
**empilement** anormal exactement sur le min/max global (≥ 3 relevés et > 2× la densité juste
à l'intérieur). Donnée continue → tomber pile sur une borne ronde n'est pas un arrondi. C'est
**distinct d'un outlier** : valeur *valide mais tronquée* (ex. température réelle ≥ borne), à
**ne pas imputer**. Le silver matérialise ce constat par un flag `*_is_saturated` (par mesure),
indépendant de `*_is_outlier`.

## Corrélations
- **Pearson** : liens **linéaires** entre mesures (grandeurs continues).
- **Spearman** : liens **monotones** (sur les rangs), robustes aux outliers et aux
  non-linéarités. Les comparer distingue une vraie relation d'un artefact.

## Profils temporels
Axes dérivés du `timestamp` : `jour`, `heure`, `weekday`, `shift` (matin 06–14 ·
après-midi 14–22 · nuit sinon). Les profils horaire et par équipe sont **centrés-réduits
par variable** (z-score) pour comparer des grandeurs d'échelles différentes sur un même
graphe.
