# Datasheet for Datasets — InduSense 4.0

> Suit la trame *Datasheets for Datasets* (Gebru et al., 2018), adaptée au dataset
> **gold** de maintenance prédictive. Rédigée pour Sprint 01 (cadrage, US 1.1) et
> mise à jour au fil du pipeline. Chiffres tirés des runs journalisés dans
> `artifacts/analyses/*/runs.md` et `artifacts/gold/runs.md` (derniers runs, 2026-06-23).

## 1. Motivation

**Pour quel usage le dataset a-t-il été créé ?**
Entraîner un modèle de **maintenance prédictive** : prédire, à partir de télémétrie
capteur (température, pression, tension, rotation, production) et du contexte
opérationnel (historique d'incidents, de maintenance), si une machine subira un
**incident** dans les 6h / 12h / 24h / 48h suivantes, et estimer sa **durée de vie
résiduelle** (RUL).

**Qui a créé le dataset et pour le compte de qui ?**
Données fournies dans le cadre de la formation *Concevoir et implémenter une
solution IA* (INDUSENSE 4.0 — cas d'usage pédagogique), assemblées et
transformées (bronze → silver → gold) par l'apprenant (architecte IA du projet).

**Financement.** Sans objet (jeu de données pédagogique, pas de collecte terrain).

## 2. Composition

**Sources brutes** (`data/raw/`, immuables) :

| Fichier | Contenu | Volumétrie (bronze) |
|---|---|---:|
| `telemetry.csv` | Relevés capteurs horaires (température, pression, tension, rotation, pièces produites) | 134 280 lignes × 11 colonnes, 15 machines |
| `releves_incidents.csv` | Incidents déclarés par les opérateurs (sévérité, 9 flags de typologie, commentaire libre) | 900 lignes brutes → **1 245** après dédoublonnage, 27 colonnes, 15 machines |
| `machine.sql` | Référentiel machines + historique de maintenance | 15 machines, **1 562** interventions |

**Unité d'analyse du dataset gold** : une ligne = une machine × une heure (hors
fenêtres de maintenance, exclues). Volumétrie finale : **130 613 lignes × 127
colonnes** (~118 features + identifiants/labels).

**Le dataset est-il exhaustif ou un échantillon ?** Exhaustif sur la période
couverte : toute la télémétrie et tous les incidents/maintenances fournis sont
intégrés (pas d'échantillonnage). La période couverte (~12 mois) est elle-même un
échantillon de l'activité réelle des machines.

**Étiquettes / cibles.** `label_incident_{6,12,24,48}h` (binaire, incident dans
`(t, t+H]`) + `rul_hours` / `rul_censored` (censure à droite si aucun incident
futur observé dans l'horizon disponible). Taux de positifs observés :

| Horizon | 6h | 12h | 24h | 48h |
|---|---:|---:|---:|---:|
| Taux positif | 4,3 % | 8,5 % | 16,32 % | 25,09 % |

→ **déséquilibre de classes marqué**, plus sévère aux horizons courts (US 2.1 :
Accuracy trompeuse, préférer Recall/F1/PR-AUC). RUL censuré sur 3 258 lignes.

**Données manquantes.**
- Télémétrie : 2 812 valeurs manquantes détectées (~2 %, pannes capteur en blocs
  de plusieurs heures) → imputation par interpolation intra-segment, secours
  médiane globale résiduelle (**point de fuite identifié, cf. §7**).
- Incidents : 0 valeur manquante après nettoyage (confiance moyenne de parsing
  0,998).

**Le dataset contient-il des données sensibles ?** Oui — voir §3.

## 3. Confidentialité, éthique, conformité (RGPD)

**PII identifiées** (`bronze.incident`, cf. `reports/dictionnaire_donnees.md`) :

| Colonne | Nature | Traitement retenu |
|---|---|---|
| `operator_name` | Nom de l'opérateur | À **supprimer/anonymiser** avant tout usage modèle ou exposition |
| `operator_badge` | Identifiant badge | Idem |
| `comment` | Texte libre | PII potentielle (noms, propos nominatifs) — à filtrer ou supprimer |

**Risque de ré-identification.** Combinaison `shift` × `machine_id` × `date` /
`severity` pourrait ré-identifier un opérateur même sans nom (faible volumétrie
par créneau). → minimiser : ne pas exposer ces colonnes dans le gold dataset
(elles ne le sont pas actuellement — confirmé par les tests anti-fuite).

**Base légale / minimisation (RGPD).** Les colonnes PII ne sont **pas** utilisées
en feature engineering ni persistées dans `gold.dataset` : seules les
caractéristiques agrégées de la machine et l'historique d'incidents (comptages,
sévérité, ancienneté) sont conservées. Aucune donnée à caractère personnel ne
transite dans la couche gold.

**Biais connus.**
- **Maintenance très majoritairement réactive** (1 472 / 1 562 = **94,2 %**) vs
  proactive (90 / 1 562 = 5,8 %) → le modèle apprendra sur un contexte où la
  maintenance préventive est rare ; une politique de maintenance différente en
  production changerait la distribution du signal (`days_since_last_maintenance`
  serait structurellement plus faible).
- **Répartition machines** : à vérifier (nombre d'incidents/lignes par
  `machine_id`) avant modélisation — pas de rééquilibrage appliqué au gold ;
  laissé au pipeline d'entraînement (`class_weight` / SMOTE, cf. `plan_de_lecture_ml.md` §Phase 4).
- **Saturation capteur** : 154 valeurs identifiées comme saturées (bornes
  instrument) sur la télémétrie — flag `is_saturated` conservé, pas neutralisé
  au gold (backlog ouvert).
- **Unités hétérogènes** : suspicion de °F sur certains capteurs de température,
  vérifiée non significative (ratio d'amplitude inter-machines 1,20, sous le
  seuil de 1,5) → pas de biais °C/°F confirmé.

## 4. Collecte

**Comment les données ont-elles été acquises ?** Fichiers fournis en kit de
démarrage pédagogique (CSV + SQL), simulant capteurs industriels + saisie
manuelle d'incidents + référentiel machine. Pas de collecte active par
l'apprenant.

**Période de collecte.** ~12 mois de télémétrie (l'énoncé initial mentionne
6 mois — écart noté dans `README.md`).

## 5. Prétraitement / nettoyage / labellisation

Pipeline bronze → silver → gold (cf. `gold_roadmap.md` pour le détail
pédagogique, `src/indusense/data/{telemetry,silver,gold}.py` pour le code) :

1. **Bronze** : chargement typé, contrat de schéma (`indusense-schema`).
2. **Silver** : dédoublonnage (1 346 doublons retirés), contrôle d'unités
   (`check_unit_consistency`), détection saturation (`detect_saturation`,
   154 flags), imputation 2 étapes (interpolation intra-segment + médiane de
   secours globale — **fuite résiduelle identifiée**, cf. §7), normalisation
   3FN (mesures en forme longue + clés surrogates).
3. **Gold** : agrégation horaire, features dynamiques (rolling
   mean/std/min/max/pente, fenêtres 6/12/24/48h, *segment-aware* et *trailing
   strict*), compteurs qualité, features historiques (incidents/maintenance),
   labels anti-fuite, split **chronologique** 70/15/15
   (train 91 389 / val 19 571 / test 19 653).

**Les données brutes sont-elles conservées ?** Oui, `data/raw/` est immuable et
versionné hors pipeline (lecture seule). `data/interim/`, `data/processed/` sont
régénérables et ignorés par git.

## 6. Usages

**Le dataset a-t-il déjà été utilisé ?** Non — première construction (Sprint 01).

**Usages prévus.** Entraînement/évaluation de modèles de classification binaire
(incident à H) et, en option, régression de survie / RUL (Sprint 02). Baseline
prévue : régression logistique, Random Forest, Gradient Boosting
(`plan_de_lecture_ml.md`).

**Usages à éviter.** Ré-identification d'opérateurs ; toute décision RH/disciplinaire
basée sur les données d'incidents (hors périmètre et hors base légale du
traitement) ; extrapolation à des machines/lignes non représentées dans le
référentiel (15 machines).

## 7. Limites connues et points ouverts

- **Fuite résiduelle d'imputation** : la médiane de secours (silver) est
  calculée sur l'ensemble du dataset (train+val+test), pas seulement sur le
  train. Impact mesuré **faible** (~2 752 valeurs imputées par ce mécanisme sur
  134 280, ~60 NaN résiduels au final). **Décision validée (2026-07-06,
  option A)** : le silver laissera le NaN résiduel ; un imputer médiane
  **fit-train** sera ajouté au pipeline modèle (même discipline que
  scaling/PCA). *Reste à implémenter.*
- **Traitement de la saturation au gold** : les valeurs saturées sont
  flaguées (compteurs glissants disponibles) mais pas neutralisées/plafonnées.
  Décision à prendre en Sprint 02 selon le modèle retenu.
- **Représentativité maintenance** : très faible part de maintenance proactive
  (5,8 %) — cf. §3 biais.

## 8. Distribution & maintenance

**Distribution.** Interne au projet — dataset gold persisté dans PostgreSQL
(`gold.dataset`, hors migrations Alembic, régénéré en bloc). Pas de diffusion
externe prévue à ce stade.

**Versioning.** Pas encore en place — **DVC prévu Sprint 3** (US 3.1). En
attendant, traçabilité assurée par les journaux de runs
(`artifacts/*/runs.md`) et les tests de non-régression (`tests/test_gold.py`).

**Qui maintient le dataset ?** L'apprenant / architecte IA du projet, sur la
durée du parcours (Sprints 1 à 4).
