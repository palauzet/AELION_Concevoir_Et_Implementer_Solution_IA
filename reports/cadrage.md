# Note de cadrage — InduSense 4.0 (Sprint 01)

> Artefact de la phase **Cadrage** (`suivi_projet_ia.md`, §1). Complète la
> [datasheet](datasheet.md) (données) avec le cadrage métier, les parties
> prenantes, les risques et les critères d'acceptation.

## 1. Besoin métier

**Problème.** Les pannes machines sur les lignes de production sont aujourd'hui
détectées **après coup** (incident déclaré par l'opérateur) ; la maintenance est
**très majoritairement réactive** (94,2 % des interventions, cf. datasheet §3).
Cela génère des arrêts de production non planifiés et des coûts de réparation
plus élevés qu'une intervention préventive.

**Objectif de la solution IA.** Prédire, à partir de la télémétrie capteur et du
contexte opérationnel, la probabilité qu'une machine subisse un incident dans un
horizon donné (6h / 12h / 24h / 48h), afin de **basculer une partie de la
maintenance réactive vers du préventif planifié**.

**Cas d'usage retenu (parmi ceux du sprint 2 de l'énoncé) :** maintenance
prédictive tabulaire (capteurs). Le volet vision (défauts sur pièces produites,
MVTec AD) est un cas d'usage complémentaire traité en Sprint 2, hors périmètre
de ce cadrage.

## 2. Parties prenantes et impacts

| Partie prenante | Impact direct/indirect | Attente |
|---|---|---|
| Équipe maintenance | Direct | Alerte fiable et actionnable (pas de sur-alerte) |
| Opérateurs de production | Direct | Pas d'arrêt de production non anticipé ; pas d'usage disciplinaire de leurs relevés d'incidents |
| Responsable production / planification | Indirect | Fenêtres de maintenance planifiables (horizon 24-48h) |
| Direction / finance | Indirect | ROI maintenance préventive vs coût des pannes |
| Apprenant / équipe IA (nous) | Direct | Modèle reproductible, traçable, non biaisé |

## 3. Risques éthiques, sociétaux et réglementaires

Détaillés dans la [datasheet](datasheet.md) §3 ; synthèse actionnable ici :

| Risque | Traitement |
|---|---|
| Ré-identification d'opérateurs (`operator_name`, `operator_badge`, `comment`) | Colonnes exclues de la couche gold — confirmé par tests anti-fuite (`tests/test_gold.py`) |
| Usage disciplinaire des données d'incidents | **Hors périmètre** — à documenter explicitement dans la release notes / gouvernance d'usage |
| Biais maintenance réactive (94,2 %) → sous-représentation du préventif | Documenté ; à surveiller lors de l'évaluation (le modèle ne doit pas halluciner un pattern « préventif » qu'il n'a quasi jamais vu) |
| Sur-confiance dans un horizon court (6h, 4,3 % positifs) | Le protocole d'évaluation (Sprint 2) doit reporter PR-AUC/Recall par horizon, pas seulement l'Accuracy |

## 4. Critères d'acceptation (Sprint 01 — Gold Dataset)

- [x] Schéma relationnel 3FN silver, migrations Alembic réversibles.
- [x] Détection et flag de la saturation capteur + cohérence d'unités.
- [x] Gold dataset construit : 130 613 lignes × 127 colonnes, anti-fuite vérifié
      (aucune colonne décrivant l'incident/la maintenance réactive dans les
      features, lectures pendant maintenance exclues, split chronologique
      disjoint et ordonné).
- [x] Labels multi-horizons (6/12/24/48h) + RUL censuré, imbrication vérifiée.
- [ ] Imputation anti-fuite train-only (option A validée le 2026-07-06,
      implémentation restant à faire — cf. backlog).
- [ ] Décision de traitement de la saturation au gold.
- [ ] Datasheet et note de cadrage tenues à jour à chaque changement de
      version des données *(ce document)*.

## 5. KPI

### 5.1 KPI métier (pilotage de la solution en production)

| KPI | Définition | Cible indicative |
|---|---|---|
| Taux de pannes anticipées | % d'incidents réels précédés d'une alerte dans l'horizon | à définir avec la maintenance (Sprint 2) |
| Délai moyen d'anticipation | Temps entre l'alerte et l'incident réel | maximiser sans dégrader le taux de faux positifs |
| Taux de fausses alertes | % d'alertes non suivies d'incident dans l'horizon | sous un seuil acceptable métier (coût mobilisation équipe) |
| Ratio maintenance préventive / réactive | Suivi de bascule réactif → préventif après déploiement | augmenter par rapport à la baseline 5,8 % |
| Coût évité estimé | Coût panne non planifiée − coût intervention préventive, pondéré par le taux de détection | ROI (Sprint 4, US 4.1) |

### 5.2 KPI modèle (protocole d'évaluation, Sprint 2)

| KPI | Pourquoi (vs Accuracy) |
|---|---|
| **PR-AUC** par horizon | Référence sur classes déséquilibrées (4,3 %–25 % de positifs) — l'Accuracy est trompeuse |
| **Recall** (rappel) | Coût métier d'un faux négatif (panne manquée) ≫ coût d'un faux positif (US 2.1) |
| Precision | Éviter la sur-alerte qui déprécie la confiance des équipes |
| F1-score | Compromis synthétique precision/recall |
| Courbe ROC/AUC | Comparaison de modèles (Sprint 2, US 2.5) |
| Calibration (Brier / reliability) | Nécessaire si les probabilités servent à prioriser les interventions |

## 6. Hors périmètre (Sprint 01)

- Modélisation (Sprint 2), industrialisation (Sprint 3), MLOps/monitoring
  (Sprint 3-4) : cf. `PROJECT.md`.
- Volet vision (détection de défauts sur pièces produites).
- Versioning des données (DVC) — prévu Sprint 3.
