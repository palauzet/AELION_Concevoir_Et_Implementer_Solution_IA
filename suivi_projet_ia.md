# Suivi de projet IA

> Template de suivi calé sur le cycle de vie d'un projet IA. Chaque phase liste les tâches, les **technos / outils** à utiliser et les artefacts produits. À dupliquer pour chaque nouveau projet.

## Fiche projet

| Champ | Valeur |
|---|---|
| **Nom du projet** | InduSense 4.0 — maintenance prédictive (cas d'usage pédagogique) |
| **Responsable** | Patrick Alauzet (architecte IA) |
| **Client / commanditaire** | Formation *Concevoir et implémenter une solution IA* (cas d'usage fourni) |
| **Date de début** | 2026-06-15 |
| **Échéance cible** | Planning pédagogique = 21 jours (`PROJECT.md`) ; rythme réel étalé sur plusieurs semaines — Sprint 01 en cours de clôture au 2026-07-06 |
| **Statut global** | 🟡 En cours |

**Légende statut** : `[ ]` à faire · `[~]` en cours · `[x]` terminé

---

## 1 · Cadrage *(🔵)*

**Objectif** : définir le problème métier, les critères de succès et les contraintes.

- [x] Cadrer le besoin métier et les KPI de succès
- [x] Cartographier les parties prenantes et les impacts (directs/indirects)
- [x] Identifier les risques éthiques et le cadre réglementaire
- [x] Définir les critères d'acceptation

🔧 **Technos / outils** : **Datasheet for datasets** · cadre **RGPD** (confidentialité, minimisation, traçabilité)
📦 **Artefacts** : [note de cadrage](reports/cadrage.md) · [datasheet initiale](reports/datasheet.md) · liste des KPI (`reports/cadrage.md` §5)
**Statut** : [x] terminé (à réviser si le périmètre ou les données évoluent)

---

## 2 · Données — ingestion & gouvernance *(🟡)*

**Objectif** : collecter, modéliser et versionner les données de façon traçable.

- [x] Mettre en place l'environnement et le versioning de code
- [x] Lire / ingérer les sources et modéliser le schéma
- [ ] Versionner les données
- [~] Anonymiser / pseudonymiser si données sensibles

🔧 **Technos / outils** : **Git** · **venv/pyenv** · **pandas** · **CSV** · **SQLAlchemy (ORM/SQL)** · **DVC** · **ETL** · anonymisation / **confidentialité différentielle**
📦 **Artefacts** : dataset versionné (DVC) · schéma de données · pipeline ETL
**Statut** : [~] en cours — env/git/uv + schéma bronze/silver (3FN, Alembic) faits ; PII (`operator_name`/`badge`/`comment`) exclues du gold mais **non anonymisées à la source** ; DVC reporté Sprint 3

---

## 3 · Exploration & profiling *(🟡)*

**Objectif** : comprendre les données et détecter les risques tôt.

- [x] Profiling : valeurs manquantes, doublons, distributions, corrélations
- [x] Détecter le déséquilibre de classes et les biais d'échantillonnage
- [x] Mettre à jour la datasheet (qualité, limites, biais)

🔧 **Technos / outils** : **pandas** · profiling de données
📦 **Artefacts** : rapport d'exploration (`notebooks/01_analyse_exploratoire.ipynb`, `artifacts/analyses/`) · [datasheet mise à jour](reports/datasheet.md)
**Statut** : [x] terminé

---

## 4 · Préparation & nettoyage *(🟡)*

**Objectif** : produire des données propres et fiables.

- [x] Nettoyer (règles, normalisation) et journaliser les transformations
- [~] Traiter les outliers et les valeurs manquantes
- [x] Mettre en place des tests de qualité (seuils, alertes)

🔧 **Technos / outils** : **Z-score** · **IQR** · imputation **mean/median/KNN** · normalisation
📦 **Artefacts** : dataset nettoyé · tests de qualité
**Statut** : [~] en cours — dédoublonnage, unités, saturation (flag) et imputation interpolation faits ; fuite résiduelle corrigée côté silver (médiane de secours globale retirée, 2026-07-06) ; **imputeur médiane fit-train restant à ajouter au pipeline modèle** (Sprint 02)

---

## 5 · Feature engineering *(🟡)*

**Objectif** : construire les variables, sans fuite, et préparer le *gold dataset*.

- [x] Construire les features (lags, rolling windows, agrégations si séries temporelles)
- [ ] Sélection de features et réduction de dimension
- [ ] Gérer le déséquilibre (augmentation de données)
- [x] Vérifier l'absence de *leakage*

🔧 **Technos / outils** : **TimeSeriesSplit** · `train_test_split` · **PCA** · **SMOTE / ADASYN** · **scaling** · feature selection
📦 **Artefacts** : gold dataset (130 613 × 127, [`src/indusense/data/gold.py`](src/indusense/data/gold.py)) · pipeline de features
**Statut** : [~] en cours — features + labels + split chronologique + vérifications anti-fuite faits ; **PCA/feature selection et SMOTE/ADASYN reportés au pipeline modèle** (Sprint 02, US 1.5 restante, fit-train uniquement)

---

## 6 · Baseline & protocole d'évaluation *(🟡)*

**Objectif** : poser une référence et un protocole reproductible.

- [ ] Définir les splits train/test/val reproductibles (seeds)
- [ ] Entraîner une baseline simple
- [ ] Fixer le protocole d'évaluation offline et les métriques de référence

🔧 **Technos / outils** : **split train/test/val** · **seeds / artefacts** · **baseline metrics**
📦 **Artefacts** : baseline · protocole d'évaluation
**Statut** : _____

---

## 7 · Choix & entraînement du modèle *(🟡)*

**Objectif** : sélectionner et entraîner le modèle adapté, avec tracking.

- [ ] Choisir le modèle adapté (tabulaire / vision / anomalie)
- [ ] Entraîner avec tracking des expériences (paramètres, métriques, artefacts)
- [ ] Sauvegarder checkpoints et gérer l'early stopping

🔧 **Technos / outils** : **CNN** · **autoencodeur** · **YOLO** (vision) · **cross-validation** · **régularisation** · **MLflow** (tracking) · **checkpoints** · **early stopping**
📦 **Artefacts** : runs MLflow · modèle entraîné · checkpoints
**Statut** : _____

---

## 8 · Optimisation & fine-tuning *(🟡)*

**Objectif** : optimiser les hyperparamètres en maîtrisant l'empreinte.

- [ ] Définir l'espace de recherche et lancer l'optimisation
- [ ] Surveiller l'overfitting au tuning
- [ ] Mesurer l'empreinte énergie/carbone et arbitrer performance/impact

🔧 **Technos / outils** : **Optuna (TPE)** · pruning · **CodeCarbon** · **green AI**
📦 **Artefacts** : étude Optuna · rapport d'empreinte
**Statut** : _____

---

## 9 · Validation & explicabilité *(🟡)*

**Objectif** : valider la performance et rendre le modèle interprétable.

- [ ] Évaluer sur le jeu de test avec les bonnes métriques
- [ ] Traduire en coût métier (faux négatifs vs faux positifs)
- [ ] Produire l'explicabilité et un récit métier

🔧 **Technos / outils** : **accuracy** · **precision/recall** · **F1-score** · **matrice de confusion** · **SHAP**
📦 **Artefacts** : rapport de validation · analyses SHAP
**Statut** : _____

---

## 10 · Déploiement *(🟢)*

**Objectif** : mettre le modèle en production de façon traçable.

- [ ] Empaqueter le modèle et exposer une API
- [ ] Promouvoir via le model registry (staging → prod)
- [ ] Documenter la version déployée

🔧 **Technos / outils** : **MLflow Model Registry** (staging → prod) · API · versioning (hash, seeds)
📦 **Artefacts** : modèle promu · API · documentation de release
**Statut** : _____

---

## 11 · Suivi & ré-entraînement *(🟢 → boucle)*

**Objectif** : surveiller en production et boucler vers les données si dérive.

- [ ] Monitorer la performance et la dérive (data drift / concept drift)
- [ ] Définir les seuils d'alerte et la fréquence de contrôle
- [ ] Déclencher le ré-entraînement et revenir à la phase **Données** si nécessaire

🔧 **Technos / outils** : monitoring de dérive · **MLflow** (comparaison de versions) · boucle de ré-entraînement
📦 **Artefacts** : tableau de bord de suivi · journal des ré-entraînements
**Statut** : _____

---

## Récapitulatif techno par phase

| Phase | Technos clés |
|---|---|
| Cadrage | Datasheet, RGPD |
| Données | Git, pandas, SQLAlchemy, DVC, ETL |
| Exploration | pandas (profiling) |
| Préparation | Z-score, IQR, KNN |
| Feature engineering | TimeSeriesSplit, PCA, SMOTE/ADASYN, scaling |
| Baseline | split train/test/val, baseline metrics |
| Entraînement | CNN, autoencodeur, YOLO, MLflow, early stopping |
| Optimisation | **Optuna (TPE)**, CodeCarbon, green AI |
| Validation | accuracy, precision/recall, F1, SHAP |
| Déploiement | MLflow Model Registry, API |
| Suivi | monitoring de dérive, MLflow |
