# Méthodologie — entraînement ML (TP B5)

## Rôle
Premier POC de classification binaire (« panne dans les prochaines Hh ? ») à partir du
**gold** InduSense (`gold.dataset` / `data/processed/gold_dataset.parquet`) — pas du dataset
générique décrit dans le brief du TP.

## Colonnes exclues des features (anti-fuite)
- Identifiants : `machine_pk`, `machine_code`, `timestamp`, `split`.
- RUL (`rul_hours`, `rul_censored`) : encode directement la cible (heures jusqu'au prochain
  incident) — cible alternative (survie), jamais une feature.
- Les 4 `label_incident_{6,12,24,48}h` : un seul horizon devient `y`, les 3 autres sont
  retirés de `X` (sinon fuite directe de la cible).

## Prétraitement (fit train uniquement, dans la Pipeline)
- Numérique : imputation **médiane** — referme le correctif anti-fuite du silver (la
  médiane de secours globale, fuyante, avait été retirée du silver au profit de cet
  imputeur, fitté sur le train seul). Standardisation additionnelle pour la régression
  logistique (modèle linéaire).
- Catégoriel (`model`, `criticality`, `production_line`, `location`, `shift`) : imputation
  la plus fréquente + one-hot (`handle_unknown="ignore"`).

## Déséquilibre de classes
`class_weight="balanced"` (régression logistique, Random Forest) ; `scale_pos_weight`
calculé sur le **train uniquement** (XGBoost).

## Split & validation
Split **chronologique** existant (`split` : train/val/test, 70/15/15) — pas de split
aléatoire. Modèles fittés sur train, évalués une fois sur test. Estimation robuste
complémentaire par `TimeSeriesSplit` (5 folds, tri par horodatage seul).

## Métriques
**PR-AUC** en primaire (déséquilibre de classes — ne jamais lire l'accuracy seule).
ROC-AUC en secondaire. Précision/rappel/F1 + matrice de confusion à seuil fixe
(0.5 — le choix motivé du seuil est renvoyé au module B7).

## Suivi
Tracking MLflow (backend SQLite local, `artifacts/ml/mlflow.db`, hors versionnement) +
run versionné (`artifacts/ml/<run_id>/`, figures + tableau comparatif).
