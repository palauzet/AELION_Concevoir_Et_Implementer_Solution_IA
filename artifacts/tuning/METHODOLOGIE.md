# Méthodologie — optimisation d'hyperparamètres (B7)

## Rôle
Optimise les hyperparamètres du gagnant du POC B5 — **XGBoost, horizon 24h** — avec
**Optuna** (TPE), plutôt que les valeurs "sobres" non optimisées de `training.build_models`.

## Validation croisée — strictement train-only (écart volontaire vs B5)
`training.time_series_cv` calcule sa CV sur **tout** le dataset (train+val+test réunis) :
acceptable pour une estimation de robustesse informative en B5, mais fuiterait val/test
dans la sélection de modèle si réutilisé ici. Ce module construit donc ses folds
(`TimeSeriesSplit`, tri par `timestamp` seul) **uniquement sur le train** — val et test
restent totalement non vus par la recherche d'hyperparamètres.

## Pertinence statistique des folds — calculée une fois, jamais par essai
Le découpage des folds ne dépend que des données (timestamps), pas des hyperparamètres
testés : `compare_fold_partitions` (taux de positifs inner-train vs inner-val, écart
standardisé — SMD, Austin 2009 — maximal sur les features numériques) est calculé **une
seule fois avant `study.optimize(...)`**, jamais recalculé à chaque essai. Diagnostic (écrit
en CSV + figure), pas un filtre : un écart notable n'exclut aucun fold.

## Espace de recherche (9 hyperparamètres XGBoost)
`n_estimators` [100-800], `max_depth` [3-10], `learning_rate` [0.01-0.3, log], `subsample`
[0.5-1.0], `colsample_bytree` [0.5-1.0], `min_child_weight` [1-10], `gamma` [0.0-5.0],
`reg_alpha`/`reg_lambda` [1e-3-10, log]. Fixés : `scale_pos_weight` (train), `eval_metric`,
`n_jobs`, `random_state`.

## Objectif et budget
PR-AUC moyen sur les folds (maximiser), 50 essais par défaut, sampler TPE (graine fixe),
**sans pruning** (les folds chronologiques ont des tailles de train croissantes, pas une
vraie courbe d'apprentissage — un pruning médian écarterait à tort un essai difficile sur un
seul fold).

## Réentraînement final et comparaison
Le pipeline aux meilleurs hyperparamètres est réentraîné sur le **train complet**, évalué
une fois sur **test** (jamais vu pendant la recherche), et comparé au XGBoost "sobre" B5
(même train/test) pour un delta honnête.

## Suivi
Un run MLflow imbriqué par essai (params + PR-AUC CV, sans artefact modèle) sous un run
parent "étude", + un run distinct pour le modèle final (avec artefact). Étude Optuna
journalisée dans un backend SQLite séparé (`artifacts/tuning/optuna.db`). Les deux backends
(mlflow.db, optuna.db) sont hors versionnement.
