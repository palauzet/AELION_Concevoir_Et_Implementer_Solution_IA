# Méthodologie — explicabilité SHAP (B8)

## Rôle
Compare l'importance globale des features entre le XGBoost "sobre" (B5,
`training.build_models`) et le XGBoost tuné par Optuna (B7, meilleurs hyperparamètres du
dernier run `indusense-tune`) — répond à « le tuning a-t-il changé ce que le modèle
regarde ? », pas seulement « le score a-t-il changé ? » (déjà répondu par B7).

## Portée : explications globales uniquement
Résumé SHAP (beeswarm) + importance (|SHAP| moyen, part normalisée, rang) par feature, sur
l'ensemble du split **test**. Pas d'explication locale (une observation précise) ni de
dependence plot dans cette première itération — hors périmètre, pourrait être ajouté ensuite
sans changer cette base.

## Biais d'échelle entre modèles — pourquoi une part normalisée, pas les valeurs brutes
Les valeurs SHAP sont en marge (log-odds). B7 est un modèle plus conservateur que B5
(learning_rate, profondeur et nombre d'arbres plus faibles) : il produit donc mécaniquement
des |SHAP| bruts plus petits sur **toutes** les features, sans que ça signifie qu'il
"regarde moins" les features — un artefact d'échelle, pas un changement d'attention. La
comparaison utilise donc une **part normalisée** (`importance_share`, somme = 1 par modèle,
invariante à l'échelle) et le **rang** (`rank_b5`/`rank_b7`/`delta_rank`) comme signaux
robustes de comparaison inter-modèles. Les valeurs brutes restent dans les CSV
(`importance_b5_sobre.csv`/`importance_b7_tune.csv`) pour qui veut les consulter, mais ni la
figure de comparaison ni la lecture notebook ne s'appuient dessus pour comparer les modèles.

## Calcul sur le test complet, affichage sur un échantillon
Le |SHAP| moyen (et donc part/rang) est calculé sur les **19653 lignes du test complet**
(`TreeExplainer` est rapide et exact pour les modèles arbres, aucun échantillon de fond
nécessaire). Seul le nuage de points des figures de résumé (beeswarm) est sous-échantillonné
(`config.EXPLAIN_BEESWARM_SAMPLE`, graine 42, indices communs aux deux modèles) pour rester
lisible et léger en SVG — indiqué explicitement dans la légende de chaque figure.

## SHAP sur les features transformées, pas le pipeline complet
`shap.TreeExplainer` explique le modèle arbre (`XGBClassifier`) lui-même : les deux
pipelines sont donc décomposés (`preprocessor.transform(X_test)` puis `TreeExplainer` sur
`named_steps["model"]`), avec les noms de features post-imputation/one-hot
(`preprocessor.get_feature_names_out()`, préfixes `num__`/`cat__` retirés pour la lisibilité).
Valeurs en marge (log-odds), pas en probabilité — comportement par défaut de
`TreeExplainer` pour XGBoost binaire, exact, suffisant pour une lecture d'importance
relative. Les deux pipelines partagent le même `build_preprocessor` sur le même `X_train` :
`run()` vérifie explicitement que les noms de features transformées sont identiques entre
les deux modèles avant de construire les figures de comparaison.

## Modèles réentraînés, pas rechargés depuis MLflow
Les deux pipelines (B5 sobre, B7 tuné) sont reconstruits et réentraînés sur le **train**
directement dans ce module (mêmes fonctions que `training.py`/`tuning.py`), plutôt que
rechargés depuis leurs artefacts MLflow respectifs. Avec `random_state=42` fixe et les mêmes
données train, ce réentraînement est déterministe bit-à-bit (à versions de librairies
identiques) — ce n'est donc pas "expliquer un autre modèle", mais bien celui suivi par
`training`/`tuning`. Plus simple, et évite une dépendance à la sérialisation MLflow pour un
module d'analyse.

## Pas de tracking MLflow
Ce module ne produit ni n'enregistre de nouveau modèle : seulement des artefacts d'analyse
(figures + tableaux d'importance). Le suivi MLflow reste celui de `training.py`/`tuning.py`.

## Sélection du run de tuning et couplage à l'horizon
Par défaut, le run `indusense-tune` le plus récent (`artifacts/tuning/<run_id>/`, tri
chronologique par `run_id`) — surchargeable via `--tuning-run-id`. Les meilleurs
hyperparamètres B7 proviennent d'un tuning réalisé à l'horizon **24h** (seul horizon tuné à
ce jour) : les appliquer à un autre horizon (`--horizon`) reste possible (cohérence de
l'interface avec les autres modules) mais est exploratoire — `run_metadata.json` porte alors
un champ `note_horizon` signalant ce couplage.
