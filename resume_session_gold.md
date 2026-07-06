# Résumé de session — InduSense 4.0 (Sprint 01 : enquête & nettoyage, couche GOLD)

> Fichier de contexte à réinjecter dans une autre conversation. État arrêté au 2026-07-06.

## 1. Cadre projet

- **Projet** : InduSense 4.0 — maintenance prédictive ML.
- **Architecture médaillon** : raw → bronze → silver → gold, sur **PostgreSQL** (Docker `docker/pgdocker`).
- **Sprint 01** : « enquête et nettoyage de données ». Phase courante = **feature engineering → dataset gold**.
- **Sécurité / conventions** :
  - mot de passe PG hors code via `.env` + `python-dotenv` (`.env` gitignored) ;
  - `sql/` (scripts de contrôle) et `dependances.md` **non commités** (choix assumé) ;
  - table `gold.dataset` **hors Alembic** (régénérée en bloc via `to_sql(replace)`).

## 2. Stack technique (cf. `dependances.md`, racine, non commité)

pandas≥2.2, numpy≥2.1, scipy≥1.14, scikit-learn≥1.5 (déclaré, **pas encore importé** — réservé
modélisation), SQLAlchemy≥2.0 (ORM), Alembic≥1.13, psycopg[binary]≥3.2, matplotlib≥3.9 (Agg),
seaborn≥0.13, pyarrow≥17.0, python-dotenv≥1.0. Dev : jupyterlab, ipykernel, pytest≥8.3,
ruff≥0.6 (ligne 100), mypy≥1.11. Build : hatchling, uv, nbconvert.
**imbalanced-learn (SMOTE) NON installé.**

## 3. Concepts clés maîtrisés

- **Data leakage** : toute statistique *fittée* (médiane/moyenne d'imputation, scaling, PCA)
  doit l'être **sur train uniquement**, puis appliquée à val/test. Splits **chronologiques**
  (non aléatoires) pour séries temporelles. Fenêtres glissantes **trailing strict**. Features
  **segment-aware** (ne franchissent pas une fenêtre de maintenance).
- **Saturation capteur** (valeurs censurées aux bornes de l'instrument) ≠ outliers IQR ;
  détection par pile-up sur bornes rondes.
- **Piège timestamp** : PostgreSQL `timestamp` = microseconde ; pandas `read_sql` →
  `datetime64[us]`. A causé un bug de pente ×1000 (corrigé).

## 4. État du code (tout commité sauf `sql/` et `dependances.md`)

### `src/indusense/data/gold.py` — pipeline complet
~118 features, **127 colonnes × 130 613 lignes**. Fonctions : `load_silver`, `build_base`,
`_assign_segment`, `add_dynamic_features` (rolling mean/std/min/max/slope par capteur×fenêtre,
segment-aware, trailing), `add_quality_counts` (sommes glissantes n_imputed/outlier/saturated),
`add_production_features`, `add_history_features`, `add_static_and_cyclical`, `add_labels`,
`assign_split` (chronologique 70/15/15), `build_gold`, `make_figures` (3 figs), `write_gold`
(→ `gold.dataset`), `run`, `main`.
- **Fenêtres** : 6/12/24/48 h. **Horizons labels** : 6/12/24/48 h.
- **Correctif pente** (données en µs, pas ns) :
  `__t_h = groupby(GROUP)["timestamp"].transform(lambda s: (s - s.min())/np.timedelta64(1,"h"))`.
  Vérifié 2413.7 → 2.414 °C/h.
- **`time_since_last_maintenance_h`** ancré sur la **reprise** (fin = `maintenance_at + duration_hours`) :
  `prev = searchsorted(starts, t, "right")-1 ; tsl = max(0, (t-ends[prev])/np.timedelta64(1,"h"))`.
- **Labels** : `label_incident_{6,12,24,48}h` (incident dans `(t, t+H]`), `rul_hours` + `rul_censored`
  (=1 si aucun incident futur → `rul_hours` NULL). Lignes `during_maintenance` exclues.
- **Métriques observées** : taux positifs {6h:0.043, 12h:0.085, 24h:0.163, 48h:0.251} ;
  RUL censurés 3258 ; split train 91389 / val 19571 / test 19653.
- Colonnes décrivant l'incident/la maintenance réactive **exclues** (= la cible).

### `src/indusense/data/silver.py`
- `impute_telemetry(df)` : imputation 2 étapes — (1) interpolation linéaire par
  (machine, segment) [locale, comble la majorité] ; (2) **médiane de secours globale** pour
  segments entièrement NaN → **c'est la fuite résiduelle identifiée** (médiane calculée sur tout
  le dataset avant split). Impact **faible** (fallback rare : ~2752 imputés, ~60 NaN résiduels).
- `flag_saturation(df)` (via `tel.detect_saturation`, pose `<capteur>_is_saturated`), porté dans
  `build_measurement` ; métrique `n_satures_flagges`. `_maintenance_windows` utilise
  `start + pd.to_timedelta(duration_hours, unit="h")`.

### `src/indusense/data/telemetry.py` — QC bronze
- `check_unit_consistency(df, cols, max_ratio=1.5)` : ratio d'amplitude inter-machines, flag
  `unite_suspecte` si >1.5 (attrape °C/°F). Verdict : cohérent (ratio temp 1.20).
- `detect_saturation(df, cols, width=1.0)` : confirme si comptage sur borne ≥ `SATURATION_MIN_HITS`(3)
  ET > `SATURATION_RATIO`(2.0)×densité bin adjacent. Constantes : `UNIT_AMPLITUDE_MAX`=1.5,
  `SATURATION_MIN_HITS`=3, `SATURATION_RATIO`=2.0.

### Autres fichiers
- `models/silver_tables.py` (renommé depuis silver.py ; `is_saturated: Mapped[bool]`),
  `models/bronze_tables.py` (renommé). Renommage fait pour éviter la collision de noms de modules.
- `config.py` : `GOLD_DIR`, `ROLLING_WINDOWS=(6,12,24,48)`, `LABEL_HORIZONS=(6,12,24,48)`, `SENSOR_UNITS`.
- Alembic : `0139f5eed4ed` (3FN) + `f1b64dacb88b` (is_saturated). `gold.dataset` hors Alembic.
- `tests/test_gold.py` (5 tests) : labels binaire+RUL ; dynamic features segment-aware & trailing
  (verrouille pente ×1 pas ×1000, cold-start std NaN/slope 0) ; time_since depuis reprise (2h fin,
  pas 4h début) ; split chronologique disjoint & ordonné ; smoke `build_gold` (80−3=77 lignes, pas
  de colonnes de fuite, `not any(c.endswith("_was_imputed"))`).
- `notebooks/` : `01_analyse_exploratoire.ipynb` (référentiel rafraîchi ~94% réactif),
  `02_silver.ipynb` (ERD + 3FN), `03_gold.ipynb` (build via `gold.run()`, 3 SVG, checks anti-fuite,
  top-10 features corrélées à label_incident_24h : rul_hours −0.380, n_incidents_7d 0.184…).
- `reports/silver_erd.png` + `.graphml`. `.gitignore` : ignore data/interim, data/processed,
  *.parquet, artifacts/*/, .env*, notebooks/*.html.

## 5. Vérifications déjà passées

FK intégrité ; imbrication labels (6h⊆12h⊆24h⊆48h : 0 violation) ; RUL null⇔censuré ; split
chronologique disjoint ; aucune colonne décrivant l'incident ; lectures capteur pendant
maintenance exclues (0 ligne gold sur fenêtre de maintenance) ; les **événements** de maintenance
(horodatages) restent utilisés pour reset segment + features time_since.

## 6. Point traité le 2026-07-06 — imputation anti-fuite (option A validée et implémentée)

Go de l'utilisateur reçu pour l'**option A**. Implémenté dans
`src/indusense/data/silver.py::impute_telemetry` : suppression de la médiane de secours globale
(fittée sur tout le dataset → fuite) ; les segments entièrement NaN restent **NaN résiduels**.
Docstring mise à jour pour expliquer le renvoi vers un imputeur **fit-train** côté pipeline
modèle (Sprint 02, pas encore implémenté — pas de pipeline modèle existant à ce stade).
`uv run pytest` : 43/43 passés après le changement (aucun test ne verrouillait l'ancien
comportement de secours). **Non revérifié en conditions réelles** (rebuild silver/gold contre
PostgreSQL) : Docker Desktop non démarré pendant la session — à relancer (`docker compose -f
docker/pgdocker/docker-compose.yml up -d` puis `uv run indusense-silver` / `-gold`) pour
confirmer le nombre de NaN résiduels (attendu proche des ~60 déjà observés + les anciens
fallbacks médiane, soit ~2 812 NaN télémétrie avant imputation).

## 7. Docs / cadrage produits le 2026-07-06

- [`reports/datasheet.md`](reports/datasheet.md) : Datasheet for Datasets (Gebru et al.),
  chiffrée à partir des runs (`artifacts/*/runs.md`) — composition, PII/RGPD, biais (maintenance
  94,2 % réactive, saturation, unités), limites (fuite résiduelle, saturation non traitée),
  usages à éviter.
- [`reports/cadrage.md`](reports/cadrage.md) : note de cadrage — besoin métier, parties
  prenantes, risques éthiques/RGPD, critères d'acceptation Sprint 01, **KPI métier** (taux de
  pannes anticipées, délai d'anticipation, fausses alertes, ratio préventif/réactif, coût évité)
  et **KPI modèle** (PR-AUC, Recall, Precision, F1, ROC/AUC, calibration — priorité Recall sur
  Accuracy, cf. US 2.1).
- `suivi_projet_ia.md` : fiche projet complétée (nom, responsable, dates, statut) ; phases 1
  (Cadrage) à 5 (Feature engineering) cochées/nuancées avec liens vers les artefacts réels ;
  cases restantes explicitement notées **[ ]** ou **[~]** (DVC, anonymisation à la source, PCA/
  SMOTE, imputeur fit-train modèle).

## 8. Tâches restantes (backlog)

- **Vérifier en réel** l'effet du retrait de la médiane de secours (Docker + rebuild silver/gold)
  et mettre à jour les chiffres NaN résiduels dans `reports/datasheet.md` §2/§7 si besoin.
- Imputeur médiane **fit-train** dans le futur pipeline d'entraînement (Sprint 02).
- Traitement saturation au gold (neutraliser les niveaux censurés ? compteurs déjà présents).
- Décision de commit pour `sql/` et `dependances.md`.
- Anonymisation/pseudonymisation à la source des PII incidents (`operator_name`/`badge`/
  `comment`) — actuellement seulement exclues du gold, pas traitées en bronze/silver.
- US 1.5 / PCA au pipeline modèle (fit-train).
- Versioning données **DVC** (prévu Sprint 3).
- Baseline modélisation (suivi §6/§7) : TimeSeriesSplit, scaling/PCA/SMOTE fit-train, métriques.
- Décider si `reports/datasheet.md` et `reports/cadrage.md` sont commités (a priori oui, docs de
  projet — à confirmer avec le même choix que `sql/`/`dependances.md`).
