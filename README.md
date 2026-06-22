# InduSense 4.0

Solution d'IA industrielle (maintenance prédictive) construite de façon
**cumulative** sur 4 sprints (compétences C1–C9). Un seul projet `uv`, qui
grandit sprint après sprint ; les jalons sont marqués par des tags/branches git.

Énoncé complet : voir [`PROJECT.md`](PROJECT.md).

**Sprint courant : 01 — Enquête et nettoyage de données.**
À partir de données brutes, hétérogènes et « sales », transformer le chaos en
une base de connaissance fiable et un **Gold Dataset** prêt pour l'entraînement.

## Objectifs du sprint (User Stories)

| US  | Intitulé                                   | Compétences | Module                       |
|-----|--------------------------------------------|-------------|------------------------------|
| 1.1 | Enquête + anonymisation                    | C1, C2      | `data/loaders`, `data/anonymize` |
| 1.2 | Amalgamation & BDD relationnelle (ORM)     | C1, C3      | `data/db`                    |
| 1.3 | Outliers + imputation                      | C3, C4      | `data/cleaning`              |
| 1.4 | Data leakage & séries temporelles          | C3, C4      | `data/leakage`               |
| 1.5 | Réduction dimensionnelle → Gold Dataset    | C3          | `data/features`              |

## Structure

```
. (racine du dépôt)
├── pyproject.toml          # projet uv, Python 3.13
├── .python-version         # 3.13
├── PROJECT.md              # énoncé complet (4 sprints, C1–C9)
├── src/indusense/          # code source (package)
│   ├── config.py           # chemins & constantes
│   └── data/               # un module par User Story
├── notebooks/              # 00_acquisition, 01_analyse_exploratoire (silver/gold à venir)
├── data/                   # convention Cookiecutter Data Science
│   ├── raw/                #   ARTEFACTS BRUTS (immuables — ne pas modifier)
│   │   ├── telemetry.csv
│   │   ├── releves_incidents.csv
│   │   └── machine.sql
│   ├── interim/            #   données intermédiaires (ignoré par git)
│   └── processed/          #   Gold Dataset (ignoré par git)
├── reports/figures/        # boxplots & visualisations
└── tests/                  # pytest
```

> **3 artefacts bruts** dans `data/raw/` (kit de départ — **immuables**, lecture
> seule). Correspondance avec les artefacts de l'énoncé
> (cf. [`PROJECT.md`](PROJECT.md) § Artefacts) :
>
> | Fichier | Artefact(s) PROJECT.md couvert(s) |
> |---|---|
> | `telemetry.csv` | capteurs **température** + **pression hydraulique** + **production** (+ tension, rotation). Granularité horaire. |
> | `releves_incidents.csv` | relevés manuels d'**incidents**. |
> | `machine.sql` | référentiel **machines** + **maintenance** (hors énoncé, fourni en complément). |
>
> `telemetry.csv` **agrège** trois artefacts distincts de l'énoncé — il n'y a pas
> de correspondance 1 fichier ↔ 1 artefact. Les **images** de pièces (détection de
> défauts) relèvent du **Sprint 2**. Note : l'énoncé évoque « 6 derniers mois » de
> relevés capteurs alors que les données fournies couvrent ~12 mois.
>
> Chemins exposés dans [`src/indusense/config.py`](src/indusense/config.py).

## Mise en route

Environnement géré par **uv** (Python **3.13.x**, cf. `.python-version`).
Une fois le dépôt cloné :

```powershell
# 1. Créer / synchroniser l'environnement (uv installe Python 3.13 au besoin)
uv sync

# 2. Lancer les tests / le lint / JupyterLab
uv run pytest
uv run ruff check .
uv run jupyter lab
```

> Première installation de `uv` (si absent du poste) :
> `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Stack (Sprint 01)

pandas · numpy · scipy · scikit-learn · SQLAlchemy · matplotlib · seaborn ·
pyarrow — outillage dev : JupyterLab, pytest, ruff, mypy.
