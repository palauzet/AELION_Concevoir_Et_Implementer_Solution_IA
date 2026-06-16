# Méthodologie — ingestion des incidents

## Anonymisation des opérateurs (RGPD)

### Finalité du traitement (étalon de la minimisation)
Analyser les **relevés d'incidents machines** (distribution temporelle, typologie des
pannes, corrélations entre signaux, fiabilité du signalement) pour préparer une
solution de **maintenance prédictive**. Finalité **centrée machine, pas opérateur** :
c'est elle qui détermine ce qui est nécessaire.

### Principe de minimisation (RGPD art. 5.1.c) — test de nécessité
Données « adéquates, pertinentes et limitées à ce qui est nécessaire au regard des
finalités ». Appliqué champ par champ :

| Champ | Nécessaire à la finalité (machine) ? | Décision |
|---|---|---|
| `operator_name` | Non — l'identité n'explique pas la panne | **Supprimé** |
| `operator_badge` | Non — aucune analyse *par opérateur* retenue | **Supprimé** |
| `machine_id`, `date`, `shift`, signaux | Oui — cœur de l'analyse | Conservés |
| `comment` | Oui (type de panne) **et** non-DCP | Conservé |

### Proportionnalité : suppression plutôt que pseudonymisation
Un pseudonyme (hash salé) ne se justifierait que si une **finalité d'analyse par
opérateur** était retenue (ex. détecter un biais de signalement) — ce **n'est pas le
cas**. Conserver un pseudo-identifiant serait donc une donnée **excessive** et
porterait un **risque résiduel** de ré-identification. La **suppression** est l'option
la plus protectrice *et* suffisante (*privacy by design*, art. 25). Aucune table de
correspondance n'étant conservée, l'anonymisation est **irréversible** et le dataset
sort du périmètre RGPD.

### Risque de ré-identification indirecte
Les quasi-identifiants restants (`shift` × `machine_id` × `date`) sont **nécessaires**
et donc conservés. Le risque résiduel est **réduit** (plus d'identifiant direct) et
**maîtrisé** (usage interne). Une diffusion externe imposerait un **k-anonymat** sur
ces colonnes.

### Traçabilité (accountability, art. 5.2)
Décision prouvable via : cette note, la docstring de `anonymize_operators` (règle +
justification) et l'historique git (décision datée et signée).

## Indice de confiance du signalement (par incident)
Score dans [0, 1] mesurant la **qualité du relevé** :

| Composante | Règle |
|---|---|
| `coherence` | 1.0 si 1 signal actif · 0.5 si plusieurs · 0.0 si aucun |
| `comment_present` | 1.0 si commentaire présent, sinon 0.0 |
| `machine_valide` | 1.0 si `machine_id` au format référentiel (`MACH-\d+`) |
| `severity_valide` | 1.0 si `severity` ∈ [1, 5] |

`confidence = 0.4·coherence + 0.2·comment_present + 0.2·machine_valide + 0.2·severity_valide`
