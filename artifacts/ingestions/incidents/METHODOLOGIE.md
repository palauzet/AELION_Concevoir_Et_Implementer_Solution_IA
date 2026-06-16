# Méthodologie — ingestion des incidents

## Anonymisation des opérateurs (RGPD)
`operator_name` et `operator_badge` sont des **identifiants directs** (DCP). Ils ne
sont pas nécessaires aux analyses et sont donc **supprimés** (minimisation). Comme
aucune table de correspondance n'est conservée, l'anonymisation est **irréversible**
et le dataset sort du périmètre RGPD. `comment` est **conservé** (saisie guidée
décrivant le type de panne, sans donnée personnelle).

## Indice de confiance du signalement (par incident)
Score dans [0, 1] mesurant la **qualité du relevé** :

| Composante | Règle |
|---|---|
| `coherence` | 1.0 si 1 signal actif · 0.5 si plusieurs · 0.0 si aucun |
| `comment_present` | 1.0 si commentaire présent, sinon 0.0 |
| `machine_valide` | 1.0 si `machine_id` au format référentiel (`MACH-\d+`) |
| `severity_valide` | 1.0 si `severity` ∈ [1, 5] |

`confidence = 0.4·coherence + 0.2·comment_present + 0.2·machine_valide + 0.2·severity_valide`
