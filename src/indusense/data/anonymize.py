"""Anonymisation / minimisation des données (US 1.1, C2).

Objectifs : garantir l'anonymat des opérateurs, éviter la ré-identification
par combinaison de features, limiter les biais (machines sur-représentées,
pannes minimisées).
"""

from __future__ import annotations

import pandas as pd

# Identifiants directs des opérateurs présents dans les relevés d'incidents.
OPERATOR_PII_COLUMNS = ("operator_name", "operator_badge")


def anonymize_operators(
    df: pd.DataFrame, columns: tuple[str, ...] = OPERATOR_PII_COLUMNS
) -> pd.DataFrame:
    """Anonymise les opérateurs en **supprimant** leurs identifiants directs.

    Choix technique (justification) :
    
    - Rappel de la finalité du traitement : analyser les relevés d'incidents machines 
      (distribution temporelle, typologie des pannes, corrélations entre signaux, 
      fiabilité du signalement) pour préparer une solution de maintenance prédictive.

    - ``operator_name`` et ``operator_badge`` sont des **identifiants directs**
      (DCP au sens du RGPD). Ils ne sont **pas nécessaires** aux analyses visées
      (distribution temporelle, signaux, corrélations, indice de confiance par
      signalement) — donc on applique le principe de **minimisation**.
    - On retient une **anonymisation par suppression** plutôt qu'une
      pseudonymisation par hash : aucune table de correspondance n'étant
      conservée, l'opération est **irréversible** et le jeu de données résultant
      sort du périmètre RGPD (plus de donnée à caractère personnel). Le
      regroupement par opérateur n'étant pas requis, on évite le risque résiduel
      d'un pseudo-identifiant.
    - ``comment`` est **conservé** : il s'agit d'une saisie guidée décrivant le
      type de panne (ex. « chauffe anormale »), sans donnée personnelle.

    Retourne une **copie** ; sans effet de bord si les colonnes sont déjà absentes.

    A noter : 
    Les quasi-identifiants restants (shift vs machine_id vs date) sont nécessaires 
    à l'analyse et donc conservés. 
    Le risque de ré-identification indirecte est réduit (plus aucun identifiant direct) 
    et maîtrisé (usage interne, pas de diffusion). 
    Si une diffusion externe était envisagée, on appliquerait un k-anonymat sur ces colonnes.
    
    """
    to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=to_drop).copy()
