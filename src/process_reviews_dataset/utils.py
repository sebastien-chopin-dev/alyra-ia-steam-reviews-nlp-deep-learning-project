"""
Fonctions utilitaires pour le traitement des reviews Steam
"""

import re
import pandas as pd


def column_summary(df: pd.DataFrame):
    summary = []
    for col in df.columns:
        col_type = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()

        # Gérer le cas où la colonne contient des listes (unhashable)
        try:
            unique_count = df[col].nunique()
        except TypeError:
            # Si erreur (listes), convertir en string temporairement
            unique_count = df[col].astype(str).nunique()
            print(
                f"⚠️ Colonne '{col}' contient des types non-hashable (probablement des listes)"
            )

        summary.append(
            {
                "Column": col,
                "Type": str(col_type),
                "Non-Null Count": non_null,
                "Null Count": null_count,
                "Unique Values": unique_count,
            }
        )

    # Afficher le résumé des colonnes
    print("=" * 80)
    print("Résumé détaillé des colonnes:")
    print("=" * 80)
    column_summary_df = pd.DataFrame(summary)
    print(column_summary_df.to_string(index=False))
    print("\n")


def print_voted_up_count_proportion(df: pd.DataFrame):
    voted_up_counts = df["voted_up"].value_counts()
    total_reviews = len(df)

    print("voted_up (reviews positive:1 / negative:0)")
    for voted_up_value, count in voted_up_counts.items():
        proportion = (count / total_reviews) * 100
        print(
            f"voted_up = {voted_up_value}: Count = {count}, Proportion = {proportion:.2f}%"
        )

    print(
        f"\nPour obtenir une distribution équilibrée 50/50, en sous échantillonnant la classe majoritaire\n"
    )
    print(
        f"On aurait {min(voted_up_counts)} reviews par classe. Soit un total de {2 * min(voted_up_counts)} reviews."
    )
