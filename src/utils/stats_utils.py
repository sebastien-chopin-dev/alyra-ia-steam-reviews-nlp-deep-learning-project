import pandas as pd


def column_summary(df: pd.DataFrame):
    """Affiche un résumé détaillé des colonnes du DataFrame"""
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
