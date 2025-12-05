import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def show_model_train_history(history, save_path: str):
    """
    Affiche l'historique d'entraînement d'un ou plusieurs modèles

    Args:
        history: Liste de tuples (name, history_object, color)
        save_path: Chemin pour sauvegarder le graphique
    """
    n_models = len(history)

    # Créer les subplots
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))

    # Si un seul modèle, axes n'est pas automatiquement un array 2D
    if n_models == 1:
        axes = axes.reshape(2, 1)

    # Plot Loss
    for idx, (name, hist, color) in enumerate(history):
        ax = axes[0, idx]
        ax.plot(
            hist.history["loss"],
            label="Train",
            linewidth=2,
            color=color,
            alpha=0.7,
            marker="o",
        )
        ax.plot(
            hist.history["val_loss"],
            label="Val",
            linewidth=2,
            color=color,
            linestyle="--",
            marker="s",
        )
        ax.set_title(f"Loss - {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot Accuracy
    for idx, (name, hist, color) in enumerate(history):
        ax = axes[1, idx]
        ax.plot(
            hist.history["accuracy"],
            label="Train",
            linewidth=2,
            color=color,
            alpha=0.7,
            marker="o",
        )
        ax.plot(
            hist.history["val_accuracy"],
            label="Val",
            linewidth=2,
            color=color,
            linestyle="--",
            marker="s",
        )
        ax.set_title(f"Accuracy - {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Graphiques sauvegardés: {save_path}")
    plt.close()


def show_confusion_matrix(cm, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Négatif", "Positif"],
        yticklabels=["Négatif", "Positif"],
    )
    plt.title("Matrice de Confusion", fontsize=14, fontweight="bold")
    plt.ylabel("Vraie classe")
    plt.xlabel("Classe prédite")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Matrice de confusion sauvegardée: {save_path}")


def compare_evaluation_results(csv_path: str):
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 80)
    print("COMPARAISON DES MODÈLES")
    print("=" * 80)

    # Colonnes principales à afficher
    columns_to_display = [
        "timestamp",
        "model_name",
        "preset",
        "epochs",
        "test_accuracy",
        "macro_avg_f1_score",
        "test_loss",
    ]

    print(df[columns_to_display].to_string(index=False))

    # Meilleur modèle par accuracy
    best_accuracy_idx = df["test_accuracy"].idxmax()
    best_model = df.loc[best_accuracy_idx]

    print("\n" + "=" * 80)
    print("🏆 MEILLEUR MODÈLE (Accuracy)")
    print("=" * 80)
    print(f"Nom: {best_model['model_name']}")
    print(f"Preset: {best_model['preset']}")
    print(f"Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"F1-Score (macro): {best_model['macro_avg_f1_score']:.4f}")
    print(f"Date: {best_model['timestamp']}")

    # Visualisation
    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df["test_accuracy"], marker="o", linewidth=2)
    plt.xlabel("Expérience")
    plt.ylabel("Test Accuracy")
    plt.title("Évolution de l'Accuracy")
    plt.grid(True, alpha=0.3)

    # Subplot 2: F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(
        df.index, df["macro_avg_f1_score"], marker="o", linewidth=2, color="orange"
    )
    plt.xlabel("Expérience")
    plt.ylabel("Macro F1-Score")
    plt.title("Évolution du F1-Score")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n✅ Graphique de comparaison sauvegardé: model_comparison.png")

    return df
