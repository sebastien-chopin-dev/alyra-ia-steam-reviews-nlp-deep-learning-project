import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def column_summary(df: pd.DataFrame):
    """Display a detailed summary of the DataFrame columns"""
    summary = []
    for col in df.columns:
        col_type = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()

        # Handle the case where the column contains lists (unhashable)
        try:
            unique_count = df[col].nunique()
        except TypeError:
            # If error (lists), convert to string temporarily
            unique_count = df[col].astype(str).nunique()
            print(
                f"⚠️ Column '{col}' contains non-hashable types (probably lists)"
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

    # Display column summary
    print("=" * 80)
    print("Detailed column summary:")
    print("=" * 80)
    column_summary_df = pd.DataFrame(summary)
    print(column_summary_df.to_string(index=False))
    print("\n")


def show_model_train_history(history, save_path: str):
    """
    Display the training history of one or more models

    Args:
        history: List of tuples (name, history_object, color)
        save_path: Path to save the plot
    """
    n_models = len(history)

    # Create subplots
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))

    # If a single model, axes is not automatically a 2D array
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
    print(f"Plots saved: {save_path}")
    plt.close()


def show_confusion_matrix(cm, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved: {save_path}")


def compare_evaluation_results(csv_path: str):
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Main columns to display
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

    # Best model by accuracy
    best_accuracy_idx = df["test_accuracy"].idxmax()
    best_model = df.loc[best_accuracy_idx]

    print("\n" + "=" * 80)
    print("🏆 BEST MODEL (Accuracy)")
    print("=" * 80)
    print(f"Nom: {best_model['model_name']}")
    print(f"Preset: {best_model['preset']}")
    print(f"Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"F1-Score (macro): {best_model['macro_avg_f1_score']:.4f}")
    print(f"Date: {best_model['timestamp']}")

    # Visualization
    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df["test_accuracy"], marker="o", linewidth=2)
    plt.xlabel("Experiment")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy Progress")
    plt.grid(True, alpha=0.3)

    # Subplot 2: F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(
        df.index, df["macro_avg_f1_score"], marker="o", linewidth=2, color="orange"
    )
    plt.xlabel("Experiment")
    plt.ylabel("Macro F1-Score")
    plt.title("F1-Score Progress")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    return df


def show_stats_compare_train_evaluation(df: pd.DataFrame, best_count=6):
    df["duration_seconds"] = pd.to_timedelta(df["duration"]).dt.total_seconds()

    print("Results Analysis")

    print(f"Nombre de runs : {len(df)}")
    print(f"Dataset : {df['reviews_subset'].iloc[0]:,} reviews")
    print()

    # Top Configurations
    print(f"Top {best_count} Best Configurations")
    print("=" * 80)

    top6 = df.nlargest(best_count, "test_accuracy")[
        [
            "learning_rate",
            "layer_architecture",
            "callback_strategy",
            "test_accuracy",
            "test_loss",
            "macro_avg_f1_score",
            "weighted_avg_f1_score",
            "duration",
        ]
    ].copy()
    top6.index = range(1, best_count + 1)
    top6.columns = [
        "LR",
        "Arch",
        "CB",
        "Accuracy",
        "Loss",
        "Macro F1",
        "Weighted F1",
        "Duration",
    ]

    print(top6.to_string())
    print()

    # Global Statistics
    print("Global Statistics")
    print("=" * 80)

    stats_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "F1-Score (Macro)", "Loss"],
            "Best": [
                f"{df['test_accuracy'].max():.4f}",
                f"{df['macro_avg_f1_score'].max():.4f}",
                f"{df['test_loss'].min():.4f}",
            ],
            "Worst": [
                f"{df['test_accuracy'].min():.4f}",
                f"{df['macro_avg_f1_score'].min():.4f}",
                f"{df['test_loss'].max():.4f}",
            ],
            "Mean": [
                f"{df['test_accuracy'].mean():.4f}",
                f"{df['macro_avg_f1_score'].mean():.4f}",
                f"{df['test_loss'].mean():.4f}",
            ],
            "Std Dev": [
                f"{df['test_accuracy'].std():.4f}",
                f"{df['macro_avg_f1_score'].std():.4f}",
                f"{df['test_loss'].std():.4f}",
            ],
        }
    )

    print(stats_df.to_string(index=False))
    print()

    # Recommandations
    best_run = df.loc[df["test_accuracy"].idxmax()]

    recommendations = pd.DataFrame(
        {
            "Hyperparameter": [
                "Learning Rate",
                "Architecture",
                "Callback Strategy",
                "Achieved Accuracy",
            ],
            "Optimal Value": [
                f"{best_run['learning_rate']}",
                f"{int(best_run['layer_architecture'])}",
                f"{int(best_run['callback_strategy'])}",
                f"{best_run['test_accuracy']:.4f} ({best_run['test_accuracy']*100:.2f}%)",
            ],
        }
    )

    print(recommendations.to_string(index=False))
    print()


def show_plt_distribution_accuracy_result(df: pd.DataFrame):
    # Create a label for each configuration
    df["config"] = df.apply(
        lambda x: f"LR:{x['learning_rate']:.0e}\nArch:{int(x['layer_architecture'])}\nCB:{int(x['callback_strategy'])}",
        axis=1,
    )

    # Chart
    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        range(len(df)), df["test_accuracy"], color="steelblue", edgecolor="black"
    )

    plt.xlabel("Configuration")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy - All Configurations")
    plt.xticks(range(len(df)), range(1, len(df) + 1), rotation=0)

    # Zoom Y axis
    min_acc = df["test_accuracy"].min()
    max_acc = df["test_accuracy"].max()
    margin = (max_acc - min_acc) * 0.1  # 10% margin
    plt.ylim(min_acc - margin, max_acc + margin)

    plt.axhline(
        df["test_accuracy"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Moyenne: {df["test_accuracy"].mean():.4f}',
    )
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
