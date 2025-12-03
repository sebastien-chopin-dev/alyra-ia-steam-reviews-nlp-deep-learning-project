"""
Transfer Learning et Fine-Tuning avec BERT : Classification de Sentiments

Ce script implémente le Transfer Learning et le Fine-Tuning en NLP avec BERT :
- Transfer Learning - Utiliser un modèle pré-entraîné sur un language et l'adapter
- Fine-Tuning - Ré-entrainer BERT à notre tâche spécifique
- Layer Freezing : Geler certaines couches pour le transfer learning
- Learning Rate Scheduling : Stratégies d'optimisation pour le fine-tuning

Utilise les fichiers Steam review anglais pour la classification de sentiments (positif/négatif).
Backend: TensorFlow avec API KerasNLP
"""

import warnings

from src.bert_review_en.keras_helpers import instantiate_bert_model_finetuned
from src.configuration.init_tf_helpers import (
    init_gpu_for_tf,
    init_graph_plt,
    init_seed,
    show_tf_keras_version_engine,
)
from src.utils.stats_utils import column_summary

warnings.filterwarnings("ignore")

# Bibliothèques principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# TensorFlow et Keras
import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

# KerasNLP pour BERT
import keras_nlp

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


init_gpu_for_tf()  # Utilisation de la GPU pour TensorFlow (si disponible
init_graph_plt()  # Initialisation de matplotlib (pour graphiques)

SEED = 42
init_seed(SEED)  # Initialisation seed reproductibilité

show_tf_keras_version_engine()

# Récupération du chemin du dataset Kaggle
kaggle_cache_path = kagglehub.dataset_download("kieranpoc/steam-reviews")
print("Path to dataset files:", kaggle_cache_path)

# Chargement des données
REVIEWS_FILE_PATH = f"{kaggle_cache_path}/reviews_en_processed.csv"
df_reviews = pd.read_csv(REVIEWS_FILE_PATH)

print(f"\nDataset chargé: {len(df_reviews)} reviews")
print(f"   - Colonnes: {list(df_reviews.columns)}")
print(f"   - Shape: {df_reviews.shape}")

# Affichage d'un échantillon
print("\n" + "=" * 80)
print("Aperçu des données:")
print("=" * 80)
print(df_reviews.head())

# Résumé des colonnes
column_summary(df_reviews)

# Rééquilibrage des données (50% positif / 50% négatif)
positive_reviews = df_reviews[df_reviews["label"] == 1]
negative_reviews = df_reviews[df_reviews["label"] == 0]

n_samples = min(len(positive_reviews), len(negative_reviews))
print("\nRééquilibrage des données:")
print(f"   - Reviews positives: {len(positive_reviews)}")
print(f"   - Reviews négatives: {len(negative_reviews)}")
print(f"   - Samples par classe: {n_samples}")

# Échantillonnage
positive_sampled = positive_reviews.sample(n=n_samples, random_state=SEED)
negative_sampled = negative_reviews.sample(n=n_samples, random_state=SEED)

# Fusion et mélange
df_balanced = pd.concat([positive_sampled, negative_sampled])
df_balanced = df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\nDataset équilibré: {len(df_balanced)} reviews")
print("   - Distribution:")
print(df_balanced["label"].value_counts())


# Extraction des features et labels
X = df_balanced["processed_text"].values
y = df_balanced["label"].values

# Split train/temp (80/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Split temp en val/test (50/50 du 20% restant = 10% chacun)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("\n Données séparées:")
print(f"   - Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   - Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Paramètres
SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5

print("\n🔧 Configuration BERT:")
print("   - Modèle: bert_small_en_uncased")
print(f"   - Sequence length: {SEQUENCE_LENGTH}")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Learning rate: {LEARNING_RATE}")


print("\nInitialisation du preprocessor BERT...")
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_small_en_uncased", sequence_length=SEQUENCE_LENGTH
)
print("Preprocessor chargé")

print("\nPreprocessing des données avec BERT...")
X_train_bert = preprocessor(X_train)
X_val_bert = preprocessor(X_val)
X_test_bert = preprocessor(X_test)
print("Preprocessing terminé")


print("\n Construction du modèle BERT...")

# bert_small_en_uncased
bert_finetuned_model = instantiate_bert_model_finetuned("bert_tiny_en_uncased", 1)

print("Modèle créé")
print("\nArchitecture du modèle:")
bert_finetuned_model.summary()


print("\nCompilation du modèle...")
bert_finetuned_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print("Modèle compilé")


# Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True, verbose=1, mode="min"
)

# Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1
)

callbacks_list = [early_stopping, reduce_lr]


print("\nDébut de l'entraînement...")
print("=" * 80)

history = bert_finetuned_model.fit(
    X_train_bert,
    y_train,
    validation_data=(X_val_bert, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1,
)

print("\nEntraînement terminé!")

print("\nVisualisation des courbes d'apprentissage...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Courbe de loss
axes[0].plot(history.history["loss"], label="Train Loss", marker="o")
axes[0].plot(history.history["val_loss"], label="Val Loss", marker="o")
axes[0].set_title("Model Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Courbe d'accuracy
axes[1].plot(history.history["accuracy"], label="Train Accuracy", marker="o")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
axes[1].set_title("Model Accuracy", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
print("Graphiques sauvegardés: training_curves.png")


print("\nÉvaluation sur le test set...")
test_loss, test_accuracy = bert_finetuned_model.evaluate(X_test, y_test, verbose=0)

print("\nRésultats sur le test set:")
print(f"   - Test Loss: {test_loss:.4f}")
print(f"   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")


print("\nCalcul des prédictions...")
y_pred_prob = bert_finetuned_model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print("=" * 80)
print(classification_report(y_test, y_pred, target_names=["Négatif", "Positif"]))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion:")
print(cm)

# Visualisation de la matrice de confusion
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
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
print("Matrice de confusion sauvegardée: confusion_matrix.png")


# ============================================================================
# TESTS SUR NOUVELLES REVIEWS
# ============================================================================

print("\nTests sur de nouvelles reviews...")

# Reviews de test
test_reviews = [
    "This game was absolutely amazing! The graphics and gameplay was superb.",
    "Terrible game. Waste of time and money. The worst game ever.",
    "It was okay, nothing special. Some good moments but forgettable.",
    "Brilliant masterpiece! One of the best games of the decade!",
    "Boring and predictable. The actors tried but the script was awful.",
    "I loved every minute! The graphics were stunning.",
]

# Prédictions
predictions_prob = bert_finetuned_model.predict(test_reviews, verbose=0).flatten()

# Affichage des résultats
print("\n" + "=" * 90)
print("RÉSULTATS DES PRÉDICTIONS")
print("=" * 90)

for i, (review, prob) in enumerate(zip(test_reviews, predictions_prob), 1):
    sentiment = "POSITIF" if prob > 0.5 else "NÉGATIF"
    confidence = prob if prob > 0.5 else 1 - prob
    emoji = "✅" if sentiment == "POSITIF" else "❌"

    print(f'\n{i}. "{review}"')
    print(f"   {emoji} {sentiment} (Confiance: {confidence*100:.1f}%)")
    print("-" * 90)


print("\nSauvegarde du modèle...")
model_path = "bert_sentiment_model.keras"
bert_finetuned_model.save(model_path)
print(f"Modèle sauvegardé: {model_path}")


print("\n" + "=" * 80)
print("✅ SCRIPT TERMINÉ AVEC SUCCÈS!")
print("=" * 80)
print("\nFichiers générés:")
print("   - training_curves.png")
print("   - confusion_matrix.png")
print("   - bert_sentiment_model.keras")
print("=" * 80)
