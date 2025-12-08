import os
import time
from datetime import timedelta

import psutil
import tensorflow as tf

# KerasNLP pour BERT
import keras_nlp
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.file_system_utils import get_outputs_path
from src.utils.stats_utils import show_confusion_matrix, column_summary


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print("\nUtilisation mémoire:")
    print(f"   RAM utilisée: {mem_info.rss / 1024**3:.2f} GB")
    print(f"   RAM disponible: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"   RAM totale: {psutil.virtual_memory().total / 1024**3:.2f} GB")


class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print_memory_usage()


def get_preprocessor_and_backbone(config):

    preset_name = config["PREPROCESSOR_PRESET_NAME"]
    preset_lower = preset_name.lower()
    use_sequence_output = True

    # DistilBERT
    if "distil" in preset_lower and "bert" in preset_lower:
        preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
            preset_name, sequence_length=config["SEQUENCE_LENGTH"]
        )
        backbone = keras_nlp.models.DistilBertBackbone.from_preset(
            preset_name, trainable=True
        )
        use_sequence_output = False
    elif "deberta" in preset_lower:
        preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
            preset_name, sequence_length=config["SEQUENCE_LENGTH"]
        )
        backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
            preset_name, trainable=True
        )
        use_sequence_output = False
    # XLM-RoBERTa
    elif "xlm_roberta" in preset_lower:
        preprocessor = keras_nlp.models.XLMRobertaPreprocessor.from_preset(
            preset_name, sequence_length=config["SEQUENCE_LENGTH"]
        )
        backbone = keras_nlp.models.XLMRobertaBackbone.from_preset(
            preset_name, trainable=True
        )
    elif "roberta" in preset_lower:
        preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset(
            preset_name, sequence_length=config["SEQUENCE_LENGTH"]
        )
        backbone = keras_nlp.models.RobertaBackbone.from_preset(
            preset_name, trainable=True
        )
    # BERT (standard)
    elif "bert" in preset_lower:
        preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
            preset_name, sequence_length=config["SEQUENCE_LENGTH"]
        )
        backbone = keras_nlp.models.BertBackbone.from_preset(
            preset_name, trainable=True
        )

    else:
        raise ValueError(f"Preset '{preset_name}' non supporté")

    return preprocessor, backbone, use_sequence_output


def create_tf_dataset(X_preprocessed, y, config):
    # Evite de charger tout en mémoire pour les gros modèles
    preset_name = config["PREPROCESSOR_PRESET_NAME"]
    preset_lower = preset_name.lower()

    # DistilBERT - deberta - PAS de segment_ids
    if "distil" in preset_lower or "deberta_v3" in preset_lower:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "token_ids": X_preprocessed["token_ids"],
                    "padding_mask": X_preprocessed["padding_mask"],
                },
                y,
            )
        )
    elif "xlm_roberta" in preset_lower:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "token_ids": X_preprocessed["token_ids"],
                    "segment_ids": X_preprocessed["segment_ids"],
                    "padding_mask": X_preprocessed["padding_mask"],
                },
                y,
            )
        )
    elif "roberta" in preset_lower:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "token_ids": X_preprocessed["token_ids"],
                    "padding_mask": X_preprocessed["padding_mask"],
                },
                y,
            )
        )
    # BERT AVEC segment_ids
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "token_ids": X_preprocessed["token_ids"],
                    "segment_ids": X_preprocessed["segment_ids"],
                    "padding_mask": X_preprocessed["padding_mask"],
                },
                y,
            )
        )

    # Shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(config["BATCH_SIZE"])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_model_inputs(config):

    sequence_length = config["SEQUENCE_LENGTH"]
    preset_name = config["PREPROCESSOR_PRESET_NAME"]
    preset_lower = preset_name.lower()

    # DistilBERT - deberta - PAS de segment_ids
    if "distil" in preset_lower or "deberta_v3" in preset_lower:
        inputs = {
            "token_ids": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="token_ids"
            ),
            "padding_mask": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="padding_mask"
            ),
        }
    elif "xlm_roberta" in preset_lower:
        inputs = {
            "token_ids": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="token_ids"
            ),
            "segment_ids": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="segment_ids"
            ),
            "padding_mask": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="padding_mask"
            ),
        }
    elif "roberta" in preset_lower:
        inputs = {
            "token_ids": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="token_ids"
            ),
            "padding_mask": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="padding_mask"
            ),
        }
    # BERT AVEC segment_ids
    else:
        inputs = {
            "token_ids": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="token_ids"
            ),
            "segment_ids": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="segment_ids"
            ),
            "padding_mask": layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="padding_mask"
            ),
        }

    return inputs


def load_and_verif_reviews_datas(config: dict):
    # Récupération du chemin du dataset Kaggle
    kaggle_cache_path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    print("Path to dataset files:", kaggle_cache_path)

    # Chargement des données
    reviews_file_path = f"{kaggle_cache_path}/{config["REVIEWS_DATA_FILE"]}"
    df_reviews = pd.read_csv(reviews_file_path)

    print(f"\nDataset chargé: {len(df_reviews)} reviews")
    column_summary(df_reviews)  # Résumé des colonnes

    # Affichage d'un échantillon
    print("\n" + "=" * 80)
    print("Aperçu des données:")
    print("=" * 80)
    print(df_reviews.head())

    return df_reviews


def create_train_test_eval_split(df: pd.DataFrame, config: dict):
    if config["REVIEWS_SUBSET"] != -1:
        df_reviews_sample = df.sample(n=config["REVIEWS_SUBSET"], random_state=42)
    else:
        df_reviews_sample = df.copy()

    config["REVIEWS_SUBSET"] = len(df_reviews_sample)

    # Extraction des features et labels
    X = df_reviews_sample["review"].values
    y = df_reviews_sample["voted_up"].values

    # Split train/temp (80/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=config["SEED"], stratify=y
    )

    # Split temp en val/test (50/50 du 20% restant = 10% chacun)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config["SEED"], stratify=y_temp
    )

    print("\n Données split:")
    print(f"   - Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   - Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def keras_preset_preprocess_data(X_train, X_val, X_test, config: dict):
    print("\nInitialisation du preprocessor BERT...")
    preprocessor, backbone, use_sequence_output = get_preprocessor_and_backbone(config)
    print("Preprocessor chargé")

    print("\nPreprocessing des données avec BERT...")
    X_train_bert = preprocessor(X_train)
    X_val_bert = preprocessor(X_val)
    X_test_bert = preprocessor(X_test)
    print("Preprocessing terminé")

    return X_train_bert, X_val_bert, X_test_bert


def instantiate_keras_preset_model_finetuned(config: dict):
    # Paramètres
    preset_name = config["MODEL_PRESET_NAME"]
    architecture_layer = config["LAYER_ARCHITECTURE"]

    print(f"\n Construction du modèle Preset Keras {preset_name}...")

    # BERT backbone - TOUS LES POIDS ENTRAÎNABLES
    preprocessor, bert_backbone, use_sequence_output = get_preprocessor_and_backbone(
        config
    )

    # Inputs (données déjà preprocessée en amont)
    inputs = create_model_inputs(config)

    # Construction du modèle avec preprocess intégré mais à chaque epochs
    # inputs = keras.Input(shape=(), dtype="string", name="text_input")
    # x = preprocessor(inputs)

    if use_sequence_output:  # BERT
        bert_output = bert_backbone(inputs)["sequence_output"]
    else:  # DistillBERT
        bert_output = bert_backbone(inputs)

    # Extraction du token [CLS] (premier token)
    cls_token = bert_output[:, 0, :]

    if (
        architecture_layer == 1
    ):  # Architecture vu pendant la formation avec petit dataset de démo
        x = layers.Dropout(0.3)(cls_token)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif architecture_layer == 2:  # Architecture plus équilibré
        x = layers.Dropout(0.2)(cls_token)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    elif architecture_layer == 3:  # Architecture plus complexe
        x = layers.Dropout(0.3)(cls_token)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    else:
        x = layers.Dropout(0.1)(cls_token)

    # Nombre de couche Version auto avec optuna
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)

    # Dernière couche commune pour la classification
    output = layers.Dense(1, activation="sigmoid", name="classifier")(x)
    model = keras.Model(inputs, output)

    return model


def get_callback_from_config(config):
    if config["CALLBACK_OPTION"] == 0:
        # Pour tests rapides agressif
        callback_es = EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        )
        callback_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1
        )
    elif config["CALLBACK_OPTION"] == 1:
        # Pour fine-tuning BERT (équilibré)
        callback_es = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1,
        )
        callback_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            cooldown=1,
            min_delta=0.001,
            verbose=1,
        )
    elif config["CALLBACK_OPTION"] == 2:
        # Pour fine-tuning BERT (équilibré un petit peu plus patient)
        callback_es = EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1,
        )
        callback_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            cooldown=1,
            min_delta=0.001,
            verbose=1,
        )
    else:
        # Pour entraînement long (patient)
        callback_es = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            min_delta=0.0005,
            verbose=1,
        )
        callback_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-8,
            cooldown=2,
            verbose=1,
        )

    return callback_es, callback_lr


def compile_and_train_model(
    bert_finetuned_model: keras.Model,
    X_train_bert,
    y_train,
    X_val_bert,
    y_val,
    config: dict,
):
    bert_finetuned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print("Modèle compilé")

    tensorboard = TensorBoard(
        log_dir=f"{config["SAVE_FOLDER"]}/logs", histogram_freq=1, write_graph=True
    )

    callback_es, callback_lr = get_callback_from_config(config)

    callbacks_list = [callback_es, callback_lr, tensorboard, MemoryCallback()]

    print("\nDébut de l'entraînement...")
    print("=" * 80)

    if config["USE_DATASET"] is True:
        train_dataset = create_tf_dataset(X_train_bert, y_train, config)
        val_dataset = create_tf_dataset(X_val_bert, y_val, config)

        history_bert_finetuned = bert_finetuned_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config["EPOCHS"],
            callbacks=callbacks_list,
            verbose=1,
        )
    else:
        history_bert_finetuned = bert_finetuned_model.fit(
            X_train_bert,
            y_train,
            validation_data=(X_val_bert, y_val),
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            callbacks=callbacks_list,
            verbose=1,
        )

    print("\nEntraînement terminé!")

    return history_bert_finetuned


def evaluate_model(finetuned_model: keras.Model, X_test_bert, y_test, config: dict):
    print("\nÉvaluation sur le test set...")
    test_loss, test_accuracy = finetuned_model.evaluate(X_test_bert, y_test, verbose=0)

    print("\nRésultats sur le test set:")
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    print("\nCalcul des prédictions...")
    y_pred_prob = finetuned_model.predict(X_test_bert, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Classification report
    print("\nClassification Report:")
    print("=" * 80)
    report_dict = classification_report(
        y_test, y_pred, target_names=["Négatif", "Positif"], output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=["Négatif", "Positif"]))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion:")
    print(cm)

    show_confusion_matrix(cm, f"{config['SAVE_FOLDER']}/confusion_matrix.png")

    return test_loss, test_accuracy, report_dict, cm


def test_on_new_reviews(finetuned_model: keras.Model):
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
    predictions_prob = finetuned_model.predict(test_reviews, verbose=0).flatten()

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


def save_model_spec_and_eval(
    config: dict, test_loss, test_accuracy, report_dict, cm, duration_str
):
    eval_results = {
        "timestamp": pd.Timestamp.now(),
        "model_name": config.get("NAME_TRAIN_CONFIG", "bert_model"),
        "preset": config.get("MODEL_PRESET_NAME", "bert_small_en_uncased"),
        "reviews_subset": config.get("REVIEWS_SUBSET", -1),
        "sequence_length": config.get("SEQUENCE_LENGTH", 128),
        "batch_size": config.get("BATCH_SIZE", 32),
        "epochs": config.get("EPOCHS", 3),
        "learning_rate": config.get("LEARNING_RATE", 5e-5),
        "layer_architecture": config.get("LAYER_ARCHITECTURE", 0),
        "callback_strategy": config.get("CALLBACK_OPTION", 0),
        # Métriques globales
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        # Métriques par classe - Négatif
        "negative_precision": report_dict["Négatif"]["precision"],
        "negative_recall": report_dict["Négatif"]["recall"],
        "negative_f1_score": report_dict["Négatif"]["f1-score"],
        "negative_support": report_dict["Négatif"]["support"],
        # Métriques par classe - Positif
        "positive_precision": report_dict["Positif"]["precision"],
        "positive_recall": report_dict["Positif"]["recall"],
        "positive_f1_score": report_dict["Positif"]["f1-score"],
        "positive_support": report_dict["Positif"]["support"],
        # Moyennes
        "macro_avg_precision": report_dict["macro avg"]["precision"],
        "macro_avg_recall": report_dict["macro avg"]["recall"],
        "macro_avg_f1_score": report_dict["macro avg"]["f1-score"],
        "weighted_avg_precision": report_dict["weighted avg"]["precision"],
        "weighted_avg_recall": report_dict["weighted avg"]["recall"],
        "weighted_avg_f1_score": report_dict["weighted avg"]["f1-score"],
        # Matrice de confusion
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        # Durée d'entraînement
        "duration": duration_str,
    }

    # Créer un DataFrame avec une seule ligne
    eval_df = pd.DataFrame([eval_results])

    outputs_dir = get_outputs_path()

    # Chemin du fichier CSV
    csv_path = f"{outputs_dir}/evaluation_results_{config['PHASE_NAME']}.csv"

    # Sauvegarder (append si le fichier existe déjà)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        eval_df = pd.concat([existing_df, eval_df], ignore_index=True)

    # Sauvegarder
    eval_df.to_csv(csv_path, index=False)
