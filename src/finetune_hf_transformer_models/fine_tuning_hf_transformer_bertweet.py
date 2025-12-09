# Nécessite un autre environnement avec keras 2
# pip install tf-keras

import warnings
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
from datetime import timedelta

from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import keras
from keras import layers

from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_helpers import (
    MemoryCallback,
    create_train_test_eval_split,
    evaluate_model,
    get_callback_from_config,
    load_and_verif_reviews_datas,
    save_model_spec_and_eval,
)

from src.configuration.init_tf_helpers import (
    init_gpu_for_tf,
    init_graph_plt,
    init_seed,
    show_tf_keras_version_engine,
)

from src.utils.file_system_utils import get_outputs_path
from src.utils.stats_utils import show_model_train_history

warnings.filterwarnings("ignore")

bert_base_model_config = {
    "NAME_TRAIN_CONFIG": "BERT Fine-Tuned keras base",
    "SEED": 42,
    "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
    "SAVE_FOLDER": "bert_base_en_poc",
    "REVIEWS_SUBSET": 20000,
    "PREPROCESSOR_PRESET_NAME": "bert_base_en_uncased",
    "MODEL_PRESET_NAME": "bert_tiny_en_uncased",
    "SEQUENCE_LENGTH": 128,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LEARNING_RATE": 3e-5,
    "CALLBACK_OPTION": 1,
    "LAYER_ARCHITECTURE": 0,
    "USE_TF_DATASET": False,
    "PLT_COLOR": "green",
}


def train_bert_tweet_model(config: dict):

    print(f"Start model with {config['NAME_TRAIN_CONFIG']}")
    print(f"   - Preset model: {config["MODEL_PRESET_NAME"]}")
    print(f"   - Subset: {config["REVIEWS_SUBSET"]}")
    print(f"   - Architecture: {config["LAYER_ARCHITECTURE"]}")
    print(f"   - Sequence length: {config["SEQUENCE_LENGTH"]}")
    print(f"   - Batch size: {config["BATCH_SIZE"]}")
    print(f"   - Epochs: {config["EPOCHS"]}")
    print(f"   - Learning rate: {config["LEARNING_RATE"]}")
    print(f"   - Callback strategie: {config["CALLBACK_OPTION"]}")

    start_time = time.time()

    init_gpu_for_tf()  # Utilisation de la GPU pour TensorFlow (si disponible)
    init_graph_plt()  # Initialisation de matplotlib (pour graphiques)
    init_seed(config["SEED"])  # Initialisation seed reproductibilité
    show_tf_keras_version_engine()

    config["SAVE_FOLDER"] = get_outputs_path(f'reports/{config["SAVE_FOLDER"]}')
    print(f"Création dossier reports: {config["SAVE_FOLDER"]}")

    df_reviews = load_and_verif_reviews_datas(config)  # Chargmenet des données

    X_train, y_train, X_val, y_val, X_test, y_test = create_train_test_eval_split(
        df_reviews, config
    )

    # Créer le modèle
    model, tokenizer = build_bertweet_model(config)

    print("\nArchitecture du modèle:")
    model.summary()

    # Tokenizer les données
    print("\nTokenization des données...")
    X_train_encoded = tokenize_texts(X_train, tokenizer, config["SEQUENCE_LENGTH"])
    X_val_encoded = tokenize_texts(X_val, tokenizer, config["SEQUENCE_LENGTH"])
    X_test_encoded = tokenize_texts(X_test, tokenizer, config["SEQUENCE_LENGTH"])

    # Compiler
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callback_es, callback_lr = get_callback_from_config(config)
    callbacks_list = [callback_es, callback_lr, MemoryCallback()]

    # Entraîner
    model.fit(
        {
            "input_ids": X_train_encoded["input_ids"],
            "attention_mask": X_train_encoded["attention_mask"],
        },
        y_train,
        validation_data=(
            {
                "input_ids": X_val_encoded["input_ids"],
                "attention_mask": X_val_encoded["attention_mask"],
            },
            y_val,
        ),
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        callbacks=callbacks_list,
        verbose=1,
    )

    # Create folder path if not exist
    if not os.path.exists(config["SAVE_FOLDER"]):
        os.makedirs(config["SAVE_FOLDER"])

    # Visualisation des courbes d'apprentissage
    models_histories = [(config["NAME_TRAIN_CONFIG"], model, config["PLT_COLOR"])]

    show_model_train_history(
        models_histories, f"{config['SAVE_FOLDER']}/train_history.png"
    )

    # Evaluation finale
    test_loss, test_accuracy, report_dict, cm = evaluate_model(
        model, X_test_encoded, y_test, config
    )

    # Calculer la durée
    training_duration = time.time() - start_time
    duration_str = str(timedelta(seconds=int(training_duration)))

    print(f"\nDurée d'entraînement: {duration_str}")

    # Sauvegarde des résultats pour comparaison ultérieur
    save_model_spec_and_eval(
        config, test_loss, test_accuracy, report_dict, cm, duration_str
    )

    model_path = f"{config['SAVE_FOLDER']}/finetuned_model.keras"
    model.save(model_path)
    print(f"Modèle sauvegardé: {model_path}")


def build_bertweet_model(config: dict):
    model_name = "vinai/bertweet-base"
    sequence_length = config["SEQUENCE_LENGTH"]

    print("\nChargement de BERTweet...")

    # Tokenizer et backbone
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = TFAutoModel.from_pretrained(model_name)

    # Inputs
    input_ids = layers.Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name="attention_mask"
    )

    # Backbone
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)

    # [CLS] token
    cls_token = outputs.last_hidden_state[:, 0, :]

    # Classification head
    layer_arch = config["LAYER_ARCHITECTURE"]

    if layer_arch == 0:
        x = layers.Dropout(0.1)(cls_token)
    elif layer_arch == 1:
        x = layers.Dropout(0.3)(cls_token)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif layer_arch == 2:
        x = layers.Dropout(0.2)(cls_token)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Créer le modèle
    model = keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)

    print("Modèle BERTweet créé")

    return model, tokenizer


def tokenize_texts(texts, tokenizer, max_length=128):
    """
    Tokenize les textes avec le tokenizer BERTweet
    """
    return tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )


# # Configuration
# bertweet_config = {
#     "MODEL_NAME": "BERTweet-Steam",
#     "SEQUENCE_LENGTH": 128,
#     "BATCH_SIZE": 16,
#     "EPOCHS": 10,
#     "LEARNING_RATE": 2e-05,
#     "LAYER_ARCHITECTURE": 1,
#     "REVIEWS_SUBSET": -1,
# }
