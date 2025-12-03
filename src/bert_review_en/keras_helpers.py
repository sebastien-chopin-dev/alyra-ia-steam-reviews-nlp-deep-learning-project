# KerasNLP pour BERT
import keras_nlp

import tensorflow as tf

from keras import layers
import keras


def instantiate_bert_model_finetuned(preset_name: str, architecture_layer=0):

    # BERT backbone - TOUS LES POIDS ENTRAÎNABLES
    bert_backbone = keras_nlp.models.BertBackbone.from_preset(
        preset_name, trainable=True  # FINE-TUNING
    )

    # Inputs (données déjà preprocessée en amont)
    inputs = {
        "token_ids": layers.Input(shape=(128,), dtype=tf.int32, name="token_ids"),
        "segment_ids": layers.Input(shape=(128,), dtype=tf.int32, name="segment_ids"),
        "padding_mask": layers.Input(shape=(128,), dtype=tf.int32, name="padding_mask"),
    }

    # Construction du modèle avec preprocess intégré mais à chaque epochs
    # inputs = keras.Input(shape=(), dtype="string", name="text_input")
    # x = preprocessor(inputs)

    # BERT
    bert_output = bert_backbone(inputs)["sequence_output"]
    # x = bert_backbone(x) sequence_output sortie par défaut

    # Extraction du token [CLS] (premier token)
    cls_token = bert_output[:, 0, :]

    if (
        architecture_layer == 1
    ):  # Architecture vu pendant la formation avec petit dataset de démo
        x = layers.Dropout(0.3)(cls_token)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif architecture_layer == 2:  # Architecture plus équilibré
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    elif architecture_layer == 3:  # Architecture plus complexe
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    else:
        x = layers.Dropout(0.1)(x)

    # Nombre de couche Version auto avec optuna
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)

    # Dernière couche commune pour la classification
    output = layers.Dense(1, activation="sigmoid", name="classifier")(x)
    model = keras.Model(inputs, output)

    return model
