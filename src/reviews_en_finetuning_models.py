from src.finetune_models.fine_tuning_preset_keras_bert import train_bert_base_model


if __name__ == "__main__":
    # Premier test avec les hyper-paramétres vu pendant la formation
    # bert_base_model_config = {
    #     "NAME_TRAIN_CONFIG": "BERT Fine-Tuned Test rapide",
    #     "SEED": 42,
    #     "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
    #     "SAVE_FOLDER": "bert_base_en_test",
    #     "REVIEWS_SUBSET": 20000,
    #     "PREPROCESSOR_PRESET_NAME": "bert_base_en_uncased",
    #     "MODEL_PRESET_NAME": "bert_tiny_en_uncased",
    #     "SEQUENCE_LENGTH": 128,
    #     "BATCH_SIZE": 32,
    #     "EPOCHS": 10,
    #     "LEARNING_RATE": 2e-5,
    #     "LAYER_ARCHITECTURE": 1,
    #     "PLT_COLOR": "green",
    # }

    bert_base_model_config = {
        "NAME_TRAIN_CONFIG": "BERT Fine-Tuned Test rapide",
        "SEED": 42,
        "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
        "SAVE_FOLDER": "bert_base_en_test",
        "REVIEWS_SUBSET": -1,
        "PREPROCESSOR_PRESET_NAME": "bert_base_en_uncased",
        "MODEL_PRESET_NAME": "bert_tiny_en_uncased",
        "SEQUENCE_LENGTH": 128,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,
        "LEARNING_RATE": 2e-5,
        "LAYER_ARCHITECTURE": 1,
        "PLT_COLOR": "green",
    }

    train_bert_base_model(bert_base_model_config)
