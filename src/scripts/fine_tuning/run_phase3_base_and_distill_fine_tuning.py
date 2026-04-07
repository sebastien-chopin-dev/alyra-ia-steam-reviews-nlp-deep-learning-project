from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    run_multiple_combinaison,
)


def main(run_index=-1):
    # quick_test_presets = [
    #     "distil_bert_base_en_uncased",
    #     "bert_base_en_uncased"
    # ]

    # # Test all LR + architecture combinations
    learning_rate_list = 3e-5
    layer_architecture_list = 1
    subset_size = -1  # All reviews
    callback_s = 2  # 0 - 1 - 2 - 3 (fastest to most patient)

    finetune_model_config = {
        "NAME_TRAIN_CONFIG": "Hyperparameter Search phase 1",
        "PHASE_NAME": "en_phase3",
        "SEED": 42,
        "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
        "REVIEWS_SUBSET": subset_size,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,  # high value, early stopping will trigger
        "SEQUENCE_LENGTH": 128,
        "USE_TF_DATASET": False,  # opti batch memory
        "PLT_COLOR": "green",
    }

    combinations = []

    # combinations.append(
    #     (
    #         "bert_base_en_uncased",
    #         learning_rate_list,
    #         layer_architecture_list,
    #         callback_s,
    #     )
    # )

    combinations.append(
        (
            "distil_bert_base_en_uncased",
            learning_rate_list,
            layer_architecture_list,
            callback_s,
        )
    )

    run_multiple_combinaison(
        "en_phase3",
        combinations,
        base_config=finetune_model_config,
        run_index=run_index,
        subset_size=subset_size,
    )


if __name__ == "__main__":
    main()
