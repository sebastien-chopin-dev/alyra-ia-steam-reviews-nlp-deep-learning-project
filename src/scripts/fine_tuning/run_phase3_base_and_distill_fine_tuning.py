from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    run_multiple_combinaison,
)


def main(run_index=-1):
    # quick_test_presets = [
    #     "distil_bert_base_en_uncased",
    #     "bert_base_en_uncased"
    # ]

    # # Tester toutes les combinaisons LR + architectures
    learning_rate_list = 3e-5
    layer_architecture_list = 1
    subset_size = -1  # Toute les reviews
    callback_s = 1  # 0 - 1 - 2 - 3 (plus rapide au plus patient)

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
        "en_phase3", combinations, run_index=run_index, subset_size=subset_size
    )


if __name__ == "__main__":
    main()
