from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    run_multiple_combinaison,
)


def main(run_index=-1):
    # quick_test_presets = [
    #     "bert_small_en_uncased",
    # ]
    # # Tester toutes les combinaisons LR + architectures
    learning_rate_list = [2e-5, 3e-5, 5e-5]
    layer_architecture_list = [0, 1]
    subset_size = -1  # Toute les reviews
    callback_s = 1  # 0 - 1 - 2 (plus rapide au plus patient)

    combinations = []

    # BERT VALIDATION HYPER PARAMETRES - (6 combo)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            combinations.append(("bert_small_en_uncased", lr, arch, callback_s))

    run_multiple_combinaison(
        "phase2", combinations, run_index=run_index, subset_size=subset_size
    )


# BERT-small-arch1-lr3e-05
#    Test Accuracy: 90.21%
#    Test Loss: 0.2561
#    F1-Score macro: 90.20%
#    Config: Architecture 1 + LR 3e-05
#    Durée: 15min 11s

# BERT-small-arch1-lr5e-05
#    Test Accuracy: 90.18%
#    Test Loss: 0.2572
#    F1-Score macro: 90.17%
#    Config: Architecture 1 + LR 5e-05
#    Durée: 15min 15s

# BERT-small-arch0-lr5e-05
#    Test Accuracy: 90.07%
#    Test Loss: 0.2492 (meilleur loss)
#    F1-Score macro: 90.06%
#    Config: Architecture 0 + LR 5e-05
#    Durée: 11min 45s

if __name__ == "__main__":
    main()
