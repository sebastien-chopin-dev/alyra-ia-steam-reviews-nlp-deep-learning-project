from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    run_multiple_combinaison,
)


def main(run_index=-1):
    # quick_test_presets = [
    #     "bert_small_en_uncased",
    # ]
    # # Test all combo LR + layer architectures + callback patience
    learning_rate_list = [2e-5, 3e-5, 5e-5]
    layer_architecture_list = [0, 1, 2]
    subset_size = 50000
    callback_s_list = [0, 1, 2]  # 0 - 1 - 2 (plus rapide au plus patient)

    finetune_model_config = {
        "NAME_TRAIN_CONFIG": "Hyperparameter Search phase 1",
        "PHASE_NAME": "en_phase1",
        "SEED": 42,
        "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
        "REVIEWS_SUBSET": subset_size,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,  # pour être sur car early stopping
        "SEQUENCE_LENGTH": 128,
        "USE_TF_DATASET": False,  # opti batch memory
        "PLT_COLOR": "green",
    }

    combinations = []

    # BERT small fast finetuning - (27 combo)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            for cs in callback_s_list:
                combinations.append(("bert_small_en_uncased", lr, arch, cs))

    run_multiple_combinaison(
        "en_phase1",
        combinations,
        base_config=finetune_model_config,
        run_index=run_index,
        subset_size=subset_size,
    )


if __name__ == "__main__":
    main()

# BERT-small-arch1-lr5e-05
#    Accuracy: 88.22% | Val Accuracy: 88.22% | Loss: 0.3168
#    Config: Architecture 1 + Learning Rate 5e-05 + Callback Option 0

# BERT-small-arch0-lr3e-05
#    Accuracy: 87.78% | Val Accuracy: 87.78% | Loss: 0.3124
#    Config: Architecture 0 + Learning Rate 3e-05 + Callback Option 0

# BERT-small-arch0-lr5e-05
#    Accuracy: 87.76% | Val Accuracy: 87.76% | Loss: 0.3065
#    Config: Architecture 0 + Learning Rate 5e-05 + Callback Option 0

# Observations :
# Learning Rate 5e-05 domine (5 fois dans le top 6)
# Architecture 0 (la plus simple) est très compétitive
# Architecture 1 donne le meilleur pic de performance
# Architectures 2 & 3 (plus complexes) n'apportent pas de gain
