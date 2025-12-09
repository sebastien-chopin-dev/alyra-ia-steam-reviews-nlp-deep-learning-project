from datetime import datetime
from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    run_multiple_combinaison,
    train_bert_base_model,
)


def run_quick_test_to_find_best_hyperparameters(run_index=-1):
    # quick_test_presets = [
    #     "bert_small_en_uncased",
    # ]
    # # Tester toutes les combinaisons LR + architectures
    learning_rate_list = [2e-5, 3e-5, 5e-5]
    layer_architecture_list = [0, 1, 2, 3]
    subset_size = 50000
    callback_s = 0  # 0 - 1 - 2 (plus rapide au plus patient)

    combinations = []

    # BERT TEST RAPIDE - (12 combo)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            combinations.append(("bert_small_en_uncased", lr, arch, callback_s))

    run_multiple_combinaison(
        "phase1", combinations, run_index=run_index, subset_size=subset_size
    )


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


def run_quick_test_to_find_best_hyperparameters_phase2(run_index=-1):
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


def run_test_to_find_best_hyperparameters_phase3(run_index=-1):
    # quick_test_presets = [
    #     "distil_bert_base_en_uncased",
    #     "bert_base_en_uncased"
    # ]
    # # Tester toutes les combinaisons LR + architectures
    learning_rate_list = [3e-5]
    layer_architecture_list = [1]
    subset_size = -1  # Toute les reviews
    callback_s = 2  # 0 - 1 - 2 - 3 (plus rapide au plus patient)

    combinations = []

    # BERT VALIDATION HYPER PARAMETRES - (1 combo)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            combinations.append(("bert_base_en_uncased", lr, arch, callback_s))

    run_multiple_combinaison(
        "en_phase3", combinations, run_index=run_index, subset_size=subset_size
    )


def run_test_to_find_best_hyperparameters_final(run_index=-1):
    # quick_test_presets = [
    #     "roberta_base_en",
    #     "deberta_v3_small_en",
    # ]
    # # Tester toutes les combinaisons LR + architectures
    learning_rate_list = [3e-5]
    layer_architecture_list = [1]
    subset_size = 100000  # Erreur allocation mémoire
    callback_s = 2  # 0 - 1 - 2 - 3 (plus rapide au plus patient)

    combinations = []

    # BERT VALIDATION HYPER PARAMETRES - (1 combo)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            combinations.append(("deberta_v3_small_en", lr, arch, callback_s))

    bert_base_model_config = {
        "NAME_TRAIN_CONFIG": "Hyperparameter Search",
        "PHASE_NAME": "en_final",
        "SEED": 42,
        "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
        "SAVE_FOLDER": "hyperparam_search",
        "REVIEWS_SUBSET": subset_size,
        "BATCH_SIZE": 16,
        "EPOCHS": 10,  # pour être sur car early stopping
        "SEQUENCE_LENGTH": 128,
        "USE_TF_DATASET": True,
        "PLT_COLOR": "green",
    }

    run_multiple_combinaison(
        "en_final",
        combinations,
        base_config=bert_base_model_config,
        run_index=run_index,
        subset_size=subset_size,
    )


if __name__ == "__main__":

    # validation_presets = [
    #     "distil_bert_base_en_uncased",
    #     "bert_base_en_uncased",
    # ]
    # # Tester avec meilleurs LR + architectures de Phase 1

    # final_presets = [
    #     "roberta_base_en",
    #     "deberta_v3_small_en",
    # ]
    # Utiliser la meilleure config de Phase 2

    # run_quick_test_to_find_best_hyperparameters()
    run_test_to_find_best_hyperparameters_final()

    # evaluation_path_result = os.path.join(get_outputs_path(), "evaluation_results.csv")
    # compare_evaluation_results(evaluation_path_result)
