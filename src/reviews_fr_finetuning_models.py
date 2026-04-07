from datetime import datetime
import os
from pdb import run

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typer.cli import callback
from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    train_bert_base_model,
)
from src.utils.file_system_utils import get_outputs_path
from src.utils.stats_utils import compare_evaluation_results

import keras_nlp


def run_multiple_combinaison(
    phase_name: str, combinations, run_index=-1, subset_size=-1
):
    # bert_base_model_config = {
    #     "NAME_TRAIN_CONFIG": "Finetune BERT tiny en-l1-r2e5",
    #     "SEED": 42,
    #     "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
    #     "SAVE_FOLDER": "bert_tiny_en_arch_1_rate_2e5",
    #     "REVIEWS_SUBSET": -1,
    #     "PREPROCESSOR_PRESET_NAME": "bert_base_en_uncased",
    #     "MODEL_PRESET_NAME": "bert_tiny_en_uncased",
    #     "SEQUENCE_LENGTH": 128,
    #     "BATCH_SIZE": 32,
    #     "EPOCHS": 15,
    #     "LEARNING_RATE": 2e-5,
    #     "LAYER_ARCHITECTURE": 1,
    #     "PLT_COLOR": "green",
    # }

    # Layer architecture
    # 0 Simplest
    #     x = layers.Dropout(0.1)(x)
    # 1 Architecture from training with small demo dataset
    #     x = layers.Dropout(0.3)(cls_token)
    #     x = layers.Dense(64, activation="relu")(x)
    #     x = layers.Dropout(0.3)(x)
    # 2 More balanced architecture
    #     x = layers.Dropout(0.2)(x)
    #     x = layers.Dense(128, activation="relu")(x)
    #     x = layers.Dropout(0.2)(x)
    # 3 More complex architecture
    #     x = layers.Dropout(0.3)(x)
    #     x = layers.Dense(256, activation="relu")(x)
    #     x = layers.Dropout(0.3)(x)
    #     x = layers.Dense(64, activation="relu")(x)
    #     x = layers.Dropout(0.3)(x)

    bert_base_model_config = {
        "NAME_TRAIN_CONFIG": "Hyperparameter Search",
        "PHASE_NAME": phase_name,
        "SEED": 42,
        "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
        "SAVE_FOLDER": "hyperparam_search",
        "REVIEWS_SUBSET": subset_size,
        "BATCH_SIZE": 32,
        "EPOCHS": 15,  # high value, early stopping will trigger
        "SEQUENCE_LENGTH": 128,
        "PLT_COLOR": "green",
    }

    print("\nTraining plan:")
    for i, (variant, lr, arch, callback_s) in enumerate(combinations, 1):

        if run_index != -1 and i < run_index:
            continue

        print(
            f"   {i:2d}. {variant:25s} | LR: {lr:.0e} | Arch: {arch} | Callback: {callback_s}"
        )

    for i, (variant, lr, arch, callback_s) in enumerate(combinations, 1):

        if run_index != -1 and i < run_index:
            continue

        print(f"\n{'='*80}")
        print(f"Training {i}/{len(combinations)}")
        print(f"   Variant: {variant}")
        print(f"   Learning Rate: {lr}")
        print(f"   Architecture: {arch}")
        print(f"   Callback strategy: {callback_s}")
        print(f"{'='*80}")

        # Create config for this run
        config = bert_base_model_config.copy()
        config.update(
            {
                "NAME_TRAIN_CONFIG": f"BERT-{variant.split('_')[1]}-arch{arch}-lr{lr:.0e}",
                "SAVE_FOLDER": f"{variant}_arch{arch}_lr{lr:.0e}call{callback_s}_on_en_{subset_size}",
                "MODEL_PRESET_NAME": variant,
                "PREPROCESSOR_PRESET_NAME": variant,
                "LEARNING_RATE": lr,
                "LAYER_ARCHITECTURE": arch,
                "CALLBACK_OPTION": callback_s,
            }
        )

        try:
            # Train the model
            start_time = datetime.now()

            train_bert_base_model(config)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Store results
            result = {
                "run_number": i,
                "variant": variant,
                "learning_rate": lr,
                "architecture": arch,
                "training_duration": duration,
                "status": "success",
            }

            print(f"\nRun {i} done - {result}")

        except Exception as e:
            print(f"\nError during run {i}: {e} - config {config}")

        # Show progress
        print(
            f"\nProgress: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)"
        )


def run_quick_test_to_find_best_hyperparameters(run_index=-1):
    # quick_test_presets = [
    #     "bert_small_en_uncased",
    # ]
    # # Test all LR + architecture combinations
    learning_rate_list = [2e-5, 3e-5, 5e-5]
    layer_architecture_list = [0, 1, 2, 3]
    subset_size = 50000
    callback_s = 0  # 0 - 1 - 2 (fastest to most patient)

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

# Observations:
# Learning Rate 5e-05 dominates (5 times in top 6)
# Architecture 0 (simplest) is very competitive
# Architecture 1 yields the best peak performance
# Architectures 2 & 3 (more complex) bring no improvement


def run_quick_test_to_find_best_hyperparameters_phase2(run_index=-1):
    # quick_test_presets = [
    #     "bert_small_en_uncased",
    # ]
    # # Test all LR + architecture combinations
    learning_rate_list = [2e-5, 3e-5, 5e-5]
    layer_architecture_list = [0, 1]
    subset_size = -1  # All reviews
    callback_s = 1  # 0 - 1 - 2 (fastest to most patient)

    combinations = []

    # BERT HYPERPARAMETER VALIDATION - (6 combos)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            combinations.append(("bert_small_en_uncased", lr, arch, callback_s))

    run_multiple_combinaison(
        "phase2", combinations, run_index=run_index, subset_size=subset_size
    )


# BERT-small-arch0-lr5e-05
#    Test Accuracy: 90.11%
#    Test Loss: 0.2491
#    F1-Score (macro): 90.11%
#    Config: Architecture 0 (simple) + LR 5e-05 + Callback option 0
#    Duration: 9min 14s

# BERT-small-arch1-lr3e-05
#    Test Accuracy: 90.20%
#    Test Loss: 0.2565
#    F1-Score (macro): 90.19%
#    Config: Architecture 1 + LR 3e-05 + Callback option 0
#    Duration: 12min 21s


def run_test_to_find_best_hyperparameters_phase3(run_index=-1):
    # quick_test_presets = [
    #     "distil_bert_base_en_uncased",
    #     "bert_base_en_uncased"
    # ]
    # # Test all LR + architecture combinations
    learning_rate_list = [5e-5]
    layer_architecture_list = [0, 1]
    subset_size = -1  # All reviews
    callback_s = 1  # 0 - 1 - 2 (fastest to most patient)

    combinations = []

    # BERT HYPERPARAMETER VALIDATION - (6 combos)
    for arch in layer_architecture_list:
        for lr in learning_rate_list:
            combinations.append(("bert_small_en_uncased", lr, arch, callback_s))

    run_multiple_combinaison(
        "phase2", combinations, run_index=run_index, subset_size=subset_size
    )


if __name__ == "__main__":

    # final_presets = [
    #     "bert_base_multi",
    #     "xlm_roberta_base_multi",
    # ]

    run_quick_test_to_find_best_hyperparameters()
    # run_quick_test_to_find_best_hyperparameters_phase2(run_index=6)

    # evaluation_path_result = os.path.join(get_outputs_path(), "evaluation_results.csv")
    # compare_evaluation_results(evaluation_path_result)
