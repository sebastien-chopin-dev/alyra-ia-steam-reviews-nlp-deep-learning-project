from datetime import datetime
import os

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typer.cli import callback
from src.finetune_models.fine_tuning_preset_keras_bert import train_bert_base_model
from src.utils.file_system_utils import get_outputs_path
from src.utils.stats_utils import compare_evaluation_results


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
    # 0 Le plus simple
    #     x = layers.Dropout(0.1)(x)
    # 1 Architecture vu pendant la formation avec petit dataset de démo
    #     x = layers.Dropout(0.3)(cls_token)
    #     x = layers.Dense(64, activation="relu")(x)
    #     x = layers.Dropout(0.3)(x)
    # 2 Architecture plus équilibré
    #     x = layers.Dropout(0.2)(x)
    #     x = layers.Dense(128, activation="relu")(x)
    #     x = layers.Dropout(0.2)(x)
    # 3 Architecture plus complexe
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
        "EPOCHS": 15,  # pour être sur car early stopping
        "SEQUENCE_LENGTH": 128,
        "PLT_COLOR": "green",
    }

    print("\nPlan d'entraînement:")
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
        print(f"Entraînement {i}/{len(combinations)}")
        print(f"   Variant: {variant}")
        print(f"   Learning Rate: {lr}")
        print(f"   Architecture: {arch}")
        print(f"   Callback strategy: {callback_s}")
        print(f"{'='*80}")

        # Créer la config pour ce run
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
            # Entraîner le modèle
            start_time = datetime.now()

            train_bert_base_model(config)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Stocker les résultats
            result = {
                "run_number": i,
                "variant": variant,
                "learning_rate": lr,
                "architecture": arch,
                "training_duration": duration,
                "status": "success",
            }

            print(f"\nRun {i} terminé - {result}")

        except Exception as e:
            print(f"\nErreur durant le run {i}: {e} - confg {config}")

        # Afficher progression
        print(
            f"\nProgression: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)"
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
    run_quick_test_to_find_best_hyperparameters_phase2()

    # evaluation_path_result = os.path.join(get_outputs_path(), "evaluation_results.csv")
    # compare_evaluation_results(evaluation_path_result)
