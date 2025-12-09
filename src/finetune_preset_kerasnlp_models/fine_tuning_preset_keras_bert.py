# Transfer Learning et Fine-Tuning avec BERT : Classification de Sentiments
# Utilise les fichiers Steam review anglais pour la classification de sentiments (positif/négatif).
# Backend: TensorFlow avec API KerasNLP

import warnings
import os
import time
from datetime import datetime, timedelta

from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_helpers import (
    compile_and_train_model,
    create_train_test_eval_split,
    evaluate_model,
    instantiate_keras_preset_model_finetuned,
    load_and_verif_reviews_datas,
    keras_preset_preprocess_data,
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

# bert_base_model_config_example = {
#     "NAME_TRAIN_CONFIG": "BERT Fine-Tuned keras base",
#     "SEED": 42,
#     "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
#     "SAVE_FOLDER": "bert_base_en_poc",
#     "REVIEWS_SUBSET": 20000,
#     "PREPROCESSOR_PRESET_NAME": "bert_base_en_uncased",
#     "MODEL_PRESET_NAME": "bert_tiny_en_uncased",
#     "SEQUENCE_LENGTH": 128,
#     "BATCH_SIZE": 32,
#     "EPOCHS": 10,
#     "LEARNING_RATE": 3e-5,
#     "CALLBACK_OPTION": 1,
#     "LAYER_ARCHITECTURE": 0,
#     "USE_TF_DATASET": False,
#     "PLT_COLOR": "green",
# }


def train_bert_base_model(config: dict):

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

    X_train_bert, X_val_bert, X_test_bert = keras_preset_preprocess_data(
        X_train, X_val, X_test, config
    )

    # bert_small_en_uncased
    bert_finetuned_model = instantiate_keras_preset_model_finetuned(config)

    print("\nArchitecture du modèle:")
    bert_finetuned_model.summary()

    history_bert_finetuned = compile_and_train_model(
        bert_finetuned_model,
        X_train_bert,
        y_train,
        X_val_bert,
        y_val,
        config,
    )

    # Create folder path if not exist
    if not os.path.exists(config["SAVE_FOLDER"]):
        os.makedirs(config["SAVE_FOLDER"])

    # Visualisation des courbes d'apprentissage
    models_histories = [
        (config["NAME_TRAIN_CONFIG"], history_bert_finetuned, config["PLT_COLOR"])
    ]

    show_model_train_history(
        models_histories, f"{config['SAVE_FOLDER']}/train_history.png"
    )

    # Evaluation finale
    test_loss, test_accuracy, report_dict, cm = evaluate_model(
        bert_finetuned_model, X_test_bert, y_test, config
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
    bert_finetuned_model.save(model_path)
    print(f"Modèle sauvegardé: {model_path}")


def run_multiple_combinaison(
    phase_name: str,
    combinations,
    base_config=None,
    run_index=-1,
    subset_size=-1,
):

    if base_config is not None:
        bert_base_model_config = base_config
    else:
        bert_base_model_config = {
            "NAME_TRAIN_CONFIG": "Hyperparameter Search",
            "PHASE_NAME": phase_name,
            "SEED": 42,
            "REVIEWS_DATA_FILE": "reviews_en_processed.csv",
            "SAVE_FOLDER": "hyperparam_search",
            "REVIEWS_SUBSET": subset_size,
            "BATCH_SIZE": 32,
            "EPOCHS": 10,  # pour être sur car early stopping
            "SEQUENCE_LENGTH": 128,
            "USE_TF_DATASET": False,
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
                "NAME_TRAIN_CONFIG": f"{variant}-arch{arch}-lr{lr:.0e}-call{callback_s}-on-en-{subset_size}",
                "SAVE_FOLDER": f"{variant}_arch{arch}_lr{lr:.0e}_call{callback_s}_on_en_{subset_size}",
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
