from datetime import datetime
import kagglehub
import pandas as pd

from src.finetune_preset_kerasnlp_models.fine_tuning_preset_keras_bert import (
    train_bert_base_model,
)
from src.process_reviews_dataset.extract_reviews_lang_from_large_dataset import (
    extract_lang,
)
from src.process_reviews_dataset.process_reviews_en_fr import (
    process_reviews,
)
from src.utils.file_system_utils import get_outputs_path, list_files_recursively
from src.utils.stats_utils import show_stats_compare_train_evaluation

# config_extract_en = {
#     "LANG": "english",
#     "OUTPUT_LANG_REVIEW_FILE_NAME": "en_weighted_score_above_06.csv",
#     "MIN_WEIGHTED_SCORE": 0.6,
# }

# process_reviews_config_example = {
#     "PLAY_TIME_FOREVER": 30,  # minutes passées sur le jeux avant d'écrire la review
#     "MIN_WORD_COUNT": 10,  # compromis pour ne pas perdre le signal du sentiment si trop court
#     "MAX_WORD_COUNT": 400,  # essayer de garder des reviews completes non tronqués (contrainte taille max tokenizer)
#     "WEIGHTED_VOTE_SCORE": 0.5,  # on filtre les reviews avec un score de qualité donné par steam (minimiser celles écritent par des bots)
#     "STEAM_PURCHASE": True,  # Joueur légitime achat vérifié
#     "RECEIVED_FOR_FREE": False,  # Pas reçu gratuitement (ça biaise le vote)
#     "CLEAN_HTML_TAGS": True,  # On nettois la reviews des tags html (valeur faible pour le signal de sentiment)
#     "REMOVE_ASCCI_ART_REVIEWS": True,  # On supprime les reviews composé essentiellement d'ascii art
# }

EXTRACT_LANG_PASS = False
PROCESS_REVIEWS_PASS = False


def main():
    # Download latest version
    path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    print("Path to dataset files:", path)

    # Create en reviews extraction from large file
    config_extract_en = {
        "LANG": "english",
        "OUTPUT_LANG_REVIEW_FILE_NAME": "en_weighted_score_above_06.csv",
        "MIN_WEIGHTED_SCORE": 0.6,
    }
    if EXTRACT_LANG_PASS is True:
        try:
            en_review_file = extract_lang(path, config_extract_en)
        except Exception as e:

            print(f"\nError when create extraction review en file. {e}")
            return None

        print(f"En extracted file created: {en_review_file}")

    # Filter and Process english reviews to csv file
    config_reviews_en = {
        "PROCESS_NAME": "Create filtered preprocessed english reviews file",
        "INPUT_FILE_NAME": config_extract_en["OUTPUT_LANG_REVIEW_FILE_NAME"],
        "EXPORT_FILE_NAME": "complete_reviews_en_processed.csv",
        "PLAY_TIME_FOREVER": 30,  # minutes passées sur le jeu avant d'écrire la review
        "MIN_WORD_COUNT": 10,
        "MAX_WORD_COUNT": 400,
        "WEIGHTED_VOTE_SCORE": 0.5,
        "STEAM_PURCHASE": True,
        "RECEIVED_FOR_FREE": False,
        "CLEAN_HTML_TAGS": True,
        "REMOVE_ASCCI_ART_REVIEWS": True,
    }

    if PROCESS_REVIEWS_PASS is True:
        try:
            process_reviews(path, config_reviews_en)
        except Exception as e:
            print(f"\nError when process review en. {e}")
            return None

    # Check new created files
    print("\n" + "=" * 80)
    print("Check new kaggle cache files created")
    print("=" * 80)
    list_files_recursively(path)

    preset_model_name = "distil_bert_base_en_uncased"

    bert_model_config = {
        "NAME_TRAIN_CONFIG": "Complete pipeline en for prod",
        "SAVE_FOLDER": "complete_en_for_prod",
        "PHASE_NAME": "complete_en_prod",
        "SEED": 42,
        "REVIEWS_DATA_FILE": config_reviews_en["EXPORT_FILE_NAME"],
        "MODEL_PRESET_NAME": preset_model_name,
        "PREPROCESSOR_PRESET_NAME": preset_model_name,
        "LEARNING_RATE": 3e-5,
        "LAYER_ARCHITECTURE": 1,
        "CALLBACK_OPTION": 2,
        "REVIEWS_SUBSET": -1,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,  # pour être sur car early stopping
        "SEQUENCE_LENGTH": 128,
        "USE_TF_DATASET": False,
        "PLT_COLOR": "green",
    }

    try:
        # Entraîner le modèle
        start_time = datetime.now()

        model_finetuned_path = train_bert_base_model(bert_model_config)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Stocker les résultats
        result = {
            "run_number": bert_model_config["NAME_TRAIN_CONFIG"],
            "variant": preset_model_name,
            "learning_rate": bert_model_config["LEARNING_RATE"],
            "architecture": bert_model_config["LAYER_ARCHITECTURE"],
            "training_duration": duration,
            "status": "success",
        }

        print(f"\nRun complete pipeline ended - {result}")

    except Exception as e:
        print(
            f"\nErreur durant le run complete pipeline: {e} - confg {bert_model_config}"
        )

    # outputs_dir = get_outputs_path()
    # csv_path = f"{outputs_dir}/evaluation_results_{bert_model_config['PHASE_NAME']}.csv"
    # df = pd.read_csv(csv_path)


if __name__ == "__main__":
    main()
