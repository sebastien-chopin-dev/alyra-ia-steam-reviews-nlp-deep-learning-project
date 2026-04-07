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
#     "PLAY_TIME_FOREVER": 30,  # minutes of playtime before writing the review
#     "MIN_WORD_COUNT": 10,  # trade-off to keep sentiment signal (too short = unreliable)
#     "MAX_WORD_COUNT": 400,  # keep complete reviews, avoid truncation (tokenizer max size constraint)
#     "WEIGHTED_VOTE_SCORE": 0.5,  # filter by Steam quality score (minimize bot-written reviews)
#     "STEAM_PURCHASE": True,  # legitimate player with verified purchase
#     "RECEIVED_FOR_FREE": False,  # not received for free (biases the vote)
#     "CLEAN_HTML_TAGS": True,  # strip HTML tags from reviews (low sentiment signal value)
#     "REMOVE_ASCCI_ART_REVIEWS": True,  # remove reviews composed mostly of ASCII art
# }

EXTRACT_LANG_PASS = False
PROCESS_REVIEWS_PASS = True


def main():
    # Download latest version
    path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    print("Path to dataset files:", path)

    # Create en reviews extraction from large file
    config_extract_fr = {
        "LANG": "french",
        "OUTPUT_LANG_REVIEW_FILE_NAME": "fr_all_reviews.csv",
        "MIN_WEIGHTED_SCORE": 0.0,
    }
    if EXTRACT_LANG_PASS is True:
        try:
            en_review_file = extract_lang(path, config_extract_fr)
        except Exception as e:

            print(f"\nError when create extraction review fr file. {e}")
            return None

        print(f"fr extracted file created: {en_review_file}")

    # Filter and Process french reviews to csv file
    config_reviews_fr = {
        "PROCESS_NAME": "Create filtered preprocessed fr reviews file",
        "INPUT_FILE_NAME": "fr_all_reviews.csv",
        "EXPORT_FILE_NAME": "reviews_fr_processed_prod.csv",
        "PLAY_TIME_FOREVER": 30,
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
            process_reviews(path, config_reviews_fr)
        except Exception as e:
            print(f"\nError when process review fr. {e}")
            return None

    # Check new created files
    print("\n" + "=" * 80)
    print("Check new kaggle cache files created")
    print("=" * 80)
    list_files_recursively(path)

    preset_model_name = "bert_base_multi"

    bert_model_config = {
        "NAME_TRAIN_CONFIG": "Complete pipeline fr for prod",
        "SAVE_FOLDER": "complete_fr_for_prod",
        "PHASE_NAME": "complete_fr_prod",
        "SEED": 42,
        "REVIEWS_DATA_FILE": config_reviews_fr["EXPORT_FILE_NAME"],
        "MODEL_PRESET_NAME": preset_model_name,
        "PREPROCESSOR_PRESET_NAME": preset_model_name,
        "LEARNING_RATE": 3e-5,
        "LAYER_ARCHITECTURE": 1,
        "CALLBACK_OPTION": 1,
        "REVIEWS_SUBSET": -1,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,  # high value, early stopping will trigger
        "SEQUENCE_LENGTH": 128,
        "USE_TF_DATASET": False,
        "PLT_COLOR": "green",
    }

    try:
        # Train the model
        start_time = datetime.now()

        model_finetuned_path = train_bert_base_model(bert_model_config)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Store results
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
            f"\nError during complete pipeline run: {e} - config {bert_model_config}"
        )

    # outputs_dir = get_outputs_path()
    # csv_path = f"{outputs_dir}/evaluation_results_{bert_model_config['PHASE_NAME']}.csv"
    # df = pd.read_csv(csv_path)


if __name__ == "__main__":
    main()
