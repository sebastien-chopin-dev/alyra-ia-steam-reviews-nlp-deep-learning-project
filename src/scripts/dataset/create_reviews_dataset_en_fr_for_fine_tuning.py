import kagglehub

from src.process_reviews_dataset.process_reviews_en_fr import (
    process_reviews,
)
from src.utils.file_system_utils import list_files_recursively

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


def main():
    # Download latest version
    path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    print("Path to dataset files:", path)

    # Filter and Process english reviews to csv file
    config_reviews_en = {
        "PROCESS_NAME": "Create filtered preprocessed english reviews file",
        "INPUT_FILE_NAME": "en_weighted_score_above_06.csv",
        "EXPORT_FILE_NAME": "reviews_en_processed.csv",
        "PLAY_TIME_FOREVER": 30,  # minutes of playtime before writing the review
        "MIN_WORD_COUNT": 10,
        "MAX_WORD_COUNT": 400,
        "WEIGHTED_VOTE_SCORE": 0.5,
        "STEAM_PURCHASE": True,
        "RECEIVED_FOR_FREE": False,
        "CLEAN_HTML_TAGS": True,
        "REMOVE_ASCCI_ART_REVIEWS": True,
    }

    try:
        process_reviews(path, config_reviews_en)
    except Exception as e:
        print(f"\nError while preprocessing English review file. {e}")

    # Filter and Process french reviews to csv file
    config_reviews_fr = {
        "PROCESS_NAME": "Create filtered preprocessed fr reviews file",
        "INPUT_FILE_NAME": "fr_all_reviews.csv",
        "EXPORT_FILE_NAME": "reviews_fr_processed.csv",
        "PLAY_TIME_FOREVER": 30,
        "MIN_WORD_COUNT": 10,
        "MAX_WORD_COUNT": 400,
        "WEIGHTED_VOTE_SCORE": 0.5,
        "STEAM_PURCHASE": True,
        "RECEIVED_FOR_FREE": False,
        "CLEAN_HTML_TAGS": True,
        "REMOVE_ASCCI_ART_REVIEWS": True,
    }

    try:
        process_reviews(path, config_reviews_fr)
    except Exception as e:
        print(f"\nError when process review fr. {e}")

    # Check created files
    print("\n" + "=" * 80)
    print("Check new kaggle cache files created")
    print("=" * 80)
    list_files_recursively(path)


if __name__ == "__main__":
    main()
