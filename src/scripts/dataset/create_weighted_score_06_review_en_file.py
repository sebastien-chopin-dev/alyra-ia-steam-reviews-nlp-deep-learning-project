import kagglehub

from src.process_reviews_dataset.extract_reviews_lang_from_large_dataset import (
    extract_lang,
)


def create_file():
    # Download latest version
    path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    # Create filtered review file
    config_lang = {
        "LANG": "english",
        "OUTPUT_LANG_REVIEW_FILE_NAME": "en_weighted_score_above_06.csv",
        "MIN_WEIGHTED_SCORE": 0.6,
    }

    en_review_file = extract_lang(path, config_lang)
    print(f"Created file: {en_review_file}")


if __name__ == "__main__":
    create_file()
