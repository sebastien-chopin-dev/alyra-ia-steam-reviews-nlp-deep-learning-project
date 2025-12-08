import kagglehub

from src.process_reviews_dataset.extract_reviews_lang_from_large_dataset import (
    extract_lang,
)


def create_file():
    # Download latest version
    path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    # Create filtered fr review file
    config_lang = {
        "LANG": "french",
        "OUTPUT_LANG_REVIEW_FILE_NAME": "fr_all_reviews.csv",
        "MIN_WEIGHTED_SCORE": 0.0,
    }
    fr_all_review_file = extract_lang(path, config_lang)
    print(f"Fichier créé: {fr_all_review_file}")


if __name__ == "__main__":
    create_file()
