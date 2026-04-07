"""
Full processing pipeline for French and English Steam reviews

This script:
1. Loads intermediate review files (fr and en)
2. Applies cleaning filters
3. Balances classes (50/50 positive/negative)
4. Exports processed files for fine-tuning
"""

import os
import re
import pandas as pd

from src.process_reviews_dataset.utils import (
    print_voted_up_count_proportion,
)


def clean_review_with_all_processing(df: pd.DataFrame, config):
    df_clean = df.copy()

    # Filter by weighted_vote_score

    df_clean = df_clean[
        (df_clean["weighted_vote_score"] > config["WEIGHTED_VOTE_SCORE"])
    ]

    want_steam_purchase = 1
    if config["STEAM_PURCHASE"] is False:
        want_steam_purchase = 0

    want_received_for_free = 0
    if config["RECEIVED_FOR_FREE"] is True:
        want_received_for_free = 1

    # Filter for legitimate player reviews
    df_clean = df_clean[
        (df_clean["steam_purchase"] == want_steam_purchase)  # Verified purchase
        & (
            df_clean["received_for_free"] == want_received_for_free
        )  # Not received for free (biases the vote)
        & (
            df_clean["author_playtime_forever"] > config["PLAY_TIME_FOREVER"]
        )  # At least 30 minutes of playtime
    ]

    # Clean html tags
    if config["CLEAN_HTML_TAGS"] is True:
        df_clean["review"] = df_clean["review"].apply(clean_tags)

    # Filter ASCII Art reviews
    if config["REMOVE_ASCCI_ART_REVIEWS"] is True:
        df_clean = df_clean[df_clean["review"].apply(has_enough_letters)]

    # Filter by min and max size
    df_clean = df_clean[
        (df_clean["word_count"] >= config["MIN_WORD_COUNT"])
        & (df_clean["word_count"] <= config["MAX_WORD_COUNT"])
    ]

    # Remove duplicate reviews
    df_clean = df_clean.drop_duplicates(subset=["review"])
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


# clean html tags
def clean_tags(text: str):
    if not isinstance(text, str):
        return text
    text = re.sub(r"\[/?[a-zA-Z0-9]+[^\]]*\]", "", text)
    text = re.sub(r"</?[a-zA-Z0-9]+[^>]*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# clean ascii art
def has_enough_letters(text: str, min_ratio=0.5):

    if not isinstance(text, str) or len(text) == 0:
        return False

    letters = sum(c.isalpha() for c in text)
    return (letters / len(text)) >= min_ratio


# Class balancing by undersampling
def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    # Count each class
    count_0 = (df["voted_up"] == 0).sum()
    count_1 = (df["voted_up"] == 1).sum()

    # Undersample majority class to match minority
    df_class_0 = df[df["voted_up"] == 0]
    df_class_1 = df[df["voted_up"] == 1].sample(n=count_0, random_state=42)

    # Combine and shuffle
    df_balanced = (
        pd.concat([df_class_0, df_class_1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return df_balanced


def process_reviews(path, config):
    """
    Process reviews: filter, clean, balance, and export
    """
    print("\n" + "=" * 80)
    print(f"PROCESS: {config["PROCESS_NAME"]}")
    print("=" * 80)

    # Load data
    print(f"\nLoading file {config["INPUT_FILE_NAME"]}...")
    df_reviews_lang = pd.read_csv(
        os.path.join(path, config["INPUT_FILE_NAME"]), low_memory=True
    )

    print(f"Number of reviews loaded: {len(df_reviews_lang)}")
    print("\nData preview:")
    print(df_reviews_lang.head(3))

    # Initial statistics
    print("\nInitial statistics")
    print_voted_up_count_proportion(df_reviews_lang)

    bins = [0, 10, 20, 50, 100, 300, 512, float("inf")]
    labels = ["<10", "10-20", "20-50", "50-100", "100-300", "300-512", ">512"]
    df_reviews_lang["tranche"] = pd.cut(
        df_reviews_lang["word_count"], bins=bins, labels=labels
    )

    print("\nAll word count ranges:")
    print(df_reviews_lang["tranche"].value_counts().sort_index())

    # Cleaning
    print("\nApplying cleaning filters")
    df_reviews_lang_cleaned = clean_review_with_all_processing(df_reviews_lang, config)

    print(f"Before: {len(df_reviews_lang)}")
    print(f"After:  {len(df_reviews_lang_cleaned)}")

    print_voted_up_count_proportion(df_reviews_lang_cleaned)

    # Check a few positive reviews
    print("\nPositive review examples")
    for i, review in enumerate(
        df_reviews_lang_cleaned[
            (df_reviews_lang_cleaned["voted_up"] == 1)
            & (df_reviews_lang_cleaned["word_count"] <= 30)
        ]["review"].head(5),
        1,
    ):
        print(f"--- Review {i} ---")
        print(review)
        print()

    # Check a few negative reviews
    print("\nNegative review examples")
    for i, review in enumerate(
        df_reviews_lang_cleaned[
            (df_reviews_lang_cleaned["voted_up"] == 0)
            & (df_reviews_lang_cleaned["word_count"] <= 30)
        ]["review"].head(5),
        1,
    ):
        print(f"--- Review {i} ---")
        print(review)
        print()

    # Class balancing
    print("\nBalancing classes")
    df_balanced = balance_classes(df_reviews_lang_cleaned)

    print(f"\nBalanced dataset: {len(df_balanced)} reviews (50/50)")
    print(df_balanced["voted_up"].value_counts())

    # Export
    output_file = os.path.join(path, config["EXPORT_FILE_NAME"])
    keep_columns = ["voted_up", "review", "weighted_vote_score"]

    df_balanced[keep_columns].to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nCleaned reviews exported to: {output_file}")

    return df_balanced
