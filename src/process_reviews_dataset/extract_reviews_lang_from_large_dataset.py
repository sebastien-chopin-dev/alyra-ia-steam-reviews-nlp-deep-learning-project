import os
import pandas as pd
import kagglehub


def extract_lang(kaggle_cache_path: str, config):
    # Languages to keep
    languages = [config["LANG"]]

    # 100 million rows, processing in chunks of 500k rows
    chunk_size = 500000
    input_file = os.path.join(kaggle_cache_path, "all_reviews", "all_reviews.csv")
    output_file = os.path.join(
        kaggle_cache_path, config["OUTPUT_LANG_REVIEW_FILE_NAME"]
    )

    # Traiter par chunks
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):

        # Compute word count
        chunk["word_count"] = chunk["review"].str.split().str.len()
        # Filter rows by language and non-empty reviews
        filtered = chunk[
            (chunk["language"].isin(languages))
            & (chunk["word_count"] > 0)
            & (chunk["weighted_vote_score"] > config["MIN_WEIGHTED_SCORE"])
        ]

        # Write to output file
        if first_chunk:
            filtered.to_csv(output_file, index=False, encoding="utf-8")
            first_chunk = False
        else:
            filtered.to_csv(
                output_file, mode="a", index=False, header=False, encoding="utf-8"
            )

        print(f"Processed {len(chunk)} rows, kept {len(filtered)}")

    return output_file
