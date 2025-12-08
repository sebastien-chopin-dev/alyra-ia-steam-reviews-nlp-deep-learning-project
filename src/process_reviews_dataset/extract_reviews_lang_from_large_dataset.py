import os
import pandas as pd
import kagglehub


def extract_lang(kaggle_cache_path: str, config):
    # Langues à garder
    languages = [config["LANG"]]

    # 100 millions de lignes, on prend des chunks de 500 milles lignes
    chunk_size = 500000
    input_file = os.path.join(kaggle_cache_path, "all_reviews", "all_reviews.csv")
    output_file = os.path.join(
        kaggle_cache_path, config["OUTPUT_LANG_REVIEW_FILE_NAME"]
    )

    # Traiter par chunks
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):

        # Calculer le nombre de mots
        chunk["word_count"] = chunk["review"].str.split().str.len()
        # Filtrer les lignes en fonction de la langue et avec des reviews non vides
        filtered = chunk[
            (chunk["language"].isin(languages))
            & (chunk["word_count"] > 0)
            & (chunk["weighted_vote_score"] > config["MIN_WEIGHTED_SCORE"])
        ]

        # Écrire dans le fichier de sortie
        if first_chunk:
            filtered.to_csv(output_file, index=False, encoding="utf-8")
            first_chunk = False
        else:
            filtered.to_csv(
                output_file, mode="a", index=False, header=False, encoding="utf-8"
            )

        print(f"Traité {len(chunk)} lignes, gardé {len(filtered)}")

    return output_file
