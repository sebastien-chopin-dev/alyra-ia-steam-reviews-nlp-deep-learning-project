import os
import pandas as pd
import kagglehub


def create_en_weighted_score_above_06_file(path: str):
    # Langues à garder
    languages = ["english"]

    # 100 millions de lignes, on prend des chunks de 500 milles lignes
    chunk_size = 500000
    input_file = os.path.join(path, "all_reviews", "all_reviews.csv")
    output_file = os.path.join(path, "en_weighted_score_above_06.csv")

    # Traiter par chunks
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):

        # Calculer le nombre de mots
        chunk["word_count"] = chunk["review"].str.split().str.len()
        # Filtrer les lignes reviews non vides et avec un weighted_vote_score > 0.6
        filtered = chunk[
            (chunk["language"].isin(languages))
            & (chunk["word_count"] > 0)
            & (chunk["weighted_vote_score"] > 0.6)
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


def create_file():
    # Download latest version
    path = kagglehub.dataset_download("kieranpoc/steam-reviews")
    # Create filtered review file
    en_review_file = create_en_weighted_score_above_06_file(path)
    print(f"Fichier créé: {en_review_file}")


if __name__ == "__main__":
    create_file()
