"""
Script de traitement complet des reviews Steam françaises et anglaises

Ce script :
1. Charge les fichiers de reviews intermédiaires (fr et en)
2. Applique les filtres de nettoyage
3. Équilibre les classes (50/50 positif/négatif)
4. Exporte les fichiers traités pour le fine-tuning
"""

import os
import re
import pandas as pd

from src.process_reviews_dataset.utils import (
    print_voted_up_count_proportion,
)


def clean_review_with_all_processing(df: pd.DataFrame, config):
    df_clean = df.copy()

    # filtre weighted_vote_score

    df_clean = df_clean[
        (df_clean["weighted_vote_score"] > config["WEIGHTED_VOTE_SCORE"])
    ]

    want_steam_purchase = 1
    if config["STEAM_PURCHASE"] is False:
        want_steam_purchase = 0

    want_received_for_free = 0
    if config["RECEIVED_FOR_FREE"] is True:
        want_received_for_free = 1

    # Filtre les reviews des joueurs légitimes
    df_clean = df_clean[
        (df_clean["steam_purchase"] == want_steam_purchase)  # Achat vérifié
        & (
            df_clean["received_for_free"] == want_received_for_free
        )  # Pas reçu gratuitement (ça biaise le vote)
        & (
            df_clean["author_playtime_forever"] > config["PLAY_TIME_FOREVER"]
        )  # Au moins 30 minutes de jeu
    ]

    # Clean html tags
    if config["CLEAN_HTML_TAGS"] is True:
        df_clean["review"] = df_clean["review"].apply(clean_tags)

    # Filtre les reviews ASCII Art
    if config["REMOVE_ASCCI_ART_REVIEWS"] is True:
        df_clean = df_clean[df_clean["review"].apply(has_enough_letters)]

    # filtre size min et max
    df_clean = df_clean[
        (df_clean["word_count"] >= config["MIN_WORD_COUNT"])
        & (df_clean["word_count"] <= config["MAX_WORD_COUNT"])
    ]

    # Supprimer reviews en double
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


# équilibrage des classes par sous échantillonnage
def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    # Compter chaque classe
    count_0 = (df["voted_up"] == 0).sum()
    count_1 = (df["voted_up"] == 1).sum()

    # Sous-échantillonner la classe majoritaire pour égaler la minoritaire
    df_class_0 = df[df["voted_up"] == 0]
    df_class_1 = df[df["voted_up"] == 1].sample(n=count_0, random_state=42)

    # Combiner et mélanger
    df_balanced = (
        pd.concat([df_class_0, df_class_1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return df_balanced


def process_reviews(path, config):
    """
    Traitement des reviews
    """
    print("\n" + "=" * 80)
    print(f"PROCESS: {config["PROCESS_NAME"]}")
    print("=" * 80)

    # Chargement des données
    print(f"\nChargement du fichier {config["INPUT_FILE_NAME"]}...")
    df_reviews_lang = pd.read_csv(
        os.path.join(path, config["INPUT_FILE_NAME"]), low_memory=True
    )

    print(f"Nombre de reviews chargées : {len(df_reviews_lang)}")
    print("\nAperçu des données :")
    print(df_reviews_lang.head(3))

    # Statistiques initiales
    print("\nStatistiques initiales")
    print_voted_up_count_proportion(df_reviews_lang)

    bins = [0, 10, 20, 50, 100, 300, 512, float("inf")]
    labels = ["<10", "10-20", "20-50", "50-100", "100-300", "300-512", ">512"]
    df_reviews_lang["tranche"] = pd.cut(
        df_reviews_lang["word_count"], bins=bins, labels=labels
    )

    print("\nAffichage de toute les tranches de longueurs:")
    print(df_reviews_lang["tranche"].value_counts().sort_index())

    # Nettoyage
    print("\nApplication des filtres de nettoyage")
    df_reviews_lang_cleaned = clean_review_with_all_processing(df_reviews_lang, config)

    print(f"Avant : {len(df_reviews_lang)}")
    print(f"Après : {len(df_reviews_lang_cleaned)}")

    print_voted_up_count_proportion(df_reviews_lang_cleaned)

    # Vérification de quelques reviews positives
    print("\nExemples de reviews positives")
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

    # Vérification de quelques reviews négatives
    print("\nExemples de reviews négatives")
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

    # Équilibrage des classes
    print("\nEquilibrage des classes")
    df_balanced = balance_classes(df_reviews_lang_cleaned)

    print(f"\nDataset équilibré : {len(df_balanced)} reviews (50/50)")
    print(df_balanced["voted_up"].value_counts())

    # Export
    output_file = os.path.join(path, config["EXPORT_FILE_NAME"])
    keep_columns = ["voted_up", "review", "weighted_vote_score"]

    df_balanced[keep_columns].to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nFichier des reviews nettoyées exporté vers : {output_file}")

    return df_balanced
