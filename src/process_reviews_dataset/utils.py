"""
Utility functions for Steam reviews processing
"""

import re
import pandas as pd


def column_summary(df: pd.DataFrame):
    summary = []
    for col in df.columns:
        col_type = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()

        # Handle the case where the column contains lists (unhashable)
        try:
            unique_count = df[col].nunique()
        except TypeError:
            # If error (lists), convert to string temporarily
            unique_count = df[col].astype(str).nunique()
            print(
                f"⚠️ Column '{col}' contains non-hashable types (probably lists)"
            )

        summary.append(
            {
                "Column": col,
                "Type": str(col_type),
                "Non-Null Count": non_null,
                "Null Count": null_count,
                "Unique Values": unique_count,
            }
        )

    # Display column summary
    print("=" * 80)
    print("Detailed column summary:")
    print("=" * 80)
    column_summary_df = pd.DataFrame(summary)
    print(column_summary_df.to_string(index=False))
    print("\n")


def print_voted_up_count_proportion(df: pd.DataFrame):
    voted_up_counts = df["voted_up"].value_counts()
    total_reviews = len(df)

    print("voted_up (reviews positive:1 / negative:0)")
    for voted_up_value, count in voted_up_counts.items():
        proportion = (count / total_reviews) * 100
        print(
            f"voted_up = {voted_up_value}: Count = {count}, Proportion = {proportion:.2f}%"
        )

    print(
        "\nTo get a balanced 50/50 distribution by undersampling the majority class:\n"
    )
    print(
        f"We would have {min(voted_up_counts)} reviews per class, for a total of {2 * min(voted_up_counts)} reviews."
    )
