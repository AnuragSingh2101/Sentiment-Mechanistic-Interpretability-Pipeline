"""
Phase 4 — Create and Validate Paired Sentiment Dataset.

Purpose
-------
Create a controlled clean/corrupted sentiment benchmark for the
activation-patching experiment.

Each pair differs primarily in the sentiment-bearing word.

Example:

    CLEAN:
        The movie was excellent.

    CORRUPTED:
        The movie was terrible.

The goal is to create minimal contrasts so that later activation
differences are easier to interpret causally.

Dataset:
    20 positive/negative pairs
    40 total examples

This script does NOT:
    - load the model
    - perform activation patching
    - rank attention heads

It only creates and validates the dataset.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = OUTPUT_DIR / "sentiment_pairs.csv"


# ---------------------------------------------------------------------
# Paired minimal-contrast examples
# ---------------------------------------------------------------------
#
# The wording is intentionally kept nearly identical within each pair.
# Only the sentiment-bearing expression changes.
#
# All clean examples are positive.
# All corrupted examples are negative.
# ---------------------------------------------------------------------

PAIRS = [
    (
        "The movie was excellent.",
        "The movie was terrible.",
    ),
    (
        "The film was wonderful.",
        "The film was awful.",
    ),
    (
        "The acting was amazing.",
        "The acting was horrible.",
    ),
    (
        "The story was brilliant.",
        "The story was dreadful.",
    ),
    (
        "The ending was satisfying.",
        "The ending was disappointing.",
    ),
    (
        "The performance was outstanding.",
        "The performance was terrible.",
    ),
    (
        "The director was talented.",
        "The director was incompetent.",
    ),
    (
        "The soundtrack was beautiful.",
        "The soundtrack was unpleasant.",
    ),
    (
        "The characters were engaging.",
        "The characters were boring.",
    ),
    (
        "The dialogue was clever.",
        "The dialogue was stupid.",
    ),
    (
        "The screenplay was impressive.",
        "The screenplay was disappointing.",
    ),
    (
        "The cinematography was stunning.",
        "The cinematography was ugly.",
    ),
    (
        "The pacing was perfect.",
        "The pacing was awful.",
    ),
    (
        "The acting was convincing.",
        "The acting was unconvincing.",
    ),
    (
        "The plot was fascinating.",
        "The plot was tedious.",
    ),
    (
        "The film was enjoyable.",
        "The film was unpleasant.",
    ),
    (
        "The experience was memorable.",
        "The experience was forgettable.",
    ),
    (
        "The movie was entertaining.",
        "The movie was boring.",
    ),
    (
        "The production was polished.",
        "The production was sloppy.",
    ),
    (
        "The final scene was fantastic.",
        "The final scene was horrible.",
    ),
]


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def normalize(text: str) -> str:
    """
    Normalize text for comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def word_difference(
    clean_text: str,
    corrupted_text: str,
) -> list[tuple[str, str]]:
    """
    Return word-level differences between the two texts.

    This is intentionally simple because the pairs are short and
    manually controlled.
    """

    clean_words = normalize(clean_text).split()
    corrupted_words = normalize(corrupted_text).split()

    differences = []

    max_length = max(
        len(clean_words),
        len(corrupted_words),
    )

    for index in range(max_length):

        clean_word = (
            clean_words[index]
            if index < len(clean_words)
            else "<missing>"
        )

        corrupted_word = (
            corrupted_words[index]
            if index < len(corrupted_words)
            else "<missing>"
        )

        if clean_word != corrupted_word:

            differences.append(
                (
                    clean_word,
                    corrupted_word,
                )
            )

    return differences


def validate_pair(
    pair_id: int,
    clean_text: str,
    corrupted_text: str,
) -> bool:
    """
    Validate a single clean/corrupted pair.
    """

    differences = word_difference(
        clean_text,
        corrupted_text,
    )

    print(
        f"\nPair {pair_id}:"
    )

    print(
        f"  CLEAN:     {clean_text}"
    )

    print(
        f"  CORRUPTED: {corrupted_text}"
    )

    print(
        f"  Word differences: {len(differences)}"
    )

    for clean_word, corrupted_word in differences:

        print(
            f"    {clean_word!r} → {corrupted_word!r}"
        )

    return len(differences) > 0


# ---------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------


def create_dataset() -> None:

    print("=" * 70)
    print("PHASE 4 — CREATE PAIRED SENTIMENT DATASET")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Validate number of pairs
    # ---------------------------------------------------------------

    expected_pairs = 20

    if len(PAIRS) != expected_pairs:

        raise RuntimeError(
            f"Expected {expected_pairs} pairs, "
            f"found {len(PAIRS)}."
        )

    print(
        f"\nPairs: {len(PAIRS)}"
    )

    print(
        f"Total examples: {len(PAIRS) * 2}"
    )

    # ---------------------------------------------------------------
    # Validate every pair
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIR VALIDATION")
    print("=" * 70)

    valid_pairs = 0

    for pair_id, (
        clean_text,
        corrupted_text,
    ) in enumerate(
        PAIRS,
        start=1,
    ):

        if validate_pair(
            pair_id,
            clean_text,
            corrupted_text,
        ):

            valid_pairs += 1

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("WRITING DATASET")
    print("=" * 70)

    rows = []

    for pair_id, (
        clean_text,
        corrupted_text,
    ) in enumerate(
        PAIRS,
        start=1,
    ):

        rows.append(
            {
                "pair_id": pair_id,
                "clean_text": clean_text,
                "corrupted_text": corrupted_text,
                "clean_label": "positive",
                "corrupted_label": "negative",
            }
        )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "clean_text",
                "corrupted_text",
                "clean_label",
                "corrupted_label",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    print(
        f"\nValid pairs: "
        f"{valid_pairs}/{len(PAIRS)}"
    )

    if valid_pairs != len(PAIRS):

        raise RuntimeError(
            "One or more dataset pairs failed validation."
        )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ---------------------------------------------------------------
    # Dataset summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Pairs:              {len(rows)}"
    )

    print(
        f"Clean examples:     {len(rows)}"
    )

    print(
        f"Corrupted examples: {len(rows)}"
    )

    print(
        "Clean label:        positive"
    )

    print(
        "Corrupted label:    negative"
    )

    # ---------------------------------------------------------------
    # Show examples
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE PAIRS")
    print("=" * 70)

    for row in rows[:5]:

        print(
            f"\nPair {row['pair_id']}"
        )

        print(
            f"  CLEAN:     {row['clean_text']}"
        )

        print(
            f"  CORRUPTED: {row['corrupted_text']}"
        )

    # ---------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4 DATASET CREATION: PASS")
    print("=" * 70)

    print(
        "\nThe controlled paired dataset is ready "
        "for behavioral validation."
    )


if __name__ == "__main__":
    create_dataset()