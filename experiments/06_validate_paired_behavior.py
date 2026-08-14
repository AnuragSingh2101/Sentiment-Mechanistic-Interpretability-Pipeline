"""
Phase 4.5 — Validate Clean/Corrupted Paired Behavior.

Purpose
-------
Validate that changing only the sentiment-bearing word in each
minimal-contrast pair produces a measurable behavioral change in
GPT-2-small.

For each pair we calculate:

    LD_clean
    LD_corrupted
    behavioral_gap = LD_clean - LD_corrupted

where:

    LD = logit(" positive") - logit(" negative")

Expected behavior:

    clean positive sentence
        -> higher LD

    corrupted negative sentence
        -> lower LD

This script establishes the behavioral baseline required before
activation patching.

No activation patching is performed here.
"""

from __future__ import annotations

import csv
import gc
import statistics
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "gpt2"

POSITIVE_TOKEN = " positive"
NEGATIVE_TOKEN = " negative"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentiment_pairs.csv"
)


# ---------------------------------------------------------------------
# Token utility
# ---------------------------------------------------------------------


def get_single_token_id(
    bridge: TransformerBridge,
    text: str,
) -> int:
    """
    Convert a token string into exactly one token ID.
    """

    token_ids = bridge.to_tokens(
        text,
        prepend_bos=False,
    )

    if token_ids.ndim == 2:
        token_ids = token_ids[0]

    if len(token_ids) != 1:
        raise ValueError(
            f"{text!r} does not tokenize to exactly one token.\n"
            f"Token IDs: {token_ids.tolist()}"
        )

    return int(token_ids[0].item())


# ---------------------------------------------------------------------
# Logit difference
# ---------------------------------------------------------------------


def calculate_logit_difference(
    bridge: TransformerBridge,
    text: str,
    positive_id: int,
    negative_id: int,
) -> tuple[float, float, float]:
    """
    Calculate positive logit, negative logit, and their difference.
    """

    prompt = (
        f"Review: {text}\n"
        "Sentiment:"
    )

    tokens = bridge.to_tokens(
        prompt,
        prepend_bos=True,
    )

    tokens = tokens.to(bridge.cfg.device)

    with torch.no_grad():

        logits = bridge(tokens)

    next_token_logits = logits[0, -1]

    positive_logit = float(
        next_token_logits[positive_id].item()
    )

    negative_logit = float(
        next_token_logits[negative_id].item()
    )

    logit_difference = (
        positive_logit - negative_logit
    )

    del tokens
    del logits

    return (
        positive_logit,
        negative_logit,
        logit_difference,
    )


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------


def load_dataset() -> list[dict[str, str]]:
    """
    Load the paired sentiment CSV.
    """

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Run:\n"
            "python experiments/05_create_paired_dataset.py"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    required_columns = {
        "pair_id",
        "clean_text",
        "corrupted_text",
        "clean_label",
        "corrupted_label",
    }

    missing_columns = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(rows) == 0:

        raise ValueError(
            "Dataset contains zero rows."
        )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 70)
    print("PHASE 4.5 — CLEAN/CORRUPTED BEHAVIOR VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    rows = load_dataset()

    print(
        f"\nDataset: {DATASET_PATH}"
    )

    print(
        f"Pairs loaded: {len(rows)}"
    )

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL LOADING")
    print("=" * 70)

    bridge = TransformerBridge.boot_transformers(
        MODEL_NAME,
        device=device,
    )

    bridge.eval()

    print(
        "GPT-2-small loaded successfully."
    )

    # ---------------------------------------------------------------
    # Validate sentiment tokens
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SENTIMENT TOKEN VALIDATION")
    print("=" * 70)

    positive_id = get_single_token_id(
        bridge,
        POSITIVE_TOKEN,
    )

    negative_id = get_single_token_id(
        bridge,
        NEGATIVE_TOKEN,
    )

    print(
        f"{POSITIVE_TOKEN!r} → token ID {positive_id}"
    )

    print(
        f"{NEGATIVE_TOKEN!r} → token ID {negative_id}"
    )

    # ---------------------------------------------------------------
    # Evaluate pairs
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIRED BEHAVIOR EVALUATION")
    print("=" * 70)

    results = []

    clean_correct = 0
    corrupted_correct = 0

    expected_direction_count = 0

    clean_logit_differences = []
    corrupted_logit_differences = []
    behavioral_gaps = []

    for row in rows:

        pair_id = int(row["pair_id"])
        clean_text = row["clean_text"]
        corrupted_text = row["corrupted_text"]
        clean_label = row["clean_label"]
        corrupted_label = row["corrupted_label"]

        # Clean
        (
            clean_positive_logit,
            clean_negative_logit,
            clean_ld,
        ) = calculate_logit_difference(
            bridge,
            clean_text,
            positive_id,
            negative_id,
        )

        # -----------------------------------------------------------
        # Corrupted
        # -----------------------------------------------------------

        (
            corrupted_positive_logit,
            corrupted_negative_logit,
            corrupted_ld,
        ) = calculate_logit_difference(
            bridge,
            corrupted_text,
            positive_id,
            negative_id,
        )

        # -----------------------------------------------------------
        # Predictions
        # -----------------------------------------------------------

        clean_prediction = (
            "positive"
            if clean_ld > 0
            else "negative"
        )

        corrupted_prediction = (
            "positive"
            if corrupted_ld > 0
            else "negative"
        )

        clean_is_correct = (
            clean_prediction == clean_label
        )

        corrupted_is_correct = (
            corrupted_prediction == corrupted_label
        )

        if clean_is_correct:
            clean_correct += 1

        if corrupted_is_correct:
            corrupted_correct += 1

        # -----------------------------------------------------------
        # Behavioral gap
        # -----------------------------------------------------------

        behavioral_gap = (
            clean_ld - corrupted_ld
        )

        clean_logit_differences.append(
            clean_ld
        )

        corrupted_logit_differences.append(
            corrupted_ld
        )

        behavioral_gaps.append(
            behavioral_gap
        )

        # -----------------------------------------------------------
        # Expected direction
        # -----------------------------------------------------------
        #
        # Since clean is positive and corrupted is negative,
        # we expect:
        #
        #     clean LD > corrupted LD
        #
        # -----------------------------------------------------------

        expected_direction = (
            clean_ld > corrupted_ld
        )

        if expected_direction:
            expected_direction_count += 1

        # -----------------------------------------------------------
        # Store result
        # -----------------------------------------------------------

        result = {
            "pair_id": int(pair_id),
            "clean_text": clean_text,
            "corrupted_text": corrupted_text,
            "clean_label": clean_label,
            "corrupted_label": corrupted_label,
            "clean_positive_logit": clean_positive_logit,
            "clean_negative_logit": clean_negative_logit,
            "clean_logit_difference": clean_ld,
            "clean_prediction": clean_prediction,
            "clean_correct": clean_is_correct,
            "corrupted_positive_logit": corrupted_positive_logit,
            "corrupted_negative_logit": corrupted_negative_logit,
            "corrupted_logit_difference": corrupted_ld,
            "corrupted_prediction": corrupted_prediction,
            "corrupted_correct": corrupted_is_correct,
            "behavioral_gap": behavioral_gap,
            "expected_direction": expected_direction,
        }

        results.append(result)

        # -----------------------------------------------------------
        # Print
        # -----------------------------------------------------------

        print(
            f"{pair_id:02d}. "
            f"clean LD={clean_ld:+.4f} | "
            f"corrupted LD={corrupted_ld:+.4f} | "
            f"gap={behavioral_gap:+.4f} | "
            f"clean={clean_prediction:>8} "
            f"({'✓' if clean_is_correct else '✗'}) | "
            f"corrupted={corrupted_prediction:>8} "
            f"({'✓' if corrupted_is_correct else '✗'})"
        )

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------

    total_pairs = len(results)

    clean_accuracy = (
        clean_correct / total_pairs
    )

    corrupted_accuracy = (
        corrupted_correct / total_pairs
    )

    expected_direction_rate = (
        expected_direction_count
        / total_pairs
    )

    # ---------------------------------------------------------------
    # Mean metrics
    # ---------------------------------------------------------------

    mean_clean_ld = statistics.mean(
        clean_logit_differences
    )

    mean_corrupted_ld = statistics.mean(
        corrupted_logit_differences
    )

    mean_behavioral_gap = statistics.mean(
        behavioral_gaps
    )

    median_behavioral_gap = statistics.median(
        behavioral_gaps
    )

    positive_gap_count = sum(
        gap > 0
        for gap in behavioral_gaps
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEHAVIORAL RESULTS")
    print("=" * 70)

    print(
        f"Pairs: {total_pairs}"
    )

    print(
        f"\nClean accuracy: "
        f"{clean_correct}/{total_pairs} "
        f"({clean_accuracy * 100:.2f}%)"
    )

    print(
        f"Corrupted accuracy: "
        f"{corrupted_correct}/{total_pairs} "
        f"({corrupted_accuracy * 100:.2f}%)"
    )

    print(
        f"\nMean clean LD: "
        f"{mean_clean_ld:+.4f}"
    )

    print(
        f"Mean corrupted LD: "
        f"{mean_corrupted_ld:+.4f}"
    )

    print(
        f"Mean behavioral gap: "
        f"{mean_behavioral_gap:+.4f}"
    )

    print(
        f"Median behavioral gap: "
        f"{median_behavioral_gap:+.4f}"
    )

    print(
        f"Pairs with clean LD > corrupted LD: "
        f"{expected_direction_count}/{total_pairs} "
        f"({expected_direction_rate * 100:.2f}%)"
    )

    print(
        f"Pairs with positive behavioral gap: "
        f"{positive_gap_count}/{total_pairs} "
        f"({positive_gap_count / total_pairs * 100:.2f}%)"
    )

    # ---------------------------------------------------------------
    # Identify weak pairs
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("WEAKEST BEHAVIORAL PAIRS")
    print("=" * 70)

    weakest = sorted(
        results,
        key=lambda result: result["behavioral_gap"],
    )[:5]

    for result in weakest:

        print(
            f"\nPair {result['pair_id']}"
        )

        print(
            f"  CLEAN:     "
            f"{result['clean_text']}"
        )

        print(
            f"  CORRUPTED: "
            f"{result['corrupted_text']}"
        )

        print(
            f"  Clean LD: "
            f"{result['clean_logit_difference']:+.4f}"
        )

        print(
            f"  Corrupted LD: "
            f"{result['corrupted_logit_difference']:+.4f}"
        )

        print(
            f"  Gap: "
            f"{result['behavioral_gap']:+.4f}"
        )

    # ---------------------------------------------------------------
    # Strongest behavioral pairs
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRONGEST BEHAVIORAL PAIRS")
    print("=" * 70)

    strongest = sorted(
        results,
        key=lambda result: result["behavioral_gap"],
        reverse=True,
    )[:5]

    for result in strongest:

        print(
            f"\nPair {result['pair_id']}"
        )

        print(
            f"  CLEAN:     "
            f"{result['clean_text']}"
        )

        print(
            f"  CORRUPTED: "
            f"{result['corrupted_text']}"
        )

        print(
            f"  Clean LD: "
            f"{result['clean_logit_difference']:+.4f}"
        )

        print(
            f"  Corrupted LD: "
            f"{result['corrupted_logit_difference']:+.4f}"
        )

        print(
            f"  Gap: "
            f"{result['behavioral_gap']:+.4f}"
        )

    # ---------------------------------------------------------------
    # Scientific gate
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEHAVIORAL GATE")
    print("=" * 70)

    if expected_direction_rate >= 0.70:

        print(
            "PASS: At least 70% of pairs show the "
            "expected behavioral direction."
        )

        print(
            "The dataset provides a usable behavioral "
            "contrast for activation patching."
        )

    else:

        print(
            "WARNING: Fewer than 70% of pairs show "
            "the expected behavioral direction."
        )

        print(
            "The dataset may need refinement before "
            "activation patching."
        )

    # ---------------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------------

    if device == "cuda":

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print("\n" + "=" * 70)
        print("GPU MEMORY")
        print("=" * 70)

        print(
            f"Allocated: {allocated:.2f} GB"
        )

        print(
            f"Reserved:  {reserved:.2f} GB"
        )

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    del bridge

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4.5 COMPLETE")
    print("=" * 70)

    print(
        "\nNo activation patching was performed."
    )

    print(
        "This phase only establishes whether the "
        "clean/corrupted dataset produces a measurable "
        "behavioral contrast."
    )


if __name__ == "__main__":
    main()