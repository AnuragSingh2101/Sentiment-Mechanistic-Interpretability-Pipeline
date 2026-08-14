"""
PHASE 11 — EXPANDED DATASET BEHAVIORAL VALIDATION

Purpose
-------
Validate that the expanded 100-pair sentiment dataset produces a
measurable clean/corrupted behavioral contrast before expensive
activation-patching experiments are performed.

Dataset
-------
100 total paired examples:

    60 discovery pairs
    40 held-out pairs

Each pair contains:

    clean_text      -> positive sentiment
    corrupted_text  -> negative sentiment

Primary metric
--------------
logit_difference =
    logit(" positive") - logit(" negative")

Behavioral gap
--------------
clean_logit_difference - corrupted_logit_difference

This script does NOT:
    - perform activation patching
    - select candidate heads
    - modify model weights
    - use the held-out set for head selection

It only validates the behavioral signal.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from transformer_lens.model_bridge import TransformerBridge


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

SPLITS_DIR = DATA_DIR / "splits"

RESULTS_DIR = PROJECT_ROOT / "results" / "expanded_behavior"

FIGURES_DIR = RESULTS_DIR / "figures"

DISCOVERY_PATH = (
    SPLITS_DIR / "discovery_pairs.csv"
)

HELDOUT_PATH = (
    SPLITS_DIR / "heldout_pairs.csv"
)

ALL_DATASET_PATH = (
    DATA_DIR
    / "raw"
    / "expanded_sentiment_pairs.csv"
)

PAIR_RESULTS_PATH = (
    RESULTS_DIR
    / "expanded_behavior_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "expanded_behavior_summary.json"
)

DOMAIN_RESULTS_PATH = (
    RESULTS_DIR
    / "domain_behavior_results.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "gpt2"

EXPECTED_TOTAL_PAIRS = 100

EXPECTED_DISCOVERY_PAIRS = 60

EXPECTED_HELDOUT_PAIRS = 40

EXPECTED_LAYERS = 12

EXPECTED_HEADS = 12

POSITIVE_TOKEN = " positive"

NEGATIVE_TOKEN = " negative"


# ============================================================================
# PRINTING
# ============================================================================


def section(title: str) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# DEVICE
# ============================================================================


def get_device() -> torch.device:

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================================
# MODEL
# ============================================================================


def load_model(
    device: torch.device,
):

    section("MODEL LOADING")

    print(
        f"Loading model: {MODEL_NAME}"
    )

    bridge = TransformerBridge.boot_transformers(
        MODEL_NAME,
        device=device,
    )

    bridge.eval()

    print(
        "GPT-2-small loaded successfully."
    )

    return bridge


# ============================================================================
# MODEL VALIDATION
# ============================================================================


def validate_model(
    model,
):

    section("MODEL VALIDATION")

    if model.cfg.n_layers != EXPECTED_LAYERS:

        raise ValueError(
            "Unexpected number of layers: "
            f"{model.cfg.n_layers}"
        )

    if model.cfg.n_heads != EXPECTED_HEADS:

        raise ValueError(
            "Unexpected number of attention heads: "
            f"{model.cfg.n_heads}"
        )

    print(
        f"Layers:       {model.cfg.n_layers}"
    )

    print(
        f"Heads/layer:  {model.cfg.n_heads}"
    )

    print(
        f"Hidden size:  {model.cfg.d_model}"
    )

    print(
        f"Vocabulary:   {model.cfg.d_vocab}"
    )

    print(
        f"Context size: {model.cfg.n_ctx}"
    )

    print(
        "Architecture validation: PASS"
    )


# ============================================================================
# TOKEN VALIDATION
# ============================================================================


def validate_sentiment_tokens(
    model,
):

    section("SENTIMENT TOKEN VALIDATION")

    positive_tokens = model.to_tokens(
        POSITIVE_TOKEN,
        prepend_bos=False,
    )

    negative_tokens = model.to_tokens(
        NEGATIVE_TOKEN,
        prepend_bos=False,
    )

    if positive_tokens.numel() != 1:

        raise ValueError(
            f"{POSITIVE_TOKEN!r} must map to "
            "exactly one token."
        )

    if negative_tokens.numel() != 1:

        raise ValueError(
            f"{NEGATIVE_TOKEN!r} must map to "
            "exactly one token."
        )

    positive_id = int(
        positive_tokens.flatten()[0].item()
    )

    negative_id = int(
        negative_tokens.flatten()[0].item()
    )

    print(
        f"{POSITIVE_TOKEN!r} → token ID "
        f"{positive_id}"
    )

    print(
        f"{NEGATIVE_TOKEN!r} → token ID "
        f"{negative_id}"
    )

    if positive_id == negative_id:

        raise ValueError(
            "Positive and negative sentiment "
            "tokens are identical."
        )

    print(
        "Sentiment token validation: PASS"
    )

    return positive_id, negative_id


# ============================================================================
# CSV LOADING
# ============================================================================


REQUIRED_COLUMNS = {
    "pair_id",
    "domain",
    "clean_text",
    "corrupted_text",
    "clean_label",
    "corrupted_label",
    "split",
}


def load_pairs(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            raise ValueError(
                f"No CSV header found in {path}"
            )

        missing = (
            REQUIRED_COLUMNS
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                f"Missing columns in {path}: "
                f"{sorted(missing)}"
            )

        rows = list(reader)

    return rows


# ============================================================================
# DATASET VALIDATION
# ============================================================================


def validate_datasets(
    all_rows,
    discovery_rows,
    heldout_rows,
):

    section("DATASET VALIDATION")

    print(
        f"All dataset pairs:      {len(all_rows)}"
    )

    print(
        f"Discovery pairs:        {len(discovery_rows)}"
    )

    print(
        f"Held-out pairs:        {len(heldout_rows)}"
    )

    if len(all_rows) != EXPECTED_TOTAL_PAIRS:

        raise ValueError(
            f"Expected {EXPECTED_TOTAL_PAIRS} "
            f"total pairs, got {len(all_rows)}."
        )

    if len(discovery_rows) != EXPECTED_DISCOVERY_PAIRS:

        raise ValueError(
            f"Expected {EXPECTED_DISCOVERY_PAIRS} "
            f"discovery pairs, got "
            f"{len(discovery_rows)}."
        )

    if len(heldout_rows) != EXPECTED_HELDOUT_PAIRS:

        raise ValueError(
            f"Expected {EXPECTED_HELDOUT_PAIRS} "
            f"held-out pairs, got "
            f"{len(heldout_rows)}."
        )

    all_ids = {
        int(row["pair_id"])
        for row in all_rows
    }

    discovery_ids = {
        int(row["pair_id"])
        for row in discovery_rows
    }

    heldout_ids = {
        int(row["pair_id"])
        for row in heldout_rows
    }

    if len(all_ids) != len(all_rows):

        raise ValueError(
            "Duplicate pair IDs found."
        )

    overlap = (
        discovery_ids
        & heldout_ids
    )

    if overlap:

        raise ValueError(
            "Discovery/held-out leakage detected: "
            f"{sorted(overlap)}"
        )

    if (
        discovery_ids
        | heldout_ids
    ) != all_ids:

        raise ValueError(
            "Discovery + held-out sets do not "
            "cover the complete dataset."
        )

    print(
        "Pair ID integrity: PASS"
    )

    print(
        "Discovery/held-out separation: PASS"
    )

    # ------------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------------

    for row in all_rows:

        if row["clean_label"] != "positive":

            raise ValueError(
                f"Unexpected clean label for "
                f"pair {row['pair_id']}: "
                f"{row['clean_label']}"
            )

        if row["corrupted_label"] != "negative":

            raise ValueError(
                f"Unexpected corrupted label for "
                f"pair {row['pair_id']}: "
                f"{row['corrupted_label']}"
            )

    print(
        "Label validation: PASS"
    )

    # ------------------------------------------------------------------------
    # Text integrity
    # ------------------------------------------------------------------------

    clean_texts = set()

    corrupted_texts = set()

    for row in all_rows:

        clean = row["clean_text"].strip()

        corrupted = (
            row["corrupted_text"].strip()
        )

        if not clean:

            raise ValueError(
                f"Empty clean text in pair "
                f"{row['pair_id']}"
            )

        if not corrupted:

            raise ValueError(
                f"Empty corrupted text in pair "
                f"{row['pair_id']}"
            )

        if clean.lower() == corrupted.lower():

            raise ValueError(
                f"Clean/corrupted text identical "
                f"in pair {row['pair_id']}"
            )

        clean_texts.add(
            clean.lower()
        )

        corrupted_texts.add(
            corrupted.lower()
        )

    if len(clean_texts) != len(all_rows):

        raise ValueError(
            "Duplicate clean sentences detected."
        )

    if len(corrupted_texts) != len(all_rows):

        raise ValueError(
            "Duplicate corrupted sentences detected."
        )

    print(
        "Text uniqueness: PASS"
    )

    print(
        "Dataset validation: PASS"
    )


# ============================================================================
# LOGIT DIFFERENCE
# ============================================================================


@torch.no_grad()
def calculate_logit_difference(
    model,
    prompt: str,
    positive_token_id: int,
    negative_token_id: int,
):

    tokens = model.to_tokens(
        prompt,
        prepend_bos=True,
    )

    logits = model(
        tokens
    )

    # Last token position predicts the next token.
    next_token_logits = logits[
        0,
        -1,
    ]

    positive_logit = float(
        next_token_logits[
            positive_token_id
        ].item()
    )

    negative_logit = float(
        next_token_logits[
            negative_token_id
        ].item()
    )

    logit_difference = (
        positive_logit
        - negative_logit
    )

    return (
        logit_difference,
        positive_logit,
        negative_logit,
        int(tokens.shape[1]),
    )


# ============================================================================
# SINGLE PAIR EVALUATION
# ============================================================================


def evaluate_pair(
    model,
    row,
    positive_token_id,
    negative_token_id,
):

    clean_prompt = (
        "Review: "
        + row["clean_text"]
        + "\nSentiment:"
    )

    corrupted_prompt = (
        "Review: "
        + row["corrupted_text"]
        + "\nSentiment:"
    )

    (
        clean_ld,
        clean_positive_logit,
        clean_negative_logit,
        clean_sequence_length,
    ) = calculate_logit_difference(
        model,
        clean_prompt,
        positive_token_id,
        negative_token_id,
    )

    (
        corrupted_ld,
        corrupted_positive_logit,
        corrupted_negative_logit,
        corrupted_sequence_length,
    ) = calculate_logit_difference(
        model,
        corrupted_prompt,
        positive_token_id,
        negative_token_id,
    )

    behavioral_gap = (
        clean_ld
        - corrupted_ld
    )

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

    clean_correct = (
        clean_prediction == "positive"
    )

    corrupted_correct = (
        corrupted_prediction == "negative"
    )

    return {
        "pair_id":
            int(row["pair_id"]),

        "domain":
            row["domain"],

        "split":
            row["split"],

        "clean_text":
            row["clean_text"],

        "corrupted_text":
            row["corrupted_text"],

        "clean_ld":
            clean_ld,

        "corrupted_ld":
            corrupted_ld,

        "behavioral_gap":
            behavioral_gap,

        "clean_positive_logit":
            clean_positive_logit,

        "clean_negative_logit":
            clean_negative_logit,

        "corrupted_positive_logit":
            corrupted_positive_logit,

        "corrupted_negative_logit":
            corrupted_negative_logit,

        "clean_prediction":
            clean_prediction,

        "corrupted_prediction":
            corrupted_prediction,

        "clean_correct":
            clean_correct,

        "corrupted_correct":
            corrupted_correct,

        "clean_sequence_length":
            clean_sequence_length,

        "corrupted_sequence_length":
            corrupted_sequence_length,
    }


# ============================================================================
# WRITE RESULTS CSV
# ============================================================================


def write_results_csv(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"Saved: {path}")


# ============================================================================
# STATISTICS
# ============================================================================


def mean(values):

    if not values:

        return 0.0

    return float(
        statistics.mean(values)
    )


def median(values):

    if not values:

        return 0.0

    return float(
        statistics.median(values)
    )


def summarize_rows(
    rows,
):

    clean_ld = [
        float(row["clean_ld"])
        for row in rows
    ]

    corrupted_ld = [
        float(row["corrupted_ld"])
        for row in rows
    ]

    gaps = [
        float(row["behavioral_gap"])
        for row in rows
    ]

    clean_correct = sum(
        bool(row["clean_correct"])
        for row in rows
    )

    corrupted_correct = sum(
        bool(row["corrupted_correct"])
        for row in rows
    )

    clean_higher = sum(
        gap > 0
        for gap in gaps
    )

    return {
        "pairs":
            len(rows),

        "clean_correct":
            clean_correct,

        "clean_accuracy":
            (
                clean_correct
                / len(rows)
                if rows
                else 0.0
            ),

        "corrupted_correct":
            corrupted_correct,

        "corrupted_accuracy":
            (
                corrupted_correct
                / len(rows)
                if rows
                else 0.0
            ),

        "mean_clean_ld":
            mean(clean_ld),

        "median_clean_ld":
            median(clean_ld),

        "mean_corrupted_ld":
            mean(corrupted_ld),

        "median_corrupted_ld":
            median(corrupted_ld),

        "mean_behavioral_gap":
            mean(gaps),

        "median_behavioral_gap":
            median(gaps),

        "positive_behavioral_gap_pairs":
            clean_higher,

        "positive_behavioral_gap_fraction":
            (
                clean_higher
                / len(rows)
                if rows
                else 0.0
            ),
    }


# ============================================================================
# DOMAIN ANALYSIS
# ============================================================================


def analyze_domains(
    rows,
):

    grouped = defaultdict(list)

    for row in rows:

        grouped[
            row["domain"]
        ].append(row)

    results = []

    section("DOMAIN-LEVEL BEHAVIOR")

    for domain in sorted(
        grouped
    ):

        summary = summarize_rows(
            grouped[domain]
        )

        result = {
            "domain":
                domain,
            **summary,
        }

        results.append(
            result
        )

        print(
            f"{domain:<15} "
            f"pairs={summary['pairs']:3d} | "
            f"clean_acc="
            f"{summary['clean_accuracy'] * 100:6.2f}% | "
            f"corrupted_acc="
            f"{summary['corrupted_accuracy'] * 100:6.2f}% | "
            f"mean_gap="
            f"{summary['mean_behavioral_gap']:+.4f} | "
            f"gap>0="
            f"{summary['positive_behavioral_gap_fraction'] * 100:6.2f}%"
        )

    return results


# ============================================================================
# WEAK / STRONG PAIRS
# ============================================================================


def print_extreme_pairs(
    rows,
):

    sorted_rows = sorted(
        rows,
        key=lambda row:
        float(row["behavioral_gap"]),
    )

    section("WEAKEST BEHAVIORAL PAIRS")

    for row in sorted_rows[:10]:

        print(
            f"Pair {row['pair_id']:03d} "
            f"[{row['split']:<9}] "
            f"{row['domain']:<12} "
            f"gap="
            f"{float(row['behavioral_gap']):+.4f}"
        )

        print(
            f"  CLEAN:     "
            f"{row['clean_text']}"
        )

        print(
            f"  CORRUPTED: "
            f"{row['corrupted_text']}"
        )

    section("STRONGEST BEHAVIORAL PAIRS")

    for row in reversed(
        sorted_rows[-10:]
    ):

        print(
            f"Pair {row['pair_id']:03d} "
            f"[{row['split']:<9}] "
            f"{row['domain']:<12} "
            f"gap="
            f"{float(row['behavioral_gap']):+.4f}"
        )

        print(
            f"  CLEAN:     "
            f"{row['clean_text']}"
        )

        print(
            f"  CORRUPTED: "
            f"{row['corrupted_text']}"
        )


# ============================================================================
# BEHAVIORAL GATE
# ============================================================================


def behavioral_gate(
    all_summary,
    discovery_summary,
    heldout_summary,
):

    section("BEHAVIORAL GATE")

    criteria = {}

    # ------------------------------------------------------------------------
    # Criterion 1
    # ------------------------------------------------------------------------

    criteria[
        "clean_accuracy_above_80_percent"
    ] = (
        all_summary[
            "clean_accuracy"
        ]
        >= 0.80
    )

    # ------------------------------------------------------------------------
    # Criterion 2
    # ------------------------------------------------------------------------

    criteria = {
    "clean_accuracy_above_80_percent":
        all_summary["clean_accuracy"] >= 0.80,

    "mean_behavioral_gap_positive":
        all_summary["mean_behavioral_gap"] > 0,

    "positive_gap_majority":
        all_summary["positive_behavioral_gap_fraction"] >= 0.70,

    "discovery_gap_positive":
        discovery_summary["mean_behavioral_gap"] > 0,

    "heldout_gap_positive":
        heldout_summary["mean_behavioral_gap"] > 0,

    "heldout_positive_gap_majority":
        heldout_summary["positive_behavioral_gap_fraction"] >= 0.60,
    }

    # ------------------------------------------------------------------------
    # Criterion 3
    # ------------------------------------------------------------------------

    criteria[
        "mean_behavioral_gap_positive"
    ] = (
        all_summary[
            "mean_behavioral_gap"
        ]
        > 0
    )

    # ------------------------------------------------------------------------
    # Criterion 4
    # ------------------------------------------------------------------------

    criteria[
        "positive_gap_majority"
    ] = (
        all_summary[
            "positive_behavioral_gap_fraction"
        ]
        >= 0.70
    )

    # ------------------------------------------------------------------------
    # Criterion 5
    # ------------------------------------------------------------------------

    criteria[
        "discovery_gap_positive"
    ] = (
        discovery_summary[
            "mean_behavioral_gap"
        ]
        > 0
    )

    # ------------------------------------------------------------------------
    # Criterion 6
    # ------------------------------------------------------------------------

    criteria[
        "heldout_gap_positive"
    ] = (
        heldout_summary[
            "mean_behavioral_gap"
        ]
        > 0
    )

    # ------------------------------------------------------------------------
    # Criterion 7
    # ------------------------------------------------------------------------

    criteria[
        "heldout_positive_gap_majority"
    ] = (
        heldout_summary[
            "positive_behavioral_gap_fraction"
        ]
        >= 0.60
    )

    for name, passed in criteria.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    passed_count = sum(
        criteria.values()
    )

    gate_passed = (
        passed_count
        == len(criteria)
    )

    print()

    if gate_passed:

        decision = "PASS"

        print(
            "Overall behavioral gate: PASS"
        )

    else:

        decision = "FAIL"

        print(
            "Overall behavioral gate: FAIL"
        )

    return {
        "decision":
            decision,
        "criteria":
            {
                key: bool(value)
                for key, value
                in criteria.items()
            },
        "passed":
            int(passed_count),
        "total":
            int(len(criteria)),
    }


# ============================================================================
# SAVE JSON
# ============================================================================


def save_json(
    path: Path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    print(
        f"Saved: {path}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    section(
        "PHASE 11 — EXPANDED DATASET BEHAVIORAL VALIDATION"
    )

    device = get_device()

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    # ------------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------------

    section("LOADING DATASETS")

    all_rows = load_pairs(
        ALL_DATASET_PATH
    )

    discovery_rows = load_pairs(
        DISCOVERY_PATH
    )

    heldout_rows = load_pairs(
        HELDOUT_PATH
    )

    validate_datasets(
        all_rows,
        discovery_rows,
        heldout_rows,
    )

    # ------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------

    model = load_model(
        device
    )

    validate_model(
        model
    )

    (
        positive_token_id,
        negative_token_id,
    ) = validate_sentiment_tokens(
        model
    )

    # ------------------------------------------------------------------------
    # Behavioral evaluation
    # ------------------------------------------------------------------------

    section(
        "EVALUATING 100 PAIRED EXAMPLES"
    )

    results = []

    total = len(all_rows)

    for index, row in enumerate(
        all_rows,
        start=1,
    ):

        result = evaluate_pair(
            model,
            row,
            positive_token_id,
            negative_token_id,
        )

        results.append(
            result
        )

        clean_symbol = (
            "✓"
            if result["clean_correct"]
            else "✗"
        )

        corrupted_symbol = (
            "✓"
            if result["corrupted_correct"]
            else "✗"
        )

        print(
            f"{index:03d}/{total} "
            f"[{result['split']:<9}] "
            f"{result['domain']:<12} "
            f"clean LD="
            f"{result['clean_ld']:+.4f} | "
            f"corrupted LD="
            f"{result['corrupted_ld']:+.4f} | "
            f"gap="
            f"{result['behavioral_gap']:+.4f} | "
            f"clean={result['clean_prediction']} "
            f"({clean_symbol}) | "
            f"corrupted="
            f"{result['corrupted_prediction']} "
            f"({corrupted_symbol})"
        )

    # ------------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------------

    all_summary = summarize_rows(
        results
    )

    discovery_results = [
        row
        for row in results
        if row["split"] == "discovery"
    ]

    heldout_results = [
        row
        for row in results
        if row["split"] == "heldout"
    ]

    discovery_summary = summarize_rows(
        discovery_results
    )

    heldout_summary = summarize_rows(
        heldout_results
    )

    # ------------------------------------------------------------------------
    # Overall results
    # ------------------------------------------------------------------------

    section("OVERALL BEHAVIORAL RESULTS")

    print(
        f"Pairs: "
        f"{all_summary['pairs']}"
    )

    print(
        f"Clean accuracy: "
        f"{all_summary['clean_correct']}/"
        f"{all_summary['pairs']} "
        f"("
        f"{all_summary['clean_accuracy'] * 100:.2f}%"
        f")"
    )

    print(
        f"Corrupted accuracy: "
        f"{all_summary['corrupted_correct']}/"
        f"{all_summary['pairs']} "
        f"("
        f"{all_summary['corrupted_accuracy'] * 100:.2f}%"
        f")"
    )

    print()

    print(
        f"Mean clean LD: "
        f"{all_summary['mean_clean_ld']:+.4f}"
    )

    print(
        f"Median clean LD: "
        f"{all_summary['median_clean_ld']:+.4f}"
    )

    print(
        f"Mean corrupted LD: "
        f"{all_summary['mean_corrupted_ld']:+.4f}"
    )

    print(
        f"Median corrupted LD: "
        f"{all_summary['median_corrupted_ld']:+.4f}"
    )

    print()

    print(
        f"Mean behavioral gap: "
        f"{all_summary['mean_behavioral_gap']:+.4f}"
    )

    print(
        f"Median behavioral gap: "
        f"{all_summary['median_behavioral_gap']:+.4f}"
    )

    print(
        f"Pairs with clean LD > corrupted LD: "
        f"{all_summary['positive_behavioral_gap_pairs']}/"
        f"{all_summary['pairs']} "
        f"("
        f"{all_summary['positive_behavioral_gap_fraction'] * 100:.2f}%"
        f")"
    )

    # ------------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------------

    section(
        "DISCOVERY SPLIT RESULTS"
    )

    print(
        f"Pairs: "
        f"{discovery_summary['pairs']}"
    )

    print(
        f"Clean accuracy: "
        f"{discovery_summary['clean_accuracy'] * 100:.2f}%"
    )

    print(
        f"Corrupted accuracy: "
        f"{discovery_summary['corrupted_accuracy'] * 100:.2f}%"
    )

    print(
        f"Mean behavioral gap: "
        f"{discovery_summary['mean_behavioral_gap']:+.4f}"
    )

    print(
        f"Positive gap pairs: "
        f"{discovery_summary['positive_behavioral_gap_fraction'] * 100:.2f}%"
    )

    # ------------------------------------------------------------------------
    # Held-out
    # ------------------------------------------------------------------------

    section(
        "HELD-OUT SPLIT RESULTS"
    )

    print(
        f"Pairs: "
        f"{heldout_summary['pairs']}"
    )

    print(
        f"Clean accuracy: "
        f"{heldout_summary['clean_accuracy'] * 100:.2f}%"
    )

    print(
        f"Corrupted accuracy: "
        f"{heldout_summary['corrupted_accuracy'] * 100:.2f}%"
    )

    print(
        f"Mean behavioral gap: "
        f"{heldout_summary['mean_behavioral_gap']:+.4f}"
    )

    print(
        f"Positive gap pairs: "
        f"{heldout_summary['positive_behavioral_gap_fraction'] * 100:.2f}%"
    )

    # ------------------------------------------------------------------------
    # Domain analysis
    # ------------------------------------------------------------------------

    domain_results = analyze_domains(
        results
    )

    # ------------------------------------------------------------------------
    # Extreme examples
    # ------------------------------------------------------------------------

    print_extreme_pairs(
        results
    )

    # ------------------------------------------------------------------------
    # Behavioral gate
    # ------------------------------------------------------------------------

    gate = behavioral_gate(
        all_summary,
        discovery_summary,
        heldout_summary,
    )

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------

    section(
        "SAVING PHASE 11 RESULTS"
    )

    write_results_csv(
        PAIR_RESULTS_PATH,
        results,
    )

    write_results_csv(
        DOMAIN_RESULTS_PATH,
        domain_results,
    )

    summary = {
        "phase": 11,

        "name":
            "expanded_dataset_behavioral_validation",

        "model":
            MODEL_NAME,

        "total_pairs":
            EXPECTED_TOTAL_PAIRS,

        "discovery_pairs":
            EXPECTED_DISCOVERY_PAIRS,

        "heldout_pairs":
            EXPECTED_HELDOUT_PAIRS,

        "positive_token":
            POSITIVE_TOKEN,

        "negative_token":
            NEGATIVE_TOKEN,

        "overall":
            all_summary,

        "discovery":
            discovery_summary,

        "heldout":
            heldout_summary,

        "domains":
            domain_results,

        "behavioral_gate":
            gate,

        "activation_patching_performed":
            False,

        "model_weights_modified":
            False,
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    # ------------------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------------------

    if device.type == "cuda":

        section("GPU MEMORY")

        print(
            f"Allocated: "
            f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
        )

        print(
            f"Reserved:  "
            f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB"
        )

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    section(
        "PHASE 11 BEHAVIORAL VALIDATION COMPLETE"
    )

    print(
        f"Behavioral gate: "
        f"{gate['decision']}"
    )

    print()

    if gate["decision"] == "PASS":

        print(
            "The expanded dataset provides a "
            "usable clean/corrupted behavioral contrast."
        )

        print()
        print(
            "The next step is the discovery-split "
            "activation-patching experiment."
        )

    else:

        print(
            "The expanded dataset did NOT pass "
            "the behavioral gate."
        )

        print()
        print(
            "Do NOT proceed to activation patching "
            "until the behavioral contrast is investigated."
        )

    print()
    print(
        f"Pair results:"
    )

    print(
        f"  {PAIR_RESULTS_PATH}"
    )

    print()
    print(
        f"Summary:"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print()
    print(
        f"Domain results:"
    )

    print(
        f"  {DOMAIN_RESULTS_PATH}"
    )

    print()
    print(
        "No model weights were modified."
    )

    print(
        "No activation patching was performed."
    )


if __name__ == "__main__":
    main()