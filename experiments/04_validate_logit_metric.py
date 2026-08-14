"""
Phase 3.5 — Logit-Difference Behavioral Metric Validation.

Purpose
-------
Validate a more suitable behavioral metric for the mechanistic
interpretability experiment.

Instead of comparing softmax probabilities directly, we compare
the logits assigned to:

    " positive"
    " negative"

Metric:

    logit_difference =
        logit(" positive") - logit(" negative")

Interpretation:

    positive logit difference -> model favors positive
    negative logit difference -> model favors negative

This metric is useful for causal activation-patching experiments
because it gives us a continuous quantity that can be compared
between:

    CLEAN
    CORRUPTED
    PATCHED

No activation patching is performed in this script.
"""

from __future__ import annotations

import gc
import statistics

import torch
from transformer_lens.model_bridge import TransformerBridge


MODEL_NAME = "gpt2"

POSITIVE_TOKEN = " positive"
NEGATIVE_TOKEN = " negative"


# ---------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------

EXAMPLES = [
    ("This movie was wonderful.", "positive"),
    ("The film was fantastic.", "positive"),
    ("The acting was excellent.", "positive"),
    ("I really enjoyed this movie.", "positive"),
    ("The story was beautiful.", "positive"),
    ("The performance was amazing.", "positive"),
    ("This was a brilliant film.", "positive"),
    ("The movie was enjoyable.", "positive"),
    ("The ending was satisfying.", "positive"),
    ("I loved every minute of it.", "positive"),

    ("This movie was terrible.", "negative"),
    ("The film was awful.", "negative"),
    ("The acting was dreadful.", "negative"),
    ("I really disliked this movie.", "negative"),
    ("The story was boring.", "negative"),
    ("The performance was horrible.", "negative"),
    ("This was a terrible film.", "negative"),
    ("The movie was disappointing.", "negative"),
    ("The ending was frustrating.", "negative"),
    ("I hated every minute of it.", "negative"),
]


# ---------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------


def get_single_token_id(
    bridge: TransformerBridge,
    text: str,
) -> int:
    """
    Return the token ID for a string that must tokenize to one token.
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
# Logit-difference calculation
# ---------------------------------------------------------------------


def get_logit_difference(
    bridge: TransformerBridge,
    text: str,
    positive_id: int,
    negative_id: int,
) -> tuple[float, float, float]:
    """
    Calculate:

        positive logit
        negative logit
        positive - negative logit difference
    """

    prompt = f"Review: {text}\nSentiment:"

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
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 70)
    print("PHASE 3.5 — LOGIT-DIFFERENCE METRIC VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

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

    print("GPT-2-small loaded successfully.")

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

    print("Sentiment token validation: PASS")

    # ---------------------------------------------------------------
    # Evaluate examples
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGIT-DIFFERENCE EVALUATION")
    print("=" * 70)

    results = []

    correct = 0

    positive_differences = []
    negative_differences = []

    for index, (text, expected) in enumerate(
        EXAMPLES,
        start=1,
    ):

        (
            positive_logit,
            negative_logit,
            logit_difference,
        ) = get_logit_difference(
            bridge,
            text,
            positive_id,
            negative_id,
        )

        # -----------------------------------------------------------
        # Prediction
        # -----------------------------------------------------------

        if logit_difference > 0:
            predicted = "positive"
        else:
            predicted = "negative"

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        # -----------------------------------------------------------
        # Store by class
        # -----------------------------------------------------------

        if expected == "positive":
            positive_differences.append(
                logit_difference
            )
        else:
            negative_differences.append(
                logit_difference
            )

        result = {
            "id": index,
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "positive_logit": positive_logit,
            "negative_logit": negative_logit,
            "logit_difference": logit_difference,
            "correct": is_correct,
        }

        results.append(result)

        print(
            f"{index:02d}. "
            f"{expected:>8} | "
            f"predicted={predicted:>8} | "
            f"LD={logit_difference:+.4f} | "
            f"P-logit={positive_logit:+.4f} | "
            f"N-logit={negative_logit:+.4f} | "
            f"{'✓' if is_correct else '✗'}"
        )

    # ---------------------------------------------------------------
    # Overall accuracy
    # ---------------------------------------------------------------

    total = len(results)

    accuracy = correct / total

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"Correct:  {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ---------------------------------------------------------------
    # Class-level results
    # ---------------------------------------------------------------

    positive_correct = sum(
        result["correct"]
        for result in results
        if result["expected"] == "positive"
    )

    negative_correct = sum(
        result["correct"]
        for result in results
        if result["expected"] == "negative"
    )

    positive_total = len(positive_differences)
    negative_total = len(negative_differences)

    positive_accuracy = (
        positive_correct / positive_total
    )

    negative_accuracy = (
        negative_correct / negative_total
    )

    positive_mean = statistics.mean(
        positive_differences
    )

    negative_mean = statistics.mean(
        negative_differences
    )

    positive_median = statistics.median(
        positive_differences
    )

    negative_median = statistics.median(
        negative_differences
    )

    print(
        f"\nPositive accuracy: "
        f"{positive_correct}/{positive_total} "
        f"({positive_accuracy * 100:.2f}%)"
    )

    print(
        f"Negative accuracy: "
        f"{negative_correct}/{negative_total} "
        f"({negative_accuracy * 100:.2f}%)"
    )

    # ---------------------------------------------------------------
    # Logit-difference distributions
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGIT-DIFFERENCE DISTRIBUTION")
    print("=" * 70)

    print(
        f"Positive mean LD:   {positive_mean:+.4f}"
    )

    print(
        f"Positive median LD: {positive_median:+.4f}"
    )

    print(
        f"Negative mean LD:   {negative_mean:+.4f}"
    )

    print(
        f"Negative median LD: {negative_median:+.4f}"
    )

    # ---------------------------------------------------------------
    # Separation
    # ---------------------------------------------------------------

    separation = (
        positive_mean - negative_mean
    )

    print(
        f"\nClass separation "
        f"(positive mean - negative mean): "
        f"{separation:+.4f}"
    )

    # ---------------------------------------------------------------
    # Check whether the metric is behaving sensibly
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("METRIC SANITY CHECK")
    print("=" * 70)

    if positive_mean > negative_mean:
        print(
            "PASS: Positive examples have a higher "
            "mean logit difference than negative examples."
        )
    else:
        print(
            "WARNING: Positive and negative examples "
            "are not correctly separated by the metric."
        )

    if (
        positive_mean > 0
        and negative_mean < 0
    ):
        print(
            "PASS: Class means fall on opposite "
            "sides of zero."
        )
    else:
        print(
            "NOTE: Class means do not fall on opposite "
            "sides of zero."
        )

    # ---------------------------------------------------------------
    # Behavioral metric recommendation
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("METRIC RECOMMENDATION")
    print("=" * 70)

    print(
        "Primary causal metric:"
    )

    print(
        "    logit_difference = "
        "logit(' positive') - logit(' negative')"
    )

    print(
        "\nThis continuous metric should be used alongside "
        "classification accuracy in later activation-patching "
        "experiments."
    )

    # ---------------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------------

    if device == "cuda":

        allocated = (
            torch.cuda.memory_allocated() / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved() / 1024**3
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
    # Final status
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3.5 COMPLETE")
    print("=" * 70)

    print(
        "\nThe logit-difference metric has been evaluated."
    )

    print(
        "Next step: decide whether the prompt/metric "
        "needs adjustment before constructing the full "
        "clean/corrupted activation-patching dataset."
    )


if __name__ == "__main__":
    main()
    