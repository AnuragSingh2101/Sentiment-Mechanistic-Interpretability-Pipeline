"""
Phase 3 — Behavioral Sentiment Validation.

Purpose
-------
Validate that GPT-2-small can distinguish positive and negative sentiment
using controlled next-token probabilities.

This phase establishes the behavioral metric that will later be used for
activation-patching experiments.

The model is NOT modified.
No activation patching is performed.

Metric
------
For each prompt:

    Review: <text>
    Sentiment:

we compare:

    P(" positive")
    P(" negative")

The predicted sentiment is whichever token receives the higher probability.

This gives us a reproducible behavioral measurement for later causal
interventions.
"""

from __future__ import annotations

import gc

import torch
from transformer_lens.model_bridge import TransformerBridge


MODEL_NAME = "gpt2"

POSITIVE_TOKEN = " positive"
NEGATIVE_TOKEN = " negative"


# ---------------------------------------------------------------------
# Small validation dataset
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
# Helpers
# ---------------------------------------------------------------------


def get_single_token_id(
    bridge: TransformerBridge,
    text: str,
) -> int:
    """
    Return the token ID for a string that must tokenize to exactly one token.
    """

    token_ids = bridge.to_tokens(
        text,
        prepend_bos=False,
    )

    if token_ids.ndim == 2:
        token_ids = token_ids[0]

    if len(token_ids) != 1:
        raise ValueError(
            f"{text!r} does not tokenize to exactly one token. "
            f"Token IDs: {token_ids.tolist()}"
        )

    return int(token_ids[0].item())


def sentiment_probabilities(
    bridge: TransformerBridge,
    text: str,
    positive_id: int,
    negative_id: int,
) -> tuple[float, float]:
    """
    Calculate P(positive) and P(negative) for the next token.
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

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    positive_probability = float(
        probabilities[positive_id].item()
    )

    negative_probability = float(
        probabilities[negative_id].item()
    )

    del tokens
    del logits
    del probabilities

    return positive_probability, negative_probability


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 70)
    print("PHASE 3 — BEHAVIORAL SENTIMENT VALIDATION")
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
    # Verify sentiment tokens
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
    print("BEHAVIORAL EVALUATION")
    print("=" * 70)

    correct = 0

    results = []

    for index, (text, expected) in enumerate(
        EXAMPLES,
        start=1,
    ):

        positive_probability, negative_probability = (
            sentiment_probabilities(
                bridge,
                text,
                positive_id,
                negative_id,
            )
        )

        if positive_probability > negative_probability:
            predicted = "positive"
        else:
            predicted = "negative"

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        results.append(
            {
                "id": index,
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "positive_probability": positive_probability,
                "negative_probability": negative_probability,
                "correct": is_correct,
            }
        )

        print(
            f"{index:02d}. "
            f"{expected:>8} | "
            f"predicted={predicted:>8} | "
            f"P(pos)={positive_probability:.4f} | "
            f"P(neg)={negative_probability:.4f} | "
            f"{'✓' if is_correct else '✗'}"
        )

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------

    total = len(results)

    accuracy = correct / total

    print("\n" + "=" * 70)
    print("BEHAVIORAL RESULTS")
    print("=" * 70)

    print(f"Correct:  {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ---------------------------------------------------------------
    # Separate positive / negative accuracy
    # ---------------------------------------------------------------

    positive_results = [
        result
        for result in results
        if result["expected"] == "positive"
    ]

    negative_results = [
        result
        for result in results
        if result["expected"] == "negative"
    ]

    positive_correct = sum(
        result["correct"]
        for result in positive_results
    )

    negative_correct = sum(
        result["correct"]
        for result in negative_results
    )

    positive_accuracy = (
        positive_correct / len(positive_results)
    )

    negative_accuracy = (
        negative_correct / len(negative_results)
    )

    print(
        f"Positive accuracy: "
        f"{positive_correct}/{len(positive_results)} "
        f"({positive_accuracy * 100:.2f}%)"
    )

    print(
        f"Negative accuracy: "
        f"{negative_correct}/{len(negative_results)} "
        f"({negative_accuracy * 100:.2f}%)"
    )

    # ---------------------------------------------------------------
    # Sanity requirement
    # ---------------------------------------------------------------

    if accuracy == 0:
        raise RuntimeError(
            "Behavioral validation failed: "
            "the model predicted every example incorrectly."
        )

    print("\nBehavioral metric validation: PASS")

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

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")

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
    print("PHASE 3 VALIDATION: PASS")
    print("=" * 70)

    print(
        "\nThe sentiment probability metric is ready "
        "for activation-patching experiments."
    )


if __name__ == "__main__":
    main()