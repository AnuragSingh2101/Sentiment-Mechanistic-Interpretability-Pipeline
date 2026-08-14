"""
PHASE 15 — HELD-OUT GENERALIZATION EVALUATION

Frozen circuit selected during discovery:

    L10H4 + L8H9 + L9H2

Frozen baseline head:

    L10H4

The 40 held-out examples are evaluated exactly once.

IMPORTANT:
    No head selection is performed here.
    No circuit search is performed here.
    No optimization is performed here.
    No model weights are modified.

Purpose:
    Determine whether the discovery-selected circuit
    generalizes to previously unseen examples.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer_lens.model_bridge import TransformerBridge


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

SPLITS_DIR = DATA_DIR / "splits"

HELDOUT_DATASET = (
    SPLITS_DIR / "heldout_pairs.csv"
)

ACTIVATION_DIR = (
    DATA_DIR
    / "expanded_activations"
    / "heldout"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "heldout_generalization"
)

FIGURES_DIR = (
    RESULTS_DIR / "figures"
)

RESULTS_CSV = (
    RESULTS_DIR
    / "heldout_generalization_results.csv"
)

SUMMARY_CSV = (
    RESULTS_DIR
    / "heldout_generalization_summary.csv"
)

SUMMARY_JSON = (
    RESULTS_DIR
    / "heldout_generalization_summary.json"
)

MODEL_NAME = "gpt2"

EXPECTED_HELDOUT_PAIRS = 40

N_LAYERS = 12

N_HEADS = 12

D_HEAD = 64

POSITIVE_TEXT = " positive"

NEGATIVE_TEXT = " negative"

EPSILON = 1e-8

HOOK_TEMPLATE = (
    "blocks.{layer}.attn.hook_z"
)

# ============================================================================
# FROZEN DISCOVERY CIRCUIT
# ============================================================================

FROZEN_HEAD = (
    10,
    4,
    "L10H4",
)

FROZEN_CIRCUIT = (
    (10, 4, "L10H4"),
    (8, 9, "L8H9"),
    (9, 2, "L9H2"),
)


# ============================================================================
# PRINTING
# ============================================================================


def section(title: str):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


# ============================================================================
# DIRECTORIES
# ============================================================================


def ensure_directories():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# DEVICE
# ============================================================================


def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================================
# DATASET
# ============================================================================


def load_heldout_dataset():

    if not HELDOUT_DATASET.exists():

        raise FileNotFoundError(
            "Held-out dataset not found:\n"
            f"{HELDOUT_DATASET}"
        )

    with HELDOUT_DATASET.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(reader)

    if len(rows) != EXPECTED_HELDOUT_PAIRS:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_HELDOUT_PAIRS} "
            f"held-out pairs, "
            f"found {len(rows)}."
        )

    required = {
        "pair_id",
        "domain",
        "clean_text",
        "corrupted_text",
        "clean_label",
        "corrupted_label",
        "split",
    }

    missing = (
        required
        - set(reader.fieldnames or [])
    )

    if missing:

        raise ValueError(
            "Missing dataset columns: "
            f"{sorted(missing)}"
        )

    for row in rows:

        if row["split"] != "heldout":

            raise ValueError(
                f"Non-heldout row found: "
                f"{row['pair_id']}"
            )

    print(
        f"Held-out pairs loaded: "
        f"{len(rows)}"
    )

    print(
        "Held-out dataset validation: PASS"
    )

    return rows


# ============================================================================
# MODEL
# ============================================================================


def load_model(device):

    section(
        "MODEL LOADING"
    )

    model = (
        TransformerBridge
        .boot_transformers(
            MODEL_NAME,
            device=device,
        )
    )

    model.eval()

    print(
        "GPT-2-small loaded successfully."
    )

    return model


# ============================================================================
# ARCHITECTURE
# ============================================================================


def validate_architecture(model):

    section(
        "ARCHITECTURE VALIDATION"
    )

    print(
        f"Layers:      {model.cfg.n_layers}"
    )

    print(
        f"Heads/layer: {model.cfg.n_heads}"
    )

    print(
        f"Head dim:    {model.cfg.d_head}"
    )

    if model.cfg.n_layers != N_LAYERS:

        raise ValueError(
            "Unexpected layer count."
        )

    if model.cfg.n_heads != N_HEADS:

        raise ValueError(
            "Unexpected head count."
        )

    if model.cfg.d_head != D_HEAD:

        raise ValueError(
            "Unexpected head dimension."
        )

    print(
        "Architecture validation: PASS"
    )


# ============================================================================
# TOKENS
# ============================================================================


def validate_sentiment_tokens(model):

    section(
        "SENTIMENT TOKEN VALIDATION"
    )

    positive_tokens = model.to_tokens(
        POSITIVE_TEXT,
        prepend_bos=False,
    )

    negative_tokens = model.to_tokens(
        NEGATIVE_TEXT,
        prepend_bos=False,
    )

    if positive_tokens.numel() != 1:

        raise ValueError(
            "' positive' is not one token."
        )

    if negative_tokens.numel() != 1:

        raise ValueError(
            "' negative' is not one token."
        )

    positive_id = int(
        positive_tokens.flatten()[0]
        .item()
    )

    negative_id = int(
        negative_tokens.flatten()[0]
        .item()
    )

    print(
        f"' positive' → {positive_id}"
    )

    print(
        f"' negative' → {negative_id}"
    )

    return (
        positive_id,
        negative_id,
    )


# ============================================================================
# PROMPT
# ============================================================================


def make_prompt(text):

    return (
        "Review: "
        + text
        + "\nSentiment:"
    )


# ============================================================================
# LOGIT DIFFERENCE
# ============================================================================


@torch.no_grad()
def get_logit_difference(
    model,
    text,
    positive_id,
    negative_id,
):

    tokens = model.to_tokens(
        make_prompt(text),
        prepend_bos=True,
    )

    logits = model(
        tokens
    )

    final_logits = logits[
        0,
        -1,
    ]

    value = (
        final_logits[positive_id]
        -
        final_logits[negative_id]
    )

    return float(
        value.detach()
        .cpu()
        .item()
    )


# ============================================================================
# ACTIVATION CACHE
# ============================================================================


def cache_path(
    pair_id,
    condition,
):

    return (
        ACTIVATION_DIR
        / f"pair_{pair_id:03d}_{condition}.pt"
    )


def load_cache(
    pair_id,
    condition,
):

    path = cache_path(
        pair_id,
        condition,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing held-out activation:\n"
            f"{path}"
        )

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if cache.get(
        "split"
    ) != "heldout":

        raise ValueError(
            f"Cache is not held-out:\n"
            f"{path}"
        )

    return cache


# ============================================================================
# CACHE VALIDATION
# ============================================================================


def validate_cache(
    cache,
    pair_id,
    condition,
):

    activations = cache[
        "activations"
    ]

    if len(activations) != N_LAYERS:

        raise ValueError(
            f"Pair {pair_id}: "
            f"incorrect layer count."
        )

    sequence_length = int(
        cache[
            "sequence_length"
        ]
    )

    for layer in range(
        N_LAYERS
    ):

        hook_name = (
            HOOK_TEMPLATE.format(
                layer=layer
            )
        )

        if hook_name not in activations:

            raise ValueError(
                f"Missing {hook_name}"
            )

        activation = activations[
            hook_name
        ]

        expected = (
            sequence_length,
            N_HEADS,
            D_HEAD,
        )

        if tuple(
            activation.shape
        ) != expected:

            raise ValueError(
                f"Invalid shape for "
                f"{hook_name}: "
                f"{tuple(activation.shape)} "
                f"expected {expected}"
            )


# ============================================================================
# RECOVERY
# ============================================================================


def calculate_recovery(
    clean_ld,
    corrupted_ld,
    patched_ld,
):

    denominator = (
        clean_ld
        -
        corrupted_ld
    )

    if abs(
        denominator
    ) < EPSILON:

        return float("nan")

    return (
        patched_ld
        -
        corrupted_ld
    ) / denominator


# ============================================================================
# PATCH HOOK
# ============================================================================


def create_patch_hooks(
    clean_cache,
    heads,
):

    hooks = []

    for (
        layer,
        head,
        label,
    ) in heads:

        hook_name = (
            HOOK_TEMPLATE.format(
                layer=layer
            )
        )

        clean_activation = (
            clean_cache[
                "activations"
            ][hook_name]
        )

        clean_vector = (
            clean_activation[
                -1,
                head,
                :
            ]
            .detach()
        )

        def make_hook(
            head_index,
            clean_vector,
        ):

            def patch_hook(
                activation,
                hook,
            ):

                patched = (
                    activation.clone()
                )

                final_position = (
                    activation.shape[1]
                    - 1
                )

                patched[
                    0,
                    final_position,
                    head_index,
                    :
                ] = (
                    clean_vector.to(
                        device=activation.device,
                        dtype=activation.dtype,
                    )
                )

                return patched

            return patch_hook

        hooks.append(
            (
                hook_name,
                make_hook(
                    head,
                    clean_vector,
                ),
            )
        )

    return hooks


# ============================================================================
# PATCH
# ============================================================================


@torch.no_grad()
def run_patch(
    model,
    corrupted_text,
    clean_cache,
    heads,
    positive_id,
    negative_id,
):

    tokens = model.to_tokens(
        make_prompt(
            corrupted_text
        ),
        prepend_bos=True,
    )

    hooks = create_patch_hooks(
        clean_cache,
        heads,
    )

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=hooks,
    )

    final_logits = logits[
        0,
        -1,
    ]

    patched_ld = (
        final_logits[
            positive_id
        ]
        -
        final_logits[
            negative_id
        ]
    )

    return float(
        patched_ld.detach()
        .cpu()
        .item()
    )


# ============================================================================
# MAIN HELD-OUT EVALUATION
# ============================================================================


def run_evaluation(
    model,
    rows,
    positive_id,
    negative_id,
):

    section(
        "HELD-OUT GENERALIZATION EVALUATION"
    )

    print(
        "Frozen single head: L10H4"
    )

    print(
        "Frozen circuit: "
        "L10H4 + L8H9 + L9H2"
    )

    print(
        "No head selection will be performed."
    )

    print(
        "No circuit search will be performed."
    )

    results = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        pair_id = int(
            row["pair_id"]
        )

        clean_cache = (
            load_cache(
                pair_id,
                "clean",
            )
        )

        corrupted_cache = (
            load_cache(
                pair_id,
                "corrupted",
            )
        )

        validate_cache(
            clean_cache,
            pair_id,
            "clean",
        )

        validate_cache(
            corrupted_cache,
            pair_id,
            "corrupted",
        )

        clean_ld = (
            get_logit_difference(
                model,
                row["clean_text"],
                positive_id,
                negative_id,
            )
        )

        corrupted_ld = (
            get_logit_difference(
                model,
                row["corrupted_text"],
                positive_id,
                negative_id,
            )
        )

        single_patched_ld = (
            run_patch(
                model=model,
                corrupted_text=(
                    row["corrupted_text"]
                ),
                clean_cache=(
                    clean_cache
                ),
                heads=(
                    (FROZEN_HEAD,)
                ),
                positive_id=(
                    positive_id
                ),
                negative_id=(
                    negative_id
                ),
            )
        )

        circuit_patched_ld = (
            run_patch(
                model=model,
                corrupted_text=(
                    row["corrupted_text"]
                ),
                clean_cache=(
                    clean_cache
                ),
                heads=(
                    FROZEN_CIRCUIT
                ),
                positive_id=(
                    positive_id
                ),
                negative_id=(
                    negative_id
                ),
            )
        )

        single_recovery = (
            calculate_recovery(
                clean_ld,
                corrupted_ld,
                single_patched_ld,
            )
        )

        circuit_recovery = (
            calculate_recovery(
                clean_ld,
                corrupted_ld,
                circuit_patched_ld,
            )
        )

        improvement = (
            circuit_recovery
            -
            single_recovery
        )

        results.append(
            {
                "pair_id":
                    pair_id,

                "domain":
                    row["domain"],

                "clean_ld":
                    clean_ld,

                "corrupted_ld":
                    corrupted_ld,

                "single_patched_ld":
                    single_patched_ld,

                "circuit_patched_ld":
                    circuit_patched_ld,

                "single_recovery":
                    single_recovery,

                "circuit_recovery":
                    circuit_recovery,

                "circuit_minus_single":
                    improvement,
            }
        )

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"Pair {pair_id:03d} | "
            f"single={single_recovery:+.4f} | "
            f"circuit={circuit_recovery:+.4f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================================
# SUMMARY
# ============================================================================


def create_summary(
    dataframe,
):

    section(
        "HELD-OUT RESULTS"
    )

    rows = []

    metrics = [
        (
            "single_head",
            "single_recovery",
        ),
        (
            "three_head_circuit",
            "circuit_recovery",
        ),
        (
            "circuit_minus_single",
            "circuit_minus_single",
        ),
    ]

    for name, column in metrics:

        values = (
            dataframe[
                column
            ]
            .dropna()
            .to_numpy()
        )

        rows.append(
            {
                "condition":
                    name,

                "mean":
                    float(
                        np.mean(values)
                    ),

                "median":
                    float(
                        np.median(values)
                    ),

                "std":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "min":
                    float(
                        np.min(values)
                    ),

                "max":
                    float(
                        np.max(values)
                    ),

                "positive_fraction":
                    float(
                        np.mean(
                            values > 0
                        )
                    ),

                "n":
                    len(values),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    return summary


# ============================================================================
# GENERALIZATION DECISION
# ============================================================================


def create_decision(
    dataframe,
):

    section(
        "GENERALIZATION DECISION"
    )

    single = (
        dataframe[
            "single_recovery"
        ]
        .dropna()
        .to_numpy()
    )

    circuit = (
        dataframe[
            "circuit_recovery"
        ]
        .dropna()
        .to_numpy()
    )

    improvement = (
        circuit
        -
        single
    )

    mean_single = float(
        np.mean(single)
    )

    mean_circuit = float(
        np.mean(circuit)
    )

    mean_improvement = float(
        np.mean(improvement)
    )

    positive_circuit = float(
        np.mean(
            circuit > 0
        )
    )

    positive_improvement = float(
        np.mean(
            improvement > 0
        )
    )

    circuit_beats_single = (
        mean_circuit
        >
        mean_single
    )

    print(
        f"Mean single-head recovery: "
        f"{mean_single:.6f}"
    )

    print(
        f"Mean circuit recovery: "
        f"{mean_circuit:.6f}"
    )

    print(
        f"Mean circuit improvement: "
        f"{mean_improvement:.6f}"
    )

    print(
        f"Circuit positive recovery: "
        f"{positive_circuit * 100:.2f}%"
    )

    print(
        f"Circuit > single per pair: "
        f"{positive_improvement * 100:.2f}%"
    )

    print()

    if mean_circuit > mean_single:

        print(
            "circuit_beats_single: PASS"
        )

    else:

        print(
            "circuit_beats_single: FAIL"
        )

    if positive_circuit > 0.5:

        print(
            "positive_circuit_majority: PASS"
        )

    else:

        print(
            "positive_circuit_majority: FAIL"
        )

    if positive_improvement > 0.5:

        print(
            "circuit_better_on_majority: PASS"
        )

    else:

        print(
            "circuit_better_on_majority: FAIL"
        )

    if (
        circuit_beats_single
        and positive_circuit > 0.5
        and positive_improvement > 0.5
    ):

        decision = (
            "HELDOUT_GENERALIZATION_SUPPORT"
        )

    else:

        decision = (
            "HELDOUT_GENERALIZATION_WEAK"
        )

    print()

    print(
        f"Overall decision: {decision}"
    )

    return {
        "decision":
            decision,

        "mean_single_recovery":
            mean_single,

        "mean_circuit_recovery":
            mean_circuit,

        "mean_circuit_improvement":
            mean_improvement,

        "positive_circuit_fraction":
            positive_circuit,

        "circuit_better_fraction":
            positive_improvement,

        "frozen_single_head":
            "L10H4",

        "frozen_circuit":
            [
                "L10H4",
                "L8H9",
                "L9H2",
            ],

        "heldout_pairs":
            EXPECTED_HELDOUT_PAIRS,

        "model_weights_modified":
            False,
    }


# ============================================================================
# PLOT
# ============================================================================


def create_plot(
    dataframe,
):

    section(
        "CREATING GENERALIZATION PLOT"
    )

    x = np.arange(
        len(dataframe)
    )

    width = 0.38

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        x - width / 2,
        dataframe[
            "single_recovery"
        ],
        width,
        label="L10H4",
    )

    plt.bar(
        x + width / 2,
        dataframe[
            "circuit_recovery"
        ],
        width,
        label=(
            "L10H4 + L8H9 + L9H2"
        ),
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Held-out Pair"
    )

    plt.ylabel(
        "Recovery"
    )

    plt.title(
        "Held-Out Generalization: "
        "Single Head vs Frozen Circuit"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "heldout_single_vs_circuit.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        path
    )


# ============================================================================
# JSON
# ============================================================================


def save_json(
    summary,
    decision,
):

    payload = {
        "phase":
            15,

        "name":
            "heldout_generalization",

        "model":
            MODEL_NAME,

        "evaluation_split":
            "heldout",

        "heldout_pairs":
            EXPECTED_HELDOUT_PAIRS,

        "frozen_circuit":
            [
                "L10H4",
                "L8H9",
                "L9H2",
            ],

        "frozen_single_head":
            "L10H4",

        "selection_source":
            "Phase 13 and Phase 14 discovery split",

        "heldout_used_for_selection":
            False,

        "heldout_used_for_optimization":
            False,

        "model_weights_modified":
            False,

        "summary":
            summary.to_dict(
                orient="records"
            ),

        "decision":
            decision,
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

    print()

    print(
        f"JSON saved:"
    )

    print(
        f"  {SUMMARY_JSON}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    section(
        "PHASE 15 — HELD-OUT GENERALIZATION"
    )

    print(
        "Frozen circuit:"
    )

    print(
        "  L10H4 + L8H9 + L9H2"
    )

    print()

    print(
        "Frozen baseline:"
    )

    print(
        "  L10H4"
    )

    print()

    print(
        "The 40 held-out examples are "
        "being evaluated without any "
        "further model/circuit selection."
    )

    ensure_directories()

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    rows = (
        load_heldout_dataset()
    )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    section(
        "DEVICE"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model = load_model(
        device
    )

    validate_architecture(
        model
    )

    (
        positive_id,
        negative_id,
    ) = validate_sentiment_tokens(
        model
    )

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------

    dataframe = (
        run_evaluation(
            model=model,
            rows=rows,
            positive_id=positive_id,
            negative_id=negative_id,
        )
    )

    # ------------------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------------------

    dataframe.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary = create_summary(
        dataframe
    )

    # ------------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------------

    decision = create_decision(
        dataframe
    )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    create_plot(
        dataframe
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    save_json(
        summary,
        decision,
    )

    # ------------------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------------------

    if device.type == "cuda":

        section(
            "GPU MEMORY"
        )

        print(
            "Allocated:",
            f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB",
        )

        print(
            "Reserved:",
            f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB",
        )

    # ------------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------------

    section(
        "PHASE 15 COMPLETE"
    )

    print(
        "Held-out generalization evaluation completed."
    )

    print()

    print(
        f"Pairs evaluated: "
        f"{len(dataframe)}"
    )

    print()

    print(
        "Detailed results:"
    )

    print(
        f"  {RESULTS_CSV}"
    )

    print()

    print(
        "Summary:"
    )

    print(
        f"  {SUMMARY_CSV}"
    )

    print()

    print(
        "JSON:"
    )

    print(
        f"  {SUMMARY_JSON}"
    )

    print()

    print(
        "Figure:"
    )

    print(
        f"  {FIGURES_DIR / 'heldout_single_vs_circuit.png'}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The circuit was frozen before "
        "held-out evaluation."
    )

    print(
        "No model weights were modified."
    )


if __name__ == "__main__":

    main()