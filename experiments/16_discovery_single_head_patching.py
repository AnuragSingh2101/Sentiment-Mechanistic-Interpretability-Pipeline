"""
PHASE 13 — DISCOVERY SINGLE-HEAD ACTIVATION PATCHING

Goal
----
Evaluate all 144 GPT-2-small attention heads using ONLY the
60 discovery pairs from the expanded dataset.

For every pair and every attention head:

    1. Run the clean prompt.
    2. Run the corrupted prompt.
    3. Measure clean and corrupted logit difference.
    4. Patch the selected head's activation at the FINAL
       prediction position of the corrupted prompt using the
       corresponding clean activation.
    5. Measure the patched logit difference.
    6. Calculate normalized recovery.

Metric
------
logit_difference =
    logit(" positive") - logit(" negative")

Recovery
--------
recovery =
    (patched_LD - corrupted_LD)
    /
    (clean_LD - corrupted_LD)

Interpretation
--------------
0.0  -> no recovery
1.0  -> full recovery
>1.0 -> over-recovery
<0.0 -> patch made behavior worse

Important
---------
ONLY the discovery split is used.

The 40 held-out pairs are NEVER loaded.

Because clean/corrupted prompts can have different token lengths,
this experiment patches the FINAL prediction position only.

This avoids invalid tensor-shape assumptions and ensures the
activation being patched corresponds to the position whose logits
are being measured.

No model weights are modified.
"""

from __future__ import annotations

import csv
import json
import math
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

DISCOVERY_DATASET = (
    SPLITS_DIR / "discovery_pairs.csv"
)

ACTIVATION_DIR = (
    DATA_DIR
    / "expanded_activations"
    / "discovery"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "expanded_single_head"
)

FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_CSV = (
    RESULTS_DIR
    / "discovery_single_head_results.csv"
)

RANKING_CSV = (
    RESULTS_DIR
    / "discovery_head_ranking.csv"
)

SUMMARY_JSON = (
    RESULTS_DIR
    / "discovery_head_summary.json"
)

MODEL_NAME = "gpt2"

EXPECTED_PAIRS = 60

N_LAYERS = 12

N_HEADS = 12

D_HEAD = 64

TOTAL_HEADS = (
    N_LAYERS * N_HEADS
)

POSITIVE_TEXT = " positive"

NEGATIVE_TEXT = " negative"

HOOK_TEMPLATE = (
    "blocks.{layer}.attn.hook_z"
)

EPSILON = 1e-8


# ============================================================================
# UTILITIES
# ============================================================================


def section(title: str) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_directories() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_device() -> torch.device:

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================================
# DATASET
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


def load_discovery_dataset():

    if not DISCOVERY_DATASET.exists():

        raise FileNotFoundError(
            "Discovery dataset not found:\n"
            f"{DISCOVERY_DATASET}"
        )

    with DISCOVERY_DATASET.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            raise ValueError(
                "Discovery CSV has no header."
            )

        missing = (
            REQUIRED_COLUMNS
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                "Discovery dataset is missing "
                f"columns: {sorted(missing)}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_PAIRS:

        raise ValueError(
            f"Expected {EXPECTED_PAIRS} "
            f"discovery pairs, found {len(rows)}."
        )

    pair_ids = [
        int(row["pair_id"])
        for row in rows
    ]

    if len(set(pair_ids)) != len(pair_ids):

        raise ValueError(
            "Duplicate discovery pair IDs detected."
        )

    for row in rows:

        if row["split"] != "discovery":

            raise ValueError(
                f"Non-discovery row found: "
                f"pair {row['pair_id']}"
            )

        if row["clean_label"] != "positive":

            raise ValueError(
                f"Unexpected clean label in "
                f"pair {row['pair_id']}"
            )

        if row["corrupted_label"] != "negative":

            raise ValueError(
                f"Unexpected corrupted label in "
                f"pair {row['pair_id']}"
            )

    return rows


# ============================================================================
# MODEL
# ============================================================================


def load_model(device):

    section("MODEL LOADING")

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
        )

        print(
            "GPU memory:",
            f"{total_memory / (1024 ** 3):.2f} GB",
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
# ARCHITECTURE VALIDATION
# ============================================================================


def validate_architecture(model):

    section("ARCHITECTURE VALIDATION")

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
            f"Expected {N_LAYERS} layers."
        )

    if model.cfg.n_heads != N_HEADS:

        raise ValueError(
            f"Expected {N_HEADS} heads."
        )

    if model.cfg.d_head != D_HEAD:

        raise ValueError(
            f"Expected d_head={D_HEAD}."
        )

    print(
        "GPT-2-small architecture: PASS"
    )


# ============================================================================
# TOKEN VALIDATION
# ============================================================================


def validate_sentiment_tokens(model):

    section("SENTIMENT TOKEN VALIDATION")

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
            "' positive' must tokenize to "
            "exactly one token."
        )

    if negative_tokens.numel() != 1:

        raise ValueError(
            "' negative' must tokenize to "
            "exactly one token."
        )

    positive_id = int(
        positive_tokens.flatten()[0].item()
    )

    negative_id = int(
        negative_tokens.flatten()[0].item()
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


def make_prompt(text: str) -> str:

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
    text: str,
    positive_id: int,
    negative_id: int,
) -> float:

    prompt = make_prompt(text)

    tokens = model.to_tokens(
        prompt,
        prepend_bos=True,
    )

    logits = model(tokens)

    final_logits = logits[
        0,
        -1,
    ]

    positive_logit = final_logits[
        positive_id
    ]

    negative_logit = final_logits[
        negative_id
    ]

    value = (
        positive_logit
        - negative_logit
    )

    return float(
        value.detach().cpu().item()
    )


# ============================================================================
# CACHE LOADING
# ============================================================================


def get_cache_path(
    pair_id: int,
    condition: str,
) -> Path:

    return (
        ACTIVATION_DIR
        / f"pair_{pair_id:03d}_{condition}.pt"
    )


def load_activation_cache(
    pair_id: int,
    condition: str,
):

    path = get_cache_path(
        pair_id,
        condition,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing activation cache:\n"
            f"{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if data["split"] != "discovery":

        raise ValueError(
            f"Cache {path} does not belong "
            "to discovery split."
        )

    if int(data["pair_id"]) != pair_id:

        raise ValueError(
            f"Pair ID mismatch in {path}"
        )

    if data["condition"] != condition:

        raise ValueError(
            f"Condition mismatch in {path}"
        )

    return data


# ============================================================================
# CACHE VALIDATION
# ============================================================================


def validate_cache_structure(
    data,
    pair_id: int,
    condition: str,
):

    activations = data[
        "activations"
    ]

    if len(activations) != N_LAYERS:

        raise ValueError(
            f"Pair {pair_id} {condition}: "
            f"expected {N_LAYERS} layers, "
            f"found {len(activations)}"
        )

    sequence_length = int(
        data["sequence_length"]
    )

    for layer in range(N_LAYERS):

        hook_name = (
            HOOK_TEMPLATE.format(
                layer=layer
            )
        )

        if hook_name not in activations:

            raise ValueError(
                f"Missing {hook_name} "
                f"for pair {pair_id}"
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
                f"Wrong shape for "
                f"{hook_name}: "
                f"{tuple(activation.shape)} "
                f"expected {expected}"
            )


# ============================================================================
# PATCH HOOK
# ============================================================================


def create_single_head_patch_hook(
    clean_activation: torch.Tensor,
    layer: int,
    head: int,
):
    """
    Create a hook that replaces the selected head
    at the FINAL prediction position.

    clean_activation shape:

        [clean_sequence, n_heads, d_head]

    Runtime hook activation shape:

        [batch, corrupted_sequence, n_heads, d_head]
    """

    if clean_activation.ndim != 3:

        raise ValueError(
            "Expected clean activation "
            "shape [sequence, heads, d_head], "
            f"got {tuple(clean_activation.shape)}"
        )

    if clean_activation.shape[1] != N_HEADS:

        raise ValueError(
            "Unexpected number of heads "
            f"in clean activation: "
            f"{clean_activation.shape[1]}"
        )

    if clean_activation.shape[2] != D_HEAD:

        raise ValueError(
            "Unexpected head dimension "
            f"in clean activation: "
            f"{clean_activation.shape[2]}"
        )

    clean_final = (
        clean_activation[
            -1,
            head,
            :
        ]
        .detach()
    )

    hook_name = (
        HOOK_TEMPLATE.format(
            layer=layer
        )
    )

    def patch_hook(
        activation,
        hook,
    ):

        if activation.ndim != 4:

            raise ValueError(
                f"Runtime activation for "
                f"{hook_name} has unexpected "
                f"shape {tuple(activation.shape)}"
            )

        if activation.shape[0] != 1:

            raise ValueError(
                "This experiment expects "
                "batch size 1."
            )

        patched = activation.clone()

        final_position = (
            activation.shape[1] - 1
        )

        patched[
            0,
            final_position,
            head,
            :
        ] = clean_final.to(
            device=activation.device,
            dtype=activation.dtype,
        )

        return patched

    return (
        hook_name,
        patch_hook,
    )


# ============================================================================
# SINGLE PATCH
# ============================================================================


@torch.no_grad()
def run_single_patch(
    model,
    corrupted_text: str,
    clean_activation: torch.Tensor,
    layer: int,
    head: int,
    positive_id: int,
    negative_id: int,
) -> float:

    prompt = make_prompt(
        corrupted_text
    )

    tokens = model.to_tokens(
        prompt,
        prepend_bos=True,
    )

    hook_name, patch_hook = (
        create_single_head_patch_hook(
            clean_activation,
            layer,
            head,
        )
    )

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (
                hook_name,
                patch_hook,
            )
        ],
    )

    final_logits = logits[
        0,
        -1,
    ]

    patched_ld = (
        final_logits[
            positive_id
        ]
        - final_logits[
            negative_id
        ]
    )

    return float(
        patched_ld
        .detach()
        .cpu()
        .item()
    )


# ============================================================================
# RECOVERY
# ============================================================================


def calculate_recovery(
    clean_ld: float,
    corrupted_ld: float,
    patched_ld: float,
) -> float:

    denominator = (
        clean_ld
        - corrupted_ld
    )

    if abs(denominator) < EPSILON:

        return float("nan")

    return (
        patched_ld
        - corrupted_ld
    ) / denominator


# ============================================================================
# BASELINE VALIDATION
# ============================================================================


def calculate_baselines(
    model,
    rows,
    positive_id,
    negative_id,
):

    section(
        "DISCOVERY BASELINE BEHAVIOR"
    )

    baseline_rows = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        pair_id = int(
            row["pair_id"]
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

        gap = (
            clean_ld
            - corrupted_ld
        )

        baseline_rows.append(
            {
                "pair_id":
                    pair_id,

                "domain":
                    row["domain"],

                "clean_ld":
                    clean_ld,

                "corrupted_ld":
                    corrupted_ld,

                "behavioral_gap":
                    gap,
            }
        )

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"Pair {pair_id:03d} | "
            f"clean={clean_ld:+.4f} | "
            f"corrupted={corrupted_ld:+.4f} | "
            f"gap={gap:+.4f}"
        )

    gaps = [
        row["behavioral_gap"]
        for row in baseline_rows
    ]

    print()

    print(
        f"Mean clean LD: "
        f"{np.mean([r['clean_ld'] for r in baseline_rows]):+.4f}"
    )

    print(
        f"Mean corrupted LD: "
        f"{np.mean([r['corrupted_ld'] for r in baseline_rows]):+.4f}"
    )

    print(
        f"Mean behavioral gap: "
        f"{np.mean(gaps):+.4f}"
    )

    if not all(
        gap > 0
        for gap in gaps
    ):

        print(
            "WARNING: Not every discovery pair "
            "has a positive behavioral gap."
        )

    else:

        print(
            "All discovery pairs have "
            "positive behavioral gaps."
        )

    return baseline_rows


# ============================================================================
# HEAD PATCHING
# ============================================================================


def run_all_head_patches(
    model,
    rows,
    baseline_rows,
    positive_id,
    negative_id,
):

    section(
        "DISCOVERY SINGLE-HEAD PATCHING"
    )

    print(
        f"Pairs: {len(rows)}"
    )

    print(
        f"Layers: {N_LAYERS}"
    )

    print(
        f"Heads/layer: {N_HEADS}"
    )

    print(
        f"Total heads: {TOTAL_HEADS}"
    )

    print()

    baseline_lookup = {
        row["pair_id"]: row
        for row in baseline_rows
    }

    results = []

    total_operations = (
        len(rows)
        * TOTAL_HEADS
    )

    completed = 0

    for pair_index, row in enumerate(
        rows,
        start=1,
    ):

        pair_id = int(
            row["pair_id"]
        )

        clean_cache = (
            load_activation_cache(
                pair_id,
                "clean",
            )
        )

        corrupted_cache = (
            load_activation_cache(
                pair_id,
                "corrupted",
            )
        )

        validate_cache_structure(
            clean_cache,
            pair_id,
            "clean",
        )

        validate_cache_structure(
            corrupted_cache,
            pair_id,
            "corrupted",
        )

        clean_ld = baseline_lookup[
            pair_id
        ]["clean_ld"]

        corrupted_ld = baseline_lookup[
            pair_id
        ]["corrupted_ld"]

        for layer in range(
            N_LAYERS
        ):

            hook_name = (
                HOOK_TEMPLATE.format(
                    layer=layer
                )
            )

            clean_layer_activation = (
                clean_cache[
                    "activations"
                ][hook_name]
            )

            for head in range(
                N_HEADS
            ):

                patched_ld = (
                    run_single_patch(
                        model=model,
                        corrupted_text=(
                            row["corrupted_text"]
                        ),
                        clean_activation=(
                            clean_layer_activation
                        ),
                        layer=layer,
                        head=head,
                        positive_id=positive_id,
                        negative_id=negative_id,
                    )
                )

                recovery = (
                    calculate_recovery(
                        clean_ld,
                        corrupted_ld,
                        patched_ld,
                    )
                )

                results.append(
                    {
                        "pair_id":
                            pair_id,

                        "domain":
                            row["domain"],

                        "layer":
                            layer,

                        "head":
                            head,

                        "head_label":
                            f"L{layer}H{head}",

                        "clean_ld":
                            clean_ld,

                        "corrupted_ld":
                            corrupted_ld,

                        "patched_ld":
                            patched_ld,

                        "behavioral_gap":
                            (
                                clean_ld
                                - corrupted_ld
                            ),

                        "recovery":
                            recovery,
                    }
                )

                completed += 1

        print(
            f"[{pair_index:02d}/{len(rows):02d}] "
            f"Pair {pair_id:03d} complete "
            f"({completed}/{total_operations} "
            f"head patches)"
        )

    return results


# ============================================================================
# SAVE RESULTS
# ============================================================================


def save_results(
    results,
):

    section(
        "SAVING SINGLE-HEAD RESULTS"
    )

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print(
        f"Results saved:"
    )

    print(
        f"  {RESULTS_CSV}"
    )

    return dataframe


# ============================================================================
# RANK HEADS
# ============================================================================


def rank_heads(
    dataframe,
):

    section(
        "RANKING ATTENTION HEADS"
    )

    grouped = (
        dataframe
        .groupby(
            [
                "layer",
                "head",
                "head_label",
            ],
            as_index=False,
        )
        .agg(
            mean_recovery=(
                "recovery",
                "mean",
            ),

            median_recovery=(
                "recovery",
                "median",
            ),

            std_recovery=(
                "recovery",
                "std",
            ),

            min_recovery=(
                "recovery",
                "min",
            ),

            max_recovery=(
                "recovery",
                "max",
            ),

            positive_recovery_fraction=(
                "recovery",
                lambda x: np.mean(
                    np.asarray(x)
                    > 0
                ),
            ),

            strong_recovery_fraction=(
                "recovery",
                lambda x: np.mean(
                    np.asarray(x)
                    >= 0.25
                ),
            ),

            full_recovery_fraction=(
                "recovery",
                lambda x: np.mean(
                    np.asarray(x)
                    >= 0.50
                ),
            ),

            n_pairs=(
                "recovery",
                "count",
            ),
        )
    )

    grouped = grouped.sort_values(
        by=[
            "mean_recovery",
            "median_recovery",
        ],
        ascending=False,
    )

    grouped[
        "rank"
    ] = np.arange(
        1,
        len(grouped) + 1,
    )

    columns = [
        "rank",
        "layer",
        "head",
        "head_label",
        "mean_recovery",
        "median_recovery",
        "std_recovery",
        "min_recovery",
        "max_recovery",
        "positive_recovery_fraction",
        "strong_recovery_fraction",
        "full_recovery_fraction",
        "n_pairs",
    ]

    grouped = grouped[
        columns
    ]

    grouped.to_csv(
        RANKING_CSV,
        index=False,
    )

    print(
        f"Ranking saved:"
    )

    print(
        f"  {RANKING_CSV}"
    )

    print()

    print(
        "TOP 20 HEADS"
    )

    print()

    print(
        grouped.head(20).to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    return grouped


# ============================================================================
# SUMMARY
# ============================================================================


def create_summary(
    ranking,
    dataframe,
):

    top_n = min(
        20,
        len(ranking),
    )

    top_heads = []

    for _, row in ranking.head(
        top_n
    ).iterrows():

        top_heads.append(
            {
                "rank":
                    int(row["rank"]),

                "layer":
                    int(row["layer"]),

                "head":
                    int(row["head"]),

                "head_label":
                    row["head_label"],

                "mean_recovery":
                    float(
                        row["mean_recovery"]
                    ),

                "median_recovery":
                    float(
                        row["median_recovery"]
                    ),

                "positive_recovery_fraction":
                    float(
                        row[
                            "positive_recovery_fraction"
                        ]
                    ),

                "strong_recovery_fraction":
                    float(
                        row[
                            "strong_recovery_fraction"
                        ]
                    ),
            }
        )

    summary = {
        "phase": 13,

        "name":
            "discovery_single_head_activation_patching",

        "model":
            MODEL_NAME,

        "dataset": {
            "split":
                "discovery",

            "pairs":
                EXPECTED_PAIRS,

            "heads_evaluated":
                TOTAL_HEADS,

            "layers":
                N_LAYERS,

            "heads_per_layer":
                N_HEADS,
        },

        "patching": {
            "activation":
                "attention_head_output",

            "hook":
                "blocks.{layer}.attn.hook_z",

            "position":
                "final_prediction_position",

            "alignment_strategy":
                "final_position_only",

            "metric":
                "logit_difference",

            "recovery_formula":
                "(patched - corrupted) / "
                "(clean - corrupted)",
        },

        "results": {
            "rows":
                int(len(dataframe)),

            "expected_rows":
                EXPECTED_PAIRS
                * TOTAL_HEADS,

            "top_heads":
                top_heads,
        },

        "heldout_protection": {
            "heldout_loaded":
                False,

            "heldout_used_for_selection":
                False,

            "heldout_used_for_ranking":
                False,
        },

        "model_weights_modified":
            False,
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()

    print(
        f"Summary saved:"
    )

    print(
        f"  {SUMMARY_JSON}"
    )

    return summary


# ============================================================================
# HEATMAP
# ============================================================================


def create_heatmap(
    ranking,
):

    section(
        "CREATING HEAD RECOVERY HEATMAP"
    )

    matrix = np.full(
        (
            N_LAYERS,
            N_HEADS,
        ),
        np.nan,
    )

    for _, row in ranking.iterrows():

        layer = int(
            row["layer"]
        )

        head = int(
            row["head"]
        )

        matrix[
            layer,
            head
        ] = row[
            "mean_recovery"
        ]

    plt.figure(
        figsize=(12, 7)
    )

    image = plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        image,
        label="Mean Recovery",
    )

    plt.xlabel(
        "Attention Head"
    )

    plt.ylabel(
        "Layer"
    )

    plt.title(
        "GPT-2-small Discovery "
        "Single-Head Activation Recovery"
    )

    plt.xticks(
        range(N_HEADS),
        [
            f"H{i}"
            for i in range(N_HEADS)
        ],
    )

    plt.yticks(
        range(N_LAYERS),
        [
            f"L{i}"
            for i in range(N_LAYERS)
        ],
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "head_recovery_heatmap.png"
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
# TOP HEAD PLOT
# ============================================================================


def create_top_heads_plot(
    ranking,
):

    top = ranking.head(
        20
    ).copy()

    top = top.sort_values(
        "mean_recovery"
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["head_label"],
        top["mean_recovery"],
    )

    plt.xlabel(
        "Mean Recovery"
    )

    plt.ylabel(
        "Attention Head"
    )

    plt.title(
        "Top 20 Discovery Attention Heads"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "top_heads.png"
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
# SANITY CHECK
# ============================================================================


def sanity_check(
    dataframe,
    ranking,
):

    section(
        "RESULT SANITY CHECK"
    )

    expected_rows = (
        EXPECTED_PAIRS
        * TOTAL_HEADS
    )

    print(
        f"Expected result rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual result rows:   "
        f"{len(dataframe)}"
    )

    if len(dataframe) != expected_rows:

        raise ValueError(
            "Unexpected number of "
            "single-head result rows."
        )

    if len(ranking) != TOTAL_HEADS:

        raise ValueError(
            "Expected exactly "
            f"{TOTAL_HEADS} ranked heads."
        )

    print(
        "Result row count: PASS"
    )

    print(
        "All 144 heads represented: PASS"
    )

    # Check that held-out data never appeared.

    if dataframe["pair_id"].isin(
        {
            1,
            8,
            12,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            26,
            27,
            32,
            33,
            34,
            37,
            41,
            42,
            50,
            52,
            54,
            55,
            58,
            61,
            63,
            64,
            66,
            70,
            73,
            74,
            78,
            79,
            85,
            86,
            87,
            88,
            91,
            92,
            96,
            98,
        }
    ).any():

        raise ValueError(
            "HELD-OUT DATA LEAKAGE DETECTED!"
        )

    print(
        "Held-out leakage check: PASS"
    )

    finite_recoveries = (
        np.isfinite(
            dataframe[
                "recovery"
            ].to_numpy()
        )
    )

    finite_fraction = (
        finite_recoveries.mean()
    )

    print(
        f"Finite recovery values: "
        f"{finite_fraction * 100:.2f}%"
    )

    if finite_fraction < 0.95:

        raise ValueError(
            "Too many non-finite "
            "recovery values."
        )

    print(
        "Recovery metric sanity check: PASS"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    section(
        "PHASE 13 — DISCOVERY "
        "SINGLE-HEAD ACTIVATION PATCHING"
    )

    print(
        "Discovery split ONLY"
    )

    print(
        f"Expected discovery pairs: "
        f"{EXPECTED_PAIRS}"
    )

    print(
        f"Attention heads: "
        f"{TOTAL_HEADS}"
    )

    print(
        "Held-out examples will NOT be loaded."
    )

    print(
        "Model weights will NOT be modified."
    )

    ensure_directories()

    # ------------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------------

    section(
        "LOADING DISCOVERY DATASET"
    )

    rows = load_discovery_dataset()

    print(
        f"Discovery pairs loaded: "
        f"{len(rows)}"
    )

    print(
        "Discovery dataset validation: PASS"
    )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    section("DEVICE")

    device = get_device()

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB",
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
    # Baselines
    # ------------------------------------------------------------------------

    baseline_rows = (
        calculate_baselines(
            model,
            rows,
            positive_id,
            negative_id,
        )
    )

    # ------------------------------------------------------------------------
    # Verify caches before patching
    # ------------------------------------------------------------------------

    section(
        "DISCOVERY ACTIVATION CACHE VALIDATION"
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):

        pair_id = int(
            row["pair_id"]
        )

        clean_cache = (
            load_activation_cache(
                pair_id,
                "clean",
            )
        )

        corrupted_cache = (
            load_activation_cache(
                pair_id,
                "corrupted",
            )
        )

        validate_cache_structure(
            clean_cache,
            pair_id,
            "clean",
        )

        validate_cache_structure(
            corrupted_cache,
            pair_id,
            "corrupted",
        )

        if index <= 3:

            print(
                f"Pair {pair_id:03d}: PASS"
            )

    print(
        f"Validated {len(rows)} discovery pairs."
    )

    print(
        "Discovery activation cache validation: PASS"
    )

    # ------------------------------------------------------------------------
    # Single-head patching
    # ------------------------------------------------------------------------

    results = (
        run_all_head_patches(
            model=model,
            rows=rows,
            baseline_rows=baseline_rows,
            positive_id=positive_id,
            negative_id=negative_id,
        )
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    dataframe = save_results(
        results
    )

    # ------------------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------------------

    ranking = rank_heads(
        dataframe
    )

    # ------------------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------------------

    sanity_check(
        dataframe,
        ranking,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    create_summary(
        ranking,
        dataframe,
    )

    # ------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------

    create_heatmap(
        ranking
    )

    create_top_heads_plot(
        ranking
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
            "Reserved: ",
            f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB",
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    section(
        "PHASE 13 COMPLETE"
    )

    print(
        "Discovery single-head patching completed."
    )

    print()

    print(
        f"Pairs evaluated: "
        f"{EXPECTED_PAIRS}"
    )

    print(
        f"Heads evaluated: "
        f"{TOTAL_HEADS}"
    )

    print(
        f"Total patch evaluations: "
        f"{len(dataframe)}"
    )

    print()

    print(
        "Results:"
    )

    print(
        f"  {RESULTS_CSV}"
    )

    print()

    print(
        "Head ranking:"
    )

    print(
        f"  {RANKING_CSV}"
    )

    print()

    print(
        "Summary:"
    )

    print(
        f"  {SUMMARY_JSON}"
    )

    print()

    print(
        "Figures:"
    )

    print(
        f"  {FIGURES_DIR}"
    )

    print()

    print(
        "Held-out examples were NOT used."
    )

    print(
        "Model weights were NOT modified."
    )

    print()


if __name__ == "__main__":
    main()