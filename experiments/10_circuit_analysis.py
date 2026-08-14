"""
PHASE 8 — CIRCUIT / MECHANISM ANALYSIS

Purpose
-------
Analyze the mechanistic evidence from Phases 5–7.

This phase does NOT:
    - modify model weights
    - train the model
    - perform activation patching
    - create new causal interventions

Instead, Phase 8:

    1. Loads GPT-2-small.
    2. Recomputes attention-head activations for clean/corrupted pairs.
    3. Extracts activations from:
           L8H9
           L9H2
           L10H4
    4. Measures clean-vs-corrupted activation differences.
    5. Analyzes token positions.
    6. Examines layer-wise progression.
    7. Relates activation differences to Phase 6 causal results.
    8. Summarizes Phase 7 multi-head evidence.
    9. Produces CSV, JSON, and plots.

Important
---------
The Phase 5 activation cache contains layer-level tensors with shape:

    (sequence_length, 64)

That cache is NOT sufficient to directly separate all 12 attention heads.

Therefore this phase uses TransformerLens run_with_cache() to obtain:

    blocks.X.attn.hook_z

whose shape is:

    (batch, sequence, heads, d_head)

For GPT-2-small:

    heads = 12
    d_head = 64

Candidate heads:

    L8H9
    L9H2
    L10H4
"""

from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens.model_bridge import TransformerBridge


# PROJECT CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "gpt2"

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentiment_pairs.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "activations"
)

PHASE6_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "patching"
    / "single_head_results.csv"
)

PHASE6_RANKING_PATH = (
    PROJECT_ROOT
    / "results"
    / "patching"
    / "head_ranking.csv"
)

PHASE7_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "patching"
    / "phase7"
    / "multi_head_results.csv"
)

PHASE7_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "patching"
    / "phase7"
    / "multi_head_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "circuit"
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)


# ============================================================================
# GPT-2-SMALL ARCHITECTURE
# ============================================================================

N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64


# ============================================================================
# CANDIDATE HEADS FROM PHASE 6 / PHASE 7
# ============================================================================

CANDIDATE_HEADS = [
    (8, 9),
    (9, 2),
    (10, 4),
]


# ============================================================================
# NUMERICAL SETTINGS
# ============================================================================

EPSILON = 1e-8


# ============================================================================
# HELPERS
# ============================================================================


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def pearson_correlation(x, y):
    """
    Calculate Pearson correlation without requiring scipy.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return float("nan")

    if np.std(x) < EPSILON:
        return float("nan")

    if np.std(y) < EPSILON:
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ============================================================================
# DATASET
# ============================================================================


def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "pair_id",
        "clean_text",
        "corrupted_text",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================================
# PROMPT
# ============================================================================


def build_prompt(text: str) -> str:

    return (
        f"Review: {text}\n"
        "Sentiment:"
    )


# ============================================================================
# MODEL TOKENIZATION
# ============================================================================


def tokenize_prompt(
    bridge: TransformerBridge,
    prompt: str,
):

    tokens = bridge.to_tokens(
        prompt,
        prepend_bos=True,
    )

    return tokens.to(
        bridge.cfg.device
    )


def get_token_strings(
    bridge: TransformerBridge,
    tokens: torch.Tensor,
):

    return bridge.to_str_tokens(
        tokens[0]
    )


# ============================================================================
# HOOK NAME
# ============================================================================


def hook_name_for_layer(
    layer: int,
) -> str:

    return (
        f"blocks.{layer}.attn.hook_z"
    )


# ============================================================================
# RUN AND CAPTURE ATTENTION HEAD ACTIVATIONS
# ============================================================================


def run_and_capture_candidate_heads(
    bridge: TransformerBridge,
    prompt: str,
):

    """
    Run the model and capture hook_z for the candidate layers.

    Returned tensors have shape:

        [sequence, heads, d_head]

    for each requested layer.
    """

    tokens = tokenize_prompt(
        bridge,
        prompt,
    )

    hook_names = [
        hook_name_for_layer(
            layer
        )
        for layer, _ in CANDIDATE_HEADS
    ]

    with torch.no_grad():

        logits, cache = (
            bridge.run_with_cache(
                tokens,
                names_filter=hook_names,
            )
        )

    captured = {}

    for layer, head in CANDIDATE_HEADS:

        hook_name = (
            hook_name_for_layer(
                layer
            )
        )

        if hook_name not in cache:

            raise KeyError(
                f"Expected hook "
                f"{hook_name!r} "
                f"was not found in cache."
            )

        activation = cache[
            hook_name
        ]

        # Expected:
        # [batch, sequence, heads, d_head]

        if activation.ndim != 4:

            raise ValueError(
                f"{hook_name}: expected "
                f"4D activation, got "
                f"{activation.ndim}D with "
                f"shape {tuple(activation.shape)}"
            )

        if activation.shape[0] != 1:

            raise ValueError(
                f"{hook_name}: expected "
                f"batch size 1, got "
                f"{activation.shape[0]}"
            )

        if activation.shape[2] != N_HEADS:

            raise ValueError(
                f"{hook_name}: expected "
                f"{N_HEADS} heads, got "
                f"{activation.shape[2]}"
            )

        if activation.shape[3] != D_HEAD:

            raise ValueError(
                f"{hook_name}: expected "
                f"d_head={D_HEAD}, got "
                f"{activation.shape[3]}"
            )

        # Remove batch dimension.
        activation = (
            activation[0]
            .detach()
            .float()
            .cpu()
        )

        # Select requested head.
        head_activation = (
            activation[:, head, :]
        )

        captured[
            (layer, head)
        ] = {
            "all_heads": activation,
            "head": head_activation,
        }

    token_strings = get_token_strings(
        bridge,
        tokens,
    )

    del logits
    del cache
    del tokens

    return captured, token_strings


# ============================================================================
# ACTIVATION DIFFERENCE
# ============================================================================


def calculate_activation_difference(
    clean_activation: torch.Tensor,
    corrupted_activation: torch.Tensor,
):
    """
    Compare clean and corrupted activations using only the
    overlapping token positions.

    Clean/corrupted prompts can have different tokenized lengths.
    For example:

        clean      -> [12, 64]
        corrupted  -> [15, 64]

    We therefore compare:

        clean      -> [12, 64]
        corrupted  -> [12, 64]

    rather than requiring identical sequence lengths.

    The feature dimension (d_head=64) must still match.
    """

    if clean_activation.ndim != 2:
        raise ValueError(
            "Expected clean activation shape "
            "[sequence, d_head]. Got "
            f"{tuple(clean_activation.shape)}"
        )

    if corrupted_activation.ndim != 2:
        raise ValueError(
            "Expected corrupted activation shape "
            "[sequence, d_head]. Got "
            f"{tuple(corrupted_activation.shape)}"
        )

    # The feature/head dimension must match.
    if clean_activation.shape[1] != corrupted_activation.shape[1]:
        raise ValueError(
            "Clean/corrupted d_head mismatch: "
            f"{clean_activation.shape[1]} vs "
            f"{corrupted_activation.shape[1]}"
        )

    # Compare only overlapping token positions.
    sequence_length = min(
        clean_activation.shape[0],
        corrupted_activation.shape[0],
    )

    if sequence_length == 0:
        raise ValueError(
            "Cannot compare activations with zero sequence length."
        )

    clean = clean_activation[
        :sequence_length
    ]

    corrupted = corrupted_activation[
        :sequence_length
    ]

    difference = (
        clean - corrupted
    )

    difference_norm = torch.linalg.vector_norm(
        difference,
        dim=-1,
    )

    clean_norm = torch.linalg.vector_norm(
        clean,
        dim=-1,
    )

    corrupted_norm = torch.linalg.vector_norm(
        corrupted,
        dim=-1,
    )

    return {
        "clean": clean,
        "corrupted": corrupted,
        "difference": difference,
        "difference_norm": difference_norm,
        "clean_norm": clean_norm,
        "corrupted_norm": corrupted_norm,
        "sequence_length": sequence_length,
    }

# POSITION STATISTICS

def build_position_rows(
    pair_id: int,
    layer: int,
    head: int,
    token_strings: list[str],
    stats: dict,
):

    values = (
        stats[
            "difference_norm"
        ]
        .detach()
        .cpu()
        .tolist()
    )

    rows = []

    for position, value in enumerate(
        values
    ):

        token = (
            token_strings[position]
            if position < len(
                token_strings
            )
            else "<UNKNOWN>"
        )

        rows.append(
            {
                "pair_id": pair_id,
                "layer": layer,
                "head": head,
                "head_label": head_label(
                    layer,
                    head,
                ),
                "position": position,
                "token": token,
                "difference_norm": float(
                    value
                ),
            }
        )

    return rows


# ============================================================================
# PHASE 6 RESULT PARSING
# ============================================================================


def load_phase6_results():

    if not PHASE6_RESULTS_PATH.exists():

        print(
            "\nWARNING: Phase 6 results not found:"
        )

        print(
            PHASE6_RESULTS_PATH
        )

        return None

    df = pd.read_csv(
        PHASE6_RESULTS_PATH
    )

    return df


# ============================================================================
# IDENTIFY PHASE 6 COLUMNS
# ============================================================================


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
):

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================================
# BUILD PHASE 6 HEAD RECOVERY
# ============================================================================


def extract_candidate_head_recovery(
    phase6_df: pd.DataFrame,
):

    if phase6_df is None:
        return pd.DataFrame()

    head_column = find_column(
        phase6_df,
        [
            "head",
            "head_label",
            "condition",
        ],
    )

    pair_column = find_column(
        phase6_df,
        [
            "pair_id",
            "pair",
            "id",
        ],
    )

    recovery_column = find_column(
        phase6_df,
        [
            "recovery",
            "normalized_recovery",
            "recovery_fraction",
        ],
    )

    if (
        head_column is None
        or pair_column is None
        or recovery_column is None
    ):

        print(
            "\nWARNING: Could not identify "
            "Phase 6 columns needed for "
            "activation/recovery correlation."
        )

        print(
            "Available columns:",
            list(
                phase6_df.columns
            ),
        )

        return pd.DataFrame()

    output_rows = []

    for layer, head in CANDIDATE_HEADS:

        label = head_label(
            layer,
            head,
        )

        subset = phase6_df[
            phase6_df[
                head_column
            ].astype(str)
            .str.strip()
            == label
        ].copy()

        if subset.empty:
            continue

        subset["pair_id"] = (
            pd.to_numeric(
                subset[
                    pair_column
                ],
                errors="coerce",
            )
        )

        subset["recovery"] = (
            pd.to_numeric(
                subset[
                    recovery_column
                ],
                errors="coerce",
            )
        )

        subset = subset[
            [
                "pair_id",
                "recovery",
            ]
        ].dropna()

        subset[
            "head_label"
        ] = label

        output_rows.append(
            subset
        )

    if not output_rows:
        return pd.DataFrame()

    return pd.concat(
        output_rows,
        ignore_index=True,
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 70)
    print(
        "PHASE 8 — CIRCUIT / MECHANISM ANALYSIS"
    )
    print("=" * 70)

    # ========================================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # LOAD DATASET
    # ========================================================================

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    dataset = load_dataset()

    print(
        f"Pairs loaded: {len(dataset)}"
    )

    # ========================================================================
    # CANDIDATE HEADS
    # ========================================================================

    print("\n" + "=" * 70)
    print("CANDIDATE HEADS")
    print("=" * 70)

    for layer, head in CANDIDATE_HEADS:

        print(
            f"{head_label(layer, head)}"
        )

    # ========================================================================
    # DEVICE
    # ========================================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("DEVICE")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ========================================================================
    # PHASE 6
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 6 RESULTS")
    print("=" * 70)

    phase6_df = load_phase6_results()

    if phase6_df is not None:

        print(
            f"Single-head results: "
            f"{len(phase6_df)} rows"
        )

    phase6_recovery_df = (
        extract_candidate_head_recovery(
            phase6_df
        )
    )

    # ========================================================================
    # PHASE 7
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 7 RESULTS")
    print("=" * 70)

    phase7_df = None
    phase7_summary_df = None

    if PHASE7_RESULTS_PATH.exists():

        phase7_df = pd.read_csv(
            PHASE7_RESULTS_PATH
        )

        print(
            f"Multi-head results: "
            f"{len(phase7_df)} rows"
        )

    else:

        print(
            "WARNING: Phase 7 detailed "
            "results not found."
        )

    if PHASE7_SUMMARY_PATH.exists():

        phase7_summary_df = pd.read_csv(
            PHASE7_SUMMARY_PATH
        )

        print(
            f"Multi-head summary: "
            f"{len(phase7_summary_df)} rows"
        )

    else:

        print(
            "WARNING: Phase 7 summary "
            "not found."
        )

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    print("\n" + "=" * 70)
    print("MODEL LOADING")
    print("=" * 70)

    bridge = (
        TransformerBridge.boot_transformers(
            MODEL_NAME,
            device=device,
        )
    )

    bridge.eval()

    print(
        "GPT-2-small loaded successfully."
    )

    # ========================================================================
    # ARCHITECTURE VALIDATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("ARCHITECTURE VALIDATION")
    print("=" * 70)

    print(
        f"Layers:      {bridge.cfg.n_layers}"
    )

    print(
        f"Heads/layer: {bridge.cfg.n_heads}"
    )

    print(
        f"Head dim:    {bridge.cfg.d_head}"
    )

    if (
        bridge.cfg.n_layers != N_LAYERS
        or bridge.cfg.n_heads != N_HEADS
        or bridge.cfg.d_head != D_HEAD
    ):

        raise RuntimeError(
            "Unexpected GPT-2-small architecture."
        )

    print(
        "Architecture validation: PASS"
    )

    # ========================================================================
    # VERIFY HOOKS
    # ========================================================================

    print("\n" + "=" * 70)
    print("ATTENTION HOOK VALIDATION")
    print("=" * 70)

    for layer, head in CANDIDATE_HEADS:

        hook_name = (
            hook_name_for_layer(
                layer
            )
        )

        hook = bridge.get_hook_point(
            hook_name
        )

        if hook is None:

            raise RuntimeError(
                f"Hook not found: "
                f"{hook_name}"
            )

        print(
            f"{hook_name}: PASS"
        )

    # ========================================================================
    # ACTIVATION ANALYSIS
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "CLEAN vs CORRUPTED ATTENTION-HEAD ACTIVATIONS"
    )
    print("=" * 70)

    head_rows = []
    position_rows = []

    # Keep one representative example.
    representative_data = None

    for index, row in dataset.iterrows():

        pair_id = int(
            row["pair_id"]
        )

        clean_prompt = build_prompt(
            row["clean_text"]
        )

        corrupted_prompt = build_prompt(
            row["corrupted_text"]
        )

        # ------------------------------------------------------------
        # CLEAN
        # ------------------------------------------------------------

        clean_captured, clean_tokens = (
            run_and_capture_candidate_heads(
                bridge,
                clean_prompt,
            )
        )

        # ------------------------------------------------------------
        # CORRUPTED
        # ------------------------------------------------------------

        corrupted_captured, corrupted_tokens = (
            run_and_capture_candidate_heads(
                bridge,
                corrupted_prompt,
            )
        )

        # ------------------------------------------------------------
        # TOKEN ALIGNMENT
        # ------------------------------------------------------------

        if len(clean_tokens) != len(
            corrupted_tokens
        ):

            print(
                f"\nWARNING Pair {pair_id}: "
                f"clean sequence length="
                f"{len(clean_tokens)}, "
                f"corrupted sequence length="
                f"{len(corrupted_tokens)}"
            )

            print(
                "Using the overlapping token "
                "positions for activation "
                "difference analysis."
            )

        for layer, head in CANDIDATE_HEADS:

            clean_activation = (
                clean_captured[
                    (layer, head)
                ]["head"]
            )

            corrupted_activation = (
                corrupted_captured[
                    (layer, head)
                ]["head"]
            )

            stats = (
                calculate_activation_difference(
                    clean_activation,
                    corrupted_activation,
                )
            )

            difference = stats[
                "difference"
            ]

            difference_norm = stats[
                "difference_norm"
            ]

            mean_signed_difference = (
                float(
                    difference
                    .mean()
                    .item()
                )
            )

            mean_absolute_difference = (
                float(
                    difference
                    .abs()
                    .mean()
                    .item()
                )
            )

            mean_difference_norm = (
                float(
                    difference_norm
                    .mean()
                    .item()
                )
            )

            max_difference_norm = (
                float(
                    difference_norm
                    .max()
                    .item()
                )
            )

            max_position = int(
                torch.argmax(
                    difference_norm
                ).item()
            )

            max_token = (
                clean_tokens[
                    max_position
                ]
                if max_position
                < len(clean_tokens)
                else "<UNKNOWN>"
            )

            head_rows.append(
                {
                    "pair_id": pair_id,
                    "layer": layer,
                    "head": head,
                    "head_label": head_label(
                        layer,
                        head,
                    ),
                    "sequence_length": int(
                        difference_norm.shape[0]
                    ),
                    "mean_signed_difference": (
                        mean_signed_difference
                    ),
                    "mean_absolute_difference": (
                        mean_absolute_difference
                    ),
                    "mean_difference_norm": (
                        mean_difference_norm
                    ),
                    "max_difference_norm": (
                        max_difference_norm
                    ),
                    "max_difference_position": (
                        max_position
                    ),
                    "max_difference_token": (
                        max_token
                    ),
                }
            )

            position_rows.extend(
                build_position_rows(
                    pair_id,
                    layer,
                    head,
                    clean_tokens,
                    stats,
                )
            )

        if representative_data is None:

            representative_data = {
                "pair_id": pair_id,
                "clean_text": row[
                    "clean_text"
                ],
                "corrupted_text": row[
                    "corrupted_text"
                ],
                "clean_tokens": clean_tokens,
                "corrupted_tokens": corrupted_tokens,
                "clean_captured": clean_captured,
                "corrupted_captured": corrupted_captured,
            }

        print(
            f"[{index + 1:02d}/{len(dataset):02d}] "
            f"Pair {pair_id:02d} analyzed"
        )

        # Free temporary tensors.
        del clean_captured
        del corrupted_captured

    # ========================================================================
    # DATAFRAMES
    # ========================================================================

    head_activation_df = pd.DataFrame(
        head_rows
    )

    position_df = pd.DataFrame(
        position_rows
    )

    # ========================================================================
    # HEAD SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "CANDIDATE HEAD ACTIVATION SUMMARY"
    )
    print("=" * 70)

    head_summary = (
        head_activation_df
        .groupby(
            [
                "layer",
                "head",
                "head_label",
            ],
            as_index=False,
        )
        .agg(
            mean_difference_norm=(
                "mean_difference_norm",
                "mean",
            ),
            median_difference_norm=(
                "mean_difference_norm",
                "median",
            ),
            mean_max_difference_norm=(
                "max_difference_norm",
                "mean",
            ),
            mean_absolute_difference=(
                "mean_absolute_difference",
                "mean",
            ),
            mean_signed_difference=(
                "mean_signed_difference",
                "mean",
            ),
        )
        .sort_values(
            "mean_difference_norm",
            ascending=False,
        )
    )

    print(
        head_summary.to_string(
            index=False
        )
    )

    # ========================================================================
    # LAYER SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "LAYER-WISE CANDIDATE ANALYSIS"
    )
    print("=" * 70)

    layer_summary = (
        head_activation_df
        .groupby(
            "layer",
            as_index=False,
        )
        .agg(
            mean_difference_norm=(
                "mean_difference_norm",
                "mean",
            ),
            mean_absolute_difference=(
                "mean_absolute_difference",
                "mean",
            ),
            max_difference_norm=(
                "max_difference_norm",
                "max",
            ),
        )
        .sort_values(
            "layer"
        )
    )

    print(
        layer_summary.to_string(
            index=False
        )
    )

    # ========================================================================
    # POSITION SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "TOKEN-POSITION ANALYSIS"
    )
    print("=" * 70)

    position_summary = (
        position_df
        .groupby(
            [
                "head_label",
                "position",
                "token",
            ],
            as_index=False,
        )
        .agg(
            mean_difference_norm=(
                "difference_norm",
                "mean",
            ),
            max_difference_norm=(
                "difference_norm",
                "max",
            ),
        )
        .sort_values(
            "mean_difference_norm",
            ascending=False,
        )
    )

    for layer, head in CANDIDATE_HEADS:

        label = head_label(
            layer,
            head,
        )

        subset = (
            position_summary[
                position_summary[
                    "head_label"
                ]
                == label
            ]
            .sort_values(
                "mean_difference_norm",
                ascending=False,
            )
            .head(5)
        )

        print(
            f"\n{label} strongest token positions:"
        )

        print(
            subset.to_string(
                index=False
            )
        )

    # ========================================================================
    # REPRESENTATIVE EXAMPLE
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "REPRESENTATIVE TOKEN ANALYSIS"
    )
    print("=" * 70)

    if representative_data is not None:

        print(
            f"Pair: "
            f"{representative_data['pair_id']}"
        )

        print(
            f"Clean: "
            f"{representative_data['clean_text']}"
        )

        print(
            f"Corrupted: "
            f"{representative_data['corrupted_text']}"
        )

        print(
            "\nClean tokens:"
        )

        for position, token in enumerate(
            representative_data[
                "clean_tokens"
            ]
        ):

            print(
                f"  {position:02d}: "
                f"{token!r}"
            )

        print(
            "\nCorrupted tokens:"
        )

        for position, token in enumerate(
            representative_data[
                "corrupted_tokens"
            ]
        ):

            print(
                f"  {position:02d}: "
                f"{token!r}"
            )

        representative_rows = []

        clean_tokens = (
            representative_data[
                "clean_tokens"
            ]
        )

        corrupted_tokens = (
            representative_data[
                "corrupted_tokens"
            ]
        )

        for layer, head in CANDIDATE_HEADS:

            clean_head = (
                representative_data[
                    "clean_captured"
                ][
                    (layer, head)
                ]["head"]
            )

            corrupted_head = (
                representative_data[
                    "corrupted_captured"
                ][
                    (layer, head)
                ]["head"]
            )

            stats = (
                calculate_activation_difference(
                    clean_head,
                    corrupted_head,
                )
            )

            norms = (
                stats[
                    "difference_norm"
                ]
                .tolist()
            )

            for position, value in enumerate(
                norms
            ):

                clean_token = (
                    clean_tokens[position]
                    if position
                    < len(clean_tokens)
                    else "<UNKNOWN>"
                )

                corrupted_token = (
                    corrupted_tokens[position]
                    if position
                    < len(corrupted_tokens)
                    else "<UNKNOWN>"
                )

                representative_rows.append(
                    {
                        "pair_id": representative_data[
                            "pair_id"
                        ],
                        "layer": layer,
                        "head": head,
                        "head_label": head_label(
                            layer,
                            head,
                        ),
                        "position": position,
                        "clean_token": clean_token,
                        "corrupted_token": corrupted_token,
                        "difference_norm": float(
                            value
                        ),
                    }
                )

        representative_position_df = (
            pd.DataFrame(
                representative_rows
            )
        )

    else:

        representative_position_df = (
            pd.DataFrame()
        )

    # ========================================================================
    # ACTIVATION ↔ PHASE 6 RECOVERY
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "ACTIVATION <-> PHASE 6 RECOVERY"
    )
    print("=" * 70)

    correlation_rows = []

    if not phase6_recovery_df.empty:

        for layer, head in CANDIDATE_HEADS:

            label = head_label(
                layer,
                head,
            )

            activation_subset = (
                head_activation_df[
                    head_activation_df[
                        "head_label"
                    ]
                    == label
                ][
                    [
                        "pair_id",
                        "mean_difference_norm",
                    ]
                ]
            )

            recovery_subset = (
                phase6_recovery_df[
                    phase6_recovery_df[
                        "head_label"
                    ]
                    == label
                ][
                    [
                        "pair_id",
                        "recovery",
                    ]
                ]
            )

            merged = pd.merge(
                activation_subset,
                recovery_subset,
                on="pair_id",
                how="inner",
            )

            correlation = (
                pearson_correlation(
                    merged[
                        "mean_difference_norm"
                    ],
                    merged[
                        "recovery"
                    ],
                )
                if len(merged) >= 3
                else float("nan")
            )

            correlation_rows.append(
                {
                    "head": label,
                    "n_pairs": len(merged),
                    "pearson_r": correlation,
                }
            )

            print(
                f"{label}: "
                f"n={len(merged)}, "
                f"r={correlation:+.4f}"
                if math.isfinite(
                    correlation
                )
                else
                f"{label}: "
                f"n={len(merged)}, "
                f"r=NaN"
            )

    else:

        print(
            "Phase 6 recovery data could not "
            "be parsed; correlation skipped."
        )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    # ========================================================================
    # PHASE 7 EVIDENCE
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "PHASE 7 MULTI-HEAD EVIDENCE"
    )
    print("=" * 70)

    best_phase7 = None

    if phase7_summary_df is not None:

        print(
            phase7_summary_df.to_string(
                index=False
            )
        )

        if (
            "mean_recovery"
            in phase7_summary_df.columns
        ):

            ranked = (
                phase7_summary_df
                .sort_values(
                    "mean_recovery",
                    ascending=False,
                )
            )

            best_phase7 = (
                ranked.iloc[0]
            )

            print(
                "\nBest multi-head condition:"
            )

            print(
                f"  {best_phase7['condition']}"
            )

            print(
                f"  Mean recovery: "
                f"{float(best_phase7['mean_recovery']) * 100:.2f}%"
            )

    else:

        print(
            "Phase 7 summary unavailable."
        )

    # ========================================================================
    # MECHANISTIC SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print(
        "MECHANISTIC SUMMARY"
    )
    print("=" * 70)

    top_activation_head = (
        head_summary.iloc[0]
    )

    print(
        "Largest candidate-head "
        "activation difference:"
    )

    print(
        f"  {top_activation_head['head_label']}"
    )

    print(
        f"  Mean difference norm: "
        f"{float(top_activation_head['mean_difference_norm']):.6f}"
    )

    if best_phase7 is not None:

        print(
            "\nStrongest Phase 7 causal combination:"
        )

        print(
            f"  {best_phase7['condition']}"
        )

        print(
            f"  Mean recovery: "
            f"{float(best_phase7['mean_recovery']) * 100:.2f}%"
        )

    print(
        "\nInterpretation rule:"
    )

    print(
        "Activation differences identify where "
        "clean and corrupted inputs differ internally."
    )

    print(
        "Phase 6/7 patching results provide the "
        "causal evidence."
    )

    print(
        "Phase 8 combines these two sources of "
        "evidence but does not claim a complete "
        "circuit solely from activation differences."
    )

    # ========================================================================
    # SAVE CSV RESULTS
    # ========================================================================

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    head_path = (
        OUTPUT_DIR
        / "candidate_head_activation_summary.csv"
    )

    position_path = (
        OUTPUT_DIR
        / "token_position_activation.csv"
    )

    layer_path = (
        OUTPUT_DIR
        / "layer_summary.csv"
    )

    representative_path = (
        OUTPUT_DIR
        / "representative_token_positions.csv"
    )

    correlation_path = (
        OUTPUT_DIR
        / "activation_behavior_correlations.csv"
    )

    head_activation_df.to_csv(
        head_path,
        index=False,
    )

    position_summary.to_csv(
        position_path,
        index=False,
    )

    layer_summary.to_csv(
        layer_path,
        index=False,
    )

    representative_position_df.to_csv(
        representative_path,
        index=False,
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    print(
        f"  {head_path}"
    )

    print(
        f"  {position_path}"
    )

    print(
        f"  {layer_path}"
    )

    print(
        f"  {representative_path}"
    )

    print(
        f"  {correlation_path}"
    )

    # ========================================================================
    # COPY PHASE 7 SUMMARY
    # ========================================================================

    if phase7_summary_df is not None:

        phase7_copy_path = (
            OUTPUT_DIR
            / "phase7_condition_summary.csv"
        )

        phase7_summary_df.to_csv(
            phase7_copy_path,
            index=False,
        )

        print(
            f"  {phase7_copy_path}"
        )

    # ========================================================================
    # SAVE JSON REPORT
    # ========================================================================

    report = {
        "phase": 8,
        "experiment": (
            "circuit_mechanism_analysis"
        ),
        "model": MODEL_NAME,
        "device": device,
        "dataset_pairs": int(
            len(dataset)
        ),
        "candidate_heads": [
            head_label(
                layer,
                head,
            )
            for layer, head
            in CANDIDATE_HEADS
        ],
        "method": (
            "TransformerLens "
            "run_with_cache on "
            "blocks.X.attn.hook_z"
        ),
        "phase6_results_available": (
            phase6_df is not None
        ),
        "phase7_results_available": (
            phase7_summary_df is not None
        ),
        "head_activation_summary": (
            head_summary
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "layer_summary": (
            layer_summary
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "activation_behavior_correlations": (
            correlation_df
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "circuit_analysis_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"  {json_path}"
    )

    # ========================================================================
    # PLOTS
    # ========================================================================

    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    try:

        import matplotlib.pyplot as plt

        # ------------------------------------------------------------
        # Plot 1 — Candidate head activation differences
        # ------------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            head_summary[
                "head_label"
            ],
            head_summary[
                "mean_difference_norm"
            ],
        )

        plt.xlabel(
            "Candidate Attention Head"
        )

        plt.ylabel(
            "Mean Clean-Corrupted Activation Difference Norm"
        )

        plt.title(
            "Candidate Head Activation Differences"
        )

        plt.tight_layout()

        path = (
            FIGURES_DIR
            / "candidate_head_activation_difference.png"
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()

        print(
            f"  {path}"
        )

        # ------------------------------------------------------------
        # Plot 2 — Layer progression
        # ------------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            layer_summary[
                "layer"
            ],
            layer_summary[
                "mean_difference_norm"
            ],
            marker="o",
        )

        plt.xlabel(
            "Layer"
        )

        plt.ylabel(
            "Mean Activation Difference Norm"
        )

        plt.title(
            "Candidate Mechanism Across Layers"
        )

        plt.xticks(
            layer_summary[
                "layer"
            ]
        )

        plt.tight_layout()

        path = (
            FIGURES_DIR
            / "layer_progression.png"
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()

        print(
            f"  {path}"
        )

        # ------------------------------------------------------------
        # Plot 3 — Token-position activation
        # ------------------------------------------------------------

        plt.figure(
            figsize=(11, 6)
        )

        for layer, head in CANDIDATE_HEADS:

            label = head_label(
                layer,
                head,
            )

            subset = (
                position_summary[
                    position_summary[
                        "head_label"
                    ]
                    == label
                ]
                .sort_values(
                    "position"
                )
            )

            if subset.empty:
                continue

            plt.plot(
                subset[
                    "position"
                ],
                subset[
                    "mean_difference_norm"
                ],
                marker="o",
                label=label,
            )

        plt.xlabel(
            "Token Position"
        )

        plt.ylabel(
            "Mean Activation Difference Norm"
        )

        plt.title(
            "Candidate Head Activation "
            "Difference by Token Position"
        )

        plt.legend()

        plt.tight_layout()

        path = (
            FIGURES_DIR
            / "token_position_activation.png"
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()

        print(
            f"  {path}"
        )

        # ------------------------------------------------------------
        # Plot 4 — Phase 7 comparison
        # ------------------------------------------------------------

        if (
            phase7_summary_df is not None
            and "condition"
            in phase7_summary_df.columns
            and "mean_recovery"
            in phase7_summary_df.columns
        ):

            plt.figure(
                figsize=(11, 6)
            )

            ordered = (
                phase7_summary_df
                .sort_values(
                    "mean_recovery",
                    ascending=False,
                )
            )

            plt.bar(
                ordered[
                    "condition"
                ],
                ordered[
                    "mean_recovery"
                ] * 100,
            )

            plt.xlabel(
                "Patching Condition"
            )

            plt.ylabel(
                "Mean Recovery (%)"
            )

            plt.title(
                "Phase 7 Multi-Head Recovery"
            )

            plt.xticks(
                rotation=30,
                ha="right",
            )

            plt.tight_layout()

            path = (
                FIGURES_DIR
                / "phase7_multi_head_recovery.png"
            )

            plt.savefig(
                path,
                dpi=200,
            )

            plt.close()

            print(
                f"  {path}"
            )

    except ImportError:

        print(
            "matplotlib is not available. "
            "Plots skipped."
        )

    # ========================================================================
    # CLEANUP
    # ========================================================================

    del bridge

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================================
    # FINAL
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("=" * 70)

    print(
        "\nAnalyzed candidate heads:"
    )

    for layer, head in CANDIDATE_HEADS:

        print(
            f"  {head_label(layer, head)}"
        )

    print(
        "\nMethod:"
    )

    print(
        "  Recomputed blocks.X.attn.hook_z "
        "with TransformerLens"
    )

    print(
        "  Clean vs corrupted activation analysis"
    )

    print(
        "  Token-position analysis"
    )

    print(
        "  Layer progression analysis"
    )

    print(
        "  Phase 6 recovery correlation"
    )

    print(
        "  Phase 7 multi-head comparison"
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nNo model weights were modified."
    )

    print(
        "No activation patching was performed "
        "in Phase 8."
    )

if __name__ == "__main__":
    main()