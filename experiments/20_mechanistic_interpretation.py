"""
Phase 17 — Mechanistic Interpretation

Purpose
-------
Interpret the already-discovered frozen candidate heads using the
DISCOVERY split only.

Frozen circuit:
    L10H4
    L8H9
    L9H2

Important experimental rules
----------------------------
1. No head selection.
2. No circuit search.
3. No activation patching.
4. No model-weight modification.
5. Held-out examples are never loaded.
6. Phase 13/14 results are treated as existing evidence.
7. Activation differences are interpreted as correlational/mechanistic
   evidence, not as new causal evidence.
8. Causal evidence comes from the completed Phase 13/14/15/16 experiments.

Outputs
-------
results/mechanistic/
    candidate_head_activation_summary.csv
    token_position_activation.csv
    layer_progression.csv
    phase13_candidate_head_evidence.csv
    phase14_circuit_evidence.csv
    mechanistic_summary.json
    mechanistic_interpretation.md
    figures/
        candidate_head_activation_difference.png
        token_position_activation_difference.png
        layer_progression.png
        circuit_vs_single_discovery.png
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"

EXPANDED_DATASET_PATH = RAW_DIR / "expanded_sentiment_pairs.csv"
DISCOVERY_DATASET_PATH = SPLIT_DIR / "discovery_pairs.csv"
HELDOUT_DATASET_PATH = SPLIT_DIR / "heldout_pairs.csv"

EXPANDED_ACTIVATION_DIR = (
    DATA_DIR / "expanded_activations"
)

DISCOVERY_ACTIVATION_DIR = (
    EXPANDED_ACTIVATION_DIR / "discovery"
)

HELDOUT_ACTIVATION_DIR = (
    EXPANDED_ACTIVATION_DIR / "heldout"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "mechanistic"
)

FIGURES_DIR = RESULTS_DIR / "figures"

# Phase 13
PHASE13_DIR = (
    PROJECT_ROOT / "results" / "expanded_single_head"
)

PHASE13_RANKING_PATH = (
    PHASE13_DIR / "discovery_head_ranking.csv"
)

PHASE13_RESULTS_PATH = (
    PHASE13_DIR / "discovery_single_head_results.csv"
)

# Phase 14
PHASE14_DIR = (
    PROJECT_ROOT / "results" / "expanded_multi_head"
)

PHASE14_RESULTS_PATH = (
    PHASE14_DIR / "discovery_multi_head_results.csv"
)

PHASE14_SUMMARY_PATH = (
    PHASE14_DIR / "discovery_multi_head_summary.csv"
)

# Phase 15
PHASE15_DIR = (
    PROJECT_ROOT / "results" / "heldout_generalization"
)

PHASE15_SUMMARY_PATH = (
    PHASE15_DIR / "heldout_generalization_summary.csv"
)

# Phase 16
PHASE16_DIR = (
    PROJECT_ROOT / "results" / "heldout_statistics"
)

PHASE16_SUMMARY_PATH = (
    PHASE16_DIR / "phase16_statistical_summary.json"
)


# ============================================================================
# FROZEN EXPERIMENTAL CONFIGURATION
# ============================================================================

MODEL_NAME = "gpt2"

FROZEN_HEADS = [
    (10, 4),
    (8, 9),
    (9, 2),
]

FROZEN_HEAD_LABELS = [
    "L10H4",
    "L8H9",
    "L9H2",
]

FROZEN_CIRCUIT = "L10H4 + L8H9 + L9H2"

EXPECTED_DISCOVERY_PAIRS = 60
EXPECTED_HELDOUT_PAIRS = 40
EXPECTED_TOTAL_PAIRS = 100

N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64


# ============================================================================
# PRINTING HELPERS
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def subsection(title: str) -> None:
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def fail(message: str) -> None:
    raise RuntimeError(message)


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def create_output_directories() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# DATASET LOADING
# ============================================================================

def validate_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_pair_id(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["pair_id"] = pd.to_numeric(
        df["pair_id"],
        errors="raise",
    ).astype(int)

    return df


def load_discovery_dataset() -> pd.DataFrame:

    section(
        "LOADING PHASE 11 DISCOVERY DATASET"
    )

    # ------------------------------------------------------------------------
    # CRITICAL SAFETY CHECK
    # ------------------------------------------------------------------------
    #
    # Do not silently fall back to the old 20-pair pilot dataset.
    #
    # ------------------------------------------------------------------------

    if not EXPANDED_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Phase 11 expanded dataset not found:\n"
            f"{EXPANDED_DATASET_PATH}\n\n"
            "Do not use data/raw/sentiment_pairs.csv."
        )

    if not DISCOVERY_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Phase 11 discovery split not found:\n"
            f"{DISCOVERY_DATASET_PATH}"
        )

    if HELDOUT_DATASET_PATH.exists():
        print(
            "Held-out dataset exists, "
            "but it will NOT be loaded."
        )

    print(
        f"Expanded dataset:\n"
        f"  {EXPANDED_DATASET_PATH}"
    )

    print(
        f"Discovery dataset:\n"
        f"  {DISCOVERY_DATASET_PATH}"
    )

    # ------------------------------------------------------------------------
    # Read discovery split directly.
    # ------------------------------------------------------------------------

    df = pd.read_csv(DISCOVERY_DATASET_PATH)

    validate_columns(
        df,
        [
            "pair_id",
            "domain",
            "clean_text",
            "corrupted_text",
            "clean_label",
            "corrupted_label",
        ],
        "Discovery dataset",
    )

    df = df.rename(
        columns={
            "clean_text": "clean",
            "corrupted_text": "corrupted",
        }
    )

    df = normalize_pair_id(df)

    # ------------------------------------------------------------------------
    # Discovery split must contain exactly 60 examples.
    # ------------------------------------------------------------------------

    if len(df) != EXPECTED_DISCOVERY_PAIRS:
        raise ValueError(
            "Unexpected discovery dataset size.\n"
            f"Expected: {EXPECTED_DISCOVERY_PAIRS}\n"
            f"Found:    {len(df)}"
        )

    # ------------------------------------------------------------------------
    # Validate uniqueness.
    # ------------------------------------------------------------------------

    if df["pair_id"].duplicated().any():
        duplicated = (
            df.loc[
                df["pair_id"].duplicated(keep=False),
                "pair_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate pair IDs in discovery dataset: "
            f"{duplicated}"
        )

    if df["clean"].duplicated().any():
        raise ValueError(
            "Duplicate clean sentences found "
            "in discovery dataset."
        )

    if df["corrupted"].duplicated().any():
        raise ValueError(
            "Duplicate corrupted sentences found "
            "in discovery dataset."
        )

    # ------------------------------------------------------------------------
    # Confirm IDs are within Phase 11's 1..100 range.
    # ------------------------------------------------------------------------

    if not df["pair_id"].between(
        1,
        EXPECTED_TOTAL_PAIRS,
    ).all():

        raise ValueError(
            "Discovery pair IDs fall outside "
            "the Phase 11 1..100 range."
        )

    print()
    print(
        f"Discovery pairs loaded: {len(df)}"
    )

    print(
        f"Unique domains: "
        f"{df['domain'].nunique()}"
    )

    print()
    print("Domain distribution:")

    domain_counts = (
        df["domain"]
        .value_counts()
        .sort_index()
    )

    for domain, count in domain_counts.items():
        print(
            f"  {str(domain):12s} {count:3d}"
        )

    print()
    print(
        "Discovery dataset validation: PASS"
    )

    return df.reset_index(drop=True)


# ============================================================================
# CACHE LOADING
# ============================================================================

def load_torch_cache(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"Activation cache missing:\n{path}"
        )

    try:
        cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        cache = torch.load(
            path,
            map_location="cpu",
        )

    return cache


def extract_layer_activation(cache, layer):
    """
    Extract a GPT-2 attention-head activation from the Phase 11
    expanded activation-cache format.

    Cache format:

        {
            "prompt": ...,
            "text": ...,
            "sequence_length": ...,
            "token_ids": ...,
            "tokens": ...,
            "activations": {
                "blocks.0.attn.hook_z": tensor(...),
                "blocks.1.attn.hook_z": tensor(...),
                ...
                "blocks.11.attn.hook_z": tensor(...)
            },
            ...
        }

    Returns:
        Tensor with shape [sequence, heads, d_head]
    """

    if not isinstance(cache, dict):
        raise TypeError(
            f"Expected activation cache to be a dict, "
            f"got {type(cache)}"
        )

    # Phase 11 stores activations under the "activations" key.
    if "activations" not in cache:
        raise ValueError(
            "Activation cache does not contain an 'activations' key.\n"
            f"Available top-level keys: {list(cache.keys())}"
        )

    activations = cache["activations"]

    if not isinstance(activations, dict):
        raise ValueError(
            "cache['activations'] must be a dictionary.\n"
            f"Got: {type(activations)}"
        )

    hook_name = f"blocks.{layer}.attn.hook_z"

    if hook_name not in activations:
        available = list(activations.keys())

        raise ValueError(
            f"Could not locate attention activation for layer {layer}.\n"
            f"Expected key: {hook_name}\n"
            f"Available activation keys: {available}"
        )

    activation = activations[hook_name]

    if not isinstance(activation, torch.Tensor):
        activation = torch.as_tensor(activation)

    activation = activation.detach().cpu()

    # Expected GPT-2 hook_z shape:
    #
    # [sequence, heads, d_head]
    #
    # For GPT-2-small:
    # [sequence, 12, 64]

    if activation.ndim != 3:
        raise ValueError(
            "Unexpected attention activation shape.\n"
            f"Expected [sequence, heads, d_head], "
            f"got {tuple(activation.shape)}"
        )

    expected_heads = 12
    expected_head_dim = 64

    if activation.shape[1] != expected_heads:
        raise ValueError(
            "Unexpected number of attention heads.\n"
            f"Expected {expected_heads}, "
            f"got {activation.shape[1]}"
        )

    if activation.shape[2] != expected_head_dim:
        raise ValueError(
            "Unexpected attention head dimension.\n"
            f"Expected {expected_head_dim}, "
            f"got {activation.shape[2]}"
        )

    return activation



def validate_activation_shape(
    activation,
    layer,
    source_path,
):
    """
    Validate a cached GPT-2-small attention activation.

    Expected shape:
        [sequence, heads, d_head]

    GPT-2-small:
        12 heads
        64 dimensions per head
    """

    if not isinstance(activation, torch.Tensor):
        raise TypeError(
            f"Activation loaded from {source_path} "
            f"must be a torch.Tensor, "
            f"got {type(activation)}"
        )

    if activation.ndim != 3:
        raise ValueError(
            f"Invalid activation shape for layer {layer}.\n"
            f"Source: {source_path}\n"
            f"Expected: [sequence, heads, d_head]\n"
            f"Got: {tuple(activation.shape)}"
        )

    sequence_length, n_heads, head_dim = activation.shape

    if n_heads != N_HEADS:
        raise ValueError(
            f"Invalid number of heads for layer {layer}.\n"
            f"Source: {source_path}\n"
            f"Expected heads: {N_HEADS}\n"
            f"Got heads: {n_heads}\n"
            f"Shape: {tuple(activation.shape)}"
        )

    if head_dim != D_HEAD:
        raise ValueError(
            f"Invalid head dimension for layer {layer}.\n"
            f"Source: {source_path}\n"
            f"Expected head dimension: {D_HEAD}\n"
            f"Got: {head_dim}\n"
            f"Shape: {tuple(activation.shape)}"
        )

    if sequence_length <= 0:
        raise ValueError(
            f"Invalid sequence length for layer {layer}.\n"
            f"Source: {source_path}\n"
            f"Shape: {tuple(activation.shape)}"
        )

    return activation



def load_pair_head_activation(
    pair_id,
    condition,
    layer,
    head=None,
):
    """
    Load an attention activation from the Phase 11 expanded cache.

    Parameters
    ----------
    pair_id : int
        Dataset pair ID.

    condition : str
        Either "clean" or "corrupted".

    layer : int
        GPT-2 layer index.

    head : int or None
        Attention-head index.

        If None:
            return the complete layer activation
            with shape [sequence, heads, d_head].

        If an integer:
            return that head activation
            with shape [sequence, d_head].
    """

    cache_path = (
        DISCOVERY_ACTIVATION_DIR
        / f"pair_{int(pair_id):03d}_{condition}.pt"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Activation cache not found:\n"
            f"{cache_path}"
        )

    cache = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    activation = extract_layer_activation(
        cache,
        layer,
    )

    # No head requested:
    # return the complete layer activation.
    if head is None:
        return activation

    # Specific head requested.
    if head < 0 or head >= activation.shape[1]:
        raise ValueError(
            f"Invalid head index {head} for layer {layer}. "
            f"Available heads: 0-{activation.shape[1] - 1}"
        )

    return activation[:, head, :]


# ============================================================================
# TOKENIZER
# ============================================================================

def load_tokenizer():

    """
    Load the GPT-2 tokenizer through TransformerLens.

    This is only needed for token labels in the interpretation output.
    No model weights are changed.
    """

    try:

        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(
            MODEL_NAME,
            device="cpu",
        )

        return model.tokenizer

    except Exception as first_error:

        try:

            from transformer_lens.model_bridge import (
                TransformerBridge,
            )

            bridge = (
                TransformerBridge
                .boot_transformers(
                    MODEL_NAME,
                    device="cpu",
                )
            )

            return bridge.tokenizer

        except Exception as second_error:

            warnings.warn(
                "Tokenizer could not be loaded. "
                "Token labels will use numeric positions.\n"
                f"First error: {first_error}\n"
                f"Second error: {second_error}"
            )

            return None


def tokenize_prompt(
    tokenizer,
    prompt: str,
) -> list[str]:

    if tokenizer is None:

        return [
            f"position_{i}"
            for i in range(
                len(prompt.split())
            )
        ]

    try:

        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )

    except TypeError:

        token_ids = tokenizer.encode(
            prompt
        )

    tokens = []

    for token_id in token_ids:

        try:

            token = tokenizer.decode(
                [token_id]
            )

        except Exception:

            token = str(token_id)

        tokens.append(token)

    return tokens


# ============================================================================
# ACTIVATION DIFFERENCE
# ============================================================================

def compare_head_activation(
    clean_activation: torch.Tensor,
    corrupted_activation: torch.Tensor,
    head: int,
) -> dict[str, Any]:

    """
    Compare one attention head between clean and corrupted prompts.

    Handles variable sequence lengths by comparing only the overlapping
    token positions.

    Output:
        difference vector per position
        L2 norm per position
        mean norm
        maximum norm
    """

    if clean_activation.ndim != 3:
        raise ValueError(
            "Clean activation must have shape "
            "[sequence, heads, d_head]."
        )

    if corrupted_activation.ndim != 3:
        raise ValueError(
            "Corrupted activation must have shape "
            "[sequence, heads, d_head]."
        )

    clean_seq = clean_activation.shape[0]
    corrupted_seq = corrupted_activation.shape[0]

    overlap = min(
        clean_seq,
        corrupted_seq,
    )

    clean_head = clean_activation[
        :overlap,
        head,
        :,
    ].float()

    corrupted_head = corrupted_activation[
        :overlap,
        head,
        :,
    ].float()

    difference = (
        clean_head
        - corrupted_head
    )

    difference_norm = torch.linalg.vector_norm(
        difference,
        dim=-1,
    )

    absolute_difference = (
        difference.abs().mean(dim=-1)
    )

    signed_difference = (
        difference.mean(dim=-1)
    )

    return {
        "overlap": overlap,
        "difference": difference,
        "difference_norm": difference_norm,
        "absolute_difference": absolute_difference,
        "signed_difference": signed_difference,
        "clean_seq": clean_seq,
        "corrupted_seq": corrupted_seq,
    }


# ============================================================================
# CANDIDATE HEAD ANALYSIS
# ============================================================================

def analyze_candidate_heads(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    section(
        "CLEAN vs CORRUPTED ATTENTION-HEAD ACTIVATIONS"
    )

    candidate_rows = []
    position_rows = []
    layer_rows = []

    tokenizer = load_tokenizer()

    # ------------------------------------------------------------------------
    # Analyze all 60 discovery pairs.
    # ------------------------------------------------------------------------

    for index, row in df.iterrows():

        pair_id = int(
            row["pair_id"]
        )

        clean_prompt = str(
            row["clean"]
        )

        corrupted_prompt = str(
            row["corrupted"]
        )

        clean_tokens = tokenize_prompt(
            tokenizer,
            (
                "Review: "
                + clean_prompt
                + "\nSentiment:"
            ),
        )

        corrupted_tokens = tokenize_prompt(
            tokenizer,
            (
                "Review: "
                + corrupted_prompt
                + "\nSentiment:"
            ),
        )

        for layer, head in FROZEN_HEADS:

            clean_activation = (
                load_pair_head_activation(
                    pair_id,
                    "clean",
                    layer,
                )
            )

            corrupted_activation = (
                load_pair_head_activation(
                    pair_id,
                    "corrupted",
                    layer,
                )
            )

            comparison = compare_head_activation(
                clean_activation,
                corrupted_activation,
                head,
            )

            norms = comparison[
                "difference_norm"
            ].numpy()

            absolute_values = (
                comparison[
                    "absolute_difference"
                ]
                .numpy()
            )

            signed_values = (
                comparison[
                    "signed_difference"
                ]
                .numpy()
            )

            overlap = comparison[
                "overlap"
            ]

            label = (
                f"L{layer}H{head}"
            )

            # ---------------------------------------------------------------
            # Pair-level candidate summary.
            # ---------------------------------------------------------------

            candidate_rows.append(
                {
                    "pair_id": pair_id,
                    "domain": row["domain"],
                    "layer": layer,
                    "head": head,
                    "head_label": label,
                    "clean_sequence_length":
                        comparison["clean_seq"],
                    "corrupted_sequence_length":
                        comparison["corrupted_seq"],
                    "overlap_length": overlap,
                    "mean_difference_norm":
                        float(norms.mean()),
                    "median_difference_norm":
                        float(np.median(norms)),
                    "max_difference_norm":
                        float(norms.max()),
                    "mean_absolute_difference":
                        float(
                            absolute_values.mean()
                        ),
                    "mean_signed_difference":
                        float(
                            signed_values.mean()
                        ),
                }
            )

            # ---------------------------------------------------------------
            # Token-position rows.
            # ---------------------------------------------------------------

            for position in range(overlap):

                if position < len(
                    clean_tokens
                ):

                    clean_token = (
                        clean_tokens[position]
                    )

                else:

                    clean_token = "<missing>"

                if position < len(
                    corrupted_tokens
                ):

                    corrupted_token = (
                        corrupted_tokens[position]
                    )

                else:

                    corrupted_token = "<missing>"

                position_rows.append(
                    {
                        "pair_id": pair_id,
                        "domain": row["domain"],
                        "layer": layer,
                        "head": head,
                        "head_label": label,
                        "position": position,
                        "clean_token": clean_token,
                        "corrupted_token":
                            corrupted_token,
                        "difference_norm":
                            float(norms[position]),
                        "absolute_difference":
                            float(
                                absolute_values[position]
                            ),
                        "signed_difference":
                            float(
                                signed_values[position]
                            ),
                    }
                )

        # --------------------------------------------------------------------
        # Progress.
        # --------------------------------------------------------------------

        print(
            f"[{index + 1:02d}/"
            f"{len(df):02d}] "
            f"Pair {pair_id:03d} analyzed"
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    position_df = pd.DataFrame(
        position_rows
    )

    # =========================================================================
    # Candidate summary.
    # =========================================================================

    summary_df = (
        candidate_df
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
            n_pairs=(
                "pair_id",
                "nunique",
            ),
        )
    )

    summary_df = (
        summary_df
        .sort_values(
            "mean_difference_norm",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # Layer progression.
    #
    # We analyze all layers using the cached activations.
    # This gives a descriptive view of where the candidate heads differ.
    # =========================================================================

    subsection(
        "LAYER PROGRESSION"
    )

    for layer in range(
        N_LAYERS
    ):

        layer_values = []

        for pair_id in df[
            "pair_id"
        ]:

            pair_id = int(pair_id)

            clean_path = (
                DISCOVERY_ACTIVATION_DIR
                / f"pair_{pair_id:03d}_clean.pt"
            )

            corrupted_path = (
                DISCOVERY_ACTIVATION_DIR
                / f"pair_{pair_id:03d}_corrupted.pt"
            )

            clean_cache = load_torch_cache(
                clean_path
            )

            corrupted_cache = (
                load_torch_cache(
                    corrupted_path
                )
            )

            clean_activation = (
                validate_activation_shape(
                    extract_layer_activation(
                        clean_cache,
                        layer,
                    ),
                    layer,
                    clean_path,
                )
            )

            corrupted_activation = (
                validate_activation_shape(
                    extract_layer_activation(
                        corrupted_cache,
                        layer,
                    ),
                    layer,
                    corrupted_path,
                )
            )

            overlap = min(
                clean_activation.shape[0],
                corrupted_activation.shape[0],
            )

            for head in range(
                N_HEADS
            ):

                clean_head = (
                    clean_activation[
                        :overlap,
                        head,
                        :
                    ].float()
                )

                corrupted_head = (
                    corrupted_activation[
                        :overlap,
                        head,
                        :
                    ].float()
                )

                difference = (
                    clean_head
                    - corrupted_head
                )

                norms = (
                    torch.linalg.vector_norm(
                        difference,
                        dim=-1,
                    )
                )

                layer_values.append(
                    float(norms.mean())
                )

        layer_rows.append(
            {
                "layer": layer,
                "mean_head_difference_norm":
                    float(
                        np.mean(
                            layer_values
                        )
                    ),
                "median_head_difference_norm":
                    float(
                        np.median(
                            layer_values
                        )
                    ),
                "max_head_difference_norm":
                    float(
                        np.max(
                            layer_values
                        )
                    ),
            }
        )

        print(
            f"Layer {layer:02d}: "
            f"mean={np.mean(layer_values):.6f} | "
            f"max={np.max(layer_values):.6f}"
        )

    layer_df = pd.DataFrame(
        layer_rows
    )

    return (
        summary_df,
        position_df,
        layer_df,
    )


# ============================================================================
# PHASE 13 EVIDENCE
# ============================================================================

def load_phase13_evidence() -> pd.DataFrame:

    section(
        "PHASE 13 CAUSAL EVIDENCE"
    )

    if not PHASE13_RANKING_PATH.exists():

        print(
            "Phase 13 ranking file not found."
        )

        return pd.DataFrame()

    ranking = pd.read_csv(
        PHASE13_RANKING_PATH
    )

    print(
        f"Phase 13 ranking rows: "
        f"{len(ranking)}"
    )

    validate_columns(
        ranking,
        [
            "layer",
            "head",
            "head_label",
            "mean_recovery",
        ],
        "Phase 13 ranking",
    )

    candidate_labels = set(
        FROZEN_HEAD_LABELS
    )

    candidate_ranking = (
        ranking[
            ranking[
                "head_label"
            ].isin(
                candidate_labels
            )
        ]
        .copy()
    )

    candidate_ranking = (
        candidate_ranking
        .sort_values(
            "mean_recovery",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print(
        candidate_ranking[
            [
                "rank",
                "head_label",
                "mean_recovery",
                "median_recovery",
                "std_recovery",
                "n_pairs",
            ]
        ]
        .to_string(index=False)
    )

    return candidate_ranking


# ============================================================================
# PHASE 14 EVIDENCE
# ============================================================================

def load_phase14_evidence() -> pd.DataFrame:

    section(
        "PHASE 14 MULTI-HEAD EVIDENCE"
    )

    if not PHASE14_SUMMARY_PATH.exists():

        print(
            "Phase 14 summary file not found."
        )

        return pd.DataFrame()

    summary = pd.read_csv(
        PHASE14_SUMMARY_PATH
    )

    print(
        f"Phase 14 condition rows: "
        f"{len(summary)}"
    )

    if "condition" not in summary.columns:
        raise ValueError(
            "Phase 14 summary is missing "
            "'condition'."
        )

    print()

    columns_to_show = [
        column
        for column in [
            "rank",
            "condition",
            "n_heads",
            "mean_recovery",
            "median_recovery",
            "std_recovery",
            "positive_fraction",
            "recovery_ge_025",
            "recovery_ge_050",
            "n_pairs",
        ]
        if column in summary.columns
    ]

    print(
        summary[
            columns_to_show
        ]
        .to_string(index=False)
    )

    return summary


# ============================================================================
# PHASE 15 / 16 EVIDENCE
# ============================================================================

def load_generalization_evidence() -> dict:

    evidence = {}

    if PHASE15_SUMMARY_PATH.exists():

        try:

            phase15 = pd.read_csv(
                PHASE15_SUMMARY_PATH
            )

            evidence[
                "phase15_generalization"
            ] = phase15.to_dict(
                orient="records"
            )

        except Exception as exc:

            evidence[
                "phase15_error"
            ] = str(exc)

    if PHASE16_SUMMARY_PATH.exists():

        try:

            with open(
                PHASE16_SUMMARY_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                evidence[
                    "phase16_statistics"
                ] = json.load(file)

        except Exception as exc:

            evidence[
                "phase16_error"
            ] = str(exc)

    return evidence


# ============================================================================
# ACTIVATION ↔ CAUSAL RECOVERY ANALYSIS
# ============================================================================

def activation_recovery_correlation(
    activation_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compare activation-difference magnitude with Phase 13
    causal recovery.

    This is descriptive only.

    It must NOT be interpreted as proof that activation magnitude
    causes recovery.
    """

    if not PHASE13_RESULTS_PATH.exists():

        print(
            "Phase 13 detailed results not found."
        )

        return pd.DataFrame()

    phase13 = pd.read_csv(
        PHASE13_RESULTS_PATH
    )

    required = [
        "pair_id",
        "layer",
        "head",
        "normalized_recovery",
    ]

    missing = [
        column
        for column in required
        if column not in phase13.columns
    ]

    if missing:

        print(
            "Phase 13 detailed results do not contain "
            f"required columns: {missing}"
        )

        return pd.DataFrame()

    merged = activation_df.merge(
        phase13[
            required
        ],
        on=[
            "pair_id",
            "layer",
            "head",
        ],
        how="inner",
    )

    if merged.empty:

        return pd.DataFrame()

    rows = []

    for label, group in merged.groupby(
        "head_label"
    ):

        if len(group) < 3:

            continue

        x = group[
            "mean_difference_norm"
        ].to_numpy(
            dtype=float
        )

        y = group[
            "normalized_recovery"
        ].to_numpy(
            dtype=float
        )

        if (
            np.std(x) == 0
            or np.std(y) == 0
        ):

            correlation = np.nan

        else:

            correlation = float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )

        rows.append(
            {
                "head_label": label,
                "n_pairs": len(group),
                "activation_recovery_pearson_r":
                    correlation,
                "mean_activation_difference":
                    float(x.mean()),
                "mean_recovery":
                    float(y.mean()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# TOKEN POSITION SUMMARY
# ============================================================================

def summarize_token_positions(
    position_df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        position_df
        .groupby(
            [
                "head_label",
                "layer",
                "head",
                "position",
            ],
            as_index=False,
        )
        .agg(
            mean_difference_norm=(
                "difference_norm",
                "mean",
            ),
            median_difference_norm=(
                "difference_norm",
                "median",
            ),
            max_difference_norm=(
                "difference_norm",
                "max",
            ),
            n_observations=(
                "pair_id",
                "count",
            ),
        )
    )

    return (
        summary
        .sort_values(
            [
                "head_label",
                "mean_difference_norm",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================================
# STRONGEST TOKEN POSITIONS
# ============================================================================

def print_strongest_token_positions(
    position_summary: pd.DataFrame,
) -> None:

    section(
        "STRONGEST CANDIDATE-HEAD TOKEN POSITIONS"
    )

    for label in FROZEN_HEAD_LABELS:

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
            .head(10)
        )

        print()
        print(
            f"{label} strongest token positions:"
        )

        if subset.empty:

            print(
                "  No position data available."
            )

            continue

        print(
            subset[
                [
                    "head_label",
                    "position",
                    "mean_difference_norm",
                    "max_difference_norm",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================================
# REPRESENTATIVE TOKEN ANALYSIS
# ============================================================================

def representative_token_analysis(
    df: pd.DataFrame,
) -> dict[str, Any]:

    section(
        "REPRESENTATIVE TOKEN ANALYSIS"
    )

    # Choose first discovery pair.
    row = df.iloc[0]

    pair_id = int(
        row["pair_id"]
    )

    clean_prompt = (
        "Review: "
        + str(row["clean"])
        + "\nSentiment:"
    )

    corrupted_prompt = (
        "Review: "
        + str(row["corrupted"])
        + "\nSentiment:"
    )

    tokenizer = load_tokenizer()

    clean_tokens = tokenize_prompt(
        tokenizer,
        clean_prompt,
    )

    corrupted_tokens = tokenize_prompt(
        tokenizer,
        corrupted_prompt,
    )

    print(
        f"Pair: {pair_id}"
    )

    print(
        f"Domain: {row['domain']}"
    )

    print(
        f"Clean: {row['clean']}"
    )

    print(
        f"Corrupted: {row['corrupted']}"
    )

    print()
    print("Clean tokens:")

    for index, token in enumerate(
        clean_tokens
    ):

        print(
            f"  {index:02d}: {token!r}"
        )

    print()
    print("Corrupted tokens:")

    for index, token in enumerate(
        corrupted_tokens
    ):

        print(
            f"  {index:02d}: {token!r}"
        )

    return {
        "pair_id": pair_id,
        "domain": str(row["domain"]),
        "clean": str(row["clean"]),
        "corrupted": str(row["corrupted"]),
        "clean_tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
    }


# ============================================================================
# FIGURES
# ============================================================================

def create_candidate_head_plot(
    summary_df: pd.DataFrame,
) -> None:

    path = (
        FIGURES_DIR
        / "candidate_head_activation_difference.png"
    )

    plot_df = (
        summary_df
        .copy()
    )

    # Ensure frozen heads appear in a fixed order.
    plot_df[
        "order"
    ] = plot_df[
        "head_label"
    ].map(
        {
            "L10H4": 0,
            "L8H9": 1,
            "L9H2": 2,
        }
    )

    plot_df = (
        plot_df
        .sort_values("order")
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        plot_df[
            "head_label"
        ],
        plot_df[
            "mean_difference_norm"
        ],
    )

    ax.set_xlabel(
        "Attention head"
    )

    ax.set_ylabel(
        "Mean clean-corrupted activation difference norm"
    )

    ax.set_title(
        "Candidate Head Activation Differences"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        path
    )


def create_token_position_plot(
    position_summary: pd.DataFrame,
) -> None:

    path = (
        FIGURES_DIR
        / "token_position_activation_difference.png"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for label in FROZEN_HEAD_LABELS:

        subset = (
            position_summary[
                position_summary[
                    "head_label"
                ]
                == label
            ]
            .sort_values("position")
        )

        if subset.empty:
            continue

        ax.plot(
            subset["position"],
            subset[
                "mean_difference_norm"
            ],
            marker="o",
            label=label,
        )

    ax.set_xlabel(
        "Token position"
    )

    ax.set_ylabel(
        "Mean activation difference norm"
    )

    ax.set_title(
        "Candidate Head Activation Difference by Token Position"
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        path
    )


def create_layer_progression_plot(
    layer_df: pd.DataFrame,
) -> None:

    path = (
        FIGURES_DIR
        / "layer_progression.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        layer_df["layer"],
        layer_df[
            "mean_head_difference_norm"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "GPT-2 layer"
    )

    ax.set_ylabel(
        "Mean attention-head activation difference norm"
    )

    ax.set_title(
        "Activation Difference Across Transformer Layers"
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        path
    )


def create_circuit_vs_single_plot(
    phase14_df: pd.DataFrame,
) -> None:

    path = (
        FIGURES_DIR
        / "circuit_vs_single_discovery.png"
    )

    if phase14_df.empty:
        return

    best_condition = phase14_df[
        phase14_df[
            "condition"
        ].astype(str)
        == FROZEN_CIRCUIT
    ]

    if best_condition.empty:

        # Handle alternative ordering such as
        # L10H4 + L9H2 + L8H9.
        best_condition = phase14_df[
            phase14_df[
                "condition"
            ]
            .astype(str)
            .apply(
                lambda value:
                set(
                    part.strip()
                    for part in value.split("+")
                )
                == set(FROZEN_HEAD_LABELS)
            )
        ]

    if best_condition.empty:
        return

    best_condition = best_condition.iloc[0]

    single = phase14_df[
        phase14_df[
            "condition"
        ].astype(str)
        == "L10H4"
    ]

    if single.empty:
        return

    single = single.iloc[0]

    labels = [
        "L10H4",
        "Frozen circuit",
    ]

    values = [
        float(
            single["mean_recovery"]
        ),
        float(
            best_condition[
                "mean_recovery"
            ]
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Mean normalized recovery"
    )

    ax.set_title(
        "Discovery: Frozen Circuit vs Best Single Head"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        path
    )


# ============================================================================
# MECHANISTIC INTERPRETATION
# ============================================================================

def build_mechanistic_interpretation(
    activation_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    layer_df: pd.DataFrame,
    phase13_df: pd.DataFrame,
    phase14_df: pd.DataFrame,
    generalization: dict,
) -> str:

    # ------------------------------------------------------------------------
    # Candidate head with largest activation difference.
    # ------------------------------------------------------------------------

    activation_leader = (
        activation_summary
        .sort_values(
            "mean_difference_norm",
            ascending=False,
        )
        .iloc[0]
    )

    activation_leader_label = (
        activation_leader[
            "head_label"
        ]
    )

    # ------------------------------------------------------------------------
    # Candidate head with strongest causal recovery.
    # ------------------------------------------------------------------------

    if not phase13_df.empty:

        causal_leader = (
            phase13_df
            .sort_values(
                "mean_recovery",
                ascending=False,
            )
            .iloc[0]
        )

        causal_leader_label = (
            causal_leader[
                "head_label"
            ]
        )

        causal_recovery = float(
            causal_leader[
                "mean_recovery"
            ]
        )

    else:

        causal_leader_label = (
            "Unavailable"
        )

        causal_recovery = float("nan")

    # ------------------------------------------------------------------------
    # Strongest token position for L10H4.
    # ------------------------------------------------------------------------

    l10_position = (
        position_summary[
            position_summary[
                "head_label"
            ]
            == "L10H4"
        ]
        .sort_values(
            "mean_difference_norm",
            ascending=False,
        )
    )

    if not l10_position.empty:

        strongest_position = int(
            l10_position.iloc[0][
                "position"
            ]
        )

        strongest_position_value = float(
            l10_position.iloc[0][
                "mean_difference_norm"
            ]
        )

    else:

        strongest_position = -1
        strongest_position_value = float(
            "nan"
        )

    # ------------------------------------------------------------------------
    # Phase 14 circuit result.
    # ------------------------------------------------------------------------

    circuit_recovery = None

    if not phase14_df.empty:

        exact = phase14_df[
            phase14_df[
                "condition"
            ].astype(str)
            .apply(
                lambda value:
                set(
                    part.strip()
                    for part in value.split("+")
                )
                == set(FROZEN_HEAD_LABELS)
            )
        ]

        if not exact.empty:

            circuit_recovery = float(
                exact.iloc[0][
                    "mean_recovery"
                ]
            )

    # ------------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------------

    lines = []

    lines.append(
        "# Phase 17 — Mechanistic Interpretation"
    )

    lines.append("")

    lines.append(
        "## Experimental status"
    )

    lines.append("")

    lines.append(
        "This phase interprets the frozen candidate "
        "heads discovered in the previous causal experiments. "
        "No new head selection, circuit search, activation "
        "patching, or model-weight modification was performed."
    )

    lines.append("")

    lines.append(
        "The analysis uses the 60-example discovery split only. "
        "The 40 held-out examples are not loaded."
    )

    lines.append("")

    lines.append(
        "## Frozen candidate circuit"
    )

    lines.append("")

    lines.append(
        f"- {FROZEN_CIRCUIT}"
    )

    lines.append("")

    lines.append(
        "## Activation-difference evidence"
    )

    lines.append("")

    lines.append(
        f"The largest mean clean-corrupted activation "
        f"difference among the frozen candidate heads was "
        f"observed for **{activation_leader_label}**, with "
        f"a mean difference norm of "
        f"**{activation_leader['mean_difference_norm']:.6f}**."
    )

    lines.append("")

    lines.append(
        "Activation difference is treated as descriptive "
        "evidence. It does not by itself establish causal "
        "importance."
    )

    lines.append("")

    lines.append(
        "## Causal evidence"
    )

    lines.append("")

    if not math.isnan(
        causal_recovery
    ):

        lines.append(
            f"Phase 13 identified **{causal_leader_label}** "
            f"as the strongest individual frozen candidate "
            f"according to mean normalized recovery "
            f"({causal_recovery:.4f})."
        )

    else:

        lines.append(
            "Phase 13 causal ranking data were unavailable."
        )

    lines.append("")

    if circuit_recovery is not None:

        lines.append(
            f"Phase 14 measured mean recovery of "
            f"**{circuit_recovery:.4f}** for the frozen "
            f"three-head circuit on the discovery split."
        )

    lines.append("")

    lines.append(
        "The causal interpretation therefore comes from "
        "activation patching, while the activation-difference "
        "analysis is used to characterize how the candidate "
        "heads respond to the clean/corrupted contrast."
    )

    lines.append("")

    lines.append(
        "## Token-position interpretation"
    )

    lines.append("")

    lines.append(
        f"For L10H4, the strongest average activation "
        f"difference occurred at token position "
        f"**{strongest_position}**, with a mean difference "
        f"norm of **{strongest_position_value:.6f}**."
    )

    lines.append("")

    lines.append(
        "Token-position effects should be interpreted "
        "carefully because clean and corrupted prompts can "
        "have different tokenization and sequence lengths. "
        "Only overlapping positions were compared."
    )

    lines.append("")

    lines.append(
        "## Candidate mechanism"
    )

    lines.append("")

    lines.append(
        "The evidence is consistent with a distributed "
        "attention-head mechanism in which multiple late-layer "
        "heads contribute to the model's defined sentiment "
        "behavior."
    )

    lines.append("")

    lines.append(
        "The results do not justify the stronger claim that "
        "these heads constitute a universal 'sentiment circuit'. "
        "The appropriate claim is that the frozen heads showed "
        "causal influence under the defined GPT-2-small sentiment "
        "task."
    )

    lines.append("")

    lines.append(
        "## Held-out evidence"
    )

    lines.append("")

    if (
        "phase15_generalization"
        in generalization
    ):

        lines.append(
            "Phase 15 independently evaluated the frozen "
            "circuit on the 40 held-out pairs."
        )

    else:

        lines.append(
            "Phase 15 generalization data were not found."
        )

    lines.append("")

    if (
        "phase16_statistics"
        in generalization
    ):

        lines.append(
            "Phase 16 independently tested the held-out "
            "circuit-vs-single-head comparison using paired "
            "statistical analysis."
        )

    else:

        lines.append(
            "Phase 16 statistical data were not found."
        )

    lines.append("")

    lines.append(
        "These held-out results are not used to select or "
        "reinterpret the circuit in this phase."
    )

    lines.append("")

    lines.append(
        "## Limitations"
    )

    lines.append("")

    lines.append(
        "1. The task is a controlled sentiment contrast "
        "rather than a broad natural-language benchmark."
    )

    lines.append(
        "2. The candidate circuit contains only three "
        "attention heads."
    )

    lines.append(
        "3. Activation differences do not establish "
        "causality."
    )

    lines.append(
        "4. The experiments do not establish that the same "
        "heads control sentiment outside the defined task."
    )

    lines.append(
        "5. Token-position analysis is sensitive to "
        "tokenization differences."
    )

    lines.append("")

    lines.append(
        "## Conclusion"
    )

    lines.append("")

    lines.append(
        "The completed experiments support a candidate "
        "distributed mechanism involving L10H4, L8H9, and "
        "L9H2 for the defined sentiment behavior. "
        "Activation analysis provides mechanistic context, "
        "while the previous activation-patching experiments "
        "provide the causal evidence."
    )

    return "\n".join(
        lines
    )


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def safe_float(
    value: Any,
) -> Any:

    try:

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    except Exception:

        return value


def dataframe_records_safe(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    records = []

    for record in df.to_dict(
        orient="records"
    ):

        clean_record = {}

        for key, value in record.items():

            if isinstance(
                value,
                (
                    np.integer,
                    np.floating,
                ),
            ):

                value = value.item()

            if isinstance(
                value,
                np.bool_,
            ):

                value = bool(value)

            if isinstance(
                value,
                float,
            ) and (
                math.isnan(value)
                or math.isinf(value)
            ):

                value = None

            clean_record[key] = value

        records.append(
            clean_record
        )

    return records


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    activation_summary: pd.DataFrame,
    position_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    phase13_df: pd.DataFrame,
    phase14_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    representative: dict,
    interpretation: str,
    generalization: dict,
) -> None:

    section(
        "SAVING PHASE 17 RESULTS"
    )

    activation_summary_path = (
        RESULTS_DIR
        / "candidate_head_activation_summary.csv"
    )

    position_path = (
        RESULTS_DIR
        / "token_position_activation.csv"
    )

    layer_path = (
        RESULTS_DIR
        / "layer_progression.csv"
    )

    phase13_path = (
        RESULTS_DIR
        / "phase13_candidate_head_evidence.csv"
    )

    phase14_path = (
        RESULTS_DIR
        / "phase14_circuit_evidence.csv"
    )

    correlation_path = (
        RESULTS_DIR
        / "activation_recovery_correlation.csv"
    )

    report_path = (
        RESULTS_DIR
        / "mechanistic_interpretation.md"
    )

    json_path = (
        RESULTS_DIR
        / "mechanistic_summary.json"
    )

    activation_summary.to_csv(
        activation_summary_path,
        index=False,
    )

    position_df.to_csv(
        position_path,
        index=False,
    )

    layer_df.to_csv(
        layer_path,
        index=False,
    )

    phase13_df.to_csv(
        phase13_path,
        index=False,
    )

    phase14_df.to_csv(
        phase14_path,
        index=False,
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            interpretation
        )

    summary = {
        "phase": 17,
        "name": (
            "mechanistic_interpretation"
        ),
        "model": MODEL_NAME,
        "dataset": {
            "expanded_dataset":
                str(
                    EXPANDED_DATASET_PATH
                ),
            "discovery_dataset":
                str(
                    DISCOVERY_DATASET_PATH
                ),
            "discovery_pairs":
                EXPECTED_DISCOVERY_PAIRS,
            "heldout_pairs_loaded":
                0,
            "heldout_pairs_expected":
                EXPECTED_HELDOUT_PAIRS,
        },
        "frozen_circuit": {
            "heads":
                FROZEN_HEAD_LABELS,
            "condition":
                FROZEN_CIRCUIT,
        },
        "method": {
            "new_activation_patching":
                False,
            "new_head_selection":
                False,
            "new_circuit_search":
                False,
            "model_weights_modified":
                False,
            "activation_difference":
                True,
            "token_position_analysis":
                True,
        },
        "outputs": {
            "candidate_head_activation_summary":
                str(
                    activation_summary_path
                ),
            "token_position_activation":
                str(
                    position_path
                ),
            "layer_progression":
                str(
                    layer_path
                ),
            "phase13_candidate_head_evidence":
                str(
                    phase13_path
                ),
            "phase14_circuit_evidence":
                str(
                    phase14_path
                ),
            "activation_recovery_correlation":
                str(
                    correlation_path
                ),
            "mechanistic_interpretation":
                str(
                    report_path
                ),
        },
        "representative_example":
            representative,
        "candidate_activation_summary":
            dataframe_records_safe(
                activation_summary
            ),
        "phase13_candidate_evidence":
            dataframe_records_safe(
                phase13_df
            ),
        "phase14_evidence":
            dataframe_records_safe(
                phase14_df
            ),
        "activation_recovery_correlation":
            dataframe_records_safe(
                correlation_df
            ),
        "generalization_evidence":
            generalization,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"  {activation_summary_path}"
    )

    print(
        f"  {position_path}"
    )

    print(
        f"  {layer_path}"
    )

    print(
        f"  {phase13_path}"
    )

    print(
        f"  {phase14_path}"
    )

    print(
        f"  {correlation_path}"
    )

    print(
        f"  {report_path}"
    )

    print(
        f"  {json_path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    section(
        "PHASE 17 — MECHANISTIC INTERPRETATION"
    )

    print(
        "Frozen circuit:"
    )

    for label in FROZEN_HEAD_LABELS:
        print(
            f"  {label}"
        )

    print()
    print(
        "No head selection will be performed."
    )

    print(
        "No circuit search will be performed."
    )

    print(
        "No activation patching will be performed."
    )

    print(
        "No model weights will be modified."
    )

    print(
        "Held-out examples will NOT be loaded."
    )

    # ------------------------------------------------------------------------
    # Device information.
    # ------------------------------------------------------------------------

    section(
        "DEVICE"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        try:

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            memory = (
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / (
                    1024 ** 3
                )
            )

            print(
                f"GPU memory: "
                f"{memory:.2f} GB"
            )

        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Output directories.
    # ------------------------------------------------------------------------

    create_output_directories()

    # ------------------------------------------------------------------------
    # Dataset.
    # ------------------------------------------------------------------------

    df = load_discovery_dataset()

    # ------------------------------------------------------------------------
    # Activation directory validation.
    # ------------------------------------------------------------------------

    section(
        "ACTIVATION CACHE VALIDATION"
    )

    if not DISCOVERY_ACTIVATION_DIR.exists():

        raise FileNotFoundError(
            "Discovery activation directory not found:\n"
            f"{DISCOVERY_ACTIVATION_DIR}"
        )

    print(
        f"Discovery cache:\n"
        f"  {DISCOVERY_ACTIVATION_DIR}"
    )

    print(
        "Held-out cache will NOT be accessed."
    )

    # Validate one pair for all candidate heads.
    sample_pair = int(
        df.iloc[0]["pair_id"]
    )

    for layer, head in FROZEN_HEADS:

        clean = load_pair_head_activation(
            sample_pair,
            "clean",
            layer,
        )

        corrupted = load_pair_head_activation(
            sample_pair,
            "corrupted",
            layer,
        )

        print(
            f"blocks.{layer}.attn.hook_z "
            f"(head {head}): PASS | "
            f"clean={tuple(clean.shape)} | "
            f"corrupted={tuple(corrupted.shape)}"
        )

    # ------------------------------------------------------------------------
    # Activation analysis.
    # ------------------------------------------------------------------------

    (
        activation_summary,
        position_df,
        layer_df,
    ) = analyze_candidate_heads(
        df
    )

    # ------------------------------------------------------------------------
    # Print candidate summary.
    # ------------------------------------------------------------------------

    section(
        "CANDIDATE HEAD ACTIVATION SUMMARY"
    )

    print(
        activation_summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Token position summary.
    # ------------------------------------------------------------------------

    position_summary = (
        summarize_token_positions(
            position_df
        )
    )

    print_strongest_token_positions(
        position_summary
    )

    # ------------------------------------------------------------------------
    # Representative example.
    # ------------------------------------------------------------------------

    representative = (
        representative_token_analysis(
            df
        )
    )

    # ------------------------------------------------------------------------
    # Phase 13.
    # ------------------------------------------------------------------------

    phase13_df = (
        load_phase13_evidence()
    )

    # ------------------------------------------------------------------------
    # Activation vs causal recovery.
    # ------------------------------------------------------------------------

    section(
        "ACTIVATION <-> CAUSAL RECOVERY"
    )

    correlation_df = (
        activation_recovery_correlation(
            pd.read_csv(
                PHASE13_RESULTS_PATH
            )
            if PHASE13_RESULTS_PATH.exists()
            else pd.DataFrame()
        )
    )

    if correlation_df.empty:

        print(
            "No valid correlation analysis "
            "was available."
        )

    else:

        print(
            correlation_df.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------------
    # Phase 14.
    # ------------------------------------------------------------------------

    phase14_df = (
        load_phase14_evidence()
    )

    # ------------------------------------------------------------------------
    # Phase 15 / Phase 16.
    # ------------------------------------------------------------------------

    section(
        "HELD-OUT EVIDENCE REFERENCE"
    )

    print(
        "Phase 15 and Phase 16 are referenced "
        "as completed independent evaluations."
    )

    print(
        "They are NOT used for head selection "
        "or circuit selection in Phase 17."
    )

    generalization = (
        load_generalization_evidence()
    )

    if "phase15_generalization" in generalization:

        print(
            "Phase 15 generalization results: FOUND"
        )

    else:

        print(
            "Phase 15 generalization results: NOT FOUND"
        )

    if "phase16_statistics" in generalization:

        print(
            "Phase 16 statistical results: FOUND"
        )

    else:

        print(
            "Phase 16 statistical results: NOT FOUND"
        )

    # ------------------------------------------------------------------------
    # Mechanistic interpretation.
    # ------------------------------------------------------------------------

    section(
        "MECHANISTIC INTERPRETATION"
    )

    interpretation = (
        build_mechanistic_interpretation(
            activation_summary,
            position_summary,
            layer_df,
            phase13_df,
            phase14_df,
            generalization,
        )
    )

    print(
        interpretation
    )

    # ------------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------------

    section(
        "CREATING PHASE 17 FIGURES"
    )

    create_candidate_head_plot(
        activation_summary
    )

    create_token_position_plot(
        position_summary
    )

    create_layer_progression_plot(
        layer_df
    )

    create_circuit_vs_single_plot(
        phase14_df
    )

    # ------------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------------

    save_outputs(
        activation_summary,
        position_df,
        layer_df,
        phase13_df,
        phase14_df,
        correlation_df,
        representative,
        interpretation,
        generalization,
    )

    # ------------------------------------------------------------------------
    # Final.
    # ------------------------------------------------------------------------

    section(
        "PHASE 17 COMPLETE"
    )

    print(
        "Mechanistic interpretation completed."
    )

    print()
    print(
        f"Discovery pairs analyzed: "
        f"{len(df)}"
    )

    print(
        "Held-out pairs loaded: 0"
    )

    print(
        f"Frozen circuit: "
        f"{FROZEN_CIRCUIT}"
    )

    print()
    print(
        "No new activation patching was performed."
    )

    print(
        "No heads were selected."
    )

    print(
        "No circuit search was performed."
    )

    print(
        "No model weights were modified."
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {RESULTS_DIR}"
    )

    print()
    print(
        "Figures:"
    )

    print(
        f"  {FIGURES_DIR}"
    )

if __name__ == "__main__":
    main()