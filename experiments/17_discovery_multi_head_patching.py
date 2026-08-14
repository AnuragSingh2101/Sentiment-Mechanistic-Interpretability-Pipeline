"""
PHASE 14 — DISCOVERY MULTI-HEAD ACTIVATION PATCHING

Purpose
-------
Test whether the strongest attention heads discovered in Phase 13
form a useful causal multi-head circuit.

ONLY the 60 discovery pairs are used.

The 40 held-out examples are NOT loaded and are NOT used for:
    - candidate-head selection
    - combination selection
    - ranking
    - statistical evaluation

Phase 13 provides the candidate heads.

The current experiment automatically reads the top 3 heads from:

    results/expanded_single_head/discovery_head_ranking.csv

For 3 candidate heads, all 7 non-empty combinations are evaluated:

    H1
    H2
    H3

    H1 + H2
    H1 + H3
    H2 + H3

    H1 + H2 + H3

For every combination and discovery pair:

    clean LD
    corrupted LD
    patched LD
    normalized recovery

are measured.

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

Important implementation detail
--------------------------------
Clean and corrupted prompts can have different token lengths.

Therefore this experiment patches the FINAL prediction position
of the corrupted prompt using the FINAL activation of the
corresponding clean prompt.

No model weights are modified.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer_lens.model_bridge import TransformerBridge


# ============================================================================
# PROJECT PATHS
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

PHASE13_RANKING = (
    PROJECT_ROOT
    / "results"
    / "expanded_single_head"
    / "discovery_head_ranking.csv"
)

PHASE13_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "expanded_single_head"
    / "discovery_single_head_results.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "expanded_multi_head"
)

FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_CSV = (
    RESULTS_DIR
    / "discovery_multi_head_results.csv"
)

SUMMARY_CSV = (
    RESULTS_DIR
    / "discovery_multi_head_summary.csv"
)

SUMMARY_JSON = (
    RESULTS_DIR
    / "discovery_multi_head_summary.json"
)

MODEL_NAME = "gpt2"

EXPECTED_DISCOVERY_PAIRS = 60

N_LAYERS = 12

N_HEADS = 12

D_HEAD = 64

TOTAL_HEADS = (
    N_LAYERS * N_HEADS
)

TOP_K_HEADS = 3

POSITIVE_TEXT = " positive"

NEGATIVE_TEXT = " negative"

EPSILON = 1e-8

HOOK_TEMPLATE = (
    "blocks.{layer}.attn.hook_z"
)


# ============================================================================
# PRINTING
# ============================================================================


def section(title: str) -> None:

    print()

    print(
        "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


# ============================================================================
# DIRECTORIES
# ============================================================================


def ensure_directories() -> None:

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


def get_device() -> torch.device:

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

        reader = csv.DictReader(
            file
        )

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

    if len(rows) != EXPECTED_DISCOVERY_PAIRS:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_DISCOVERY_PAIRS} "
            f"discovery pairs, "
            f"found {len(rows)}."
        )

    for row in rows:

        if row["split"] != "discovery":

            raise ValueError(
                f"Non-discovery pair found: "
                f"{row['pair_id']}"
            )

        if row["clean_label"] != "positive":

            raise ValueError(
                f"Unexpected clean label "
                f"for pair {row['pair_id']}"
            )

        if row["corrupted_label"] != "negative":

            raise ValueError(
                f"Unexpected corrupted label "
                f"for pair {row['pair_id']}"
            )

    pair_ids = [
        int(row["pair_id"])
        for row in rows
    ]

    if len(set(pair_ids)) != len(pair_ids):

        raise ValueError(
            "Duplicate discovery pair IDs detected."
        )

    return rows


# ============================================================================
# MODEL
# ============================================================================


def load_model(
    device: torch.device,
):

    section(
        "MODEL LOADING"
    )

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

        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB",
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
            "Unexpected number of layers."
        )

    if model.cfg.n_heads != N_HEADS:

        raise ValueError(
            "Unexpected number of heads."
        )

    if model.cfg.d_head != D_HEAD:

        raise ValueError(
            "Unexpected head dimension."
        )

    print(
        "GPT-2-small architecture: PASS"
    )


# ============================================================================
# SENTIMENT TOKENS
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
            "' positive' must be "
            "one token."
        )

    if negative_tokens.numel() != 1:

        raise ValueError(
            "' negative' must be "
            "one token."
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


def make_prompt(
    text: str,
) -> str:

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

    prompt = make_prompt(
        text
    )

    tokens = model.to_tokens(
        prompt,
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
        final_logits[
            positive_id
        ]
        -
        final_logits[
            negative_id
        ]
    )

    return float(
        value.detach()
        .cpu()
        .item()
    )


# ============================================================================
# ACTIVATION CACHE
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
            f"Missing discovery activation cache:\n"
            f"{path}"
        )

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if cache.get(
        "split"
    ) != "discovery":

        raise ValueError(
            f"Cache is not discovery split:\n"
            f"{path}"
        )

    if int(
        cache["pair_id"]
    ) != pair_id:

        raise ValueError(
            f"Pair ID mismatch:\n"
            f"{path}"
        )

    if cache.get(
        "condition"
    ) != condition:

        raise ValueError(
            f"Condition mismatch:\n"
            f"{path}"
        )

    return cache


# ============================================================================
# CACHE VALIDATION
# ============================================================================


def validate_cache(
    cache,
    pair_id: int,
    condition: str,
):

    activations = cache[
        "activations"
    ]

    if len(activations) != N_LAYERS:

        raise ValueError(
            f"Pair {pair_id} {condition}: "
            f"expected {N_LAYERS} layers."
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
                f"Missing hook "
                f"{hook_name}"
            )

        activation = activations[
            hook_name
        ]

        expected_shape = (
            sequence_length,
            N_HEADS,
            D_HEAD,
        )

        if tuple(
            activation.shape
        ) != expected_shape:

            raise ValueError(
                f"Invalid activation shape "
                f"for {hook_name}: "
                f"{tuple(activation.shape)} "
                f"expected {expected_shape}"
            )


# ============================================================================
# PHASE 13 RANKING
# ============================================================================


def load_candidate_heads():

    section(
        "LOADING PHASE 13 HEAD RANKING"
    )

    if not PHASE13_RANKING.exists():

        raise FileNotFoundError(
            "Phase 13 ranking not found:\n"
            f"{PHASE13_RANKING}\n\n"
            "Run Phase 13 first."
        )

    ranking = pd.read_csv(
        PHASE13_RANKING
    )

    required_columns = {
        "rank",
        "layer",
        "head",
        "head_label",
        "mean_recovery",
    }

    missing = (
        required_columns
        - set(ranking.columns)
    )

    if missing:

        raise ValueError(
            "Phase 13 ranking is missing "
            f"columns: {sorted(missing)}"
        )

    ranking = ranking.sort_values(
        "rank"
    ).reset_index(
        drop=True
    )

    if len(ranking) != TOTAL_HEADS:

        raise ValueError(
            f"Expected {TOTAL_HEADS} "
            f"heads in Phase 13 ranking, "
            f"found {len(ranking)}."
        )

    candidates = []

    for _, row in ranking.head(
        TOP_K_HEADS
    ).iterrows():

        layer = int(
            row["layer"]
        )

        head = int(
            row["head"]
        )

        label = (
            f"L{layer}H{head}"
        )

        candidates.append(
            (
                layer,
                head,
                label,
            )
        )

    section(
        "CANDIDATE HEADS"
    )

    for rank, (
        layer,
        head,
        label,
    ) in enumerate(
        candidates,
        start=1,
    ):

        row = ranking[
            (
                ranking["layer"]
                == layer
            )
            &
            (
                ranking["head"]
                == head
            )
        ].iloc[0]

        print(
            f"{rank}. "
            f"{label} | "
            f"Phase 13 mean recovery="
            f"{float(row['mean_recovery']):.4f}"
        )

    return (
        candidates,
        ranking,
    )


# ============================================================================
# COMBINATIONS
# ============================================================================


def create_combinations(
    candidates,
):

    section(
        "MULTI-HEAD CONDITIONS"
    )

    combinations = []

    for size in range(
        1,
        len(candidates) + 1,
    ):

        for combination in itertools.combinations(
            candidates,
            size,
        ):

            combinations.append(
                combination
            )

    print(
        f"Candidate heads: "
        f"{len(candidates)}"
    )

    print(
        f"Conditions: "
        f"{len(combinations)}"
    )

    for index, combination in enumerate(
        combinations,
        start=1,
    ):

        labels = [
            item[2]
            for item in combination
        ]

        print(
            f"{index}. "
            + " + ".join(labels)
        )

    return combinations


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
# MULTI-HEAD PATCH HOOK
# ============================================================================


def create_multi_head_patch_hooks(
    clean_cache,
    combination,
):
    """
    Create one hook for every distinct layer
    involved in the candidate-head combination.

    Example:

        L10H4 + L8H9 + L9H2

    creates hooks for:

        blocks.8.attn.hook_z
        blocks.9.attn.hook_z
        blocks.10.attn.hook_z

    At each layer, only the selected head is replaced.
    """

    hooks = []

    for (
        layer,
        head,
        label,
    ) in combination:

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

        if clean_activation.ndim != 3:

            raise ValueError(
                f"{label}: expected "
                "[sequence, heads, d_head], "
                f"got "
                f"{tuple(clean_activation.shape)}"
            )

        if clean_activation.shape[1] != N_HEADS:

            raise ValueError(
                f"{label}: unexpected "
                "head count."
            )

        if clean_activation.shape[2] != D_HEAD:

            raise ValueError(
                f"{label}: unexpected "
                "head dimension."
            )

        clean_final = (
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

                if activation.ndim != 4:

                    raise ValueError(
                        "Runtime hook activation "
                        f"has invalid shape "
                        f"{tuple(activation.shape)}"
                    )

                if activation.shape[0] != 1:

                    raise ValueError(
                        "Expected batch size 1."
                    )

                patched = (
                    activation.clone()
                )

                corrupted_final = (
                    activation.shape[1]
                    - 1
                )

                patched[
                    0,
                    corrupted_final,
                    head_index,
                    :
                ] = (
                    clean_vector
                    .to(
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
                    clean_final,
                ),
            )
        )

    return hooks


# ============================================================================
# RUN MULTI-HEAD PATCH
# ============================================================================


@torch.no_grad()
def run_multi_head_patch(
    model,
    corrupted_text: str,
    clean_cache,
    combination,
    positive_id: int,
    negative_id: int,
):

    prompt = make_prompt(
        corrupted_text
    )

    tokens = model.to_tokens(
        prompt,
        prepend_bos=True,
    )

    hooks = (
        create_multi_head_patch_hooks(
            clean_cache,
            combination,
        )
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
# BASELINES
# ============================================================================


def calculate_discovery_baselines(
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
            -
            corrupted_ld
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

    return baseline_rows


# ============================================================================
# MULTI-HEAD EXPERIMENT
# ============================================================================


def run_experiment(
    model,
    rows,
    baseline_rows,
    combinations,
    positive_id,
    negative_id,
):

    section(
        "DISCOVERY MULTI-HEAD PATCHING"
    )

    print(
        f"Discovery pairs: "
        f"{len(rows)}"
    )

    print(
        f"Conditions: "
        f"{len(combinations)}"
    )

    expected_rows = (
        len(rows)
        *
        len(combinations)
    )

    print(
        f"Expected result rows: "
        f"{expected_rows}"
    )

    baseline_lookup = {
        row["pair_id"]: row
        for row in baseline_rows
    }

    results = []

    total = expected_rows

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

        baseline = (
            baseline_lookup[
                pair_id
            ]
        )

        clean_ld = (
            baseline["clean_ld"]
        )

        corrupted_ld = (
            baseline["corrupted_ld"]
        )

        for condition_index, combination in enumerate(
            combinations,
            start=1,
        ):

            labels = [
                item[2]
                for item in combination
            ]

            condition_name = (
                " + ".join(labels)
            )

            patched_ld = (
                run_multi_head_patch(
                    model=model,
                    corrupted_text=(
                        row["corrupted_text"]
                    ),
                    clean_cache=(
                        clean_cache
                    ),
                    combination=(
                        combination
                    ),
                    positive_id=(
                        positive_id
                    ),
                    negative_id=(
                        negative_id
                    ),
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

                    "condition":
                        condition_name,

                    "n_heads":
                        len(combination),

                    "heads":
                        condition_name,

                    "clean_ld":
                        clean_ld,

                    "corrupted_ld":
                        corrupted_ld,

                    "patched_ld":
                        patched_ld,

                    "behavioral_gap":
                        (
                            clean_ld
                            -
                            corrupted_ld
                        ),

                    "recovery":
                        recovery,
                }
            )

            completed += 1

        print(
            f"[{pair_index:02d}/{len(rows):02d}] "
            f"Pair {pair_id:03d} complete "
            f"({completed}/{total})"
        )

    return results


# ============================================================================
# SUMMARY
# ============================================================================


def summarize_results(
    dataframe,
):

    section(
        "MULTI-HEAD SUMMARY"
    )

    summary = (
        dataframe
        .groupby(
            [
                "condition",
                "n_heads",
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

            positive_fraction=(
                "recovery",
                lambda x: np.mean(
                    np.asarray(x)
                    > 0
                ),
            ),

            recovery_ge_025=(
                "recovery",
                lambda x: np.mean(
                    np.asarray(x)
                    >= 0.25
                ),
            ),

            recovery_ge_050=(
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

    summary = summary.sort_values(
        [
            "mean_recovery",
            "median_recovery",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    summary[
        "rank"
    ] = np.arange(
        1,
        len(summary) + 1,
    )

    columns = [
        "rank",
        "condition",
        "n_heads",
        "mean_recovery",
        "median_recovery",
        "std_recovery",
        "min_recovery",
        "max_recovery",
        "positive_fraction",
        "recovery_ge_025",
        "recovery_ge_050",
        "n_pairs",
    ]

    summary = summary[
        columns
    ]

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

    print()

    print(
        f"Summary saved:"
    )

    print(
        f"  {SUMMARY_CSV}"
    )

    return summary


# ============================================================================
# COMPARE SINGLE VS MULTI
# ============================================================================


def compare_against_single_head(
    summary,
    phase13_ranking,
):

    section(
        "SINGLE-HEAD VS MULTI-HEAD"
    )

    best_single = float(
        phase13_ranking.iloc[0][
            "mean_recovery"
        ]
    )

    best_condition = (
        summary.iloc[0]
    )

    best_multi = float(
        best_condition[
            "mean_recovery"
        ]
    )

    improvement = (
        best_multi
        -
        best_single
    )

    print(
        "Best Phase 13 single head:"
    )

    print(
        f"  "
        f"{phase13_ranking.iloc[0]['head_label']} "
        f"= {best_single:.4f}"
    )

    print()

    print(
        "Best Phase 14 condition:"
    )

    print(
        f"  "
        f"{best_condition['condition']} "
        f"= {best_multi:.4f}"
    )

    print()

    print(
        f"Multi-head improvement: "
        f"{improvement:+.4f}"
    )

    if improvement > 0:

        print(
            "Multi-head condition improves "
            "over the strongest single head: PASS"
        )

    else:

        print(
            "Multi-head condition does not "
            "improve over strongest single head."
        )

    return {
        "best_single_head":
            str(
                phase13_ranking.iloc[0][
                    "head_label"
                ]
            ),

        "best_single_recovery":
            best_single,

        "best_multi_head_condition":
            str(
                best_condition[
                    "condition"
                ]
            ),

        "best_multi_recovery":
            best_multi,

        "multi_minus_single":
            improvement,

        "multi_head_better":
            bool(
                improvement > 0
            ),
    }


# ============================================================================
# SANITY CHECK
# ============================================================================


def sanity_check(
    dataframe,
    summary,
    rows,
):

    section(
        "RESULT SANITY CHECK"
    )

    expected_rows = (
        len(rows)
        * 7
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
            "multi-head result rows."
        )

    print(
        "Result row count: PASS"
    )

    if len(summary) != 7:

        raise ValueError(
            "Expected exactly 7 "
            "multi-head conditions."
        )

    print(
        "Seven non-empty combinations: PASS"
    )

    expected_pairs = {
        int(row["pair_id"])
        for row in rows
    }

    actual_pairs = set(
        dataframe[
            "pair_id"
        ].astype(int)
    )

    if actual_pairs != expected_pairs:

        raise ValueError(
            "Unexpected pair IDs "
            "in multi-head results."
        )

    print(
        "Discovery pair IDs: PASS"
    )

    if dataframe[
        "recovery"
    ].isna().all():

        raise ValueError(
            "All recovery values are NaN."
        )

    finite_fraction = np.isfinite(
        dataframe[
            "recovery"
        ].to_numpy()
    ).mean()

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
# PLOT — CONDITION COMPARISON
# ============================================================================


def create_condition_plot(
    summary,
):

    section(
        "CREATING MULTI-HEAD RECOVERY PLOT"
    )

    plot_data = (
        summary
        .sort_values(
            "mean_recovery"
        )
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        plot_data[
            "condition"
        ],
        plot_data[
            "mean_recovery"
        ],
    )

    plt.xlabel(
        "Mean Recovery"
    )

    plt.ylabel(
        "Patch Condition"
    )

    plt.title(
        "Discovery Multi-Head "
        "Activation Patching"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "multi_head_recovery.png"
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
# PLOT — COMBINATION COMPARISON
# ============================================================================


def create_combination_plot(
    summary,
):

    plot_data = (
        summary
        .sort_values(
            [
                "n_heads",
                "mean_recovery",
            ]
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    positions = np.arange(
        len(plot_data)
    )

    plt.plot(
        positions,
        plot_data[
            "mean_recovery"
        ],
        marker="o",
    )

    plt.xticks(
        positions,
        plot_data[
            "condition"
        ],
        rotation=45,
        ha="right",
    )

    plt.xlabel(
        "Head Combination"
    )

    plt.ylabel(
        "Mean Recovery"
    )

    plt.title(
        "Multi-Head Combination Comparison"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "combination_comparison.png"
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
# JSON SUMMARY
# ============================================================================


def save_json_summary(
    candidates,
    combinations,
    summary,
    comparison,
):

    conditions = []

    for _, row in summary.iterrows():

        conditions.append(
            {
                "rank":
                    int(row["rank"]),

                "condition":
                    str(
                        row["condition"]
                    ),

                "n_heads":
                    int(row["n_heads"]),

                "mean_recovery":
                    float(
                        row["mean_recovery"]
                    ),

                "median_recovery":
                    float(
                        row["median_recovery"]
                    ),

                "std_recovery":
                    float(
                        row["std_recovery"]
                    ),

                "positive_fraction":
                    float(
                        row[
                            "positive_fraction"
                        ]
                    ),

                "recovery_ge_025":
                    float(
                        row[
                            "recovery_ge_025"
                        ]
                    ),

                "recovery_ge_050":
                    float(
                        row[
                            "recovery_ge_050"
                        ]
                    ),

                "n_pairs":
                    int(
                        row["n_pairs"]
                    ),
            }
        )

    candidate_labels = [
        item[2]
        for item in candidates
    ]

    payload = {
        "phase": 14,

        "name":
            "discovery_multi_head_activation_patching",

        "model":
            MODEL_NAME,

        "dataset": {
            "split":
                "discovery",

            "pairs":
                EXPECTED_DISCOVERY_PAIRS,

            "heldout_pairs_loaded":
                False,
        },

        "candidate_selection": {
            "source":
                "Phase 13 discovery_head_ranking.csv",

            "top_k":
                TOP_K_HEADS,

            "candidate_heads":
                candidate_labels,
        },

        "combinations": [
            " + ".join(
                item[2]
                for item in combination
            )
            for combination in combinations
        ],

        "metric": {
            "name":
                "logit_difference",

            "definition":
                "logit(' positive') - "
                "logit(' negative')",

            "recovery":
                "(patched - corrupted) / "
                "(clean - corrupted)",
        },

        "patching": {
            "hook":
                "blocks.{layer}.attn.hook_z",

            "position":
                "final_prediction_position",

            "variable_sequence_lengths":
                True,

            "alignment":
                "final_position_only",
        },

        "results": {
            "conditions":
                conditions,

            "best_condition":
                conditions[0]
                if conditions
                else None,

            "comparison_to_best_single":
                comparison,
        },

        "data_leakage": {
            "heldout_loaded":
                False,

            "heldout_used_for_selection":
                False,

            "heldout_used_for_ranking":
                False,

            "heldout_used_for_evaluation":
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
            payload,
            file,
            indent=2,
        )

    print()

    print(
        f"JSON summary saved:"
    )

    print(
        f"  {SUMMARY_JSON}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    section(
        "PHASE 14 — DISCOVERY "
        "MULTI-HEAD ACTIVATION PATCHING"
    )

    print(
        "Discovery split ONLY"
    )

    print(
        "Held-out examples will NOT be loaded."
    )

    print(
        "Candidate heads come ONLY from Phase 13."
    )

    print(
        "Model weights will NOT be modified."
    )

    ensure_directories()

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    section(
        "LOADING DISCOVERY DATASET"
    )

    rows = (
        load_discovery_dataset()
    )

    print(
        f"Discovery pairs loaded: "
        f"{len(rows)}"
    )

    print(
        "Discovery dataset validation: PASS"
    )

    # ------------------------------------------------------------------------
    # Candidate heads
    # ------------------------------------------------------------------------

    (
        candidates,
        phase13_ranking,
    ) = load_candidate_heads()

    # ------------------------------------------------------------------------
    # Combinations
    # ------------------------------------------------------------------------

    combinations = (
        create_combinations(
            candidates
        )
    )

    if len(combinations) != 7:

        raise ValueError(
            "Expected 7 non-empty "
            "combinations for 3 heads."
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
    # Baseline
    # ------------------------------------------------------------------------

    baseline_rows = (
        calculate_discovery_baselines(
            model,
            rows,
            positive_id,
            negative_id,
        )
    )

    # ------------------------------------------------------------------------
    # Cache validation
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
    # Experiment
    # ------------------------------------------------------------------------

    results = (
        run_experiment(
            model=model,
            rows=rows,
            baseline_rows=baseline_rows,
            combinations=combinations,
            positive_id=positive_id,
            negative_id=negative_id,
        )
    )

    dataframe = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------------------

    section(
        "SAVING MULTI-HEAD RESULTS"
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

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary = (
        summarize_results(
            dataframe
        )
    )

    # ------------------------------------------------------------------------
    # Compare single vs multi
    # ------------------------------------------------------------------------

    comparison = (
        compare_against_single_head(
            summary,
            phase13_ranking,
        )
    )

    # ------------------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------------------

    sanity_check(
        dataframe,
        summary,
        rows,
    )

    # ------------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------------

    section(
        "CREATING PHASE 14 FIGURES"
    )

    create_condition_plot(
        summary
    )

    create_combination_plot(
        summary
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    save_json_summary(
        candidates=candidates,
        combinations=combinations,
        summary=summary,
        comparison=comparison,
    )

    # ------------------------------------------------------------------------
    # GPU
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
        "PHASE 14 COMPLETE"
    )

    print(
        "Discovery multi-head patching completed."
    )

    print()

    print(
        f"Discovery pairs: "
        f"{len(rows)}"
    )

    print(
        f"Candidate heads: "
        f"{len(candidates)}"
    )

    print(
        f"Combinations tested: "
        f"{len(combinations)}"
    )

    print(
        f"Result rows: "
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
        "Figures:"
    )

    print(
        f"  {FIGURES_DIR}"
    )

    print()

    print(
        "Held-out examples were NOT loaded."
    )

    print(
        "Model weights were NOT modified."
    )

    print()

    print(
        "Next step: Phase 15 — "
        "held-out generalization evaluation."
    )


if __name__ == "__main__":

    main()