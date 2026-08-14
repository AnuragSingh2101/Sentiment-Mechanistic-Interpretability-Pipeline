"""
PHASE 7 — TOP-HEAD VALIDATION AND MULTI-HEAD PATCHING

Purpose
-------
Validate the strongest individual heads discovered during Phase 6 and
test whether combinations of those heads produce stronger behavioral
recovery.

Phase 6 top heads:
    L10H4
    L9H2
    L8H9

Test conditions:
    1. L10H4
    2. L9H2
    3. L8H9
    4. L10H4 + L9H2
    5. L10H4 + L8H9
    6. L9H2 + L8H9
    7. L10H4 + L9H2 + L8H9

Metric
------
    LD = logit(" positive") - logit(" negative")

Recovery
--------
    recovery =
        (patched_LD - corrupted_LD)
        /
        (clean_LD - corrupted_LD)

Interpretation
--------------
    0.0  = no recovery
    0.5  = 50% recovery
    1.0  = complete recovery
    >1.0 = overshoot

No model weights are modified.
"""

from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "gpt2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

PHASE6_RANKING_PATH = (
    PROJECT_ROOT
    / "results"
    / "patching"
    / "head_ranking.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "patching"
    / "phase7"
)

POSITIVE_TOKEN = " positive"
NEGATIVE_TOKEN = " negative"

N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64

EPSILON = 1e-8


# Explicit Phase 6 top heads.
TOP_HEADS = [
    (10, 4),  # L10H4
    (9, 2),   # L9H2
    (8, 9),   # L8H9
]


# All requested experimental combinations.
HEAD_CONDITIONS = [
    [(10, 4)],

    [(9, 2)],

    [(8, 9)],

    [(10, 4), (9, 2)],

    [(10, 4), (8, 9)],

    [(9, 2), (8, 9)],

    [(10, 4), (9, 2), (8, 9)],
]


POSITIVE_ID: int | None = None
NEGATIVE_ID: int | None = None


# ============================================================================
# HELPERS
# ============================================================================


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def condition_label(
    heads: list[tuple[int, int]]
) -> str:
    return " + ".join(
        head_label(layer, head)
        for layer, head in heads
    )


# ============================================================================
# DATASET
# ============================================================================


def load_dataset() -> list[dict[str, str]]:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    required = {
        "pair_id",
        "clean_text",
        "corrupted_text",
    }

    missing = required - set(
        reader.fieldnames or []
    )

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    return rows


# ============================================================================
# PROMPT
# ============================================================================


def build_prompt(text: str) -> str:

    return (
        f"Review: {text}\n"
        "Sentiment:"
    )


# ============================================================================
# TOKEN
# ============================================================================


def get_single_token_id(
    bridge: TransformerBridge,
    token_text: str,
) -> int:

    token_ids = bridge.to_tokens(
        token_text,
        prepend_bos=False,
    )

    if token_ids.ndim == 2:
        token_ids = token_ids[0]

    if len(token_ids) != 1:
        raise ValueError(
            f"{token_text!r} does not tokenize to one token."
        )

    return int(token_ids[0].item())


# ============================================================================
# LOGIT DIFFERENCE
# ============================================================================


def calculate_ld(
    logits: torch.Tensor,
) -> float:

    if POSITIVE_ID is None or NEGATIVE_ID is None:
        raise RuntimeError(
            "Sentiment token IDs not initialized."
        )

    final_logits = logits[0, -1]

    return float(
        (
            final_logits[POSITIVE_ID]
            - final_logits[NEGATIVE_ID]
        ).item()
    )


# ============================================================================
# NORMAL FORWARD
# ============================================================================


def run_model(
    bridge: TransformerBridge,
    prompt: str,
) -> float:

    tokens = bridge.to_tokens(
        prompt,
        prepend_bos=True,
    ).to(bridge.cfg.device)

    with torch.no_grad():

        logits = bridge(tokens)

    ld = calculate_ld(logits)

    del tokens
    del logits

    return ld


# ============================================================================
# CACHE
# ============================================================================


def load_cache(
    pair_id: int,
    condition: str = "clean",
) -> dict:

    path = (
        CACHE_DIR
        / f"pair_{pair_id:02d}_{condition}.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found:\n{path}"
        )

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(cache, dict):
        raise TypeError(
            f"Expected dict cache, got {type(cache)}"
        )

    if "activation" not in cache:
        raise KeyError(
            f"Cache missing 'activation'. "
            f"Keys: {list(cache.keys())}"
        )

    return cache


def get_layer_activation(
    cache: dict,
    layer: int,
) -> torch.Tensor:

    store = cache["activation"]

    if layer in store:
        activation = store[layer]

    elif str(layer) in store:
        activation = store[str(layer)]

    else:
        raise KeyError(
            f"Layer {layer} missing from cache."
        )

    if not isinstance(
        activation,
        torch.Tensor,
    ):
        raise TypeError(
            f"Layer {layer} activation is not a tensor."
        )

    expected = (
        N_HEADS,
        D_HEAD,
    )

    if tuple(activation.shape) != expected:
        raise ValueError(
            f"Layer {layer}: expected {expected}, "
            f"got {tuple(activation.shape)}"
        )

    return activation


# ============================================================================
# MULTI-HEAD PATCH HOOK
# ============================================================================


def make_multi_head_hook(
    layer: int,
    heads: list[int],
    clean_activation: torch.Tensor,
):

    hook_name = (
        f"blocks.{layer}.attn.hook_z"
    )

    if tuple(clean_activation.shape) != (
        N_HEADS,
        D_HEAD,
    ):
        raise ValueError(
            f"Unexpected clean activation shape: "
            f"{tuple(clean_activation.shape)}"
        )

    clean_heads = (
        clean_activation[heads]
        .detach()
        .clone()
    )

    def hook_fn(
        activation: torch.Tensor,
        hook,
    ) -> torch.Tensor:

        if activation.ndim != 4:
            raise RuntimeError(
                f"Expected hook_z rank 4, "
                f"got {activation.ndim}"
            )

        if activation.shape[2] != N_HEADS:
            raise RuntimeError(
                f"Expected {N_HEADS} heads, "
                f"got {activation.shape[2]}"
            )

        if activation.shape[3] != D_HEAD:
            raise RuntimeError(
                f"Expected d_head={D_HEAD}, "
                f"got {activation.shape[3]}"
            )

        patched = activation.clone()

        replacement = (
            clean_heads
            .to(
                device=activation.device,
                dtype=activation.dtype,
            )
        )

        patched[
            :,
            -1,
            heads,
            :
        ] = replacement.unsqueeze(0)

        return patched

    return hook_name, hook_fn


# ============================================================================
# PATCH ONE CONDITION
# ============================================================================


def run_multi_head_patch(
    bridge: TransformerBridge,
    prompt: str,
    heads: list[tuple[int, int]],
    clean_cache: dict,
) -> float:

    tokens = bridge.to_tokens(
        prompt,
        prepend_bos=True,
    ).to(bridge.cfg.device)

    hooks = []

    # Group heads by layer.
    by_layer: dict[int, list[int]] = {}

    for layer, head in heads:

        by_layer.setdefault(
            layer,
            []
        ).append(head)

    for layer, layer_heads in by_layer.items():

        clean_activation = (
            get_layer_activation(
                clean_cache,
                layer,
            )
        )

        hook_name, hook_fn = (
            make_multi_head_hook(
                layer,
                layer_heads,
                clean_activation,
            )
        )

        hooks.append(
            (
                hook_name,
                hook_fn,
            )
        )

    with torch.no_grad():

        logits = bridge.run_with_hooks(
            tokens,
            fwd_hooks=hooks,
        )

    patched_ld = calculate_ld(logits)

    del tokens
    del logits

    return patched_ld


# ============================================================================
# RECOVERY
# ============================================================================


def recovery(
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
# MAIN
# ============================================================================


def main():

    global POSITIVE_ID
    global NEGATIVE_ID

    print("=" * 70)
    print("PHASE 7 — TOP-HEAD VALIDATION & MULTI-HEAD PATCHING")
    print("=" * 70)

    # ========================================================================
    # DATASET
    # ========================================================================

    rows = load_dataset()

    print(
        f"\nPairs loaded: {len(rows)}"
    )

    # ========================================================================
    # DEVICE
    # ========================================================================

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

        memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"GPU memory: {memory:.2f} GB"
        )

    # ========================================================================
    # MODEL
    # ========================================================================

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

    # ========================================================================
    # ARCHITECTURE
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
    # TOKENS
    # ========================================================================

    print("\n" + "=" * 70)
    print("SENTIMENT TOKEN VALIDATION")
    print("=" * 70)

    POSITIVE_ID = get_single_token_id(
        bridge,
        POSITIVE_TOKEN,
    )

    NEGATIVE_ID = get_single_token_id(
        bridge,
        NEGATIVE_TOKEN,
    )

    print(
        f"{POSITIVE_TOKEN!r} → {POSITIVE_ID}"
    )

    print(
        f"{NEGATIVE_TOKEN!r} → {NEGATIVE_ID}"
    )

    # ========================================================================
    # OUTPUT
    # ========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # BASELINE
    # ========================================================================

    print("\n" + "=" * 70)
    print("BASELINE BEHAVIOR")
    print("=" * 70)

    baselines = {}

    for row in rows:

        pair_id = int(
            row["pair_id"]
        )

        clean_prompt = build_prompt(
            row["clean_text"]
        )

        corrupted_prompt = build_prompt(
            row["corrupted_text"]
        )

        clean_ld = run_model(
            bridge,
            clean_prompt,
        )

        corrupted_ld = run_model(
            bridge,
            corrupted_prompt,
        )

        gap = (
            clean_ld
            - corrupted_ld
        )

        baselines[pair_id] = {
            "pair_id": pair_id,
            "clean_text": row[
                "clean_text"
            ],
            "corrupted_text": row[
                "corrupted_text"
            ],
            "clean_ld": clean_ld,
            "corrupted_ld": corrupted_ld,
            "gap": gap,
        }

        print(
            f"Pair {pair_id:02d}: "
            f"clean={clean_ld:+.4f} | "
            f"corrupted={corrupted_ld:+.4f} | "
            f"gap={gap:+.4f}"
        )

    # ========================================================================
    # HEAD CONDITIONS
    # ========================================================================

    print("\n" + "=" * 70)
    print("EXPERIMENTAL CONDITIONS")
    print("=" * 70)

    for index, heads in enumerate(
        HEAD_CONDITIONS,
        start=1,
    ):

        print(
            f"{index}. "
            f"{condition_label(heads)}"
        )

    # ========================================================================
    # PATCHING
    # ========================================================================

    detailed_results = []
    condition_summary = []

    total_conditions = len(
        HEAD_CONDITIONS
    )

    for condition_index, heads in enumerate(
        HEAD_CONDITIONS,
        start=1,
    ):

        label = condition_label(heads)

        print("\n" + "=" * 70)
        print(
            f"CONDITION {condition_index}/{total_conditions}: "
            f"{label}"
        )
        print("=" * 70)

        condition_recoveries = []

        for pair_index, row in enumerate(
            rows,
            start=1,
        ):

            pair_id = int(
                row["pair_id"]
            )

            baseline = baselines[
                pair_id
            ]

            clean_cache = load_cache(
                pair_id,
                "clean",
            )

            corrupted_prompt = build_prompt(
                row["corrupted_text"]
            )

            patched_ld = run_multi_head_patch(
                bridge,
                corrupted_prompt,
                heads,
                clean_cache,
            )

            rec = recovery(
                baseline["clean_ld"],
                baseline["corrupted_ld"],
                patched_ld,
            )

            condition_recoveries.append(
                rec
            )

            detailed_results.append(
                {
                    "condition": label,
                    "num_heads": len(heads),
                    "heads": label,
                    "pair_id": pair_id,
                    "clean_ld": baseline[
                        "clean_ld"
                    ],
                    "corrupted_ld": baseline[
                        "corrupted_ld"
                    ],
                    "patched_ld": patched_ld,
                    "baseline_gap": baseline[
                        "gap"
                    ],
                    "recovery": rec,
                }
            )

            print(
                f"{pair_index:02d}/{len(rows):02d} "
                f"Pair {pair_id:02d} | "
                f"patched LD={patched_ld:+.4f} | "
                f"recovery={rec:+.4f}"
            )

            del clean_cache

        valid = [
            x
            for x in condition_recoveries
            if math.isfinite(x)
        ]

        mean_recovery = (
            sum(valid) / len(valid)
            if valid
            else float("nan")
        )

        positive_fraction = (
            sum(
                x > 0
                for x in valid
            )
            / len(valid)
            if valid
            else float("nan")
        )

        full_recovery_fraction = (
            sum(
                x >= 1.0
                for x in valid
            )
            / len(valid)
            if valid
            else float("nan")
        )

        condition_summary.append(
            {
                "condition": label,
                "num_heads": len(heads),
                "mean_recovery": mean_recovery,
                "median_recovery": (
                    sorted(valid)[
                        len(valid) // 2
                    ]
                    if valid
                    else float("nan")
                ),
                "positive_recovery_fraction": (
                    positive_fraction
                ),
                "full_recovery_fraction": (
                    full_recovery_fraction
                ),
                "valid_pairs": len(valid),
                "total_pairs": len(rows),
            }
        )

        print(
            f"\nCondition summary: {label}"
        )

        print(
            f"Mean recovery: "
            f"{mean_recovery:+.4f}"
        )

        print(
            f"Positive recovery: "
            f"{positive_fraction * 100:.2f}%"
        )

    # ========================================================================
    # RANK CONDITIONS
    # ========================================================================

    ranked = sorted(
        condition_summary,
        key=lambda x: (
            x["mean_recovery"]
            if math.isfinite(
                x["mean_recovery"]
            )
            else float("-inf")
        ),
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("MULTI-HEAD CONDITION RANKING")
    print("=" * 70)

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{item['condition']} | "
            f"mean recovery="
            f"{item['mean_recovery']:+.4f} | "
            f"positive="
            f"{item['positive_recovery_fraction'] * 100:.1f}%"
        )

    # ========================================================================
    # SAVE DETAILED CSV
    # ========================================================================

    detailed_path = (
        OUTPUT_DIR
        / "multi_head_results.csv"
    )

    with detailed_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "num_heads",
                "heads",
                "pair_id",
                "clean_ld",
                "corrupted_ld",
                "patched_ld",
                "baseline_gap",
                "recovery",
            ],
        )

        writer.writeheader()

        writer.writerows(
            detailed_results
        )

    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================

    summary_path = (
        OUTPUT_DIR
        / "multi_head_summary.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "condition",
                "num_heads",
                "mean_recovery",
                "median_recovery",
                "positive_recovery_fraction",
                "full_recovery_fraction",
                "valid_pairs",
                "total_pairs",
            ],
        )

        writer.writeheader()

        for rank, item in enumerate(
            ranked,
            start=1,
        ):

            writer.writerow(
                {
                    "rank": rank,
                    **item,
                }
            )

    # ========================================================================
    # SAVE JSON
    # ========================================================================

    json_path = (
        OUTPUT_DIR
        / "phase7_summary.json"
    )

    json_data = {
        "experiment": (
            "top_head_validation_and_multi_head_patching"
        ),
        "model": MODEL_NAME,
        "device": device,
        "n_pairs": len(rows),
        "top_heads": [
            head_label(
                layer,
                head,
            )
            for layer, head in TOP_HEADS
        ],
        "conditions": ranked,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            indent=2,
            allow_nan=True,
        )

    # ========================================================================
    # GPU MEMORY
    # ========================================================================

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

    # ========================================================================
    # CLEANUP
    # ========================================================================

    del bridge

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # ========================================================================
    # FINAL
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE")
    print("=" * 70)

    print(
        "\nTop individual heads validated:"
    )

    for layer, head in TOP_HEADS:

        print(
            f"  - {head_label(layer, head)}"
        )

    print(
        "\nResults saved:"
    )

    print(
        f"  {detailed_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        "\nDo NOT move to Phase 8 until these "
        "results have been inspected."
    )


if __name__ == "__main__":
    main()