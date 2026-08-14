"""
Phase 5 — Cache Clean and Corrupted Activations.

Purpose
-------
Cache GPT-2-small attention-head activations for every clean/corrupted
sentiment pair.

This is the first mechanistic-interpretability phase.

For each pair we store:

    CLEAN:
        attention-head activation

    CORRUPTED:
        attention-head activation

The relevant TransformerLens hook is:

    blocks.{layer}.attn.hook_z

For GPT-2-small:

    layers = 12
    heads per layer = 12
    head dimension = 64

Therefore:

    hook_z shape =
        [batch, sequence_position, 12, 64]

We do NOT patch activations in this phase.

We only verify and cache them.

The cached activations will later be used for single-head
activation patching.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge


# =====================================================================
# CONFIGURATION
# =====================================================================

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

METADATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "metadata"
)

POSITIVE_TOKEN = " positive"
NEGATIVE_TOKEN = " negative"


# ---------------------------------------------------------------------
# Cache settings
# ---------------------------------------------------------------------

# We only need the final token position for the current causal metric.
#
# This dramatically reduces storage compared with saving every token
# position.
#
# However, we still cache the complete 12-head × 64-dimensional
# activation at the final position.

CACHE_FINAL_POSITION_ONLY = True


# =====================================================================
# DATASET LOADING
# =====================================================================


def load_dataset() -> list[dict[str, str]]:
    """
    Load the paired sentiment dataset.
    """

    import csv

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

    missing = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing:

        raise ValueError(
            "Dataset is missing required columns:\n"
            f"{sorted(missing)}"
        )

    return rows


# =====================================================================
# TOKEN VALIDATION
# =====================================================================


def get_single_token_id(
    bridge: TransformerBridge,
    text: str,
) -> int:
    """
    Return the token ID for a string that must tokenize to exactly
    one token.
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

    return int(
        token_ids[0].item()
    )


# =====================================================================
# PROMPT CONSTRUCTION
# =====================================================================


def build_prompt(text: str) -> str:
    """
    Construct the exact behavioral prompt used in Phase 4.5.

    Keeping this identical is critical.

    Phase 4.5 used:

        Review: <text>
        Sentiment:
    """

    return (
        f"Review: {text}\n"
        "Sentiment:"
    )


# =====================================================================
# ACTIVATION CACHE
# =====================================================================


def cache_activation(
    bridge: TransformerBridge,
    prompt: str,
) -> dict:
    """
    Run the model with activation caching and return the final-token
    attention-head activation.

    Returns
    -------
    dict
        {
            "hook_name": str,
            "activation": Tensor,
            "tokens": Tensor,
            "sequence_length": int
        }
    """

    tokens = bridge.to_tokens(
        prompt,
        prepend_bos=True,
    )

    tokens = tokens.to(
        bridge.cfg.device
    )

    hook_name = (
        "blocks.0.attn.hook_z"
    )

    # ---------------------------------------------------------------
    # Run once to discover / verify the hook naming.
    # ---------------------------------------------------------------

    # GPT-2-small has 12 layers.
    #
    # We collect all 12 attention hook_z activations.
    #
    # Dictionary:
    #
    # blocks.0.attn.hook_z
    # blocks.1.attn.hook_z
    # ...
    # blocks.11.attn.hook_z

    hook_names = [
        f"blocks.{layer}.attn.hook_z"
        for layer in range(12)
    ]

    with torch.no_grad():

        logits, cache = bridge.run_with_cache(
            tokens,
            names_filter=hook_names,
        )

    # ---------------------------------------------------------------
    # Extract final position
    # ---------------------------------------------------------------

    activations = {}

    sequence_length = tokens.shape[1]

    for layer in range(12):

        current_hook = (
            f"blocks.{layer}.attn.hook_z"
        )

        activation = cache[current_hook]

        # Expected:
        #
        # [batch, sequence, heads, head_dim]
        #
        # Example:
        #
        # [1, sequence_length, 12, 64]

        if activation.ndim != 4:

            raise RuntimeError(
                f"Unexpected activation shape for "
                f"{current_hook}: "
                f"{tuple(activation.shape)}"
            )

        if activation.shape[0] != 1:

            raise RuntimeError(
                f"Expected batch size 1 for "
                f"{current_hook}, got "
                f"{activation.shape[0]}"
            )

        if activation.shape[2] != 12:

            raise RuntimeError(
                f"Expected 12 heads for "
                f"{current_hook}, got "
                f"{activation.shape[2]}"
            )

        if activation.shape[3] != 64:

            raise RuntimeError(
                f"Expected head dimension 64 for "
                f"{current_hook}, got "
                f"{activation.shape[3]}"
            )

        if CACHE_FINAL_POSITION_ONLY:

            final_activation = (
                activation[
                    0,
                    -1,
                ]
                .detach()
                .cpu()
                .clone()
            )

            # Shape:
            #
            # [12, 64]

            activations[layer] = final_activation

        else:

            full_activation = (
                activation[
                    0
                ]
                .detach()
                .cpu()
                .clone()
            )

            # Shape:
            #
            # [sequence, 12, 64]

            activations[layer] = full_activation

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    del logits
    del cache
    del tokens

    return {
        "hook_name": hook_name,
        "activation": activations,
        "sequence_length": sequence_length,
    }


# =====================================================================
# SAVE ONE EXAMPLE
# =====================================================================


def save_activation_bundle(
    bundle: dict,
    output_path: Path,
) -> None:
    """
    Save activation data using torch.save.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        bundle,
        output_path,
    )


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:

    print("=" * 70)
    print("PHASE 5 — ACTIVATION CACHING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Dataset
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

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
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
    # Model architecture validation
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ARCHITECTURE VALIDATION")
    print("=" * 70)

    n_layers = int(
        bridge.cfg.n_layers
    )

    n_heads = int(
        bridge.cfg.n_heads
    )

    d_head = int(
        bridge.cfg.d_head
    )

    d_model = int(
        bridge.cfg.d_model
    )

    print(
        f"Layers:       {n_layers}"
    )

    print(
        f"Heads/layer:  {n_heads}"
    )

    print(
        f"Head dim:     {d_head}"
    )

    print(
        f"Hidden size:  {d_model}"
    )

    if n_layers != 12:

        raise RuntimeError(
            f"Expected 12 layers, got {n_layers}"
        )

    if n_heads != 12:

        raise RuntimeError(
            f"Expected 12 heads, got {n_heads}"
        )

    if d_head != 64:

        raise RuntimeError(
            f"Expected head dimension 64, got {d_head}"
        )

    print(
        "GPT-2-small architecture: PASS"
    )

    # ---------------------------------------------------------------
    # Token validation
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKEN VALIDATION")
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
        f"{POSITIVE_TOKEN!r} → {positive_id}"
    )

    print(
        f"{NEGATIVE_TOKEN!r} → {negative_id}"
    )

    # ---------------------------------------------------------------
    # Prepare directories
    # ---------------------------------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Test one example before caching everything
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SINGLE-EXAMPLE ACTIVATION TEST")
    print("=" * 70)

    first_row = rows[0]

    test_prompt = build_prompt(
        first_row["clean_text"]
    )

    print(
        f"Prompt:\n{test_prompt}"
    )

    test_bundle = cache_activation(
        bridge,
        test_prompt,
    )

    print(
        f"\nSequence length: "
        f"{test_bundle['sequence_length']}"
    )

    for layer in range(3):

        print(
            f"Layer {layer} activation shape: "
            f"{tuple(test_bundle['activation'][layer].shape)}"
        )

    expected_shape = (
        n_heads,
        d_head,
    )

    for layer in range(n_layers):

        actual_shape = tuple(
            test_bundle["activation"][layer].shape
        )

        if actual_shape != expected_shape:

            raise RuntimeError(
                f"Unexpected cached shape at layer "
                f"{layer}: {actual_shape}. "
                f"Expected {expected_shape}."
            )

    print(
        "\nSingle-example activation test: PASS"
    )

    # ---------------------------------------------------------------
    # Cache all pairs
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CACHING ALL ACTIVATIONS")
    print("=" * 70)

    metadata = []

    total_examples = len(rows) * 2

    completed = 0

    for row in rows:

        pair_id = int(
            row["pair_id"]
        )

        clean_text = row[
            "clean_text"
        ]

        corrupted_text = row[
            "corrupted_text"
        ]

        # -----------------------------------------------------------
        # CLEAN
        # -----------------------------------------------------------

        clean_prompt = build_prompt(
            clean_text
        )

        clean_bundle = cache_activation(
            bridge,
            clean_prompt,
        )

        clean_path = (
            CACHE_DIR
            / f"pair_{pair_id:02d}_clean.pt"
        )

        save_activation_bundle(
            clean_bundle,
            clean_path,
        )

        completed += 1

        print(
            f"[{completed:02d}/{total_examples}] "
            f"Pair {pair_id:02d} CLEAN cached"
        )

        # -----------------------------------------------------------
        # CORRUPTED
        # -----------------------------------------------------------

        corrupted_prompt = build_prompt(
            corrupted_text
        )

        corrupted_bundle = cache_activation(
            bridge,
            corrupted_prompt,
        )

        corrupted_path = (
            CACHE_DIR
            / f"pair_{pair_id:02d}_corrupted.pt"
        )

        save_activation_bundle(
            corrupted_bundle,
            corrupted_path,
        )

        completed += 1

        print(
            f"[{completed:02d}/{total_examples}] "
            f"Pair {pair_id:02d} CORRUPTED cached"
        )

        # -----------------------------------------------------------
        # Metadata
        # -----------------------------------------------------------

        metadata.append(
            {
                "pair_id": pair_id,
                "clean_text": clean_text,
                "corrupted_text": corrupted_text,
                "clean_path": str(
                    clean_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "corrupted_path": str(
                    corrupted_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "clean_sequence_length": (
                    clean_bundle[
                        "sequence_length"
                    ]
                ),
                "corrupted_sequence_length": (
                    corrupted_bundle[
                        "sequence_length"
                    ]
                ),
                "activation_shape": [
                    n_heads,
                    d_head,
                ],
                "layers": n_layers,
                "heads": n_heads,
                "head_dimension": d_head,
                "cache_final_position_only": (
                    CACHE_FINAL_POSITION_ONLY
                ),
            }
        )

        # -----------------------------------------------------------
        # Free memory
        # -----------------------------------------------------------

        del clean_bundle
        del corrupted_bundle

        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------------

    metadata_path = (
        METADATA_DIR
        / "activation_cache_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "model": MODEL_NAME,
                "device": device,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_head": d_head,
                "d_model": d_model,
                "positive_token": POSITIVE_TOKEN,
                "positive_token_id": positive_id,
                "negative_token": NEGATIVE_TOKEN,
                "negative_token_id": negative_id,
                "cache_hook": (
                    "blocks.{layer}.attn.hook_z"
                ),
                "cache_final_position_only": (
                    CACHE_FINAL_POSITION_ONLY
                ),
                "pairs": metadata,
            },
            file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Verify saved files
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CACHE VERIFICATION")
    print("=" * 70)

    expected_files = (
        len(rows) * 2
    )

    cached_files = list(
        CACHE_DIR.glob("pair_*.pt")
    )

    print(
        f"Expected activation files: "
        f"{expected_files}"
    )

    print(
        f"Found activation files:    "
        f"{len(cached_files)}"
    )

    if len(cached_files) != expected_files:

        raise RuntimeError(
            "Activation cache file count does not match "
            "the expected number."
        )

    # ---------------------------------------------------------------
    # Load one cached file and verify it
    # ---------------------------------------------------------------

    sample_path = (
        CACHE_DIR
        / "pair_01_clean.pt"
    )

    sample = torch.load(
        sample_path,
        map_location="cpu",
        weights_only=False,
    )

    print(
        f"\nSample file: "
        f"{sample_path}"
    )

    print(
        f"Sample sequence length: "
        f"{sample['sequence_length']}"
    )

    print(
        f"Number of cached layers: "
        f"{len(sample['activation'])}"
    )

    for layer in range(n_layers):

        shape = tuple(
            sample[
                "activation"
            ][layer].shape
        )

        if shape != (
            n_heads,
            d_head,
        ):

            raise RuntimeError(
                f"Cached sample has invalid shape "
                f"at layer {layer}: {shape}"
            )

    print(
        "Sample cache integrity: PASS"
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
    print("PHASE 5 ACTIVATION CACHING: PASS")
    print("=" * 70)

    print(
        f"\nCached {len(rows)} clean examples."
    )

    print(
        f"Cached {len(rows)} corrupted examples."
    )

    print(
        f"Total activation files: "
        f"{expected_files}"
    )

    print(
        "\nActivation files:"
    )

    print(
        f"  {CACHE_DIR}"
    )

    print(
        "\nMetadata:"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        "\nReady for the single-head activation-patching phase."
    )


if __name__ == "__main__":
    main()