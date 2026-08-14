"""
PHASE 12 — EXPANDED ACTIVATION CACHING

Purpose
-------
Cache GPT-2-small attention-head activations for the validated
100-pair expanded dataset.

Dataset split
-------------
60 discovery pairs
40 held-out pairs

For every pair we cache:

    clean activation
    corrupted activation

The discovery and held-out datasets are stored in separate directories
to prevent accidental leakage during later circuit discovery.

Important
---------
This phase ONLY caches activations.

It does NOT:

    - perform activation patching
    - rank attention heads
    - discover a circuit
    - modify model weights
    - use held-out examples for model/circuit selection

Activation
----------
Primary activation:

    blocks.{layer}.attn.hook_z

Shape:

    [sequence_length, n_heads, d_head]

For GPT-2-small:

    n_layers = 12
    n_heads = 12
    d_head = 64

Therefore each cached layer has shape:

    [sequence_length, 12, 64]

Sequence lengths are NOT assumed to be identical between
clean and corrupted examples.
"""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import torch

from transformer_lens.model_bridge import TransformerBridge


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

SPLITS_DIR = DATA_DIR / "splits"

OUTPUT_DIR = DATA_DIR / "expanded_activations"

DISCOVERY_DIR = OUTPUT_DIR / "discovery"

HELDOUT_DIR = OUTPUT_DIR / "heldout"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "expanded_activation_cache"
)

METADATA_PATH = (
    RESULTS_DIR
    / "activation_cache_metadata.json"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "activation_cache_manifest.csv"
)

ALL_DATASET_PATH = (
    DATA_DIR
    / "raw"
    / "expanded_sentiment_pairs.csv"
)

DISCOVERY_PATH = (
    SPLITS_DIR
    / "discovery_pairs.csv"
)

HELDOUT_PATH = (
    SPLITS_DIR
    / "heldout_pairs.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "gpt2"

EXPECTED_LAYERS = 12

EXPECTED_HEADS = 12

EXPECTED_HEAD_DIM = 64

EXPECTED_DISCOVERY_PAIRS = 60

EXPECTED_HELDOUT_PAIRS = 40

TOTAL_EXPECTED_PAIRS = (
    EXPECTED_DISCOVERY_PAIRS
    + EXPECTED_HELDOUT_PAIRS
)

HOOK_TEMPLATE = (
    "blocks.{layer}.attn.hook_z"
)


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
# DIRECTORY SETUP
# ============================================================================


def create_directories() -> None:

    DISCOVERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HELDOUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


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


def load_csv(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required dataset file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            raise ValueError(
                f"No CSV header found in:\n{path}"
            )

        missing = (
            REQUIRED_COLUMNS
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                f"Missing columns in {path}:\n"
                f"{sorted(missing)}"
            )

        rows = list(reader)

    return rows


# ============================================================================
# DATASET VALIDATION
# ============================================================================


def validate_dataset_splits(
    discovery_rows,
    heldout_rows,
):

    section("DATASET SPLIT VALIDATION")

    print(
        f"Discovery pairs: {len(discovery_rows)}"
    )

    print(
        f"Held-out pairs:  {len(heldout_rows)}"
    )

    if len(discovery_rows) != EXPECTED_DISCOVERY_PAIRS:

        raise ValueError(
            "Unexpected discovery-set size: "
            f"{len(discovery_rows)} "
            f"(expected "
            f"{EXPECTED_DISCOVERY_PAIRS})"
        )

    if len(heldout_rows) != EXPECTED_HELDOUT_PAIRS:

        raise ValueError(
            "Unexpected held-out-set size: "
            f"{len(heldout_rows)} "
            f"(expected "
            f"{EXPECTED_HELDOUT_PAIRS})"
        )

    discovery_ids = {
        int(row["pair_id"])
        for row in discovery_rows
    }

    heldout_ids = {
        int(row["pair_id"])
        for row in heldout_rows
    }

    overlap = (
        discovery_ids
        & heldout_ids
    )

    if overlap:

        raise ValueError(
            "DISCOVERY/HOLDOUT LEAKAGE DETECTED:\n"
            f"{sorted(overlap)}"
        )

    if len(discovery_ids) != len(
        discovery_rows
    ):

        raise ValueError(
            "Duplicate pair IDs found "
            "in discovery dataset."
        )

    if len(heldout_ids) != len(
        heldout_rows
    ):

        raise ValueError(
            "Duplicate pair IDs found "
            "in held-out dataset."
        )

    all_ids = (
        discovery_ids
        | heldout_ids
    )

    if len(all_ids) != TOTAL_EXPECTED_PAIRS:

        raise ValueError(
            "Combined split does not contain "
            f"{TOTAL_EXPECTED_PAIRS} unique pairs. "
            f"Found {len(all_ids)}."
        )

    print(
        "Pair IDs: PASS"
    )

    print(
        "Discovery/held-out separation: PASS"
    )

    # Validate labels.

    for row in (
        discovery_rows
        + heldout_rows
    ):

        if row["clean_label"] != "positive":

            raise ValueError(
                f"Unexpected clean label "
                f"for pair {row['pair_id']}: "
                f"{row['clean_label']}"
            )

        if row["corrupted_label"] != "negative":

            raise ValueError(
                f"Unexpected corrupted label "
                f"for pair {row['pair_id']}: "
                f"{row['corrupted_label']}"
            )

    print(
        "Labels: PASS"
    )

    print(
        "Dataset split validation: PASS"
    )


# ============================================================================
# MODEL LOADING
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
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
        )

        print(
            f"GPU memory: "
            f"{total_memory / (1024 ** 3):.2f} GB"
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
        f"Layers:       {model.cfg.n_layers}"
    )

    print(
        f"Heads/layer:  {model.cfg.n_heads}"
    )

    print(
        f"Head dim:     {model.cfg.d_head}"
    )

    print(
        f"Hidden size:  {model.cfg.d_model}"
    )

    if model.cfg.n_layers != EXPECTED_LAYERS:

        raise ValueError(
            f"Expected {EXPECTED_LAYERS} layers, "
            f"got {model.cfg.n_layers}"
        )

    if model.cfg.n_heads != EXPECTED_HEADS:

        raise ValueError(
            f"Expected {EXPECTED_HEADS} heads, "
            f"got {model.cfg.n_heads}"
        )

    if model.cfg.d_head != EXPECTED_HEAD_DIM:

        raise ValueError(
            f"Expected d_head={EXPECTED_HEAD_DIM}, "
            f"got {model.cfg.d_head}"
        )

    print(
        "GPT-2-small architecture: PASS"
    )


# ============================================================================
# TOKEN VALIDATION
# ============================================================================


def validate_tokens(model):

    section("TOKEN VALIDATION")

    positive = model.to_tokens(
        " positive",
        prepend_bos=False,
    )

    negative = model.to_tokens(
        " negative",
        prepend_bos=False,
    )

    if positive.numel() != 1:

        raise ValueError(
            "' positive' must be one token."
        )

    if negative.numel() != 1:

        raise ValueError(
            "' negative' must be one token."
        )

    positive_id = int(
        positive.flatten()[0].item()
    )

    negative_id = int(
        negative.flatten()[0].item()
    )

    print(
        f"' positive' → {positive_id}"
    )

    print(
        f"' negative' → {negative_id}"
    )

    print(
        "Token validation: PASS"
    )


# ============================================================================
# HOOK VALIDATION
# ============================================================================


def validate_hooks(model):

    section("ATTENTION HOOK VALIDATION")

    for layer in range(
        EXPECTED_LAYERS
    ):

        hook_name = HOOK_TEMPLATE.format(
            layer=layer
        )

        hook = model.get_hook_point(
            hook_name
        )

        if hook is None:

            raise ValueError(
                f"Missing hook: {hook_name}"
            )

        print(
            f"{hook_name}: PASS"
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
# CACHE SERIALIZATION
# ============================================================================


def save_activation_cache(
    path: Path,
    cache_data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        cache_data,
        path,
    )


# ============================================================================
# HASH
# ============================================================================


def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:

                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================================
# CACHE ONE EXAMPLE
# ============================================================================


@torch.no_grad()
def cache_example(
    model,
    text: str,
):

    prompt = make_prompt(
        text
    )

    tokens = model.to_tokens(
        prompt,
        prepend_bos=True,
    )

    logits, cache = (
        model.run_with_cache(
            tokens
        )
    )

    del logits

    activation_cache = {}

    for layer in range(
        EXPECTED_LAYERS
    ):

        hook_name = HOOK_TEMPLATE.format(
            layer=layer
        )

        activation = cache[
            hook_name
        ]

        # Expected:
        #
        # [batch, sequence, heads, d_head]
        #
        # We cache one example, so remove
        # the batch dimension.

        if activation.ndim != 4:

            raise ValueError(
                f"Unexpected activation rank "
                f"for {hook_name}: "
                f"{tuple(activation.shape)}"
            )

        if activation.shape[0] != 1:

            raise ValueError(
                f"Unexpected batch dimension "
                f"for {hook_name}: "
                f"{tuple(activation.shape)}"
            )

        activation = (
            activation[
                0
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        expected_heads = (
            EXPECTED_HEADS
        )

        expected_d_head = (
            EXPECTED_HEAD_DIM
        )

        if (
            activation.shape[1]
            != expected_heads
        ):

            raise ValueError(
                f"Unexpected head count "
                f"for {hook_name}: "
                f"{tuple(activation.shape)}"
            )

        if (
            activation.shape[2]
            != expected_d_head
        ):

            raise ValueError(
                f"Unexpected head dimension "
                f"for {hook_name}: "
                f"{tuple(activation.shape)}"
            )

        activation_cache[
            hook_name
        ] = activation

    sequence_length = int(
        tokens.shape[1]
    )

    token_ids = (
        tokens[
            0
        ]
        .detach()
        .cpu()
        .tolist()
    )

    token_strings = (
        model.to_str_tokens(
            tokens[
                0
            ]
        )
    )

    # Convert token strings into plain
    # serializable Python strings.

    token_strings = [
        str(token)
        for token in token_strings
    ]

    return {
        "prompt": prompt,

        "text": text,

        "sequence_length":
            sequence_length,

        "token_ids":
            token_ids,

        "tokens":
            token_strings,

        "activations":
            activation_cache,
    }


# ============================================================================
# CACHE ONE PAIR
# ============================================================================


def cache_pair(
    model,
    row,
    output_root,
):

    pair_id = int(
        row["pair_id"]
    )

    clean_path = (
        output_root
        / f"pair_{pair_id:03d}_clean.pt"
    )

    corrupted_path = (
        output_root
        / f"pair_{pair_id:03d}_corrupted.pt"
    )

    clean_data = cache_example(
        model,
        row["clean_text"],
    )

    corrupted_data = cache_example(
        model,
        row["corrupted_text"],
    )

    # Add metadata describing the pair.

    clean_data["pair_id"] = pair_id

    clean_data["domain"] = (
        row["domain"]
    )

    clean_data["split"] = (
        row["split"]
    )

    clean_data["label"] = "positive"

    clean_data["condition"] = "clean"

    corrupted_data["pair_id"] = pair_id

    corrupted_data["domain"] = (
        row["domain"]
    )

    corrupted_data["split"] = (
        row["split"]
    )

    corrupted_data["label"] = "negative"

    corrupted_data["condition"] = "corrupted"

    save_activation_cache(
        clean_path,
        clean_data,
    )

    save_activation_cache(
        corrupted_path,
        corrupted_data,
    )

    return (
        clean_path,
        corrupted_path,
        clean_data,
        corrupted_data,
    )


# ============================================================================
# SINGLE CACHE VALIDATION
# ============================================================================


def validate_cache_file(
    path: Path,
):

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "prompt",
        "text",
        "sequence_length",
        "token_ids",
        "tokens",
        "activations",
        "pair_id",
        "domain",
        "split",
        "label",
        "condition",
    }

    missing = (
        required_keys
        - set(data.keys())
    )

    if missing:

        raise ValueError(
            f"Missing keys in {path}: "
            f"{sorted(missing)}"
        )

    activations = (
        data["activations"]
    )

    if len(activations) != (
        EXPECTED_LAYERS
    ):

        raise ValueError(
            f"Expected {EXPECTED_LAYERS} "
            f"layers in {path}, got "
            f"{len(activations)}"
        )

    sequence_length = int(
        data["sequence_length"]
    )

    if len(
        data["token_ids"]
    ) != sequence_length:

        raise ValueError(
            f"Token ID count does not match "
            f"sequence length in {path}"
        )

    if len(
        data["tokens"]
    ) != sequence_length:

        raise ValueError(
            f"Token string count does not match "
            f"sequence length in {path}"
        )

    for layer in range(
        EXPECTED_LAYERS
    ):

        hook_name = HOOK_TEMPLATE.format(
            layer=layer
        )

        if hook_name not in activations:

            raise ValueError(
                f"Missing {hook_name} "
                f"in {path}"
            )

        activation = activations[
            hook_name
        ]

        expected_shape = (
            sequence_length,
            EXPECTED_HEADS,
            EXPECTED_HEAD_DIM,
        )

        if tuple(
            activation.shape
        ) != expected_shape:

            raise ValueError(
                f"Wrong activation shape "
                f"in {path} "
                f"for {hook_name}: "
                f"{tuple(activation.shape)} "
                f"expected {expected_shape}"
            )

    return data


# ============================================================================
# CACHE COUNTS
# ============================================================================


def count_cache_files(
    directory: Path,
):

    clean_files = sorted(
        directory.glob(
            "pair_*_clean.pt"
        )
    )

    corrupted_files = sorted(
        directory.glob(
            "pair_*_corrupted.pt"
        )
    )

    return (
        clean_files,
        corrupted_files,
    )


# ============================================================================
# MANIFEST
# ============================================================================


def build_manifest(
    discovery_rows,
    heldout_rows,
):

    rows = []

    all_rows = (
        [
            (
                row,
                DISCOVERY_DIR,
            )
            for row in discovery_rows
        ]
        +
        [
            (
                row,
                HELDOUT_DIR,
            )
            for row in heldout_rows
        ]
    )

    for row, directory in all_rows:

        pair_id = int(
            row["pair_id"]
        )

        clean_path = (
            directory
            / f"pair_{pair_id:03d}_clean.pt"
        )

        corrupted_path = (
            directory
            / f"pair_{pair_id:03d}_corrupted.pt"
        )

        clean_data = validate_cache_file(
            clean_path
        )

        corrupted_data = validate_cache_file(
            corrupted_path
        )

        clean_length = int(
            clean_data[
                "sequence_length"
            ]
        )

        corrupted_length = int(
            corrupted_data[
                "sequence_length"
            ]
        )

        rows.append(
            {
                "pair_id":
                    pair_id,

                "domain":
                    row["domain"],

                "split":
                    row["split"],

                "clean_path":
                    str(
                        clean_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "corrupted_path":
                    str(
                        corrupted_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "clean_sequence_length":
                    clean_length,

                "corrupted_sequence_length":
                    corrupted_length,

                "sequence_length_equal":
                    (
                        clean_length
                        == corrupted_length
                    ),

                "clean_sha256":
                    sha256_file(
                        clean_path
                    ),

                "corrupted_sha256":
                    sha256_file(
                        corrupted_path
                    ),
            }
        )

    rows.sort(
        key=lambda row:
        row["pair_id"]
    )

    return rows


# ============================================================================
# WRITE MANIFEST
# ============================================================================


def write_manifest(
    rows,
):

    fieldnames = [
        "pair_id",
        "domain",
        "split",
        "clean_path",
        "corrupted_path",
        "clean_sequence_length",
        "corrupted_sequence_length",
        "sequence_length_equal",
        "clean_sha256",
        "corrupted_sha256",
    ]

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ============================================================================
# METADATA
# ============================================================================


def build_metadata(
    discovery_rows,
    heldout_rows,
    manifest_rows,
    device,
):

    clean_lengths = [
        int(
            row[
                "clean_sequence_length"
            ]
        )
        for row in manifest_rows
    ]

    corrupted_lengths = [
        int(
            row[
                "corrupted_sequence_length"
            ]
        )
        for row in manifest_rows
    ]

    unequal_lengths = sum(
        not bool(
            row[
                "sequence_length_equal"
            ]
        )
        for row in manifest_rows
    )

    return {
        "phase": 12,

        "name":
            "expanded_activation_caching",

        "model":
            MODEL_NAME,

        "device":
            str(device),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),

        "dataset": {
            "total_pairs":
                len(
                    discovery_rows
                    + heldout_rows
                ),

            "discovery_pairs":
                len(discovery_rows),

            "heldout_pairs":
                len(heldout_rows),

            "discovery_activation_files":
                len(discovery_rows) * 2,

            "heldout_activation_files":
                len(heldout_rows) * 2,

            "total_activation_files":
                len(manifest_rows) * 2,
        },

        "architecture": {
            "layers":
                EXPECTED_LAYERS,

            "heads_per_layer":
                EXPECTED_HEADS,

            "head_dimension":
                EXPECTED_HEAD_DIM,

            "activation_hook":
                "blocks.{layer}.attn.hook_z",

            "activation_shape":
                "[sequence_length, heads, d_head]",
        },

        "sequence_lengths": {
            "clean_min":
                min(clean_lengths),

            "clean_max":
                max(clean_lengths),

            "corrupted_min":
                min(corrupted_lengths),

            "corrupted_max":
                max(corrupted_lengths),

            "pairs_with_different_lengths":
                unequal_lengths,
        },

        "outputs": {
            "activation_directory":
                str(
                    OUTPUT_DIR.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "discovery_directory":
                str(
                    DISCOVERY_DIR.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "heldout_directory":
                str(
                    HELDOUT_DIR.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "manifest":
                str(
                    MANIFEST_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
        },

        "data_leakage_protection": {
            "discovery_and_heldout_separated":
                True,

            "heldout_used_for_head_selection":
                False,

            "heldout_used_for_activation_patching":
                False,
        },

        "model_weights_modified":
            False,

        "activation_patching_performed":
            False,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():

    section(
        "PHASE 12 — EXPANDED ACTIVATION CACHING"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Purpose: cache attention-head activations "
        "for the 100-pair expanded dataset."
    )

    print(
        "No activation patching will be performed."
    )

    print(
        "No model weights will be modified."
    )

    # ------------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------------

    create_directories()

    # ------------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------------

    section("LOADING DATASETS")

    discovery_rows = load_csv(
        DISCOVERY_PATH
    )

    heldout_rows = load_csv(
        HELDOUT_PATH
    )

    validate_dataset_splits(
        discovery_rows,
        heldout_rows,
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
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
        )

        print(
            f"GPU memory: "
            f"{total_memory / (1024 ** 3):.2f} GB"
        )

    else:

        print(
            "CUDA is unavailable. "
            "Caching will run on CPU."
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

    validate_tokens(
        model
    )

    validate_hooks(
        model
    )

    # ------------------------------------------------------------------------
    # Cache discovery set
    # ------------------------------------------------------------------------

    section(
        "CACHING DISCOVERY ACTIVATIONS"
    )

    print(
        f"Discovery pairs: "
        f"{len(discovery_rows)}"
    )

    print(
        f"Expected files: "
        f"{len(discovery_rows) * 2}"
    )

    for index, row in enumerate(
        discovery_rows,
        start=1,
    ):

        (
            clean_path,
            corrupted_path,
            clean_data,
            corrupted_data,
        ) = cache_pair(
            model,
            row,
            DISCOVERY_DIR,
        )

        print(
            f"[{index:02d}/"
            f"{len(discovery_rows):02d}] "
            f"Pair {int(row['pair_id']):03d} "
            f"cached"
        )

        print(
            f"    CLEAN:     "
            f"seq={clean_data['sequence_length']:3d}"
        )

        print(
            f"    CORRUPTED: "
            f"seq={corrupted_data['sequence_length']:3d}"
        )

    # ------------------------------------------------------------------------
    # Cache held-out set
    # ------------------------------------------------------------------------

    section(
        "CACHING HELD-OUT ACTIVATIONS"
    )

    print(
        f"Held-out pairs: "
        f"{len(heldout_rows)}"
    )

    print(
        f"Expected files: "
        f"{len(heldout_rows) * 2}"
    )

    for index, row in enumerate(
        heldout_rows,
        start=1,
    ):

        (
            clean_path,
            corrupted_path,
            clean_data,
            corrupted_data,
        ) = cache_pair(
            model,
            row,
            HELDOUT_DIR,
        )

        print(
            f"[{index:02d}/"
            f"{len(heldout_rows):02d}] "
            f"Pair {int(row['pair_id']):03d} "
            f"cached"
        )

        print(
            f"    CLEAN:     "
            f"seq={clean_data['sequence_length']:3d}"
        )

        print(
            f"    CORRUPTED: "
            f"seq={corrupted_data['sequence_length']:3d}"
        )

    # ------------------------------------------------------------------------
    # File count validation
    # ------------------------------------------------------------------------

    section(
        "CACHE FILE COUNT VALIDATION"
    )

    discovery_clean, discovery_corrupted = (
        count_cache_files(
            DISCOVERY_DIR
        )
    )

    heldout_clean, heldout_corrupted = (
        count_cache_files(
            HELDOUT_DIR
        )
    )

    print(
        f"Discovery clean files:     "
        f"{len(discovery_clean)}"
    )

    print(
        f"Discovery corrupted files: "
        f"{len(discovery_corrupted)}"
    )

    print(
        f"Held-out clean files:      "
        f"{len(heldout_clean)}"
    )

    print(
        f"Held-out corrupted files:  "
        f"{len(heldout_corrupted)}"
    )

    if len(discovery_clean) != (
        EXPECTED_DISCOVERY_PAIRS
    ):

        raise ValueError(
            "Incorrect discovery clean "
            "cache count."
        )

    if len(discovery_corrupted) != (
        EXPECTED_DISCOVERY_PAIRS
    ):

        raise ValueError(
            "Incorrect discovery corrupted "
            "cache count."
        )

    if len(heldout_clean) != (
        EXPECTED_HELDOUT_PAIRS
    ):

        raise ValueError(
            "Incorrect held-out clean "
            "cache count."
        )

    if len(heldout_corrupted) != (
        EXPECTED_HELDOUT_PAIRS
    ):

        raise ValueError(
            "Incorrect held-out corrupted "
            "cache count."
        )

    total_files = (
        len(discovery_clean)
        + len(discovery_corrupted)
        + len(heldout_clean)
        + len(heldout_corrupted)
    )

    print(
        f"Total activation files: "
        f"{total_files}"
    )

    if total_files != (
        TOTAL_EXPECTED_PAIRS * 2
    ):

        raise ValueError(
            "Unexpected total activation "
            "file count."
        )

    print(
        "Cache file count validation: PASS"
    )

    # ------------------------------------------------------------------------
    # Full cache integrity validation
    # ------------------------------------------------------------------------

    section(
        "CACHE INTEGRITY VALIDATION"
    )

    manifest_rows = build_manifest(
        discovery_rows,
        heldout_rows,
    )

    print(
        f"Validated pairs: "
        f"{len(manifest_rows)}"
    )

    print(
        f"Validated activation files: "
        f"{len(manifest_rows) * 2}"
    )

    # ------------------------------------------------------------------------
    # Check discovery/heldout output separation
    # ------------------------------------------------------------------------

    discovery_manifest_ids = {
        row["pair_id"]
        for row in manifest_rows
        if row["split"] == "discovery"
    }

    heldout_manifest_ids = {
        row["pair_id"]
        for row in manifest_rows
        if row["split"] == "heldout"
    }

    overlap = (
        discovery_manifest_ids
        & heldout_manifest_ids
    )

    if overlap:

        raise ValueError(
            "Activation cache split leakage "
            "detected."
        )

    print(
        "Activation split separation: PASS"
    )

    # ------------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------------

    section(
        "WRITING ACTIVATION MANIFEST"
    )

    write_manifest(
        manifest_rows
    )

    print(
        f"Manifest saved:"
    )

    print(
        f"  {MANIFEST_PATH}"
    )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    metadata = build_metadata(
        discovery_rows,
        heldout_rows,
        manifest_rows,
        device,
    )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Metadata saved:"
    )

    print(
        f"  {METADATA_PATH}"
    )

    # ------------------------------------------------------------------------
    # Sample cache inspection
    # ------------------------------------------------------------------------

    section(
        "SAMPLE CACHE INSPECTION"
    )

    sample_pair_id = int(
        discovery_rows[0]["pair_id"]
    )

    sample_path = (
        DISCOVERY_DIR
        / f"pair_{sample_pair_id:03d}_clean.pt"
    )

    sample = validate_cache_file(
        sample_path
    )

    print(
        f"Sample file:"
    )

    print(
        f"  {sample_path}"
    )

    print(
        f"Pair ID:"
        f" {sample['pair_id']}"
    )

    print(
        f"Split:"
        f" {sample['split']}"
    )

    print(
        f"Condition:"
        f" {sample['condition']}"
    )

    print(
        f"Sequence length:"
        f" {sample['sequence_length']}"
    )

    print(
        f"Number of layers:"
        f" {len(sample['activations'])}"
    )

    first_hook = HOOK_TEMPLATE.format(
        layer=0
    )

    print(
        f"Layer 0 activation:"
        f" {tuple(sample['activations'][first_hook].shape)}"
    )

    print(
        "Sample cache integrity: PASS"
    )

    # ------------------------------------------------------------------------
    # Sequence length report
    # ------------------------------------------------------------------------

    section(
        "SEQUENCE LENGTH ANALYSIS"
    )

    clean_lengths = [
        int(
            row[
                "clean_sequence_length"
            ]
        )
        for row in manifest_rows
    ]

    corrupted_lengths = [
        int(
            row[
                "corrupted_sequence_length"
            ]
        )
        for row in manifest_rows
    ]

    unequal = [
        row
        for row in manifest_rows
        if not row[
            "sequence_length_equal"
        ]
    ]

    print(
        f"Clean sequence range: "
        f"{min(clean_lengths)} - "
        f"{max(clean_lengths)}"
    )

    print(
        f"Corrupted sequence range: "
        f"{min(corrupted_lengths)} - "
        f"{max(corrupted_lengths)}"
    )

    print(
        f"Pairs with different sequence lengths: "
        f"{len(unequal)}"
    )

    if unequal:

        print()

        print(
            "Note:"
        )

        print(
            "Clean/corrupted sequence lengths "
            "are allowed to differ."
        )

        print(
            "Later activation-patching code must "
            "handle token-position alignment explicitly."
        )

    # ------------------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------------------

    if device.type == "cuda":

        section(
            "GPU MEMORY"
        )

        print(
            f"Allocated: "
            f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
        )

        print(
            f"Reserved:  "
            f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB"
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    section(
        "PHASE 12 ACTIVATION CACHING: PASS"
    )

    print(
        "Cached discovery pairs:"
        f" {len(discovery_rows)}"
    )

    print(
        "Cached held-out pairs:"
        f" {len(heldout_rows)}"
    )

    print(
        "Total pairs:"
        f" {len(manifest_rows)}"
    )

    print(
        "Total activation files:"
        f" {len(manifest_rows) * 2}"
    )

    print()

    print(
        "Discovery activations:"
    )

    print(
        f"  {DISCOVERY_DIR}"
    )

    print()

    print(
        "Held-out activations:"
    )

    print(
        f"  {HELDOUT_DIR}"
    )

    print()

    print(
        "Manifest:"
    )

    print(
        f"  {MANIFEST_PATH}"
    )

    print()

    print(
        "Metadata:"
    )

    print(
        f"  {METADATA_PATH}"
    )

    print()

    print(
        "No model weights were modified."
    )

    print(
        "No activation patching was performed."
    )

    print(
        "Held-out examples were not used for "
        "head selection."
    )

    print()

    print(
        "Ready for discovery-split "
        "single-head activation patching."
    )

if __name__ == "__main__":
    main()