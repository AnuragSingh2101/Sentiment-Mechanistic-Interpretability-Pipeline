"""
PHASE 11 — EXPANDED DATASET CREATION

Purpose
-------
Create a larger 100-pair controlled sentiment dataset for the
final mechanistic-interpretability experiment.

The existing 20-pair pilot dataset is NOT modified.

Dataset design
--------------
100 total clean/corrupted pairs

60 pairs -> discovery split
40 pairs -> held-out evaluation split

Each pair contains:
    CLEAN      = positive sentiment
    CORRUPTED  = negative sentiment

The expanded dataset intentionally contains more linguistic and
domain diversity than the original 20-pair pilot dataset.

Domains
-------
1. Movie / film
2. Restaurant / food
3. Software / technology
4. Products / services
5. Books / media
6. General experiences

Design principles
-----------------
- Balanced positive/negative examples
- Controlled semantic contrast
- No duplicate pairs
- Diverse wording
- Diverse sentence structures
- No modification of the existing pilot dataset
- Deterministic discovery/held-out split
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

SPLITS_DIR = DATA_DIR / "splits"

EXPANDED_DATASET_PATH = (
    RAW_DIR / "expanded_sentiment_pairs.csv"
)

DISCOVERY_PATH = (
    SPLITS_DIR / "discovery_pairs.csv"
)

HELDOUT_PATH = (
    SPLITS_DIR / "heldout_pairs.csv"
)

MANIFEST_PATH = (
    SPLITS_DIR / "phase11_dataset_manifest.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42

TOTAL_PAIRS = 100

DISCOVERY_PAIRS = 60

HELDOUT_PAIRS = 40


# ============================================================================
# DATASET
# ============================================================================
#
# Each entry:
#
# (
#     domain,
#     clean_sentence,
#     corrupted_sentence,
# )
#
# The two sentences should express opposite sentiment while preserving
# as much context as possible.
# ============================================================================


PAIRS = [

    # ========================================================================
    # MOVIES / FILM — 25
    # ========================================================================

    (
        "movie",
        "The movie was excellent.",
        "The movie was terrible.",
    ),
    (
        "movie",
        "The film was wonderful.",
        "The film was awful.",
    ),
    (
        "movie",
        "The acting was amazing.",
        "The acting was horrible.",
    ),
    (
        "movie",
        "The story was brilliant.",
        "The story was dreadful.",
    ),
    (
        "movie",
        "The ending was satisfying.",
        "The ending was disappointing.",
    ),
    (
        "movie",
        "The performance was outstanding.",
        "The performance was terrible.",
    ),
    (
        "movie",
        "The director was talented.",
        "The director was incompetent.",
    ),
    (
        "movie",
        "The soundtrack was beautiful.",
        "The soundtrack was unpleasant.",
    ),
    (
        "movie",
        "The characters were engaging.",
        "The characters were boring.",
    ),
    (
        "movie",
        "The dialogue was clever.",
        "The dialogue was stupid.",
    ),
    (
        "movie",
        "The screenplay was impressive.",
        "The screenplay was disappointing.",
    ),
    (
        "movie",
        "The cinematography was stunning.",
        "The cinematography was ugly.",
    ),
    (
        "movie",
        "The pacing was perfect.",
        "The pacing was awful.",
    ),
    (
        "movie",
        "The acting was convincing.",
        "The acting was unconvincing.",
    ),
    (
        "movie",
        "The plot was fascinating.",
        "The plot was tedious.",
    ),
    (
        "movie",
        "The film was enjoyable.",
        "The film was unpleasant.",
    ),
    (
        "movie",
        "The experience was memorable.",
        "The experience was forgettable.",
    ),
    (
        "movie",
        "The movie was entertaining.",
        "The movie was boring.",
    ),
    (
        "movie",
        "The production was polished.",
        "The production was sloppy.",
    ),
    (
        "movie",
        "The final scene was fantastic.",
        "The final scene was horrible.",
    ),
    (
        "movie",
        "I found the movie genuinely enjoyable.",
        "I found the movie genuinely unpleasant.",
    ),
    (
        "movie",
        "The film delivered a compelling story.",
        "The film delivered a tedious story.",
    ),
    (
        "movie",
        "The performances made the film remarkable.",
        "The performances made the film forgettable.",
    ),
    (
        "movie",
        "After watching it, I felt impressed.",
        "After watching it, I felt disappointed.",
    ),
    (
        "movie",
        "The movie remained engaging from beginning to end.",
        "The movie remained boring from beginning to end.",
    ),

    # ========================================================================
    # RESTAURANT / FOOD — 15
    # ========================================================================

    (
        "restaurant",
        "The restaurant was excellent.",
        "The restaurant was terrible.",
    ),
    (
        "restaurant",
        "The food was delicious.",
        "The food was disgusting.",
    ),
    (
        "restaurant",
        "The service was friendly.",
        "The service was rude.",
    ),
    (
        "restaurant",
        "The meal was satisfying.",
        "The meal was disappointing.",
    ),
    (
        "restaurant",
        "The menu was impressive.",
        "The menu was disappointing.",
    ),
    (
        "restaurant",
        "The staff were attentive.",
        "The staff were careless.",
    ),
    (
        "restaurant",
        "The atmosphere was pleasant.",
        "The atmosphere was unpleasant.",
    ),
    (
        "restaurant",
        "The dessert was fantastic.",
        "The dessert was horrible.",
    ),
    (
        "restaurant",
        "The presentation was beautiful.",
        "The presentation was unattractive.",
    ),
    (
        "restaurant",
        "The ingredients tasted fresh.",
        "The ingredients tasted stale.",
    ),
    (
        "restaurant",
        "I found the meal delightful.",
        "I found the meal disappointing.",
    ),
    (
        "restaurant",
        "The dining experience was memorable.",
        "The dining experience was forgettable.",
    ),
    (
        "restaurant",
        "The restaurant provided excellent service.",
        "The restaurant provided terrible service.",
    ),
    (
        "restaurant",
        "The meal was carefully prepared.",
        "The meal was poorly prepared.",
    ),
    (
        "restaurant",
        "Overall, the restaurant exceeded my expectations.",
        "Overall, the restaurant failed my expectations.",
    ),

    # ========================================================================
    # SOFTWARE / TECHNOLOGY — 15
    # ========================================================================

    (
        "software",
        "The software was reliable.",
        "The software was unreliable.",
    ),
    (
        "software",
        "The application was intuitive.",
        "The application was confusing.",
    ),
    (
        "software",
        "The interface was elegant.",
        "The interface was clumsy.",
    ),
    (
        "software",
        "The system was efficient.",
        "The system was inefficient.",
    ),
    (
        "software",
        "The update was helpful.",
        "The update was harmful.",
    ),
    (
        "software",
        "The documentation was clear.",
        "The documentation was confusing.",
    ),
    (
        "software",
        "The tool was powerful.",
        "The tool was useless.",
    ),
    (
        "software",
        "The platform was responsive.",
        "The platform was sluggish.",
    ),
    (
        "software",
        "The implementation was robust.",
        "The implementation was fragile.",
    ),
    (
        "software",
        "The feature worked perfectly.",
        "The feature worked poorly.",
    ),
    (
        "software",
        "The program was easy to use.",
        "The program was difficult to use.",
    ),
    (
        "software",
        "The new version performed impressively.",
        "The new version performed poorly.",
    ),
    (
        "software",
        "The system handled the workload smoothly.",
        "The system handled the workload badly.",
    ),
    (
        "software",
        "The developer experience was excellent.",
        "The developer experience was frustrating.",
    ),
    (
        "software",
        "The product solved the problem effectively.",
        "The product failed to solve the problem effectively.",
    ),

    # ========================================================================
    # PRODUCTS / SERVICES — 15
    # ========================================================================

    (
        "product",
        "The product was excellent.",
        "The product was terrible.",
    ),
    (
        "product",
        "The device was reliable.",
        "The device was unreliable.",
    ),
    (
        "product",
        "The headphones sounded fantastic.",
        "The headphones sounded horrible.",
    ),
    (
        "product",
        "The keyboard felt comfortable.",
        "The keyboard felt uncomfortable.",
    ),
    (
        "product",
        "The laptop performed smoothly.",
        "The laptop performed poorly.",
    ),
    (
        "product",
        "The phone had an impressive camera.",
        "The phone had a disappointing camera.",
    ),
    (
        "product",
        "The packaging was attractive.",
        "The packaging was unattractive.",
    ),
    (
        "product",
        "The build quality was excellent.",
        "The build quality was poor.",
    ),
    (
        "product",
        "The customer support was helpful.",
        "The customer support was unhelpful.",
    ),
    (
        "product",
        "The delivery was remarkably fast.",
        "The delivery was painfully slow.",
    ),
    (
        "product",
        "The service was convenient.",
        "The service was inconvenient.",
    ),
    (
        "product",
        "The purchase was worthwhile.",
        "The purchase was wasteful.",
    ),
    (
        "product",
        "The product exceeded my expectations.",
        "The product disappointed my expectations.",
    ),
    (
        "product",
        "The device felt premium and durable.",
        "The device felt cheap and fragile.",
    ),
    (
        "product",
        "The overall experience was excellent.",
        "The overall experience was frustrating.",
    ),

    # ========================================================================
    # BOOKS / MEDIA — 10
    # ========================================================================

    (
        "book",
        "The book was fascinating.",
        "The book was tedious.",
    ),
    (
        "book",
        "The novel was beautifully written.",
        "The novel was poorly written.",
    ),
    (
        "book",
        "The author developed the characters brilliantly.",
        "The author developed the characters poorly.",
    ),
    (
        "book",
        "The central argument was compelling.",
        "The central argument was unconvincing.",
    ),
    (
        "book",
        "The narrative was engaging.",
        "The narrative was boring.",
    ),
    (
        "book",
        "The ideas were thought-provoking.",
        "The ideas were superficial.",
    ),
    (
        "book",
        "The writing style was elegant.",
        "The writing style was awkward.",
    ),
    (
        "book",
        "The story was memorable.",
        "The story was forgettable.",
    ),
    (
        "book",
        "The chapter was informative.",
        "The chapter was uninformative.",
    ),
    (
        "book",
        "The book provided an excellent reading experience.",
        "The book provided a disappointing reading experience.",
    ),

    # ========================================================================
    # GENERAL EXPERIENCES — 20
    # ========================================================================

    (
        "general",
        "The presentation was impressive.",
        "The presentation was disappointing.",
    ),
    (
        "general",
        "The lecture was informative.",
        "The lecture was confusing.",
    ),
    (
        "general",
        "The meeting was productive.",
        "The meeting was unproductive.",
    ),
    (
        "general",
        "The project was successful.",
        "The project was unsuccessful.",
    ),
    (
        "general",
        "The solution was effective.",
        "The solution was ineffective.",
    ),
    (
        "general",
        "The explanation was clear.",
        "The explanation was confusing.",
    ),
    (
        "general",
        "The plan was practical.",
        "The plan was impractical.",
    ),
    (
        "general",
        "The experience was rewarding.",
        "The experience was frustrating.",
    ),
    (
        "general",
        "The event was enjoyable.",
        "The event was unpleasant.",
    ),
    (
        "general",
        "The discussion was constructive.",
        "The discussion was destructive.",
    ),
    (
        "general",
        "The workshop was engaging.",
        "The workshop was boring.",
    ),
    (
        "general",
        "The result was encouraging.",
        "The result was discouraging.",
    ),
    (
        "general",
        "The decision was sensible.",
        "The decision was foolish.",
    ),
    (
        "general",
        "The process was efficient.",
        "The process was inefficient.",
    ),
    (
        "general",
        "The outcome was favorable.",
        "The outcome was unfavorable.",
    ),
    (
        "general",
        "I found the experience genuinely rewarding.",
        "I found the experience genuinely disappointing.",
    ),
    (
        "general",
        "The overall result exceeded expectations.",
        "The overall result fell below expectations.",
    ),
    (
        "general",
        "The explanation made the topic much clearer.",
        "The explanation made the topic much more confusing.",
    ),
    (
        "general",
        "The project progressed smoothly.",
        "The project progressed poorly.",
    ),
    (
        "general",
        "The final result was remarkably successful.",
        "The final result was remarkably unsuccessful.",
    ),
]


# ============================================================================
# VALIDATION
# ============================================================================


def normalize_text(text: str) -> str:

    return " ".join(
        text.lower()
        .strip()
        .split()
    )


def sentence_word_difference(
    clean: str,
    corrupted: str,
):

    clean_words = clean.split()

    corrupted_words = corrupted.split()

    max_len = max(
        len(clean_words),
        len(corrupted_words),
    )

    differences = []

    for index in range(max_len):

        clean_word = (
            clean_words[index]
            if index < len(clean_words)
            else "<MISSING>"
        )

        corrupted_word = (
            corrupted_words[index]
            if index < len(corrupted_words)
            else "<MISSING>"
        )

        if (
            clean_word.lower()
            != corrupted_word.lower()
        ):

            differences.append(
                (
                    clean_word,
                    corrupted_word,
                )
            )

    return differences


def validate_pairs(pairs):

    print("=" * 70)
    print("PAIR VALIDATION")
    print("=" * 70)

    errors = []

    seen_clean = set()

    seen_corrupted = set()

    seen_pair_keys = set()

    for index, (
        domain,
        clean,
        corrupted,
    ) in enumerate(
        pairs,
        start=1,
    ):

        clean_norm = normalize_text(
            clean
        )

        corrupted_norm = normalize_text(
            corrupted
        )

        pair_key = (
            clean_norm,
            corrupted_norm,
        )

        # Empty text
        if not clean.strip():

            errors.append(
                f"Pair {index}: empty clean sentence"
            )

        if not corrupted.strip():

            errors.append(
                f"Pair {index}: empty corrupted sentence"
            )

        # Duplicate checks
        if clean_norm in seen_clean:

            errors.append(
                f"Pair {index}: duplicate clean sentence"
            )

        if corrupted_norm in seen_corrupted:

            errors.append(
                f"Pair {index}: duplicate corrupted sentence"
            )

        if pair_key in seen_pair_keys:

            errors.append(
                f"Pair {index}: duplicate pair"
            )

        seen_clean.add(
            clean_norm
        )

        seen_corrupted.add(
            corrupted_norm
        )

        seen_pair_keys.add(
            pair_key
        )

        # Ensure clean and corrupted differ
        if clean_norm == corrupted_norm:

            errors.append(
                f"Pair {index}: clean and corrupted are identical"
            )

        differences = sentence_word_difference(
            clean,
            corrupted,
        )

        if len(differences) == 0:

            errors.append(
                f"Pair {index}: no textual difference"
            )

        print(
            f"{index:03d}. "
            f"{domain:<12} "
            f"word_differences={len(differences):2d} | "
            f"CLEAN: {clean}"
        )

    print()

    print(
        f"Total pairs: {len(pairs)}"
    )

    print(
        f"Unique clean sentences: "
        f"{len(seen_clean)}"
    )

    print(
        f"Unique corrupted sentences: "
        f"{len(seen_corrupted)}"
    )

    if errors:

        print()
        print(
            "VALIDATION ERRORS:"
        )

        for error in errors:

            print(
                f"  [ERROR] {error}"
            )

        raise ValueError(
            "Dataset validation failed."
        )

    print()
    print(
        "Dataset validation: PASS"
    )


# ============================================================================
# DOMAIN VALIDATION
# ============================================================================


def validate_domains(pairs):

    print()
    print("=" * 70)
    print("DOMAIN BALANCE")
    print("=" * 70)

    counts = {}

    for domain, _, _ in pairs:

        counts[domain] = (
            counts.get(domain, 0)
            + 1
        )

    for domain in sorted(counts):

        print(
            f"{domain:<15} "
            f"{counts[domain]:3d}"
        )

    return counts


# ============================================================================
# HASH
# ============================================================================


def pair_hash(
    pair_id: int,
    domain: str,
    clean: str,
    corrupted: str,
):

    raw = (
        f"{pair_id}|"
        f"{domain}|"
        f"{clean}|"
        f"{corrupted}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================================
# WRITE CSV
# ============================================================================


def write_csv(
    path: Path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "domain",
                "clean_text",
                "corrupted_text",
                "clean_label",
                "corrupted_label",
                "split",
                "pair_hash",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"Saved: {path}"
    )


# ============================================================================
# CREATE SPLIT
# ============================================================================


def create_split(
    pairs,
):

    print()
    print("=" * 70)
    print("CREATING DISCOVERY / HELD-OUT SPLIT")
    print("=" * 70)

    indexed = list(
        enumerate(
            pairs,
            start=1,
        )
    )

    rng = random.Random(
        RANDOM_SEED
    )

    # ------------------------------------------------------------------------
    # Stratified split by domain
    # ------------------------------------------------------------------------

    domain_to_items = {}

    for pair_id, pair in indexed:

        domain = pair[0]

        domain_to_items.setdefault(
            domain,
            [],
        ).append(
            (
                pair_id,
                pair,
            )
        )

    discovery = []

    heldout = []

    # We perform the split separately inside each domain.
    # This prevents the held-out set from accidentally containing
    # only one or two domains.

    target_heldout_fraction = (
        HELDOUT_PAIRS
        / TOTAL_PAIRS
    )

    for domain in sorted(
        domain_to_items
    ):

        items = domain_to_items[
            domain
        ].copy()

        rng.shuffle(
            items
        )

        domain_heldout_count = round(
            len(items)
            * target_heldout_fraction
        )

        domain_heldout = items[
            :domain_heldout_count
        ]

        domain_discovery = items[
            domain_heldout_count:
        ]

        heldout.extend(
            domain_heldout
        )

        discovery.extend(
            domain_discovery
        )

    # ------------------------------------------------------------------------
    # Correct any rounding difference
    # ------------------------------------------------------------------------

    rng.shuffle(
        discovery
    )

    rng.shuffle(
        heldout
    )

    while len(heldout) > HELDOUT_PAIRS:

        discovery.append(
            heldout.pop()
        )

    while len(heldout) < HELDOUT_PAIRS:

        discovery.sort()

        heldout.append(
            discovery.pop()
        )

    if len(discovery) != DISCOVERY_PAIRS:

        raise RuntimeError(
            "Discovery split size mismatch: "
            f"{len(discovery)} != "
            f"{DISCOVERY_PAIRS}"
        )

    if len(heldout) != HELDOUT_PAIRS:

        raise RuntimeError(
            "Held-out split size mismatch: "
            f"{len(heldout)} != "
            f"{HELDOUT_PAIRS}"
        )

    discovery.sort(
        key=lambda x: x[0]
    )

    heldout.sort(
        key=lambda x: x[0]
    )

    return discovery, heldout


# ============================================================================
# BUILD ROWS
# ============================================================================


def build_rows(
    indexed_pairs,
    split_name,
):

    rows = []

    for pair_id, (
        domain,
        clean,
        corrupted,
    ) in indexed_pairs:

        rows.append(
            {
                "pair_id":
                    pair_id,
                "domain":
                    domain,
                "clean_text":
                    clean,
                "corrupted_text":
                    corrupted,
                "clean_label":
                    "positive",
                "corrupted_label":
                    "negative",
                "split":
                    split_name,
                "pair_hash":
                    pair_hash(
                        pair_id,
                        domain,
                        clean,
                        corrupted,
                    ),
            }
        )

    return rows


# ============================================================================
# SPLIT VALIDATION
# ============================================================================


def validate_split(
    discovery_rows,
    heldout_rows,
):

    print()
    print("=" * 70)
    print("SPLIT VALIDATION")
    print("=" * 70)

    discovery_ids = {
        row["pair_id"]
        for row in discovery_rows
    }

    heldout_ids = {
        row["pair_id"]
        for row in heldout_rows
    }

    overlap = (
        discovery_ids
        & heldout_ids
    )

    if overlap:

        raise ValueError(
            "Discovery/held-out leakage detected: "
            f"{sorted(overlap)}"
        )

    print(
        f"Discovery pairs: "
        f"{len(discovery_rows)}"
    )

    print(
        f"Held-out pairs: "
        f"{len(heldout_rows)}"
    )

    print(
        f"Pair overlap: "
        f"{len(overlap)}"
    )

    if len(discovery_rows) != DISCOVERY_PAIRS:

        raise ValueError(
            "Incorrect discovery split size."
        )

    if len(heldout_rows) != HELDOUT_PAIRS:

        raise ValueError(
            "Incorrect held-out split size."
        )

    # Check domains
    discovery_domains = {}

    heldout_domains = {}

    for row in discovery_rows:

        domain = row["domain"]

        discovery_domains[
            domain
        ] = (
            discovery_domains.get(
                domain,
                0,
            )
            + 1
        )

    for row in heldout_rows:

        domain = row["domain"]

        heldout_domains[
            domain
        ] = (
            heldout_domains.get(
                domain,
                0,
            )
            + 1
        )

    print()
    print("Domain distribution:")
    print()

    all_domains = sorted(
        set(
            discovery_domains
        )
        | set(
            heldout_domains
        )
    )

    print(
        f"{'Domain':<15}"
        f"{'Discovery':>12}"
        f"{'Held-out':>12}"
    )

    print("-" * 40)

    for domain in all_domains:

        print(
            f"{domain:<15}"
            f"{discovery_domains.get(domain, 0):>12}"
            f"{heldout_domains.get(domain, 0):>12}"
        )

    print()
    print(
        "Split validation: PASS"
    )


# ============================================================================
# MANIFEST
# ============================================================================


def write_manifest(
    discovery_rows,
    heldout_rows,
):

    rows = []

    for row in discovery_rows:

        rows.append(
            {
                "pair_id":
                    row["pair_id"],
                "domain":
                    row["domain"],
                "split":
                    "discovery",
                "pair_hash":
                    row["pair_hash"],
            }
        )

    for row in heldout_rows:

        rows.append(
            {
                "pair_id":
                    row["pair_id"],
                "domain":
                    row["domain"],
                "split":
                    "heldout",
                "pair_hash":
                    row["pair_hash"],
            }
        )

    rows.sort(
        key=lambda row:
        int(row["pair_id"])
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "domain",
                "split",
                "pair_hash",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"Saved: {MANIFEST_PATH}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 70)
    print(
        "PHASE 11 — EXPANDED DATASET CREATION"
    )
    print("=" * 70)

    print()
    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Total pairs: {TOTAL_PAIRS}"
    )

    print(
        f"Discovery pairs: {DISCOVERY_PAIRS}"
    )

    print(
        f"Held-out pairs: {HELDOUT_PAIRS}"
    )

    # ------------------------------------------------------------------------
    # Dataset count
    # ------------------------------------------------------------------------

    if len(PAIRS) != TOTAL_PAIRS:

        raise ValueError(
            f"Expected {TOTAL_PAIRS} pairs, "
            f"but dataset contains {len(PAIRS)}."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    # ------------------------------------------------------------------------
    #
    # Never overwrite the original 20-pair pilot dataset.
    #

    pilot_path = (
        RAW_DIR
        / "sentiment_pairs.csv"
    )

    if pilot_path.exists():

        print()
        print(
            "Existing pilot dataset detected:"
        )

        print(
            pilot_path
        )

        print(
            "It will NOT be modified."
        )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    validate_pairs(
        PAIRS
    )

    domain_counts = (
        validate_domains(
            PAIRS
        )
    )

    # ------------------------------------------------------------------------
    # Create split
    # ------------------------------------------------------------------------

    discovery, heldout = (
        create_split(
            PAIRS
        )
    )

    discovery_rows = (
        build_rows(
            discovery,
            "discovery",
        )
    )

    heldout_rows = (
        build_rows(
            heldout,
            "heldout",
        )
    )

    # ------------------------------------------------------------------------
    # Validate split
    # ------------------------------------------------------------------------

    validate_split(
        discovery_rows,
        heldout_rows,
    )

    # ------------------------------------------------------------------------
    # Write expanded dataset
    # ------------------------------------------------------------------------

    all_rows = (
        discovery_rows
        + heldout_rows
    )

    all_rows.sort(
        key=lambda row:
        int(row["pair_id"])
    )

    write_csv(
        EXPANDED_DATASET_PATH,
        all_rows,
    )

    # ------------------------------------------------------------------------
    # Write split datasets
    # ------------------------------------------------------------------------

    write_csv(
        DISCOVERY_PATH,
        discovery_rows,
    )

    write_csv(
        HELDOUT_PATH,
        heldout_rows,
    )

    # ------------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------------

    write_manifest(
        discovery_rows,
        heldout_rows,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 11 DATASET SUMMARY"
    )
    print("=" * 70)

    print(
        f"Total pairs:          {len(all_rows)}"
    )

    print(
        f"Discovery pairs:      {len(discovery_rows)}"
    )

    print(
        f"Held-out pairs:       {len(heldout_rows)}"
    )

    print(
        f"Clean examples:       {len(all_rows)}"
    )

    print(
        f"Corrupted examples:   {len(all_rows)}"
    )

    print()
    print(
        "Domain counts:"
    )

    for domain in sorted(
        domain_counts
    ):

        print(
            f"  {domain:<15} "
            f"{domain_counts[domain]}"
        )

    # ------------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SAMPLE DISCOVERY PAIRS"
    )
    print("=" * 70)

    for row in discovery_rows[:5]:

        print()
        print(
            f"Pair {row['pair_id']}"
        )

        print(
            f"  Domain:    {row['domain']}"
        )

        print(
            f"  CLEAN:     {row['clean_text']}"
        )

        print(
            f"  CORRUPTED: {row['corrupted_text']}"
        )

    print()
    print("=" * 70)
    print(
        "SAMPLE HELD-OUT PAIRS"
    )
    print("=" * 70)

    for row in heldout_rows[:5]:

        print()
        print(
            f"Pair {row['pair_id']}"
        )

        print(
            f"  Domain:    {row['domain']}"
        )

        print(
            f"  CLEAN:     {row['clean_text']}"
        )

        print(
            f"  CORRUPTED: {row['corrupted_text']}"
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 11 DATASET CREATION: PASS"
    )
    print("=" * 70)

    print()
    print(
        "The original 20-pair pilot dataset was preserved."
    )

    print()
    print(
        "Expanded dataset:"
    )

    print(
        f"  {EXPANDED_DATASET_PATH}"
    )

    print()
    print(
        "Discovery dataset:"
    )

    print(
        f"  {DISCOVERY_PATH}"
    )

    print()
    print(
        "Held-out dataset:"
    )

    print(
        f"  {HELDOUT_PATH}"
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
        "Ready for Phase 11 behavioral validation."
    )


if __name__ == "__main__":
    main()