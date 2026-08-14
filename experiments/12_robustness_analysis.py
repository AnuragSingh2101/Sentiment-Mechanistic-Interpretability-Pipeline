"""
PHASE 10 — ROBUSTNESS & STATISTICAL SIGNIFICANCE ANALYSIS

Purpose
-------
Evaluate whether the causal effects observed in Phases 6–9 are
consistent across the 20 paired examples.

This phase DOES NOT:
    - modify model weights
    - perform new activation patching
    - create new activations
    - change the dataset

It analyzes the existing Phase 6 and Phase 7 results.

Main analyses
-------------
1. Candidate-head bootstrap confidence intervals
2. Best single-head vs best multi-head paired comparison
3. Wilcoxon signed-rank test
4. Paired permutation test
5. Bootstrap distribution of the multi-vs-single improvement
6. Per-pair consistency
7. Leave-one-out robustness
8. Effect-size estimates
9. Robustness summary
10. Publication-ready plots

Candidate heads:
    L8H9
    L9H2
    L10H4

Primary multi-head condition:
    L10H4 + L9H2 + L8H9
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

PATCHING_DIR = RESULTS_DIR / "patching"

PHASE7_DIR = PATCHING_DIR / "phase7"

OUTPUT_DIR = RESULTS_DIR / "robustness"

FIGURES_DIR = OUTPUT_DIR / "figures"

PHASE6_RESULTS = PATCHING_DIR / "single_head_results.csv"

PHASE7_RESULTS = PHASE7_DIR / "multi_head_results.csv"

PHASE7_SUMMARY = PHASE7_DIR / "multi_head_summary.csv"


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

CANDIDATE_HEADS = [
    "L8H9",
    "L9H2",
    "L10H4",
]

BEST_SINGLE_HEAD = "L10H4"

BEST_MULTI_CONDITION = (
    "L10H4 + L9H2 + L8H9"
)

EXPECTED_PAIRS = 20

RANDOM_SEED = 42

BOOTSTRAP_ITERATIONS = 10000

PERMUTATION_ITERATIONS = 10000


# ============================================================================
# GENERAL UTILITIES
# ============================================================================


def print_header(title: str):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def require_file(path: Path, description: str):

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def identify_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
):

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def clean_numeric(
    series: pd.Series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================================
# STATISTICS
# ============================================================================


def mean(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.mean(values)
    )


def median(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.median(values)
    )


def std(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return float("nan")

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def cohens_d_paired(
    differences,
):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) < 2:
        return float("nan")

    sd = np.std(
        differences,
        ddof=1,
    )

    if sd == 0:
        return float("nan")

    return float(
        np.mean(differences)
        / sd
    )


def bootstrap_mean_ci(
    values,
    iterations=10000,
    seed=42,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            float("nan"),
            float("nan"),
            np.array([]),
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            iterations,
            len(values),
        ),
    )

    samples = values[
        indices
    ]

    bootstrap_means = np.mean(
        samples,
        axis=1,
    )

    lower = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    return (
        lower,
        upper,
        bootstrap_means,
    )


def paired_permutation_test(
    differences,
    iterations=10000,
    seed=42,
):

    """
    Paired random-sign permutation test.

    Null hypothesis:
        mean difference = 0

    Each pair difference is randomly multiplied
    by +1 or -1.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:

        return float("nan")

    rng = np.random.default_rng(
        seed
    )

    observed = abs(
        np.mean(differences)
    )

    signs = rng.choice(
        np.array(
            [-1.0, 1.0]
        ),
        size=(
            iterations,
            len(differences),
        ),
    )

    permuted = np.mean(
        signs * differences,
        axis=1,
    )

    p_value = (
        np.sum(
            np.abs(permuted)
            >= observed
        )
        + 1
    ) / (
        iterations
        + 1
    )

    return float(
        p_value
    )


def wilcoxon_test(
    x,
    y,
):

    """
    Paired Wilcoxon signed-rank test.

    Falls back gracefully if scipy is unavailable.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) == 0:

        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "available": False,
        }

    try:

        from scipy.stats import wilcoxon

        result = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return {
            "statistic": float(
                result.statistic
            ),
            "p_value": float(
                result.pvalue
            ),
            "available": True,
        }

    except Exception as exc:

        print(
            f"WARNING: Wilcoxon test unavailable: {exc}"
        )

        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "available": False,
        }


# ============================================================================
# LOAD PHASE 6
# ============================================================================


def load_phase6():

    print_header(
        "LOADING PHASE 6 RESULTS"
    )

    require_file(
        PHASE6_RESULTS,
        "Phase 6 single-head results",
    )

    df = pd.read_csv(
        PHASE6_RESULTS
    )

    print(
        f"Rows: {len(df)}"
    )

    head_column = identify_column(
        df,
        [
            "head_label",
            "head",
        ],
        "Phase 6 head column",
    )

    pair_column = identify_column(
        df,
        [
            "pair_id",
        ],
        "Phase 6 pair column",
    )

    recovery_column = identify_column(
        df,
        [
            "recovery",
            "normalized_recovery",
            "recovery_fraction",
        ],
        "Phase 6 recovery column",
    )

    df["head_label"] = (
        df[head_column]
        .astype(str)
        .str.strip()
    )

    df["pair_id"] = pd.to_numeric(
        df[pair_column],
        errors="coerce",
    )

    df["recovery"] = clean_numeric(
        df[recovery_column]
    )

    df = df[
        df["head_label"].isin(
            CANDIDATE_HEADS
        )
    ].copy()

    print(
        f"Candidate-head rows: {len(df)}"
    )

    return df


# ============================================================================
# LOAD PHASE 7
# ============================================================================


def load_phase7():

    print_header(
        "LOADING PHASE 7 RESULTS"
    )

    require_file(
        PHASE7_RESULTS,
        "Phase 7 multi-head results",
    )

    df = pd.read_csv(
        PHASE7_RESULTS
    )

    print(
        f"Rows: {len(df)}"
    )

    condition_column = identify_column(
        df,
        [
            "condition",
            "heads",
            "head_combination",
        ],
        "Phase 7 condition column",
    )

    pair_column = identify_column(
        df,
        [
            "pair_id",
        ],
        "Phase 7 pair column",
    )

    recovery_column = identify_column(
        df,
        [
            "recovery",
            "normalized_recovery",
            "recovery_fraction",
        ],
        "Phase 7 recovery column",
    )

    df["condition"] = (
        df[condition_column]
        .astype(str)
        .str.strip()
    )

    df["pair_id"] = pd.to_numeric(
        df[pair_column],
        errors="coerce",
    )

    df["recovery"] = clean_numeric(
        df[recovery_column]
    )

    df = df[
        df["condition"]
        == BEST_MULTI_CONDITION
    ].copy()

    print(
        f"Best-condition rows: {len(df)}"
    )

    return df


# ============================================================================
# CREATE PAIRED DATA
# ============================================================================


def create_paired_data(
    phase6,
    phase7,
):

    print_header(
        "CREATING PAIRED ROBUSTNESS DATA"
    )

    single = phase6[
        phase6["head_label"]
        == BEST_SINGLE_HEAD
    ][
        [
            "pair_id",
            "recovery",
        ]
    ].copy()

    single = single.rename(
        columns={
            "recovery":
                "single_recovery"
        }
    )

    multi = phase7[
        [
            "pair_id",
            "recovery",
        ]
    ].copy()

    multi = multi.rename(
        columns={
            "recovery":
                "multi_recovery"
        }
    )

    paired = pd.merge(
        single,
        multi,
        on="pair_id",
        how="inner",
    )

    paired[
        "improvement"
    ] = (
        paired["multi_recovery"]
        - paired["single_recovery"]
    )

    paired[
        "relative_improvement"
    ] = np.where(
        paired["single_recovery"] != 0,
        paired["improvement"]
        / paired["single_recovery"],
        np.nan,
    )

    paired = paired.sort_values(
        "pair_id"
    ).reset_index(
        drop=True
    )

    print(
        f"Paired examples: {len(paired)}"
    )

    if len(paired) != EXPECTED_PAIRS:

        print(
            "WARNING: Expected "
            f"{EXPECTED_PAIRS} pairs but found "
            f"{len(paired)}."
        )

    return paired


# ============================================================================
# PRIMARY ROBUSTNESS ANALYSIS
# ============================================================================


def analyze_primary_robustness(
    paired,
):

    print_header(
        "PRIMARY ROBUSTNESS ANALYSIS"
    )

    differences = (
        paired[
            "improvement"
        ]
        .dropna()
        .to_numpy()
    )

    single = (
        paired[
            "single_recovery"
        ]
        .dropna()
        .to_numpy()
    )

    multi = (
        paired[
            "multi_recovery"
        ]
        .dropna()
        .to_numpy()
    )

    mean_difference = mean(
        differences
    )

    median_difference = median(
        differences
    )

    std_difference = std(
        differences
    )

    positive_fraction = float(
        np.mean(
            differences > 0
        )
    )

    zero_or_positive_fraction = float(
        np.mean(
            differences >= 0
        )
    )

    d = cohens_d_paired(
        differences
    )

    wilcoxon = wilcoxon_test(
        multi,
        single,
    )

    permutation_p = (
        paired_permutation_test(
            differences,
            PERMUTATION_ITERATIONS,
            RANDOM_SEED,
        )
    )

    ci_low, ci_high, _ = (
        bootstrap_mean_ci(
            differences,
            BOOTSTRAP_ITERATIONS,
            RANDOM_SEED,
        )
    )

    print(
        f"Mean improvement: "
        f"{mean_difference:.6f}"
    )

    print(
        f"Median improvement: "
        f"{median_difference:.6f}"
    )

    print(
        f"SD improvement: "
        f"{std_difference:.6f}"
    )

    print(
        f"Bootstrap 95% CI: "
        f"[{ci_low:.6f}, {ci_high:.6f}]"
    )

    print(
        f"Positive improvement: "
        f"{positive_fraction * 100:.2f}%"
    )

    print(
        f"Paired Cohen's d: "
        f"{d:.4f}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{wilcoxon['statistic']}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{wilcoxon['p_value']}"
    )

    print(
        f"Permutation p-value: "
        f"{permutation_p:.6f}"
    )

    return {
        "n_pairs": len(differences),
        "mean_single_recovery": mean(single),
        "mean_multi_recovery": mean(multi),
        "mean_improvement": mean_difference,
        "median_improvement": median_difference,
        "std_improvement": std_difference,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "positive_improvement_fraction": positive_fraction,
        "nonnegative_improvement_fraction": (
            zero_or_positive_fraction
        ),
        "paired_cohens_d": d,
        "wilcoxon_statistic": (
            wilcoxon["statistic"]
        ),
        "wilcoxon_p_value": (
            wilcoxon["p_value"]
        ),
        "permutation_p_value": permutation_p,
    }


# ============================================================================
# BOOTSTRAP ROBUSTNESS
# ============================================================================


def bootstrap_robustness(
    paired,
):

    print_header(
        "BOOTSTRAP ROBUSTNESS"
    )

    differences = (
        paired[
            "improvement"
        ]
        .dropna()
        .to_numpy()
    )

    lower, upper, bootstrap_values = (
        bootstrap_mean_ci(
            differences,
            BOOTSTRAP_ITERATIONS,
            RANDOM_SEED,
        )
    )

    probability_positive = float(
        np.mean(
            bootstrap_values > 0
        )
    )

    print(
        f"Bootstrap iterations: "
        f"{BOOTSTRAP_ITERATIONS}"
    )

    print(
        f"Bootstrap mean: "
        f"{mean(bootstrap_values):.6f}"
    )

    print(
        f"95% CI: "
        f"[{lower:.6f}, {upper:.6f}]"
    )

    print(
        f"P(bootstrap improvement > 0): "
        f"{probability_positive:.6f}"
    )

    return {
        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,
        "bootstrap_mean":
            mean(bootstrap_values),
        "ci95_low":
            lower,
        "ci95_high":
            upper,
        "probability_improvement_positive":
            probability_positive,
    }, bootstrap_values


# ============================================================================
# LEAVE-ONE-OUT ANALYSIS
# ============================================================================


def leave_one_out_analysis(
    paired,
):

    print_header(
        "LEAVE-ONE-OUT ROBUSTNESS"
    )

    rows = []

    for excluded_pair in paired[
        "pair_id"
    ]:

        subset = paired[
            paired[
                "pair_id"
            ]
            != excluded_pair
        ]

        values = (
            subset[
                "improvement"
            ]
            .dropna()
            .to_numpy()
        )

        rows.append(
            {
                "excluded_pair":
                    int(excluded_pair),
                "n_pairs":
                    len(values),
                "mean_improvement":
                    mean(values),
                "median_improvement":
                    median(values),
                "positive_fraction":
                    float(
                        np.mean(
                            values > 0
                        )
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    min_mean = (
        result[
            "mean_improvement"
        ].min()
    )

    max_mean = (
        result[
            "mean_improvement"
        ].max()
    )

    min_positive = (
        result[
            "positive_fraction"
        ].min()
    )

    print(
        f"Minimum leave-one-out mean: "
        f"{min_mean:.6f}"
    )

    print(
        f"Maximum leave-one-out mean: "
        f"{max_mean:.6f}"
    )

    print(
        f"Minimum positive fraction: "
        f"{min_positive * 100:.2f}%"
    )

    return result


# ============================================================================
# PER-PAIR CONSISTENCY
# ============================================================================


def per_pair_consistency(
    paired,
):

    print_header(
        "PER-PAIR CONSISTENCY"
    )

    result = paired.copy()

    result[
        "multi_better"
    ] = (
        result[
            "improvement"
        ] > 0
    )

    result[
        "multi_equal_or_better"
    ] = (
        result[
            "improvement"
        ] >= 0
    )

    print(
        f"Pairs where multi-head > "
        f"single-head: "
        f"{result['multi_better'].sum()}/"
        f"{len(result)}"
    )

    print(
        f"Pairs where multi-head >= "
        f"single-head: "
        f"{result['multi_equal_or_better'].sum()}/"
        f"{len(result)}"
    )

    print()
    print(
        "Smallest improvements:"
    )

    print(
        result.sort_values(
            "improvement"
        )[
            [
                "pair_id",
                "single_recovery",
                "multi_recovery",
                "improvement",
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Largest improvements:"
    )

    print(
        result.sort_values(
            "improvement",
            ascending=False,
        )[
            [
                "pair_id",
                "single_recovery",
                "multi_recovery",
                "improvement",
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    return result


# ============================================================================
# ROBUSTNESS DECISION
# ============================================================================


def make_robustness_decision(
    primary,
    loo,
):

    print_header(
        "ROBUSTNESS DECISION"
    )

    mean_improvement = (
        primary[
            "mean_improvement"
        ]
    )

    ci_low = (
        primary[
            "bootstrap_ci95_low"
        ]
    )

    ci_high = (
        primary[
            "bootstrap_ci95_high"
        ]
    )

    positive_fraction = (
        primary[
            "positive_improvement_fraction"
        ]
    )

    permutation_p = (
        primary[
            "permutation_p_value"
        ]
    )

    wilcoxon_p = (
        primary[
            "wilcoxon_p_value"
        ]
    )

    min_loo = (
        loo[
            "mean_improvement"
        ].min()
    )

    criteria = {
        "mean_improvement_positive":
            mean_improvement > 0,

        "bootstrap_ci_excludes_zero":
            ci_low > 0,

        "positive_on_majority_pairs":
            positive_fraction >= 0.5,

        "permutation_significant":
            (
                math.isfinite(
                    permutation_p
                )
                and permutation_p < 0.05
            ),

        "wilcoxon_significant":
            (
                math.isfinite(
                    wilcoxon_p
                )
                and wilcoxon_p < 0.05
            ),

        "leave_one_out_mean_positive":
            min_loo > 0,
    }

    for key, value in criteria.items():

        print(
            f"{key}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    strong_robustness = (
        criteria[
            "mean_improvement_positive"
        ]
        and criteria[
            "bootstrap_ci_excludes_zero"
        ]
        and criteria[
            "positive_on_majority_pairs"
        ]
        and criteria[
            "leave_one_out_mean_positive"
        ]
    )

    statistically_supported = (
        criteria[
            "permutation_significant"
        ]
        or criteria[
            "wilcoxon_significant"
        ]
    )

    if (
        strong_robustness
        and statistically_supported
    ):

        decision = (
            "ROBUST_SUPPORT"
        )

    elif strong_robustness:

        decision = (
            "CONSISTENT_EFFECT"
        )

    else:

        decision = (
            "LIMITED_ROBUSTNESS"
        )

    print()
    print(
        f"Overall decision: "
        f"{decision}"
    )

    return {
        "decision": decision,
        "criteria": criteria,
        "minimum_leave_one_out_mean":
            float(min_loo),
    }


# ============================================================================
# PLOTS
# ============================================================================


def create_plots(
    paired,
    bootstrap_values,
    loo,
):

    print_header(
        "CREATING PHASE 10 PLOTS"
    )

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib unavailable. "
            "Plots skipped."
        )

        return

    # ------------------------------------------------------------------------
    # Plot 1 — Per-pair improvement
    # ------------------------------------------------------------------------

    ordered = paired.sort_values(
        "pair_id"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.bar(
        np.arange(
            len(ordered)
        ),
        ordered[
            "improvement"
        ] * 100,
    )

    plt.xlabel(
        "Pair"
    )

    plt.ylabel(
        "Multi-head improvement (percentage points)"
    )

    plt.title(
        "Per-Pair Multi-Head Improvement"
    )

    plt.xticks(
        np.arange(
            len(ordered)
        ),
        ordered[
            "pair_id"
        ].astype(int),
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "per_pair_multi_head_improvement.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(
        path
    )

    # ------------------------------------------------------------------------
    # Plot 2 — Bootstrap distribution
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        bootstrap_values * 100,
        bins=50,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Bootstrap mean improvement (percentage points)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Bootstrap Distribution of Multi-Head Improvement"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "bootstrap_improvement_distribution.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(
        path
    )

    # ------------------------------------------------------------------------
    # Plot 3 — Leave-one-out
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(11, 5)
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.plot(
        loo[
            "excluded_pair"
        ],
        loo[
            "mean_improvement"
        ] * 100,
        marker="o",
    )

    plt.xlabel(
        "Excluded pair"
    )

    plt.ylabel(
        "Mean improvement (%)"
    )

    plt.title(
        "Leave-One-Out Robustness"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "leave_one_out_robustness.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(
        path
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 70)
    print(
        "PHASE 10 — ROBUSTNESS & STATISTICAL SIGNIFICANCE"
    )
    print("=" * 70)

    print()
    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Bootstrap iterations: "
        f"{BOOTSTRAP_ITERATIONS}"
    )

    print(
        f"Permutation iterations: "
        f"{PERMUTATION_ITERATIONS}"
    )

    # ------------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    phase6 = load_phase6()

    phase7 = load_phase7()

    # ------------------------------------------------------------------------
    # Create paired data
    # ------------------------------------------------------------------------

    paired = create_paired_data(
        phase6,
        phase7,
    )

    # ------------------------------------------------------------------------
    # Primary analysis
    # ------------------------------------------------------------------------

    primary = analyze_primary_robustness(
        paired
    )

    # ------------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------------

    bootstrap_summary, bootstrap_values = (
        bootstrap_robustness(
            paired
        )
    )

    # ------------------------------------------------------------------------
    # Leave-one-out
    # ------------------------------------------------------------------------

    loo = leave_one_out_analysis(
        paired
    )

    # ------------------------------------------------------------------------
    # Per-pair consistency
    # ------------------------------------------------------------------------

    consistency = per_pair_consistency(
        paired
    )

    # ------------------------------------------------------------------------
    # Robustness decision
    # ------------------------------------------------------------------------

    decision = make_robustness_decision(
        primary,
        loo,
    )

    decision = {
        "decision": str(
            decision["decision"]
        ),
        "criteria": {
            str(key): bool(value)
            for key, value
            in decision["criteria"].items()
        },
        "minimum_leave_one_out_mean": float(
            decision[
                "minimum_leave_one_out_mean"
            ]
        ),
    }

    # ------------------------------------------------------------------------
    # Save paired data
    # ------------------------------------------------------------------------

    paired_path = (
        OUTPUT_DIR
        / "paired_multi_vs_single.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save primary statistics
    # ------------------------------------------------------------------------

    primary_path = (
        OUTPUT_DIR
        / "primary_robustness_statistics.csv"
    )

    pd.DataFrame(
        [primary]
    ).to_csv(
        primary_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save bootstrap summary
    # ------------------------------------------------------------------------

    bootstrap_path = (
        OUTPUT_DIR
        / "bootstrap_statistics.csv"
    )

    pd.DataFrame(
        [bootstrap_summary]
    ).to_csv(
        bootstrap_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save leave-one-out
    # ------------------------------------------------------------------------

    loo_path = (
        OUTPUT_DIR
        / "leave_one_out_analysis.csv"
    )

    loo.to_csv(
        loo_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save per-pair consistency
    # ------------------------------------------------------------------------

    consistency_path = (
        OUTPUT_DIR
        / "per_pair_consistency.csv"
    )

    consistency.to_csv(
        consistency_path,
        index=False,
    )

        # ------------------------------------------------------------------------
    # Save decision
    # ------------------------------------------------------------------------

    decision_path = (
        OUTPUT_DIR
        / "robustness_decision.json"
    )

    # Convert NumPy scalar values to native
    # Python values before JSON serialization.
    decision_json = {
        "decision": str(
            decision["decision"]
        ),
        "criteria": {
            str(key): bool(value)
            for key, value
            in decision["criteria"].items()
        },
        "minimum_leave_one_out_mean": float(
            decision[
                "minimum_leave_one_out_mean"
            ]
        ),
    }

    with decision_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            decision_json,
            file,
            indent=2,
        )

    # Create plots

    create_plots(
        paired,
        bootstrap_values,
        loo,
    )

    # ------------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------------

    report = {
        "phase": 10,
        "name": (
            "robustness_and_statistical_significance"
        ),
        "random_seed": RANDOM_SEED,
        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,
        "permutation_iterations":
            PERMUTATION_ITERATIONS,
        "candidate_heads":
            CANDIDATE_HEADS,
        "best_single_head":
            BEST_SINGLE_HEAD,
        "best_multi_condition":
            BEST_MULTI_CONDITION,
        "expected_pairs":
            EXPECTED_PAIRS,
        "observed_pairs":
            len(paired),
        "primary_statistics":
            primary,
        "bootstrap_statistics":
            bootstrap_summary,
        "robustness_decision":
            decision,
    }

    report_path = (
        OUTPUT_DIR
        / "phase10_robustness_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------------

    print_header(
        "PHASE 10 COMPLETE"
    )

    print(
        "Robustness analysis completed."
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Generated:"
    )

    print(
        f"  {paired_path}"
    )

    print(
        f"  {primary_path}"
    )

    print(
        f"  {bootstrap_path}"
    )

    print(
        f"  {loo_path}"
    )

    print(
        f"  {consistency_path}"
    )

    print(
        f"  {decision_path}"
    )

    print(
        f"  {report_path}"
    )

    print()
    print(
        "No model weights were modified."
    )

    print(
        "No new activation patching was performed."
    )

    print(
        "Only existing Phase 6 and Phase 7 results "
        "were statistically analyzed."
    )


if __name__ == "__main__":
    main()