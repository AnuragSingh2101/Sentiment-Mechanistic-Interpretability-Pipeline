"""
PHASE 16 — HELD-OUT STATISTICAL SIGNIFICANCE

Final statistical validation of the frozen circuit on the
40 held-out examples.

Frozen circuit:
    L10H4 + L8H9 + L9H2

Baseline:
    L10H4

IMPORTANT:
    This phase does NOT:
      - select heads
      - search circuits
      - modify the model
      - perform new activation patching
      - use discovery data for statistical testing

It only analyzes the already-computed Phase 15 held-out results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    wilcoxon,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

BOOTSTRAP_ITERATIONS = 10000

PERMUTATION_ITERATIONS = 10000

EXPECTED_PAIRS = 40

ALPHA = 0.05


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "heldout_generalization"
    / "heldout_generalization_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "heldout_statistics"
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)

PRIMARY_CSV = (
    OUTPUT_DIR
    / "heldout_primary_statistics.csv"
)

BOOTSTRAP_CSV = (
    OUTPUT_DIR
    / "heldout_bootstrap_statistics.csv"
)

PERMUTATION_CSV = (
    OUTPUT_DIR
    / "heldout_permutation_statistics.csv"
)

LEAVE_ONE_OUT_CSV = (
    OUTPUT_DIR
    / "heldout_leave_one_out.csv"
)

PAIRED_CSV = (
    OUTPUT_DIR
    / "heldout_paired_statistics.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "phase16_statistical_summary.json"
)

BOOTSTRAP_FIGURE = (
    FIGURES_DIR
    / "heldout_bootstrap_distribution.png"
)

IMPROVEMENT_FIGURE = (
    FIGURES_DIR
    / "heldout_pairwise_improvement.png"
)

LOO_FIGURE = (
    FIGURES_DIR
    / "heldout_leave_one_out.png"
)


# ============================================================================
# FIXED RNG
# ============================================================================

RNG = np.random.default_rng(
    SEED
)


# ============================================================================
# UTILITIES
# ============================================================================


def section(title: str):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


def ensure_directories():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LOAD PHASE 15 RESULTS
# ============================================================================


def load_results():

    section(
        "LOADING PHASE 15 RESULTS"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Phase 15 result file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df)}"
    )

    if len(df) != EXPECTED_PAIRS:

        raise ValueError(
            f"Expected {EXPECTED_PAIRS} "
            f"held-out rows, found {len(df)}."
        )

    required_columns = {
        "pair_id",
        "single_recovery",
        "circuit_recovery",
        "circuit_minus_single",
    }

    missing = (
        required_columns
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    df = df.sort_values(
        "pair_id"
    ).reset_index(
        drop=True
    )

    if df["pair_id"].duplicated().any():

        raise ValueError(
            "Duplicate pair IDs found."
        )

    print(
        "Phase 15 result validation: PASS"
    )

    return df


# ============================================================================
# PAIRED DATA
# ============================================================================


def create_paired_data(df):

    section(
        "CREATING PAIRED STATISTICAL DATA"
    )

    paired = df[
        [
            "pair_id",
            "single_recovery",
            "circuit_recovery",
        ]
    ].copy()

    paired[
        "improvement"
    ] = (
        paired[
            "circuit_recovery"
        ]
        -
        paired[
            "single_recovery"
        ]
    )

    paired[
        "circuit_better"
    ] = (
        paired[
            "improvement"
        ] > 0
    )

    paired.to_csv(
        PAIRED_CSV,
        index=False,
    )

    print(
        f"Paired examples: "
        f"{len(paired)}"
    )

    print(
        f"Circuit > single: "
        f"{paired['circuit_better'].sum()}/"
        f"{len(paired)}"
    )

    return paired


# ============================================================================
# PRIMARY STATISTICS
# ============================================================================


def calculate_primary_statistics(
    paired
):

    section(
        "PRIMARY HELD-OUT STATISTICS"
    )

    improvement = (
        paired[
            "improvement"
        ]
        .to_numpy(
            dtype=float
        )
    )

    single = (
        paired[
            "single_recovery"
        ]
        .to_numpy(
            dtype=float
        )
    )

    circuit = (
        paired[
            "circuit_recovery"
        ]
        .to_numpy(
            dtype=float
        )
    )

    mean_improvement = float(
        np.mean(
            improvement
        )
    )

    median_improvement = float(
        np.median(
            improvement
        )
    )

    sd_improvement = float(
        np.std(
            improvement,
            ddof=1,
        )
    )

    positive_fraction = float(
        np.mean(
            improvement > 0
        )
    )

    circuit_positive_fraction = float(
        np.mean(
            circuit > 0
        )
    )

    # Paired Cohen's d.
    if sd_improvement > 0:

        cohens_d = (
            mean_improvement
            /
            sd_improvement
        )

    else:

        cohens_d = float(
            "nan"
        )

    # Wilcoxon signed-rank.
    wilcoxon_result = wilcoxon(
        circuit,
        single,
        alternative="greater",
        zero_method="wilcox",
    )

    wilcoxon_statistic = float(
        wilcoxon_result.statistic
    )

    wilcoxon_p = float(
        wilcoxon_result.pvalue
    )

    statistics = {
        "n_pairs":
            int(len(improvement)),

        "mean_single_recovery":
            float(np.mean(single)),

        "mean_circuit_recovery":
            float(np.mean(circuit)),

        "mean_improvement":
            mean_improvement,

        "median_improvement":
            median_improvement,

        "sd_improvement":
            sd_improvement,

        "positive_improvement_fraction":
            positive_fraction,

        "circuit_positive_fraction":
            circuit_positive_fraction,

        "paired_cohens_d":
            float(cohens_d),

        "wilcoxon_statistic":
            wilcoxon_statistic,

        "wilcoxon_p_value":
            wilcoxon_p,

        "alpha":
            ALPHA,
    }

    print(
        f"Mean single recovery: "
        f"{statistics['mean_single_recovery']:.6f}"
    )

    print(
        f"Mean circuit recovery: "
        f"{statistics['mean_circuit_recovery']:.6f}"
    )

    print(
        f"Mean improvement: "
        f"{mean_improvement:.6f}"
    )

    print(
        f"Median improvement: "
        f"{median_improvement:.6f}"
    )

    print(
        f"SD improvement: "
        f"{sd_improvement:.6f}"
    )

    print(
        f"Positive improvement: "
        f"{positive_fraction * 100:.2f}%"
    )

    print(
        f"Circuit positive recovery: "
        f"{circuit_positive_fraction * 100:.2f}%"
    )

    print(
        f"Paired Cohen's d: "
        f"{cohens_d:.4f}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{wilcoxon_statistic:.4f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{wilcoxon_p:.10f}"
    )

    primary_df = pd.DataFrame(
        [statistics]
    )

    primary_df.to_csv(
        PRIMARY_CSV,
        index=False,
    )

    return (
        improvement,
        statistics,
    )


# ============================================================================
# BOOTSTRAP
# ============================================================================


def bootstrap_mean(
    values,
    iterations,
):

    n = len(values)

    means = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(
        iterations
    ):

        sample = RNG.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = np.mean(
            sample
        )

    return means


def run_bootstrap(
    improvement
):

    section(
        "BOOTSTRAP ANALYSIS"
    )

    bootstrap_values = (
        bootstrap_mean(
            improvement,
            BOOTSTRAP_ITERATIONS,
        )
    )

    ci_low = float(
        np.percentile(
            bootstrap_values,
            2.5,
        )
    )

    ci_high = float(
        np.percentile(
            bootstrap_values,
            97.5,
        )
    )

    bootstrap_mean_value = float(
        np.mean(
            bootstrap_values
        )
    )

    probability_positive = float(
        np.mean(
            bootstrap_values > 0
        )
    )

    print(
        f"Iterations: "
        f"{BOOTSTRAP_ITERATIONS}"
    )

    print(
        f"Bootstrap mean: "
        f"{bootstrap_mean_value:.6f}"
    )

    print(
        f"95% CI: "
        f"[{ci_low:.6f}, {ci_high:.6f}]"
    )

    print(
        f"P(improvement > 0): "
        f"{probability_positive:.6f}"
    )

    bootstrap_df = pd.DataFrame(
        {
            "bootstrap_mean": (
                bootstrap_values
            )
        }
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_CSV,
        index=False,
    )

    return {
        "iterations":
            BOOTSTRAP_ITERATIONS,

        "mean":
            bootstrap_mean_value,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "probability_positive":
            probability_positive,
    }, bootstrap_values


# ============================================================================
# PERMUTATION TEST
# ============================================================================


def run_permutation_test(
    improvement
):

    section(
        "PAIRED PERMUTATION TEST"
    )

    observed = float(
        np.mean(
            improvement
        )
    )

    count = 0

    permutation_means = np.empty(
        PERMUTATION_ITERATIONS,
        dtype=float,
    )

    for i in range(
        PERMUTATION_ITERATIONS
    ):

        signs = RNG.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=len(
                improvement
            ),
        )

        permuted = (
            improvement
            *
            signs
        )

        value = float(
            np.mean(
                permuted
            )
        )

        permutation_means[i] = (
            value
        )

        if value >= observed:

            count += 1

    # Add-one correction.
    p_value = (
        count + 1
    ) / (
        PERMUTATION_ITERATIONS + 1
    )

    print(
        f"Observed mean improvement: "
        f"{observed:.6f}"
    )

    print(
        f"Iterations: "
        f"{PERMUTATION_ITERATIONS}"
    )

    print(
        f"Permutation p-value: "
        f"{p_value:.6f}"
    )

    permutation_df = pd.DataFrame(
        {
            "permutation_mean":
                permutation_means
        }
    )

    permutation_df.to_csv(
        PERMUTATION_CSV,
        index=False,
    )

    return {
        "iterations":
            PERMUTATION_ITERATIONS,

        "observed_mean":
            observed,

        "p_value":
            float(p_value),
    }, permutation_means


# ============================================================================
# LEAVE-ONE-OUT
# ============================================================================


def run_leave_one_out(
    improvement
):

    section(
        "LEAVE-ONE-OUT ROBUSTNESS"
    )

    n = len(
        improvement
    )

    rows = []

    for i in range(n):

        remaining = np.delete(
            improvement,
            i,
        )

        rows.append(
            {
                "excluded_index":
                    i,

                "excluded_pair_position":
                    i + 1,

                "mean_improvement":
                    float(
                        np.mean(
                            remaining
                        )
                    ),

                "positive_fraction":
                    float(
                        np.mean(
                            remaining > 0
                        )
                    ),
            }
        )

    loo = pd.DataFrame(
        rows
    )

    loo.to_csv(
        LEAVE_ONE_OUT_CSV,
        index=False,
    )

    minimum = float(
        loo[
            "mean_improvement"
        ].min()
    )

    maximum = float(
        loo[
            "mean_improvement"
        ].max()
    )

    minimum_positive = float(
        loo[
            "positive_fraction"
        ].min()
    )

    print(
        f"Minimum leave-one-out mean: "
        f"{minimum:.6f}"
    )

    print(
        f"Maximum leave-one-out mean: "
        f"{maximum:.6f}"
    )

    print(
        f"Minimum positive fraction: "
        f"{minimum_positive * 100:.2f}%"
    )

    return {
        "minimum_mean":
            minimum,

        "maximum_mean":
            maximum,

        "minimum_positive_fraction":
            minimum_positive,
    }, loo


# ============================================================================
# DECISION
# ============================================================================


def make_decision(
    primary,
    bootstrap,
    permutation,
    loo,
):

    section(
        "PHASE 16 STATISTICAL DECISION"
    )

    checks = {
        "mean_improvement_positive":
            primary[
                "mean_improvement"
            ] > 0,

        "bootstrap_ci_excludes_zero":
            (
                bootstrap[
                    "ci_low"
                ] > 0
            ),

        "bootstrap_probability_positive":
            (
                bootstrap[
                    "probability_positive"
                ] >= 0.95
            ),

        "permutation_significant":
            (
                permutation[
                    "p_value"
                ] < ALPHA
            ),

        "wilcoxon_significant":
            (
                primary[
                    "wilcoxon_p_value"
                ] < ALPHA
            ),

        "circuit_better_on_majority":
            (
                primary[
                    "positive_improvement_fraction"
                ] > 0.5
            ),

        "leave_one_out_positive":
            (
                loo[
                    "minimum_mean"
                ] > 0
            ),
    }

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    all_strong = all(
        checks.values()
    )

    if all_strong:

        decision = (
            "HELDOUT_STATISTICAL_SUPPORT"
        )

    elif (
        checks[
            "mean_improvement_positive"
        ]
        and
        checks[
            "circuit_better_on_majority"
        ]
    ):

        decision = (
            "HELDOUT_STATISTICAL_EVIDENCE"
        )

    else:

        decision = (
            "HELDOUT_STATISTICAL_SUPPORT_WEAK"
        )

    print()

    print(
        f"Overall decision: "
        f"{decision}"
    )

    return (
        decision,
        checks,
    )


# ============================================================================
# PLOTS
# ============================================================================


def create_bootstrap_plot(
    bootstrap_values,
    observed,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        bootstrap_values,
        bins=50,
    )

    plt.axvline(
        observed,
        linewidth=2,
        label="Observed mean",
    )

    plt.axvline(
        0,
        linewidth=2,
        linestyle="--",
        label="Zero",
    )

    plt.xlabel(
        "Bootstrap Mean Improvement"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Held-Out Bootstrap Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        BOOTSTRAP_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        BOOTSTRAP_FIGURE
    )


def create_improvement_plot(
    paired
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        np.arange(
            len(paired)
        ),
        paired[
            "improvement"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Held-Out Pair"
    )

    plt.ylabel(
        "Circuit − Single Recovery"
    )

    plt.title(
        "Per-Pair Held-Out Circuit Improvement"
    )

    plt.tight_layout()

    plt.savefig(
        IMPROVEMENT_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        IMPROVEMENT_FIGURE
    )


def create_loo_plot(
    loo
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        loo[
            "excluded_pair_position"
        ],
        loo[
            "mean_improvement"
        ],
        marker="o",
    )

    plt.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    plt.xlabel(
        "Excluded Pair"
    )

    plt.ylabel(
        "Mean Improvement"
    )

    plt.title(
        "Leave-One-Out Held-Out Robustness"
    )

    plt.tight_layout()

    plt.savefig(
        LOO_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        LOO_FIGURE
    )


# ============================================================================
# JSON
# ============================================================================


def save_summary(
    primary,
    bootstrap,
    permutation,
    loo,
    decision,
    checks,
):

    payload = {
        "phase": 16,

        "name":
            "heldout_statistical_significance",

        "split":
            "heldout",

        "n_pairs":
            EXPECTED_PAIRS,

        "frozen_single_head":
            "L10H4",

        "frozen_circuit":
            [
                "L10H4",
                "L8H9",
                "L9H2",
            ],

        "source":
            "Phase 15 heldout_generalization_results.csv",

        "statistical_tests": {
            "bootstrap_iterations":
                BOOTSTRAP_ITERATIONS,

            "permutation_iterations":
                PERMUTATION_ITERATIONS,

            "alpha":
                ALPHA,

            "wilcoxon_alternative":
                "greater",

            "permutation_type":
                "paired_sign_flip",
        },

        "primary":
            primary,

        "bootstrap":
            bootstrap,

        "permutation":
            permutation,

        "leave_one_out":
            loo,

        "checks":
            checks,

        "decision":
            decision,

        "data_leakage": {
            "discovery_results_used":
                False,

            "head_selection":
                False,

            "circuit_selection":
                False,

            "heldout_optimization":
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
        "PHASE 16 — HELD-OUT STATISTICAL SIGNIFICANCE"
    )

    print(
        f"Random seed: {SEED}"
    )

    print(
        f"Bootstrap iterations: "
        f"{BOOTSTRAP_ITERATIONS}"
    )

    print(
        f"Permutation iterations: "
        f"{PERMUTATION_ITERATIONS}"
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
        "Frozen circuit:"
    )

    print(
        "  L10H4 + L8H9 + L9H2"
    )

    print()

    print(
        "No new patching will be performed."
    )

    print(
        "No heads will be selected."
    )

    print(
        "No circuit optimization will be performed."
    )

    ensure_directories()

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    df = load_results()

    # ------------------------------------------------------------------------
    # Paired data
    # ------------------------------------------------------------------------

    paired = create_paired_data(
        df
    )

    # ------------------------------------------------------------------------
    # Primary
    # ------------------------------------------------------------------------

    (
        improvement,
        primary,
    ) = calculate_primary_statistics(
        paired
    )

    # ------------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------------

    (
        bootstrap,
        bootstrap_values,
    ) = run_bootstrap(
        improvement
    )

    # ------------------------------------------------------------------------
    # Permutation
    # ------------------------------------------------------------------------

    (
        permutation,
        permutation_values,
    ) = run_permutation_test(
        improvement
    )

    # ------------------------------------------------------------------------
    # Leave one out
    # ------------------------------------------------------------------------

    (
        loo,
        loo_df,
    ) = run_leave_one_out(
        improvement
    )

    # ------------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------------

    (
        decision,
        checks,
    ) = make_decision(
        primary,
        bootstrap,
        permutation,
        loo,
    )

    # ------------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------------

    section(
        "CREATING PHASE 16 FIGURES"
    )

    create_bootstrap_plot(
        bootstrap_values,
        primary[
            "mean_improvement"
        ],
    )

    create_improvement_plot(
        paired
    )

    create_loo_plot(
        loo_df
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    save_summary(
        primary,
        bootstrap,
        permutation,
        loo,
        decision,
        checks,
    )

    # ------------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------------

    section(
        "PHASE 16 COMPLETE"
    )

    print(
        "Held-out statistical significance "
        "analysis completed."
    )

    print()

    print(
        f"Pairs analyzed: "
        f"{len(paired)}"
    )

    print()

    print(
        "Generated:"
    )

    print(
        f"  {PRIMARY_CSV}"
    )

    print(
        f"  {BOOTSTRAP_CSV}"
    )

    print(
        f"  {PERMUTATION_CSV}"
    )

    print(
        f"  {LEAVE_ONE_OUT_CSV}"
    )

    print(
        f"  {PAIRED_CSV}"
    )

    print(
        f"  {SUMMARY_JSON}"
    )

    print()

    print(
        "Figures:"
    )

    print(
        f"  {BOOTSTRAP_FIGURE}"
    )

    print(
        f"  {IMPROVEMENT_FIGURE}"
    )

    print(
        f"  {LOO_FIGURE}"
    )

    print()

    print(
        "No model weights were modified."
    )

    print(
        "No new activation patching was performed."
    )

    print(
        "No heads or circuits were selected."
    )


if __name__ == "__main__":

    main()