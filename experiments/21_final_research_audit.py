"""
PHASE 18 — FINAL RESEARCH AUDIT

Audits the artifacts produced by Phases 10–17.
No model loading, patching, head selection, or circuit search.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"

ROBUSTNESS = RESULTS / "robustness"
EXPANDED_MULTI = RESULTS / "expanded_multi_head"
HELDOUT = RESULTS / "heldout_generalization"
HELDOUT_STATS = RESULTS / "heldout_statistics"
MECHANISTIC = RESULTS / "mechanistic"
AUDIT = RESULTS / "final_audit"

FROZEN_CIRCUIT = {"L10H4", "L8H9", "L9H2"}
FROZEN_BASELINE = "L10H4"
DISCOVERY_N = 60
HELDOUT_N = 40


def check(name: str, status: str, detail: str = "") -> dict[str, str]:
    print(f"{status:<5} {name}" + (f" | {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": detail}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 18 — FINAL RESEARCH AUDIT")
    print("=" * 70)
    print(f"Frozen baseline: {FROZEN_BASELINE}")
    print("Frozen circuit: " + " + ".join(sorted(FROZEN_CIRCUIT)))
    print("No model loading.")
    print("No activation patching.")
    print("No head/circuit selection.")
    print("Held-out data is not used for selection.")

    checks: list[dict[str, str]] = []

    # ---------------------------------------------------------------
    # Artifact completeness
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ARTIFACT COMPLETENESS")
    print("=" * 70)

    expected = {
        "Phase 10 decision": ROBUSTNESS / "robustness_decision.json",
        "Phase 14 results": EXPANDED_MULTI / "discovery_multi_head_results.csv",
        "Phase 14 summary": EXPANDED_MULTI / "discovery_multi_head_summary.csv",
        "Phase 14 JSON": EXPANDED_MULTI / "discovery_multi_head_summary.json",
        "Phase 15 results": HELDOUT / "heldout_generalization_results.csv",
        "Phase 15 JSON": HELDOUT / "heldout_generalization_summary.json",
        "Phase 16 primary": HELDOUT_STATS / "heldout_primary_statistics.csv",
        "Phase 16 bootstrap": HELDOUT_STATS / "heldout_bootstrap_statistics.csv",
        "Phase 16 permutation": HELDOUT_STATS / "heldout_permutation_statistics.csv",
        "Phase 16 JSON": HELDOUT_STATS / "phase16_statistical_summary.json",
        "Phase 17 activation": MECHANISTIC / "candidate_head_activation_summary.csv",
        "Phase 17 token positions": MECHANISTIC / "token_position_activation.csv",
        "Phase 17 layer progression": MECHANISTIC / "layer_progression.csv",
        "Phase 17 interpretation": MECHANISTIC / "mechanistic_interpretation.md",
        "Phase 17 summary": MECHANISTIC / "mechanistic_summary.json",
        "Phase 17 correlation": MECHANISTIC / "activation_recovery_correlation.csv",
    }

    for name, path in expected.items():
        if path.exists():
            checks.append(check(name, "PASS", str(path.relative_to(PROJECT_ROOT))))
        else:
            checks.append(check(name, "FAIL", f"Missing {path}"))

    # ---------------------------------------------------------------
    # Phase 14 circuit consistency
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 14 CIRCUIT CONSISTENCY")
    print("=" * 70)

    p14 = EXPANDED_MULTI / "discovery_multi_head_summary.json"

    if p14.exists():
        data = load_json(p14)
        candidates = set(
            data.get("candidate_selection", {}).get("candidate_heads", [])
        )
        best = data.get("results", {}).get("best_condition", {})
        condition = best.get("condition", "")
        best_heads = {x.strip() for x in condition.split("+") if x.strip()}

        if candidates == FROZEN_CIRCUIT:
            checks.append(check("Phase 14 candidate heads", "PASS", str(sorted(candidates))))
        else:
            checks.append(check("Phase 14 candidate heads", "FAIL",
                                f"Expected {sorted(FROZEN_CIRCUIT)}, got {sorted(candidates)}"))

        if best_heads == FROZEN_CIRCUIT:
            checks.append(check("Phase 14 frozen circuit", "PASS", condition))
        else:
            checks.append(check("Phase 14 frozen circuit", "FAIL", condition))

        if best.get("n_pairs") == DISCOVERY_N:
            checks.append(check("Phase 14 discovery count", "PASS", str(best["n_pairs"])))
        else:
            checks.append(check("Phase 14 discovery count", "FAIL", str(best.get("n_pairs"))))

    # ---------------------------------------------------------------
    # Phase 15 held-out count and result sanity
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 15 HELD-OUT GENERALIZATION")
    print("=" * 70)

    p15_csv = HELDOUT / "heldout_generalization_results.csv"

    if p15_csv.exists():
        df = load_csv(p15_csv)

        if "pair_id" in df.columns:
            n_pairs = df["pair_id"].nunique()
        elif "n" in df.columns:
            n_pairs = int(df["n"].max())
        else:
            n_pairs = None

        if n_pairs == HELDOUT_N:
            checks.append(check("Held-out pair count", "PASS", str(n_pairs)))
        else:
            checks.append(check("Held-out pair count", "WARN",
                                f"Expected {HELDOUT_N}, inferred {n_pairs}"))

        if "circuit_minus_single" in df.columns:
            values = pd.to_numeric(df["circuit_minus_single"], errors="coerce").dropna()
            if len(values) and (values > 0).all():
                checks.append(check("Circuit beats single on held-out pairs", "PASS",
                                    f"{len(values)}/{len(values)}"))
            else:
                checks.append(check("Circuit beats single on held-out pairs", "WARN"))

    # ---------------------------------------------------------------
    # Phase 16 statistical sanity
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 16 STATISTICAL SANITY")
    print("=" * 70)

    p16 = HELDOUT_STATS / "phase16_statistical_summary.json"

    if p16.exists():
        data = load_json(p16)
        decision = data.get("decision") or data.get("overall_decision")

        if decision == "HELDOUT_STATISTICAL_SUPPORT":
            checks.append(check("Phase 16 decision", "PASS", decision))
        else:
            checks.append(check("Phase 16 decision", "WARN", str(decision)))

    # ---------------------------------------------------------------
    # Phase 17 mechanistic outputs
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 17 MECHANISTIC OUTPUTS")
    print("=" * 70)

    candidate = MECHANISTIC / "candidate_head_activation_summary.csv"

    if candidate.exists():
        df = load_csv(candidate)
        required = {
            "layer", "head", "head_label",
            "mean_difference_norm", "n_pairs"
        }
        missing = required - set(df.columns)

        if not missing:
            checks.append(check("Candidate activation schema", "PASS", f"{len(df)} rows"))
        else:
            checks.append(check("Candidate activation schema", "FAIL",
                                f"Missing {sorted(missing)}"))

        labels = set(df["head_label"].astype(str))
        if FROZEN_CIRCUIT.issubset(labels):
            checks.append(check("Frozen heads present", "PASS", str(sorted(FROZEN_CIRCUIT))))
        else:
            checks.append(check("Frozen heads present", "FAIL", str(sorted(labels))))

    interpretation = MECHANISTIC / "mechanistic_interpretation.md"

    if interpretation.exists():
        text = interpretation.read_text(encoding="utf-8")
        for phrase in ["Activation-difference evidence", "Causal evidence",
                       "Held-out evidence", "Limitations", "Conclusion"]:
            if phrase in text:
                checks.append(check(f"Interpretation section: {phrase}", "PASS"))
            else:
                checks.append(check(f"Interpretation section: {phrase}", "WARN",
                                    "Phrase not found"))

    # ---------------------------------------------------------------
    # Known correlation issue
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ACTIVATION <-> CAUSAL CORRELATION")
    print("=" * 70)

    p13 = MECHANISTIC / "phase13_candidate_head_evidence.csv"

    if p13.exists():
        df = load_csv(p13)
        if "normalized_recovery" in df.columns:
            checks.append(check("Phase 13 normalized_recovery", "PASS"))
        else:
            checks.append(check(
                "Phase 13 normalized_recovery",
                "WARN",
                "Missing; correlation cannot be computed from this artifact.",
            ))

    # ---------------------------------------------------------------
    # Split integrity
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DISCOVERY / HELD-OUT SPLIT INTEGRITY")
    print("=" * 70)

    discovery = PROJECT_ROOT / "data" / "splits" / "discovery_pairs.csv"
    heldout = PROJECT_ROOT / "data" / "splits" / "heldout_pairs.csv"

    if discovery.exists() and heldout.exists():
        d = load_csv(discovery)
        h = load_csv(heldout)

        if "pair_id" in d.columns and "pair_id" in h.columns:
            overlap = (
                set(d["pair_id"].astype(int))
                & set(h["pair_id"].astype(int))
            )
            if not overlap:
                checks.append(check(
                    "Discovery/held-out IDs disjoint",
                    "PASS",
                    f"{len(d)} / {len(h)}",
                ))
            else:
                checks.append(check(
                    "Discovery/held-out IDs disjoint",
                    "FAIL",
                    f"Overlap: {sorted(overlap)}",
                ))
        else:
            checks.append(check("Discovery/held-out IDs disjoint", "WARN",
                                "pair_id unavailable"))
    else:
        checks.append(check("Discovery/held-out split files", "WARN",
                            "One or both files missing"))

    # ---------------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL AUDIT DECISION")
    print("=" * 70)

    failures = [x for x in checks if x["status"] == "FAIL"]
    warnings = [x for x in checks if x["status"] == "WARN"]

    if failures:
        decision = "AUDIT_REQUIRES_FIXES"
    elif warnings:
        decision = "AUDIT_PASS_WITH_WARNINGS"
    else:
        decision = "AUDIT_PASS"

    print(f"PASS: {sum(x['status'] == 'PASS' for x in checks)}")
    print(f"WARN: {len(warnings)}")
    print(f"FAIL: {len(failures)}")
    print(f"Overall decision: {decision}")

    summary = {
        "phase": 18,
        "name": "final_research_audit",
        "frozen_baseline": FROZEN_BASELINE,
        "frozen_circuit": sorted(FROZEN_CIRCUIT),
        "discovery_pairs": DISCOVERY_N,
        "heldout_pairs": HELDOUT_N,
        "decision": decision,
        "pass_count": sum(x["status"] == "PASS" for x in checks),
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "checks": checks,
        "known_issue": (
            "Phase 17 reported missing normalized_recovery in the Phase 13 "
            "detailed artifact. This audit does not invent or reconstruct it."
        ),
    }

    json_path = AUDIT / "phase18_audit_summary.json"
    csv_path = AUDIT / "phase18_audit_checks.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(checks).to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("PHASE 18 COMPLETE")
    print("=" * 70)
    print(f"Summary: {json_path}")
    print(f"Checks:  {csv_path}")
    print("No model weights were modified.")
    print("No activation patching was performed.")
    print("No head/circuit selection was performed.")


if __name__ == "__main__":
    main()