# Phase 17 — Mechanistic Interpretation

## Experimental status

This phase interprets the frozen candidate heads discovered in the previous causal experiments. No new head selection, circuit search, activation patching, or model-weight modification was performed.

The analysis uses the 60-example discovery split only. The 40 held-out examples are not loaded.

## Frozen candidate circuit

- L10H4 + L8H9 + L9H2

## Activation-difference evidence

The largest mean clean-corrupted activation difference among the frozen candidate heads was observed for **L10H4**, with a mean difference norm of **0.244635**.

Activation difference is treated as descriptive evidence. It does not by itself establish causal importance.

## Causal evidence

Phase 13 identified **L10H4** as the strongest individual frozen candidate according to mean normalized recovery (0.1770).

Phase 14 measured mean recovery of **0.4983** for the frozen three-head circuit on the discovery split.

The causal interpretation therefore comes from activation patching, while the activation-difference analysis is used to characterize how the candidate heads respond to the clean/corrupted contrast.

## Token-position interpretation

For L10H4, the strongest average activation difference occurred at token position **15**, with a mean difference norm of **1.178268**.

Token-position effects should be interpreted carefully because clean and corrupted prompts can have different tokenization and sequence lengths. Only overlapping positions were compared.

## Candidate mechanism

The evidence is consistent with a distributed attention-head mechanism in which multiple late-layer heads contribute to the model's defined sentiment behavior.

The results do not justify the stronger claim that these heads constitute a universal 'sentiment circuit'. The appropriate claim is that the frozen heads showed causal influence under the defined GPT-2-small sentiment task.

## Held-out evidence

Phase 15 independently evaluated the frozen circuit on the 40 held-out pairs.

Phase 16 independently tested the held-out circuit-vs-single-head comparison using paired statistical analysis.

These held-out results are not used to select or reinterpret the circuit in this phase.

## Limitations

1. The task is a controlled sentiment contrast rather than a broad natural-language benchmark.
2. The candidate circuit contains only three attention heads.
3. Activation differences do not establish causality.
4. The experiments do not establish that the same heads control sentiment outside the defined task.
5. Token-position analysis is sensitive to tokenization differences.

## Conclusion

The completed experiments support a candidate distributed mechanism involving L10H4, L8H9, and L9H2 for the defined sentiment behavior. Activation analysis provides mechanistic context, while the previous activation-patching experiments provide the causal evidence.