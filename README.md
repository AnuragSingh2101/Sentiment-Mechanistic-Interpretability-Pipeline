# GPT-2 Sentiment Mechanistic Interpretability Pipeline

This repository contains a complete, 18-phase scientific pipeline for mechanistically interpreting how **GPT-2-small** processes and represents sentiment. Using causal activation patching, statistical significance testing, and token-level activation-difference profiling, this project locates and characterizes the attention heads responsible for sentiment prediction.

---

## 🚀 Project Overview

Language models often show remarkable abilities in classifying sentiment, but the underlying circuits that perform this classification are typically hidden. This project focuses on a controlled sentiment prediction task:
*   **Prompt Structure**: `Review: <text>\nSentiment:`
*   **Next-Token Target**: Predict the probability of `" positive"` vs. `" negative"`.
*   **Causal Intervention**: By using **activation patching**, we copy activations from a positive ("clean") run and inject them into a negative ("corrupted") run to observe which specific attention heads causally restore the model's positive sentiment output.

Through this methodology, we discover and validate a robust **3-head attention circuit** (`L10H4 + L8H9 + L9H2`) that plays a crucial role in GPT-2's sentiment prediction pathway.

---

## 📊 Discovered Sentiment Circuit

The pipeline evaluates a candidate 3-head circuit and compares it against the best single attention head (the baseline):
*   **Baseline Head (`L10H4`)**: Layer 10, Head 4 is the strongest individual sentiment mediator, recovering **17.3%** of the sentiment signal on held-out data.
*   **Discovered Circuit (`L10H4 + L8H9 + L9H2`)**: Adding Layer 8 Head 9 and Layer 9 Head 2 to L10H4 forms a 3-head circuit that recovers **48.4%** of the sentiment signal (a **31.1% absolute improvement** over L10H4 alone).
*   **Rigor & Generalization**: The improvement is highly statistically significant ($p < 9.1 \times 10^{-13}$, Cohen's $d = 4.29$, bootstrap 95% CI $[0.288, 0.333]$) and generalizes perfectly across **100%** of the 40 held-out evaluation pairs.

Detailed analyses, tables, and visualization plots are documented in [RESEARCH.md](RESEARCH.md).

---

## ⚙️ Pipeline Architecture

The experimental progression is structured into 18 logical phases across two main pipelines:

### 1. Pilot Pipeline (Phases 2–10)
*   **Phase 2**: Model loading validation (`transformer_lens`) ([`02_validate_model.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/02_validate_model.py)).
*   **Phase 3 & 3.5**: Next-token behavior and logit-difference metric validation ([`03_validate_behavior.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/03_validate_behavior.py), [`04_validate_logit_metric.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/04_validate_logit_metric.py)).
*   **Phase 4 & 4.5**: Paired dataset creation and behavior validation ([`05_create_paired_dataset.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/05_create_paired_dataset.py), [`06_validate_paired_behavior.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/06_validate_paired_behavior.py)).
*   **Phase 5**: Activation caching on the pilot dataset ([`07_cache_activations.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/07_cache_activations.py)).
*   **Phase 6**: Single-head activation patching ([`08_single_head_patching.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/08_single_head_patching.py)).
*   **Phase 7**: Multi-head activation patching combinations ([`09_multi_head_patching.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/09_multi_head_patching.py)).
*   **Phase 8**: Pilot circuit and mechanism analysis ([`10_circuit_analysis.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/10_circuit_analysis.py)).
*   **Phase 9**: Quantitative and statistical analysis ([`11_quantitative_analysis.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/11_quantitative_analysis.py)).
*   **Phase 10**: Robustness and statistical significance analysis ([`12_robustness_analysis.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/12_robustness_analysis.py)).

### 2. Expanded Generalization Pipeline (Phases 11–18)
*   **Phase 11**: Expanded dataset creation and behavioral validation ([`13_create_expanded_dataset.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/13_create_expanded_dataset.py), [`14_validate_expanded_behavior.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/14_validate_expanded_behavior.py)).
*   **Phase 12**: Activation caching on the expanded dataset ([`15_cache_expanded_activations.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/15_cache_expanded_activations.py)).
*   **Phase 13**: Single-head patching on the 60-pair discovery split ([`16_discovery_single_head_patching.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/16_discovery_single_head_patching.py)).
*   **Phase 14**: Causal multi-head patching combinations on the discovery split ([`17_discovery_multi_head_patching.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/17_discovery_multi_head_patching.py)).
*   **Phase 15**: Generalization evaluation on the 40-pair held-out split ([`18_heldout_generalization.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/18_heldout_generalization.py)).
*   **Phase 16**: Statistical testing on the held-out split ([`19_heldout_statistical_significance.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/19_heldout_statistical_significance.py)).
*   **Phase 17**: Mechanistic token-level interpretation ([`20_mechanistic_interpretation.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/20_mechanistic_interpretation.py)).
*   **Phase 18**: Final research audit of pipeline outputs and decisions ([`21_final_research_audit.py`](file:///c:/Users/hp/OneDrive/Desktop/mechanistic-interpretability/experiments/21_final_research_audit.py)).

---

## 📁 Directory Structure

```text
mechanistic-interpretability/
├── data/
│   ├── raw/                 # Raw datasets (sentiment_pairs.csv, expanded_sentiment_pairs.csv)
│   ├── splits/              # Discovery (60 pairs) and Held-out (40 pairs) splits
│   └── activations/         # Cached activation tensors
├── experiments/             # Active Python scripts representing the pipeline phases
├── results/                 # Output CSVs, JSON reports, and generated matplotlib plots
│   ├── circuit/             # Phase 8 (Pilot Circuit) results and figures
│   ├── quantitative/        # Phase 9 (Pilot Quantitative) results and figures
│   ├── robustness/          # Phase 10 (Pilot Robustness) results and figures
│   ├── expanded_multi_head/ # Phase 14 (Discovery Multi-Head) results
│   ├── heldout_generalization/# Phase 15 (Held-out Evaluation) results
│   ├── heldout_statistics/  # Phase 16 (Statistical Significance) results and figures
│   ├── mechanistic/         # Phase 17 (Mechanistic Interpretation) reports and figures
│   └── final_audit/         # Phase 18 (Final Research Audit) summary & checklist
└── requirements.txt         # Project package dependencies
```

---

## 🛠️ How to Run

### Prerequisites
Ensure you have Python 3.9+ and install the required packages:
```bash
pip install -r requirements.txt
```
*(Dependencies: `torch`, `transformer_lens`, `pandas`, `numpy`, `matplotlib`, `scipy`)*

### Executing the Expanded Pipeline
You can run the key phases of the expanded generalization pipeline in sequence:
```bash
# 1. Generate the expanded dataset and splits
python experiments/13_create_expanded_dataset.py

# 2. Validate behavior on the expanded dataset
python experiments/14_validate_expanded_behavior.py

# 3. Cache activations
python experiments/15_cache_expanded_activations.py

# 4. Run single-head patching on the discovery split
python experiments/16_discovery_single_head_patching.py

# 5. Run multi-head patching combinations on the discovery split
python experiments/17_discovery_multi_head_patching.py

# 6. Evaluate circuit on heldout pairs
python experiments/18_heldout_generalization.py

# 7. Run statistical significance tests (generates bootstrap/permutation figures)
python experiments/19_heldout_statistical_significance.py

# 8. Run token-level mechanistic interpretation
python experiments/20_mechanistic_interpretation.py

# 9. Perform the final research audit
python experiments/21_final_research_audit.py
```
Each script will print detailed tables to the terminal and save reports/plots in the `results/` folder.
