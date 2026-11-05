# TrustMed-GNN

**Uncertainty-Aware Graph Neural Networks for Reliable Disease–Phenotype Association Prediction**

## Overview

TrustMed-GNN investigates the reliability of disease–phenotype association predictions using biomedical knowledge graphs and semantic representations.

The completed study extends a frozen MedAlign-GNN-derived prediction baseline with Monte Carlo dropout uncertainty estimation, probability calibration, and selective prediction.

The proposed uncertainty-guided adaptive graph–semantic fusion mechanism is described as future work and was not experimentally validated in the reported results.

## Dataset

The study uses biomedical knowledge graph information derived from PrimeKG and disease–phenotype annotations from HPOA.

The reconstructed experimental graph contains:

- 37,481 nodes
- 180,540 directed edges
- 20 relation types

Unobserved disease–phenotype pairs used as proxy negatives are not necessarily clinically confirmed negative associations.

## Methodology

The evaluated pipeline consists of:

1. Frozen MedAlign-GNN-derived association prediction.
2. Monte Carlo dropout using 30 stochastic prediction-head passes.
3. Uncertainty estimation using predictive entropy, least confidence, and MC-dropout standard deviation.
4. Probability calibration using temperature scaling.
5. Selective prediction using uncertainty-based retention and abstention.
6. Reproducibility and external provenance audits.

## Experimental Configuration

- Random seeds: 11, 23, 37, 53, 71
- MC-dropout passes: 30
- Evaluation: Internal validation
- Calibration partition: 400 examples per seed
- Separate within-validation evaluation partition: 600 examples per seed
- Expected calibration error: 10 bins

## Main Results

| Metric | Result |
|---|---|
| AUROC | 0.8655 |
| AUPRC | 0.8733 |
| Brier score (full validation) | 0.1549 |
| ECE (full validation) | 0.0761 |
| Predictive entropy AURC | 0.1039 |
| MC-dropout SD AURC | 0.1232 |
| Post-calibration ECE | 0.0633 |
| Post-calibration Brier score | 0.1516 |
| Retained error at full coverage | 21.37% |
| Retained error at approximately 80% coverage | 15.60% |

All reported results are based on internal validation rather than an independently approved external test set.

## Repository Contents

The main experimental scripts are:

- `trustmed_phase1_mc_dropout.py`
- `trustmed_phase2a_compare.py`
- `trustmed_phase2b_calibration_fusion.py`
- `trustmed_phase3_selective_prediction.py`
- `trustmed_phase4c_robustness.py`
- `trustmed_phase4_repro_audit.py`
- `generate_risk_coverage.py`

Supporting experimental outputs are organized in `results/`, while frozen artifact records are maintained in `manifests/`.

## Environment

The project uses Python and PyTorch-based graph learning tools.

Dependency information is maintained in:

`requirements-medalign-lock.txt`

The exact commands, required data preparation steps, and checkpoint dependencies should be verified against the scripts before claiming end-to-end reproducibility.

## Reproducibility

The repository includes experimental configurations, validation summaries, uncertainty comparisons, calibration results, selective prediction analyses, and reproducibility audits.

Some datasets and model artifacts may require separate acquisition or reconstruction because of redistribution restrictions or file-size limitations.

## Limitations

The experiments use internal validation data, and the negative examples are sampling proxies rather than verified biological negatives.

Exact equivalence between the reconstructed graph and the original training-time graph has not been fully established.

External provenance screening identified 177 graph-covered candidate associations, but no independent external test set was approved.

The proposed adaptive graph–semantic fusion mechanism remains unvalidated.

## Data Availability

PrimeKG and HPOA data should be obtained from their respective original sources, subject to applicable licenses and access conditions.

## Citation

Citation information will be added following manuscript publication.