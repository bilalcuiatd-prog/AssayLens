# Paired drug–target evaluation protocols

This module models a quantitative drug–protein response using molecular structures and protein sequences. It is a computational research tool. Do not treat numerical predictions as validated medicinal conclusions.

## Dataset contract

A multi-target CSV must contain the exact columns `smiles`, `target_id`, `target_sequence`, and `response`. Recommended: `endpoint`, `assay_id`, `source_id`, `standard_relation`, and `data_validity_comment`. Protein sequences must use one-letter standard amino-acid symbols. Datasets should have >50 independently informative pairs and many targets for cold-target testing. The local upload cap is 20,000 rows. One endpoint per file; **do not pool Ki, Kd, IC50 or unrelated benchmark scores** without an explicitly justified transformation.

Each record must represent a measured interaction. Unknown or untested drug–protein combinations must not automatically be labelled inactive. If a binary task is used, the active/inactive boundary is applied to a consistent pActivity response and should be selected from scientific rationale before testing. AssayLens does not generate putative negative samples.

Pairs are canonicalized and reconciled before partitioning. Duplicate measurements within a pair are aggregated by the median, while pairs whose observed responses span at least 1.0 on the provided response scale are excluded. The exclusion file is exported with the dataset. For important studies, harmonize assay platforms, target construct information and protocols further before modeling.

## Evaluation settings

Let D and T denote unique canonical drug structures and exact protein sequences. E denotes the set of observed pairs.

| Protocol | Train/test relationship | Key caveat |
|---|---|---|
| Warm start | Train and test have disjoint *pairs*, but each test drug and protein occurs in the training set. | Test pairs with missing training-side entity coverage are excluded and counted. |
| Cold drug | No molecular identity is shared between training and testing. | Highly similar scaffolds can remain in the other folds. |
| Cold target | No exact protein sequence is shared between training and testing. | Homologous sequences can still be present across folds. |
| Cold drug–target | Both molecular identities and protein sequences in the test set are absent from training. | Data from the cross-block quadrants are excluded for each fold and explicitly counted. |

The cold-both procedure partitions drugs and proteins independently into `k` groups. The held-out diagonal pair-block is used for testing, and train rows are restricted to the remaining drug and target groups. The off-diagonal cross-block observations are excluded from that fold, rather than reassigned to a side where they could compromise entity isolation. This protocol has reduced test coverage and cannot be compared to a full-data k-fold run without recording coverage.

The split-assignment CSV contains training, test, and excluded row labels for every fold and seed. The runtime checks drug and protein overlap before fitting. Grouped folds are repeated with 1–5 configured seeds. These repeats are correlated evaluations, not independent biological replicates.

## Deterministic molecular and protein features

The molecular representation contains Morgan fingerprints and six simple physicochemical descriptors. The protein representation contains amino-acid composition, hashed dipeptide frequencies, sequence length and coarse physicochemical composition. Simple deterministic cross-products of selected molecular and protein features form interaction features.

Ablations: `joint` (all components), `no_cross_features` (molecule plus protein), `drug_only`, and `target_only`. Drug-only and target-only baselines are deliberately weak controls for some generalization settings. These descriptors are computational baselines; there are no pretrained protein language models or graph-transformer embeddings in v3.0.

All supervised training is isolated to train rows for each fold. Scaling contained in scikit-learn estimators is fitted within each training partition; no supervised feature selection is performed over the full dataset.

## Models, metrics, and reporting

The existing v2 algorithm library is reused for DTI regression and binary classification. Models include tree ensembles, linear models, SVR/SVM and ANN/MLP; optional XGBoost, LightGBM, and CatBoost require separate installation.

Regression metrics: MAE, RMSE, and R². Classification metrics: accuracy, balanced accuracy, precision, recall, F1, MCC, AUROC, and average precision where both classes are present. Probability estimates, if provided, are not automatically calibrated, and their reported discrimination should not be confused with calibrated confidence.

All hyperparameters are fixed before each repeat. The application does not provide nested hyperparameter tuning, prospective testing, external multi-database verification, or confidence bounds representing independent biological replication. Paper-level studies should add these methods and compare against established DTI baselines and externally selected holdout benchmarks.

Each run exports `fold_metrics.csv`, `per_seed_metrics.csv`, `summary.csv`, `test_predictions.csv`, `split_assignments.csv`, `manifest.json`, `dataset_pairs.csv`, `dataset_audit.json`, and `metrics.json` in a research ZIP. The manifest identifies truncated or excluded observations and critical interpretive limitations.

## Preparing real data

You can upload a ready-to-use multi-target CSV, or combine 2–16 previously imported **single-target** datasets from the dashboard. The merge option only works for matching endpoints and uses known catalog UniProt accessions to retrieve target protein sequences. Internet access is required for sequence retrieval; no target sequences are inferred from target names. For unsupported targets, provide measured paired data and sequences in the CSV or use the API's explicit sequence mapping.

The included `examples/synthetic_multi_target_dti.csv` is an artificial integrity-test fixture. It contains no real experimental outcomes and cannot support biological or publication conclusions.

## Scientific background

- [Therapeutics Data Commons: cold-start dataset splits](https://tdcommons.ai/functions/data_split/)
- [DTIAM: warm, drug-cold, and target-cold benchmark protocols](https://www.nature.com/articles/s41467-025-57828-0)
- [PCIM-DTA: explicit pair, cold-drug, cold-target, and cold-both considerations](https://academic.oup.com/bioinformatics/article/42/9/btag646/8776791)
