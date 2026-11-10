# AssayLens scientific study protocol

## Study objective

Evaluate whether assay-context features and within-training-fold quality weighting improve the **reproducibility, calibration, and out-of-distribution robustness** of supervised molecular regression compared with chemistry-only baselines. These are hypotheses to be tested, not established improvements.

## Data sources

1. Retrieve target-specific, endpoint-specific ChEMBL data with verified UniProt accessions. Preserve database status/version, retrieval UTC time, raw records, assay metadata and SHA-256 hashes.
2. Use BindingDB only after independent target linkage, assay/measurement definition review, and source-overlap analysis. The compact REST affinity endpoint is cutoff-selected and lacks assay IDs; it must not be treated as an unbiased external benchmark or used for assay-held-out evaluation.
3. Treat PubChem BioAssay exports as raw deposited experimental records. Identify AID-specific readouts, units, measurement conditions, binary/quantitative outcomes and source licensing before conversion. Do not re-label binary screening results as exact quantitative responses.
4. Retain full filtered cohorts, not only the convenience ≤8,000-row UI imports. Report the exact criterion and sampling scheme for every benchmark cohort.

## Measurement integrity

- Enforce one exact target and one endpoint per training cohort. Do not pool Ki, Kd and IC50 as equivalent pharmacological response variables.
- Convert exact concentration measurements measured in nM to negative-log molar activity using `9 - log10(x_nM)`. Reject inequalities and nonpositive/missing measurements from ordinary regression.
- Inspect replicate spread, duplicate publication records, structural standardization, salts and fragments. Retain row-level exclusions.
- Predefine thresholds (e.g. within-assay log-unit spread) using assay reproducibility rather than selecting a threshold after observing held-out metrics.

## Predefined comparisons

- Baselines: Ridge, SVR, Random Forest, Extra Trees, histogram gradient boosting, ANN/MLP.
- Predeclared features: fingerprint only; physicochemical descriptors only; combined chemistry; chemistry + assay-format metadata; chemistry + assay-format metadata with train-only record weighting.
- Specify the exact model family, software versions, selected hyperparameters and random seeds before inspecting outer-test results.

## Evaluation design

- Default: 5-fold grouped cross-validation repeated for seeds 17, 29, 42.
- Study separate group definitions: molecular parent, Bemis–Murcko scaffold, and experimental assay ID. Assay-ID splits may contain the same molecular structure across distinct assay groups, so report molecular overlap.
- Use disjoint training, calibration and held-out test groups in every fold. Fit preprocessing and categorical encoders on training data only.
- Report per-fold test MAE, RMSE, R², calibration coverage and interval width. Aggregate fold outcomes within seed first, then report descriptive mean and standard deviation across seed means.
- Avoid treating all repeated-fold outcomes as statistically independent observations. Do not select the best model based on its held-out test metric then reuse those metrics as unbiased performance claims.
- For strong claims, conduct nested grouped hyperparameter tuning, preregistered external validation, uncertainty quantification accounting for correlated folds, and target-specific ablation replication. The current workbench does **not** implement nested hyperparameter search.

## Recommended scientific outputs

A dataset flow diagram; full exclusion table; molecular scaffold diversity report; distribution shift characterization; per-target assay coverage; model calibration diagnostics; per-fold residual distributions; paired per-target comparison tables; and complete reproducible code/data release subject to source licenses.

## Publication readiness threshold

A package passing software integration tests establishes engineering functionality, not biomedical discovery or biological generalization. Publication evidence requires appropriately sized real cohorts, credible assay provenance, independent evaluation, stability under source shift, rigorous ablation controls, justified statistical inference, and explicit reporting of limitations and negative findings. High predicted activity alone is not a treatment recommendation.
