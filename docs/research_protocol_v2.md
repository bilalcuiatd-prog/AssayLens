# AssayLens 2.0: reproducible scientific study protocol

## Data and provenance

Lock the precise data-source release and date, target ID, protein species, assay endpoint, value relation, and assay format. Retain raw payloads, rejected records, filtering rationale and the full cleaned cohort. Validate exact measurements, protonation/tautomer policies, activity unit harmonization and molecular standardization separately from learning. Do not silently pool endpoint types.

## Evaluation cohorts

For each protein–endpoint pair, publish the number of source records, excluded records by reason, curated assay–compound observations, unique parent molecules, unique scaffolds, and genuine assays. Investigate repeated measurements, same compound across documents, and highly inconsistent assays. Use a pre-specified activity threshold for binary classification. Include class prevalence and calibration quality, not only discrimination metrics.

## Pre-specified comparisons

Report chemistry-only and assay-context models under the same folds and model settings. Include fingerprint-only, descriptor-only and quality-weighted ablations. Compare with strong classical baselines, and add deep architectures only after controlling for parameter budget, training data and hyperparameter optimization.

## Validation and statistics

Use molecule- and scaffold-held-out evaluation for generalization; assay-held-out evaluation when actual assay provenance exists. Report per-fold metrics and out-of-fold predictions. Prefer external time-split or publication-disjoint validation for a high-impact study. If model tuning is used, keep it inside nested training folds. Repeated seed variability is descriptive; confidence intervals should account for shared observations and dependencies. Evaluate uncertainty under subgroup and domain shifts rather than relying solely on mean coverage.

## Classification

Record the preselected active threshold in pActivity units. Report AUROC, AUPRC, MCC, balanced accuracy, F1 and class prevalence. Present PR curves and confusion matrices. In folds with only one observed class, AUROC is undefined and should be recorded as missing, not 0.5 or 1.0.

## Regression

Report RMSE, MAE, R², prediction intervals, calibration size, interval coverage and width. Split conformal guarantees depend on exchangeability, which may fail for assay or chemical shifts. The model must not fit feature transforms on the test data; categorical assay encoders are trained on the training fold only.

## Structural evaluation

Treat PDB viewer content and RDKit-generated conformers as visualization inputs. When docking is enabled, record receptor PDB version, preparation workflow, cofactors, protonation and tautomer choices, redocking controls, grid coordinates, docking engine/version, search seeds and exhaustiveness. Vina scoring is not a measured dissociation constant. Do not report virtual docking alone as proof of a therapeutic effect.

## Product and release limits

The application is a single-user local research prototype. It does not currently support prospective wet-lab confirmation, multiclass classification, graph neural networks, automated target safety assessment, production multi-user accounts, or fully automatic protein preparation. These limitations must be explicitly disclosed in research dissemination.
