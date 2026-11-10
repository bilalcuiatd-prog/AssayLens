# Publication and external-validation protocol (prospective study plan)

**Status:** Proposed experiments, not results. Nothing in this document supports a manuscript claim until the specified experiments are executed and data are made independently auditable.

## Research question

Does incorporating coarse experimental-assay context and prespecified, traceable quality information improve **reliability** (calibration and subgroup errors), rather than merely aggregate predictive accuracy, under compound-scaffold and assay-source distribution shifts?

## Primary hypotheses

- H1: assay-context features reduce out-of-assay held-out error relative to the identical chemistry-only estimator.
- H2: prespecified data-quality weights improve held-out error or empirical calibration over the unweighted assay-aware model.
- H3: held-out conformal coverage deteriorates in low training-similarity subgroups despite acceptable aggregate coverage.

H1 and H2 may be false; publish negative outcomes and confidence intervals rather than selecting favorable targets.

## Before training

1. Acquire a set of lawful, publishable, properly licensed **experimental assay datasets** with independently documented provenance, clearly defined targets and endpoint metadata.
2. Establish exact inclusion/exclusion criteria and assay comparability per target; distinguish censoring, relation symbols, standard units, replicate identity, measurement uncertainty and confidence score.
3. Freeze dataset hashes, curation thresholds, split seeds, model hyperparameters, primary endpoints, hypotheses, and statistical comparison method.
4. Include at least one externally obtained assay collection or a genuinely forward-in-time holdout. Do not relabel a random seed change as external validation.
5. Remove overlapping parent structures across train, calibration, and test for scaffold evaluations. Report how acyclic structures and identical parents across multiple assays were grouped.

## Necessary comparators (not implemented in v0.1)

- Established descriptor + boosted-tree baseline with tuned training-only parameters.
- Strong graph-encoder baseline using public, reproducible implementations.
- Hierarchical assay random-effects model and a context-only negative control.
- Standard (unweighted) conformal, normalized conformal, and calibrated ensemble intervals.
- An assay metadata-shuffle negative control and a no-discordance-filter ablation.

The three implemented ExtraTrees conditions are **controlled preliminary ablations**, not sufficient state-of-the-art comparators.

## Experimental factors

| Axis | Mandatory comparisons |
|---|---|
| Data curation | Raw-valid, deduplicated, discordance-filtered |
| Model input | Chemical features; chemical + assay metadata |
| Training weights | Uniform; predefined quality weighting |
| Distribution shift | Scaffold-disjoint; assay-disjoint; time-disjoint where metadata exist |
| Uncertainty | Global conformal; similarity-stratified coverage diagnostics |
| Stability | At least five fixed, prespecified group splits, plus target-level replication |

## Outcomes

Primary: held-out MAE and OOD MAE, with calibration coverage deviations at a fixed nominal interval level. Secondary: RMSE, R², interval width, change in coverage between in-domain and OOD groups, effect of curation on retained record fraction, and computational cost.

Use target-level paired comparisons and uncertainty intervals that respect repeated observations and molecular/scaffold clustering. Repeated random seeds from overlapping data are not independent biological replications.

## Interpretation

Report failures and limitations by assay type, source, scaffold novelty and endpoint. Strictly distinguish predictive performance from experimental efficacy, safety, pharmacokinetics, and clinical evidence. Reliability alone does not establish causal biological relevance.

## Requirements for a publishable software paper

- Public code, versioned dependency file, automated tests, CI, documentation, changelog and tagged source release.
- Real independent users (researchers) perform a defined workflow; document usability outcomes, error rates and reproducibility from a clean installation.
- An independent reproducer recovers every main table/figure from source hashes and exported configuration.
- Explicit license compliance, authorship disclosure, funding/conflict statements, and data access rules.
- Independent validation of assay-dependent assumptions and the proposed quality weighting; clearly state where the tool underperforms.

**Potential contribution**: An empirically validated, auditable workflow joining scientific measurement provenance and chemical-shift uncertainty diagnostics. Calling it a new algorithm or breakthrough is not warranted by the current prototype.
