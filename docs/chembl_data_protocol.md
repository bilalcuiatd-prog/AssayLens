# AssayLens AI: real ChEMBL enzyme bioactivity extraction protocol

## Objective and scope

Create an auditable, single-target, single-endpoint experimental bioactivity dataset for AssayLens's **regression and assay-heterogeneity evaluation**, without treating censored records as exact or mixing different endpoint definitions.

**Recommended pilot:** Human carbonic anhydrase II (CA2; `CHEMBL205`), endpoint `Ki` in nM, transformed to `pKi = 9 − log10(Ki in nM)`. This enzyme/target is an example, *not* an assertion that measurements from all assays are directly comparable. Do not mix Ki with IC50 in one model.

Authoritative sources:
- Official ChEMBL web services: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
- Bioactivity standardization, pChEMBL, warning/duplicate flags: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions
- ChEMBL API and interactive schema: https://www.ebi.ac.uk/chembl/api/data/docs
- ChEMBL public database and release information: https://www.ebi.ac.uk/chembl/

## Windows instructions (from the extracted AssayLens_AI folder)

Open File Explorer inside the `AssayLens_AI` project folder, type `cmd` into the folder address bar, press Enter, then run:

```bat
py -3 -m pip install -r requirements.txt
py -3 scripts\download_chembl_enzyme.py --target CHEMBL205 --endpoint Ki
```

Alternatively double-click `download_chembl_windows.bat` if the project dependencies have already been installed. It uses `.venv\Scripts\python.exe` if available.

The ChEMBL downloader requires Internet access. Its runtime is controlled by the number of matching experimental records and assay groups; ChEMBL may rate-limit requests. It is designed for one dataset at a time, not for high-rate simultaneous downloads.

**Other examples (separate datasets, never pooled):**

```bat
py -3 scripts\download_chembl_enzyme.py --target CHEMBL205 --endpoint IC50 --outdir datasets\CHEMBL205_IC50
```

For a different enzyme, first verify the ChEMBL target record is a `SINGLE PROTEIN` belonging to the specified organism and find a suitable common endpoint. Then replace the target ID. The downloader will refuse mismatched target types and organisms.

## Output files

`datasets/CHEMBL205_Ki/` contains:

- `raw_activities.jsonl.gz`: all retrieved raw ChEMBL activity JSON records, without deleting records that fail filtering.
- `assays_metadata.jsonl.gz`: associated assay metadata, including target confidence and assay context.
- `excluded_rows.csv`: rejected activity IDs and **first** rejection reason for each.
- `cleaned_full.csv`: all records passing strict a-priori QC, with extra provenance fields.
- `model_eligible_full.csv`: subset of `cleaned_full` belonging to assays with at least ten unique input SMILES, default.
- **`assaylens_import.csv`: exact ten-column dataset to upload via the AssayLens web dashboard. Maximum 8,000 rows.**
- `assaylens_audit_summary.json`: a dry-run application audit report using the project's actual quality pipeline.
- `manifest.json`: extraction parameters, ChEMBL status/version, record counts, filter reasons, SHA-256 hashes, and reproducibility caveats.

## Exact import schema

```
smiles,response,assay_id,assay_type,assay_format,source_id,endpoint,target_id,standard_relation,data_validity_comment
```

- `smiles`: `canonical_smiles` from the ChEMBL activity record.
- `response`: numeric `pKi` (or pIC50/pKd for a *separate* endpoint-specific dataset), recomputed from `standard_value` in nanomolar units.
- `assay_id`: activity `assay_chembl_id` (needed for assay-group evaluation).
- `assay_type`: joined ChEMBL assay `assay_type`, permitted B or F.
- `assay_format`: joined assay `bao_label`, with format code fallback, or `UNKNOWN`.
- `source_id`: activity `document_chembl_id` (publication provenance, not source of experimental independence).
- `endpoint`: one endpoint (`Ki` by default).
- `target_id`: one target (`CHEMBL205` by default).
- `standard_relation`: `=`, exact measurement only.
- `data_validity_comment`: empty because the strict extractor excludes *any* nonempty comment.

## Filtering (fixed order; the first applicable exclusion is logged)

1. Same target ID, same target organism, same assay endpoint.
2. `standard_relation` is `=` or `==` (no `>`, `<`, `>=`, or approximate measurements).
3. Standard units must be `nM`; value must be finite and positive.
4. `data_validity_comment` must be empty. Note: **'Manually validated' also gets excluded**, due to the downstream AssayLens curator's current handling of any nonempty comment. This is conservative and **does not mean** manual validation implies bad data.
5. ChEMBL `potential_duplicate=0`, `standard_flag=1`; unknown or missing values do not silently pass.
6. Joined assay metadata must exist. Require `confidence_score=9`, `relationship_type=D`, target ID agreement, `assay_type` in `{B, F}`.
7. Original publication document ID and parseable RDKit SMILES must exist.
8. If a ChEMBL `pchembl_value` is present, computed log response must agree to within 0.15; numerical response is computed from the source standard value.
9. Only assay groups with at least 10 distinct input SMILES are retained for the default model-eligible cohort. This is a **selection rule**; vary and report it in sensitivity analyses.
10. If the eligible dataset exceeds 8,000, a seeded uniform random sample is exported solely for the **local web app**. The unsampled eligible cohort is retained as a separate CSV.

The AssayLens application then performs its own canonicalization, parent-structure deduplication, within-assay conflict audit, and training/calibration/test group partitioning. **Do not present the extracted row count as the final number of independent model observations.** Read `assaylens_audit_summary.json` for that distinction.

## Scientific caution and publication checklist

- **Do not treat API-live reproducibility as fixed-release reproducibility.** ChEMBL Web Services update; record the DB version, extraction timestamp, and file hashes. For an archival study, acquire and preserve a fixed ChEMBL release snapshot and retain DOI/version metadata.
- No experimental records can be downloaded by this tool in an offline environment. The project contains only synthetic demonstration activity labels until you execute the downloader with Internet access.
- A strict score of 9 is about *target assignment*, **not** evidence of equal assay conditions or measurement accuracy. Assess experimental protocol consistency and assay format manually or through a documented stratification.
- The local 8,000-row subset is a software workaround, not a validated benchmark cohort. For rigorous publication, lift the cap in a secured and resourced evaluation pipeline and use the eligible **full** cohort.
- Pooling measurements originating from the same publication or replicated across different sources can produce leakage; identify and group shared compounds and documents, and consider document- and time-held-out evaluations.
- Leakage controls: fit featurizers and quality-derived *predictive* statistics on training partitions only. Keep calibration separate; avoid tuning model choices on final test performance. The existing AssayLens weighting heuristic is experimental.
- Report full exclusion counts, assay counts, target identifiers, endpoint, organism, ChEMBL release, aggregate assay variability, scaffold diversity, sensitivity to thresholds, and uncertainty interval calibration by subpopulation.
- Run the included `python -m pytest tests -q` before use and inspect every real-data report.

## Benchmarking after import

Start the local web application and upload `datasets/CHEMBL205_Ki/assaylens_import.csv` via the dashboard. For repeated holdouts from the project directory:

```bat
py -3 scripts\benchmark.py --csv datasets\CHEMBL205_Ki\assaylens_import.csv --splits scaffold assay --seeds 7 17 27 37 47 --trees 96 --output datasets\CHEMBL205_Ki\benchmark.csv
```

Assay-held-out partitioning needs at least **six nonempty assay groups**; enough observations must remain in **every** train/calibration/test split. Some seeds may be skipped for small/imbalanced groups; report this explicitly, not as a selection of favorable outcomes.
