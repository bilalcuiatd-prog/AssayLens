# Changelog

## 5.2 — Brand integration and interface polish

- Adopted the supplied AssayLens logo as an unmodified RGBA image asset.
- Refined the dashboard palette, hierarchy, navigation states and responsive layout.
- Removed the classic-workbench link from the main interface.
- Corrected the dashboard stylesheet boundary to ensure consistent CSS parsing.
- Preserved the existing DTI, DDI, PPI, data retrieval, docking, and experiment modules.
- Retained scientific provenance and software license information.

# Changelog

## 3.0.0
- Added dedicated multi-target drug–protein dataset import and auditing.
- Added pair-disjoint warm start, cold drug, cold target and strict cold drug–target splits.
- Included protein sequence descriptors and paired feature ablations.
- Added repeated k-fold classification and regression evaluation, split-assignment exports, and tests.
- Added in-app combination of existing enzyme datasets with verified UniProt sequences.
- Preserved previous dataset library, models, custom-model sandbox and docking modules.

# Changelog

## 2.0.0

- New self-contained dark research dashboard and six functional areas.
- Arbitrary ChEMBL human single-protein target search and provenance-preserving endpoint extraction.
- Binary activity classification alongside pActivity regression.
- Wider model and ablation catalog, including optional boosting libraries.
- Repeated group-disjoint K-fold evaluation and per-fold/prediction exports.
- Disjoint regression conformal calibration, classification diagnostic plots.
- Upload and register Python or notebook models; optional Docker-isolated execution.
- RCSB structure retrieval, RDKit conformer generation and optional Vina docking.
- Previous 1.1 interface preserved at `/classic`.

See README.md for external tool requirements, limitations and security boundaries.

## 5.1 — Consolidated complete release

Combined the v5 research dataset hub and DTI/DDI/PPI model presets with the HuRI protein-network importer and professional molecular SVG logo. The complete source, tests, notebooks, Windows launcher and usage documentation now ship together. No upstream datasets are redistributed in the source archive.
