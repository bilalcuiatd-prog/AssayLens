# AssayLens Unified Research Workbench 5.2

Complete local research application for bioactivity, drug–target interaction (DTI), drug–drug interaction (DDI) benchmark analysis, and protein–protein interaction (PPI) research. This release consolidates the molecular ML workbench, the task-specific public dataset hub, and HuRI/STRING protein-network tools with a provided AssayLens brand artwork.

## Windows — run the complete application

1. Extract the complete ZIP into a **new** folder. Keep all subfolders together, including `assaylens/`, `web/`, `tests/`, `notebooks/`, and `examples/`.
2. Open `AssayLens_AI` and double-click `START_HERE_WINDOWS.bat`. On first launch it creates `.venv` and installs requirements.
3. Open http://127.0.0.1:8000 in the browser. The API documentation is at http://127.0.0.1:8000/api/docs.
4. Stop the server with `Ctrl+C` in the command window.

Alternative commands from the extracted folder:

```bat
py -3 -m pip install -r requirements.txt
py -3 -m uvicorn assaylens.studio_api:app --host 127.0.0.1 --port 8000
```

Optional public dataset loader (PyTDC), for the DTI BindingDB/DAVIS/KIBA and PPI HuRI datasets:

```bat
py -3 -m pip install -r requirements-datasets.txt
```

Optional XGBoost/LightGBM/CatBoost algorithms:

```bat
py -3 -m pip install -r requirements-optional.txt
```

Optional HuRI loader uses the same PyTDC package via `requirements-optional-ppi.txt`.

Run tests:

```bat
py -3 -m pytest -q
```

## Workspaces and model presets

**Enzyme bioactivity**: ChEMBL target discovery, assay-aware curation, regression and binary classification, grouped 2–10-fold CV, multiple seeds, fingerprint/assay-context ablations. See `assaylens/advanced_experiments.py` and `assaylens/experiments.py`.

**DTI**: Multi-target datasets with SMILES, target sequence, target ID and numerical response; separate algorithm presets and warm, cold-drug, cold-target, cold-both validation. DTI algorithms include ensemble regressors/classifiers and ANN/MLP, with optional boosting algorithms. DTI datasets include BindingDB Kd, Ki, IC50 and kinase benchmark entries DAVIS and KIBA. DTI source normalization preserves original benchmark scales, which must not be conflated.

**DDI**: Research-only dataset workspace with separate tree, linear, SVM, ANN model presets and symmetric molecular pair features, network summary, cold-one/cold-both pair splits, and fold exports. TWOSIDES is multi-label and available as a raw-only source in the registry; it is **not** silently transformed into a verified binary-negative dataset. DrugBank is marked licensed-upload. No clinical drug-combination advice is provided.

**PPI**: Network and pairwise model workspace with protein sequence features, explicit homology cluster columns when available, grouped splitting and ablations. Data sources include the HuRI benchmark reference via PyTDC, STRING association network retrieval, and importable BioGRID/IntAct records. The HuRI v5 import panel supports positive-only networks or **TDC-sampled benchmark reference controls**, explicitly not independently validated negative experiments.

**Custom Python models**: Optional sandboxed `.py` or `.ipynb` estimator interface; requires Docker and `BUILD_CUSTOM_SANDBOX.bat`. Never run untrusted uploaded code directly in the API process. **Structural tools**: RDKit 3D conformers; prepared Vina docking (separate Vina/Open Babel installation, prepared structures, research-only scores). Neither system has been prospectively validated for scientific findings.

## CSV schema

Bioactivity:

```csv
smiles,response,assay_id,assay_type,assay_format,source_id,endpoint,target_id,standard_relation,data_validity_comment
```

DTI (multi-target):

```csv
smiles,target_id,target_sequence,response,endpoint,assay_id,source_id,standard_relation,data_validity_comment
```

DDI (a single prespecified labeled outcome, not general medication guidance):

```csv
drug_a,drug_b,label,drug_a_id,drug_b_id,interaction_type,source_id,evidence_type
```

PPI (amino-acid sequences for supervised training):

```csv
protein_a,protein_b,label,protein_a_id,protein_b_id,protein_cluster_a,protein_cluster_b,interaction_type,source_id,evidence_type
```

Files in `examples/` use synthetic data for software testing only. Do not include them as independent scientific evidence.

## Source map

| Files | Purpose |
|---|---|
| `assaylens/studio_api.py` | FastAPI app, dataset uploads, bioactivity and DTI experiments, report APIs |
| `assaylens/research_hub.py`, `research_hub_api.py` | Task-specific dataset registry, separate model presets, PyTDC normalizers |
| `assaylens/dti.py` | DTI features, cold-start partitions and evaluation |
| `assaylens/interactions.py`, `interaction_api.py` | Pairwise DDI/PPI network and modeling modules |
| `assaylens/ppi_huri.py`, `ppi_huri_api.py` | HuRI dataset retrieval and evidence auditing |
| `assaylens/advanced_experiments.py`, `experiments.py` | Classical ML, neural networks, ablations, validation |
| `assaylens/sources.py`, `sources_dynamic.py` | ChEMBL, BindingDB and PubChem data utilities |
| `assaylens/custom_models.py`, `docking.py` | Optional custom estimator and molecular docking integration |
| `web/suite.html`, `web/assaylens-logo-original.png` | Main self-contained frontend and original supplied logo |
| `tests/`, `examples/`, `notebooks/` | Reproducibility checks, demos and dataset extraction notebooks |

## Scientific quality and security

- Group-disjoint splits isolate exact molecular/target identities, not automatically their scaffold families or homologous protein groups. Additional cluster-aware and independent external tests are needed before publication claims.
- Unknown or missing interactions are not automatically negative observations. Assay validity, labels, endpoint units, source versions and licensing require review.
- Reported docking energies are *not* measured affinity; prediction scores are *not* clinical recommendations.
- The server has no login or public-access controls; keep it bound to `127.0.0.1` on a trusted local machine.
- Live PyTDC, ChEMBL/BindingDB/STRING retrieval and Docker/Vina execution require external dependencies and were not tested end-to-end for this integrated package. Test external sources locally and retain downloaded provenance manifests.
- Automated test count refers to tests run on the source code, not evidence of biological model superiority.

## Release verification

`python -m pytest -q` passed **82 tests** in the packaging environment, including API integration checks, split validation, PPI HuRI import, and source-normalization tests. The main FastAPI homepage, DTI/DDI API catalog, v5 dataset catalog, and the uploaded PNG logo endpoint also responded in local in-process endpoint checks. A static Chromium screenshot confirmed page layout and brand contrast. Browser-to-localhost access was blocked in this environment, so live page navigation was not verified.

MIT licensed as indicated in `LICENSE`.

## Brand and interface

The dashboard uses the supplied AssayLens logo **without any image modification**. Its original RGBA PNG is stored at `web/assaylens-logo-original.png` and is served from `/assets/assaylens-logo-original.png`. The white backing in the sidebar preserves contrast for the original dark navy wordmark and emblem. Navigation, buttons, panels and tables use a navy/teal palette, and the legacy-workbench link has been removed from the dashboard.

The existing assay-quality notes, dataset provenance, scientific limitations, and LICENSE file are retained. They should not be removed when publishing the application.

## Render and GitHub deployment

See `GITHUB_RENDER_START_HERE.md` for a complete guide. The root `render_app.py` provides password-protected access, `Dockerfile` supports Docker builds and `render.yaml` documents the Python runtime alternative.
