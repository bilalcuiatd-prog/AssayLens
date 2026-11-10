# -*- coding: utf-8 -*-
# Generated from AssayLens_4_Enzyme_ChEMBL_Datasets.ipynb
# Run with: py -3 AssayLens_4_Enzyme_ChEMBL_Datasets.py

# %%
# Run this cell once in Jupyter Notebook / JupyterLab.

# If Jupyter reports new packages installed, restart the kernel once before running subsequent cells.

# %%
from __future__ import annotations
import csv, gzip, hashlib, json, math, re, time, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from IPython.display import display

BASE = "https://www.ebi.ac.uk/chembl/api/data/"
OUT = Path("assaylens_chembl_datasets")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {"slug":"CA2_Ki", "id":"CHEMBL205", "name":"Carbonic anhydrase II", "endpoint":"Ki"},
    {"slug":"EGFR_IC50", "id":"CHEMBL203", "name":"EGFR kinase", "endpoint":"IC50"},
    {"slug":"DPP4_IC50", "id":"CHEMBL284", "name":"DPP-4", "endpoint":"IC50"},
    {"slug":"PDE5_IC50", "id":"CHEMBL1827", "name":"PDE5A", "endpoint":"IC50"},
]

# False downloads every matching activity for each target (recommended for scientific work).
# True uses only the first N raw ChEMBL records, for testing connectivity and format;
# these pilots can have severe sampling bias and are NOT publication-quality datasets.
PILOT_MODE = False
PILOT_MAX_RAW = 3000

# Existing raw files are reused to reduce public-API load.
REUSE_RAW_FILES = True
RANDOM_SEED = 42
MAX_ASSAYLENS_ROWS = 8000
MIN_UNIQUE_MOLECULES_PER_ASSAY = 10
PAGE_SIZE = 500
EXPORT_COLUMNS = [
    "smiles", "response", "assay_id", "assay_type", "assay_format",
    "source_id", "endpoint", "target_id", "standard_relation",
    "data_validity_comment"
]

print("Output folder:", OUT.resolve())
print("Python packages: pandas", pd.__version__, "RDKit", rdBase.rdkitVersion)

# %%
# HTTP session: explicit retry for transient errors and rate limits.
retry = Retry(
    total=5, connect=5, read=5,
    backoff_factor=1.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))
session.headers.update({"User-Agent":"AssayLensAI-public-bioactivity-research/1.0",
                        "Accept":"application/json"})

def get_json(resource, params=None):
    url = urljoin(BASE, resource.lstrip("/"))
    response = session.get(url, params=params, timeout=(15, 90))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected ChEMBL JSON result: " + url)
    return payload

def iter_api(resource, parameters, key, max_records=0):
    """Page through ChEMBL with a stable order; fail closed on repeated IDs."""
    offset, count = 0, 0
    seen = set()
    while True:
        data = get_json(resource, {**parameters, "limit":PAGE_SIZE, "offset":offset})
        values = data.get(key)
        if not isinstance(values, list):
            raise ValueError(f"Missing {key!r} in {resource} response")
        if not values:
            break
        for value in values:
            id_key = "activity_id" if key == "activities" else "assay_chembl_id"
            identifier = value.get(id_key)
            if identifier is not None:
                if identifier in seen:
                    raise RuntimeError(f"Duplicate API pagination identifier {identifier}")
                seen.add(identifier)
            yield value
            count += 1
            if max_records and count >= max_records:
                return
        offset += len(values)
        if not data.get("page_meta", {}).get("next"):
            return

def validate_target(target):
    data = get_json(f"target/{target['id']}.json")
    if data.get("target_type") != "SINGLE PROTEIN":
        raise ValueError(f"{target['id']} is not SINGLE PROTEIN")
    if data.get("organism") != "Homo sapiens":
        raise ValueError(f"{target['id']} target organism not human")
    print("Target verified:", target["id"], data.get("pref_name"))
    return data

def write_raw_activities(target, folder):
    destination = folder / "raw_activities.jsonl.gz"
    cache_info_path = folder / "raw_download_settings.json"
    cache_info = {
        "target_id":target["id"], "endpoint":target["endpoint"],
        "pilot_mode":PILOT_MODE, "pilot_max_raw": PILOT_MAX_RAW if PILOT_MODE else 0
    }
    if REUSE_RAW_FILES and destination.exists() and destination.stat().st_size and cache_info_path.exists():
        try:
            saved = json.loads(cache_info_path.read_text(encoding="utf-8"))
            if all(saved.get(key)==value for key,value in cache_info.items()):
                print("Reusing cached raw activities:", destination)
                return destination
        except (OSError, ValueError):
            pass
    temporary = folder / "raw_activities.partial.gz"
    n = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for record in iter_api(
            "activity.json", {"target_chembl_id":target["id"],
                              "standard_type":target["endpoint"],
                              "order_by":"activity_id"},
            "activities", max_records=PILOT_MAX_RAW if PILOT_MODE else 0
        ):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
            if n % 2500 == 0:
                print(f"  {target['slug']}: {n:,} raw activity records")
    temporary.replace(destination)
    cache_info_path.write_text(json.dumps({**cache_info,
        "n_raw_records":n, "download_utc":datetime.now(timezone.utc).isoformat()},
        indent=2), encoding="utf-8")
    print(f"  {target['slug']}: completed raw download ({n:,})")
    return destination

def read_jsonl_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

# %%
def assay_metadata(assay_ids, target_folder):
    """Retrieve full metadata in batches, with direct fallback if a batch fails."""
    result = {}
    ids = sorted(set(str(x) for x in assay_ids if x))
    total_batches = math.ceil(len(ids) / 30)
    for pos in range(0, len(ids), 30):
        batch = ids[pos:pos+30]
        try:
            records = list(iter_api(
                "assay.json",
                {"assay_chembl_id__in":",".join(batch), "order_by":"assay_chembl_id"},
                "assays"
            ))
            for item in records:
                if item.get("assay_chembl_id") in batch:
                    result[item["assay_chembl_id"]] = item
        except (requests.RequestException, ValueError, RuntimeError) as err:
            print("Assay batch failed; trying individual records:", str(err)[:180])
        # Some records might be absent from a batch; fetch them individually.
        for aid in batch:
            if aid not in result:
                try:
                    item = get_json(f"assay/{aid}.json")
                    if item.get("assay_chembl_id") == aid:
                        result[aid] = item
                except requests.RequestException:
                    pass
        if (pos // 30 + 1) % 10 == 0 or pos + 30 >= len(ids):
            print(f"  Assay batches {pos//30+1}/{total_batches}, resolved {len(result):,}/{len(ids):,}")
    path = target_folder / "assays_metadata.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for aid in sorted(result):
            handle.write(json.dumps(result[aid], ensure_ascii=False, sort_keys=True) + "\n")
    return result

def yes_flag(x):
    return str(x).strip().lower() in {"1", "1.0", "true"}

def no_flag(x):
    return str(x).strip().lower() in {"0", "0.0", "false"}

def numeric(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def assess_activity(row, assay, target):
    """First-failure exclusion rules; retain activity ID and reason in audit."""
    if row.get("target_chembl_id") != target["id"]:
        return None, "different_target"
    if row.get("target_organism") != "Homo sapiens":
        return None, "different_organism"
    if row.get("standard_type") != target["endpoint"]:
        return None, "different_endpoint"
    if row.get("standard_relation") not in {"=", "=="}:
        return None, "inexact_or_censored"
    if row.get("standard_units") != "nM":
        return None, "invalid_or_missing_standard_units"
    x = numeric(row.get("standard_value"))
    if x is None or x <= 0:
        return None, "invalid_or_nonpositive_value"
    if str(row.get("data_validity_comment") or "").strip():
        return None, "data_validity_flag"
    if not no_flag(row.get("potential_duplicate")):
        return None, "potential_duplicate_or_unknown"
    if not yes_flag(row.get("standard_flag")):
        return None, "not_standardized_or_unknown"
    if not row.get("document_chembl_id"):
        return None, "missing_publication_document"
    if not assay:
        return None, "missing_assay_metadata"
    if assay.get("target_chembl_id") != target["id"]:
        return None, "assay_target_mismatch"
    if numeric(assay.get("confidence_score")) != 9:
        return None, "assay_confidence_below_9"
    if assay.get("relationship_type") != "D":
        return None, "not_direct_target_assignment"
    if assay.get("assay_type") != "B":
        return None, "not_binding_biochemical_assay"
    smiles = row.get("canonical_smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "missing_structure"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    response = 9 - math.log10(x)
    pch = numeric(row.get("pchembl_value"))
    if pch is not None and abs(pch - response) > 0.15:
        return None, "pchembl_disagreement"
    return {
        "smiles": canonical,
        "response": round(response, 6),
        "assay_id": str(row["assay_chembl_id"]),
        "assay_type": str(assay.get("assay_type") or "UNKNOWN"),
        "assay_format": str(assay.get("bao_label") or assay.get("bao_format") or "UNKNOWN"),
        "source_id": str(row["document_chembl_id"]),
        "endpoint": target["endpoint"],
        "target_id": target["id"],
        "standard_relation": "=",
        "data_validity_comment": "",
        "activity_id": str(row.get("activity_id") or ""),
        "molecule_chembl_id": str(row.get("molecule_chembl_id") or ""),
        "original_value_nM": x,
        "assay_description": str(assay.get("description") or ""),
        "document_year": row.get("document_year"),
    }, ""

# %%
def scaffold_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return s or "ACYCLIC:" + smiles

def make_upload_subset(eligible, limit=MAX_ASSAYLENS_ROWS, seed=RANDOM_SEED):
    """Deterministic upload-sized sample, keep as many well-populated assays as possible.

    This is a UI convenience, NOT an unbiased full-cohort validation design.
    """
    if len(eligible) <= limit:
        return eligible.copy().reset_index(drop=True), "complete_eligible_set"
    # Prioritize assays with enough distinct molecules to withstand group splitting.
    counts = eligible.groupby("assay_id")["smiles"].nunique().sort_values(ascending=False)
    chosen_assays = list(counts.head(100).index)
    subset = eligible[eligible["assay_id"].isin(chosen_assays)].copy()
    if len(subset) > limit:
        # First reserve 10 records for each selected assay, then sample the remainder.
        reserved = (subset.groupby("assay_id", group_keys=False)
                    .sample(n=10, random_state=seed))
        candidates = subset.drop(index=reserved.index)
        additional = candidates.sample(n=limit-len(reserved), random_state=seed)
        subset = pd.concat([reserved, additional])
    return subset.sort_values(["assay_id", "smiles", "activity_id"]).reset_index(drop=True), "top_assays_seeded_sample"

def process_target(target):
    folder = OUT / target["slug"]
    folder.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*65)
    print("Processing", target["slug"], target["id"])
    target_metadata = validate_target(target)
    raw_path = write_raw_activities(target, folder)

    identifiers = set()
    raw_count = 0
    for record in read_jsonl_gzip(raw_path):
        raw_count += 1
        if record.get("assay_chembl_id"):
            identifiers.add(record["assay_chembl_id"])
    if not raw_count:
        raise ValueError(f"No activity records returned for {target['slug']}")
    assays = assay_metadata(identifiers, folder)

    rows, excluded, reasons = [], [], Counter()
    for r in read_jsonl_gzip(raw_path):
        record, reason = assess_activity(r, assays.get(r.get("assay_chembl_id")), target)
        if reason:
            reasons[reason] += 1
            excluded.append({"activity_id":r.get("activity_id"), "reason":reason,
                             "assay_id":r.get("assay_chembl_id")})
        else:
            rows.append(record)

    full_cols = EXPORT_COLUMNS + ["activity_id", "molecule_chembl_id", "original_value_nM",
                                   "assay_description", "document_year"]
    all_clean = pd.DataFrame(rows, columns=full_cols)
    excluded_df = pd.DataFrame(excluded, columns=["activity_id", "reason", "assay_id"])
    all_clean.to_csv(folder / "cleaned_full.csv", index=False)
    excluded_df.to_csv(folder / "excluded_rows.csv", index=False)

    if all_clean.empty:
        print("No usable measurements passed filtering; see excluded_rows.csv")
        model_ready = all_clean.copy()
    else:
        counts = all_clean.groupby("assay_id")["smiles"].nunique()
        acceptable_assays = counts[counts >= MIN_UNIQUE_MOLECULES_PER_ASSAY].index
        model_ready = all_clean.loc[all_clean["assay_id"].isin(acceptable_assays)].copy()
    model_ready.to_csv(folder / "model_eligible_full.csv", index=False)
    upload, sampling = make_upload_subset(model_ready)
    upload.reindex(columns=EXPORT_COLUMNS).to_csv(folder / "assaylens_import.csv", index=False)

    scaffolds = upload.smiles.map(scaffold_key).nunique() if len(upload) else 0
    num_assays = upload.assay_id.nunique() if len(upload) else 0
    summary = {
        "dataset":target["slug"], "target_id":target["id"], "target_name":target_metadata.get("pref_name"),
        "organism":target_metadata.get("organism"), "endpoint":target["endpoint"],
        "raw_records":raw_count, "passed_quality":len(all_clean),
        "model_eligible":len(model_ready), "upload_rows":len(upload),
        "upload_unique_assays":int(num_assays), "upload_unique_scaffolds":int(scaffolds),
        "scaffold_split_potential":bool(len(upload) >= 50 and scaffolds >= 7),
        "assay_split_potential":bool(len(upload) >= 50 and num_assays >= 7),
        "pilot":PILOT_MODE, "sampling_policy":sampling,
        "upload_limit":MAX_ASSAYLENS_ROWS,
    }
    manifest = {
        **summary,
        "download_utc":datetime.now(timezone.utc).isoformat(),
        "source":"official ChEMBL REST API", "source_base":BASE,
        "filters":{
            "organism":"Homo sapiens", "target_type":"SINGLE PROTEIN", "target_confidence":9,
            "relationship":"D", "assay_type":"B", "measurement_relation":"=", "units":"nM",
            "validity_comment":"empty", "standard_flag":True, "potential_duplicate":False,
            "publication_required":True, "min_unique_structures_per_assay":MIN_UNIQUE_MOLECULES_PER_ASSAY
        },
        "exclusions":dict(reasons),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "upload_csv_sha256":hashlib.sha256((folder/"assaylens_import.csv").read_bytes()).hexdigest(),
        "rdkit_version":rdBase.rdkitVersion,
        "limitations":[
            "8,000-row upload cap and top-assay sampling can bias scientific evaluation",
            "Different assays and target mutants can remain heterogeneous despite high-confidence target mapping",
            "ChEMBL quality flags are not a replacement for manual assay-protocol review",
            "AssayLens training/calibration/test group sizes must still be checked after application-level curation"
        ],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Kept {len(all_clean):,}/{raw_count:,} quality-filtered rows; "
          f"ready CSV {len(upload):,} rows, {num_assays} assays, {scaffolds} scaffolds")
    if not summary["assay_split_potential"]:
        print("NOTICE: Assay-held-out modeling may not be possible or statistically stable.")
    return summary

# %%
# RUN THIS CELL to download and prepare all FOUR datasets.
# Requires a working Internet connection to www.ebi.ac.uk.
# A failed target will be reported in the summary; others will still be processed.

summaries = []
errors = {}
for target in TARGETS:
    try:
        summaries.append(process_target(target))
    except Exception as error:
        errors[target["slug"]] = str(error)
        print(f"ERROR in {target['slug']}: {type(error).__name__}: {error}")

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(OUT / "dataset_summary.csv", index=False)
display(summary_df)
if errors:
    print("Failed datasets:", json.dumps(errors, indent=2))
    (OUT / "download_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

# %%
# Bundle successful upload-ready CSV datasets into ONE ZIP for easy transfer.
# This cell does NOT invent data: it only archives CSVs created by the previous cell.
zip_path = OUT / "AssayLens_4_ChEMBL_Datasets.zip"
files = [OUT / t["slug"] / "assaylens_import.csv" for t in TARGETS]
files = [f for f in files if f.is_file() and pd.read_csv(f).shape[0] > 0]
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, f"{f.parent.name}_assaylens_import.csv")
    if (OUT / "dataset_summary.csv").exists():
        zf.write(OUT / "dataset_summary.csv", "dataset_summary.csv")
print("ZIP:", zip_path.resolve())
print("Nonempty datasets included:", len(files), "/ 4")
print("All files:")
for f in files:
    print(" -", f.resolve(), "(rows:", len(pd.read_csv(f)), ")")
