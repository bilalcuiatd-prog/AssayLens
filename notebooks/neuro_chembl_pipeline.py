"""NeuroAssayLens: reproducible ChEMBL Alzheimer/Parkinson enzyme downloader.

Use inside the companion Jupyter notebook, or run as a Python script.
This is a *curated panel*, not an exhaustive list of all disease-associated enzymes.
No target is accepted without live UniProt accession verification in ChEMBL.
No experimental records are bundled; download requires internet access.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

API = "https://www.ebi.ac.uk/chembl/api/data/"
UPLOAD_COLUMNS = [
    "smiles", "response", "assay_id", "assay_type", "assay_format",
    "source_id", "endpoint", "target_id", "standard_relation", "data_validity_comment"
]
# Role describes a research relationship, NOT validated clinical efficacy.
# 'protective_or_restorative' enzymes should not be assumed to benefit from inhibition.
CATALOG = [
    ("ACHE", "P22303", "Alzheimer", "symptomatic target; cholinergic"),
    ("BCHE", "P06276", "Alzheimer", "cholinergic candidate"),
    ("BACE1", "P56817", "Alzheimer", "amyloid pathway enzyme"),
    ("BACE2", "Q9Y5Z0", "Alzheimer", "amyloid pathway paralog; context-dependent"),
    ("GSK3B", "P49841", "Alzheimer|Parkinson", "kinase; mechanistic candidate"),
    ("GSK3A", "P49840", "Alzheimer", "kinase; mechanistic candidate"),
    ("CDK5", "Q00535", "Alzheimer", "neuronal kinase; mechanistic candidate"),
    ("DYRK1A", "Q13627", "Alzheimer", "kinase; mechanistic candidate"),
    ("MAPK14", "Q16539", "Alzheimer", "stress-response kinase"),
    ("PDE4D", "Q08499", "Alzheimer", "phosphodiesterase; candidate"),
    ("PDE9A", "O76083", "Alzheimer", "phosphodiesterase; candidate"),
    ("MME", "P08473", "Alzheimer", "amyloid clearance; protective_or_restorative"),
    ("IDE", "P14735", "Alzheimer", "amyloid clearance; protective_or_restorative"),
    ("ADAM10", "O14672", "Alzheimer", "alpha-secretase; protective_or_restorative"),
    ("ADAM17", "P78536", "Alzheimer", "protease; context-dependent"),
    ("LGMN", "Q99538", "Alzheimer", "asparaginyl endopeptidase; candidate"),
    ("CASP3", "P42574", "Alzheimer", "apoptosis-associated; exploratory"),
    ("CASP6", "P55212", "Alzheimer", "apoptosis-associated; exploratory"),
    ("HDAC6", "Q9UBN7", "Alzheimer|Parkinson", "protein homeostasis; candidate"),
    ("MTOR", "P42345", "Alzheimer", "autophagy pathway kinase"),
    ("OGA", "O60502", "Alzheimer", "O-GlcNAcase; mechanistic candidate"),
    ("MAOB", "P27338", "Parkinson", "symptomatic drug target"),
    ("MAOA", "P21397", "Parkinson", "monoamine metabolism; exploratory"),
    ("COMT", "P21964", "Parkinson", "symptomatic drug target"),
    ("LRRK2", "Q5S007", "Parkinson", "genetic disease-associated kinase"),
    ("GBA1", "P04062", "Parkinson", "lysosomal enzyme; protective_or_restorative"),
    ("PINK1", "Q9BXM7", "Parkinson", "mitochondrial kinase; protective_or_restorative"),
    ("PRKN", "O60260", "Parkinson", "E3 ubiquitin ligase; protective_or_restorative"),
    ("DDC", "P20711", "Parkinson", "dopamine synthesis enzyme"),
    ("TH", "P07101", "Parkinson", "dopamine synthesis enzyme"),
    ("DBH", "P09172", "Parkinson", "catecholamine synthesis enzyme"),
    ("GCH1", "P30793", "Parkinson", "tetrahydrobiopterin synthesis enzyme"),
    ("PLA2G6", "O60733", "Parkinson", "neurodegeneration-associated phospholipase"),
    ("USP30", "Q70CQ3", "Parkinson", "mitophagy-associated deubiquitinase"),
    ("USP14", "P54578", "Parkinson", "proteasome-associated deubiquitinase"),
    ("SIRT2", "Q8IXJ6", "Parkinson", "deacetylase; exploratory"),
    ("PDE10A", "Q9Y233", "Parkinson", "phosphodiesterase; exploratory"),
    ("GBA2", "Q9HCG7", "Parkinson", "glucosylceramidase paralog"),
    ("CTSD", "P07339", "Parkinson", "lysosomal protease; exploratory"),
    ("CTSB", "P07858", "Parkinson", "lysosomal protease; exploratory"),
    ("FAAH", "O00519", "Parkinson", "lipid signaling enzyme; exploratory"),
    ("NQO1", "P15559", "Parkinson", "redox enzyme; exploratory"),
]
TARGETS = [
    dict(gene=g, accession=a, diseases=d.split("|"), research_context=r)
    for g, a, d, r in CATALOG
]
assert len({t['gene'] for t in TARGETS}) == len(TARGETS)

class Config:
    # Set RUN_MODE='all' for the entire curated panel, or 'priority' for 11 targets.
    RUN_MODE = "all"
    PRIORITY_GENES = [
        "ACHE", "BCHE", "BACE1", "GSK3B", "DYRK1A",
        "MAOB", "COMT", "LRRK2", "GBA1", "PINK1", "USP30"
    ]
    ENDPOINTS = ("IC50", "Ki")  # keep as separate datasets; not a pooled response
    OUTPUT = Path("neurodegeneration_chembl_datasets")
    PILOT_RAW_LIMIT = 0       # 0: all matching raw records; 500: connectivity pilot
    PAGE_SIZE = 500
    MAX_UPLOAD_ROWS = 8000
    MIN_UNIQUE_SMILES_PER_ASSAY = 10
    RANDOM_SEED = 42
    STRICT_BINDING_ASSAYS = True  # assay_type B + direct confidence 9
    CACHE_RAW = True
    API_DELAY_SECONDS = 0.03
    MAKE_ZIPS = True
    DB_STATUS = None

C = Config()


def make_session():
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]), respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "NeuroAssayLens-research-dataset-curation/1.0",
        "Accept": "application/json",
    })
    return s

SESSION = make_session()


def api_get(resource, params=None):
    time.sleep(C.API_DELAY_SECONDS)
    url = API + resource.lstrip("/")
    response = SESSION.get(url, params=params, timeout=(15, 90))
    response.raise_for_status()
    obj = response.json()
    if not isinstance(obj, dict):
        raise ValueError("Invalid ChEMBL API response: " + url)
    return obj


def iterate(resource, query, key, limit=0):
    """Stable pagination with checks for duplicated records and unexpected API responses."""
    seen = set()
    offset = 0
    count = 0
    while True:
        page = api_get(resource, dict(query, limit=C.PAGE_SIZE, offset=offset))
        batch = page.get(key)
        if not isinstance(batch, list):
            raise ValueError(f"ChEMBL response missing {key}: {resource}")
        if not batch:
            return
        for record in batch:
            identifier = record.get("activity_id" if key == "activities" else
                                    "assay_chembl_id" if key == "assays" else "target_chembl_id")
            if identifier is not None:
                if identifier in seen:
                    raise ValueError(f"Repeated ChEMBL record in pagination: {identifier}")
                seen.add(identifier)
            yield record
            count += 1
            if limit and count >= limit:
                return
        offset += len(batch)
        if not page.get("page_meta", {}).get("next"):
            return


def resolve_target(entry):
    """Resolve the UniProt accession; reject nonhuman, non-single-protein, ambiguous matches."""
    accession = entry["accession"]
    candidates = list(iterate("target.json", {
        "target_components__accession": accession,
        "order_by": "target_chembl_id",
    }, "targets"))
    valid = []
    for t in candidates:
        accessions = [c.get("accession") for c in t.get("target_components") or []]
        if (t.get("target_type") == "SINGLE PROTEIN" and
            t.get("organism") == "Homo sapiens" and
            accessions.count(accession) == 1 and len(accessions) == 1):
            valid.append(t)
    if len(valid) != 1:
        raise ValueError(f"Expected exactly one human SINGLE PROTEIN target "
                         f"for {entry['gene']} {accession}, found {len(valid)}; manual review needed")
    return valid[0]


def sha256(file):
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf8")
    tmp.replace(path)


def fetch_activity_file(target_id, endpoint, folder):
    gz = folder / "raw_activities.jsonl.gz"
    config = folder / "raw_fetch_config.json"
    expected = dict(target_id=target_id, endpoint=endpoint, limit=C.PILOT_RAW_LIMIT,
                    chembl_db_version=(C.DB_STATUS or {}).get("chembl_db_version"))
    if (C.CACHE_RAW and gz.exists() and gz.stat().st_size and config.exists() and
            all(json.loads(config.read_text(encoding="utf8")).get(k) == v for k, v in expected.items())):
        print("    Reusing cached raw records")
        return gz
    tmp = folder / "raw_activities.partial.gz"
    n = 0
    try:
        with gzip.open(tmp, "wt", encoding="utf8") as out:
            for rec in iterate("activity.json", {
                "target_chembl_id": target_id,
                "standard_type": endpoint,
                "order_by": "activity_id",
            }, "activities", C.PILOT_RAW_LIMIT):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 5000 == 0:
                    print("    Fetched raw measurements:", n)
        tmp.replace(gz)
        atomic_json(config, {**expected, "raw_records": n,
                             "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
    finally:
        if tmp.exists():
            tmp.unlink()
    return gz


def records(gz):
    with gzip.open(gz, "rt", encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fetch_assays(ids, folder):
    """Metadata cache, query batches and fallback individual for missing assay IDs."""
    cachepath = folder / "assay_metadata.jsonl.gz"
    cache = {}
    if C.CACHE_RAW and cachepath.exists():
        try:
            for item in records(cachepath):
                cache[item["assay_chembl_id"]] = item
        except (ValueError, OSError):
            cache = {}
    missing = sorted(set(ids) - set(cache))
    for index in range(0, len(missing), 25):
        block = missing[index:index + 25]
        try:
            result = list(iterate("assay.json", {
                "assay_chembl_id__in": ",".join(block),
                "order_by": "assay_chembl_id"
            }, "assays"))
            for x in result:
                if x.get("assay_chembl_id") in block:
                    cache[x["assay_chembl_id"]] = x
        except (requests.RequestException, ValueError) as ex:
            print("    Assay batch warning:", str(ex)[:130])
        for aid in block:
            if aid not in cache:
                try:
                    data = api_get(f"assay/{aid}.json")
                    if data.get("assay_chembl_id") == aid:
                        cache[aid] = data
                except (requests.RequestException, ValueError):
                    pass
        if index % 250 == 0:
            print("    Assay IDs resolved:", len(cache), "/", len(set(ids)))
    with gzip.open(cachepath, "wt", encoding="utf8") as f:
        for aid in sorted(cache):
            f.write(json.dumps(cache[aid], ensure_ascii=False) + "\n")
    return cache


def flag_equal(v, value):
    if isinstance(v, bool):
        return v == bool(value)
    return str(v).strip().lower() in ({"1", "1.0", "true"} if value else {"0", "0.0", "false"})


def to_positive_number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) and x > 0 else None
    except (ValueError, TypeError):
        return None


def curate_record(r, assay, target_id, endpoint):
    """Strict first-failure rules; exclusions are saved with reason, not discarded silently."""
    if r.get("target_chembl_id") != target_id:
        return None, "wrong_target"
    if r.get("target_organism") != "Homo sapiens":
        return None, "wrong_organism"
    if r.get("standard_type") != endpoint:
        return None, "wrong_endpoint"
    if r.get("standard_relation") not in ("=", "=="):
        return None, "censored_or_inexact"
    if r.get("standard_units") != "nM":
        return None, "non_nanomolar_units"
    val = to_positive_number(r.get("standard_value"))
    if val is None:
        return None, "invalid_measurement"
    if str(r.get("data_validity_comment") or "").strip():
        return None, "validity_comment"
    if not flag_equal(r.get("potential_duplicate"), False):
        return None, "possible_duplicate_or_unknown"
    if not flag_equal(r.get("standard_flag"), True):
        return None, "not_standardized_or_unknown"
    if not r.get("document_chembl_id"):
        return None, "no_document"
    if not assay:
        return None, "no_assay_metadata"
    if assay.get("target_chembl_id") != target_id:
        return None, "assay_target_mismatch"
    if int(assay.get("confidence_score") or 0) != 9:
        return None, "assay_confidence_not_9"
    if assay.get("relationship_type") != "D":
        return None, "not_direct_target_assay"
    if C.STRICT_BINDING_ASSAYS and assay.get("assay_type") != "B":
        return None, "not_binding_assay"
    smi = r.get("canonical_smiles")
    if not isinstance(smi, str) or not smi.strip():
        return None, "missing_smiles"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "bad_smiles"
    response = 9 - math.log10(val)
    try:
        value = r.get("pchembl_value")
        if value is not None and abs(float(value) - response) > 0.15:
            return None, "pchembl_mismatch"
    except (TypeError, ValueError):
        return None, "invalid_pchembl"
    return {
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "response": round(response, 6),
        "assay_id": r.get("assay_chembl_id"),
        "assay_type": str(assay.get("assay_type") or "UNKNOWN"),
        "assay_format": str(assay.get("bao_label") or assay.get("bao_format") or "UNKNOWN"),
        "source_id": r.get("document_chembl_id"),
        "endpoint": endpoint, "target_id": target_id,
        "standard_relation": "=", "data_validity_comment": "",
        "activity_id": r.get("activity_id"),
        "molecule_chembl_id": r.get("molecule_chembl_id"),
        "original_value_nM": val,
        "assay_description": assay.get("description"),
        "document_year": r.get("document_year"),
    }, None


def scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "INVALID"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "ACYCLIC:" + smi


def prepare_upload(df):
    """No synthetic assay IDs. Preserve assay diversity when sampling 8k rows."""
    if len(df) <= C.MAX_UPLOAD_ROWS:
        return df.copy(), "full_eligible"
    # Choose top assays to keep model splitting feasible. This is a BIASED UI convenience sample.
    top = df.groupby("assay_id").smiles.nunique().sort_values(ascending=False).head(100).index
    sub = df[df.assay_id.isin(top)].copy()
    if len(sub) > C.MAX_UPLOAD_ROWS:
        # Reserve 10 distinct compound-assay rows per assay, then fill deterministically.
        reserved = sub.groupby("assay_id", group_keys=False).sample(n=10, random_state=C.RANDOM_SEED)
        leftover = sub.drop(index=reserved.index)
        extra = leftover.sample(n=C.MAX_UPLOAD_ROWS - len(reserved), random_state=C.RANDOM_SEED)
        sub = pd.concat([reserved, extra])
    return sub.sort_values(["assay_id", "smiles"]).reset_index(drop=True), "biased_top_assays_seeded"


def process_dataset(entry, target, endpoint):
    gene = entry["gene"]
    target_id = target["target_chembl_id"]
    folder = C.OUTPUT / gene / endpoint
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[{gene} {endpoint}] {target_id} ({', '.join(entry['diseases'])})")
    rawfile = fetch_activity_file(target_id, endpoint, folder)
    assay_ids = set()
    raw_n = 0
    for r in records(rawfile):
        raw_n += 1
        if r.get("assay_chembl_id"):
            assay_ids.add(r["assay_chembl_id"])
    if raw_n == 0:
        raise ValueError("No matching ChEMBL activity records")
    assay_lookup = fetch_assays(assay_ids, folder)
    good = []
    rejected = []
    reasons = Counter()
    for r in records(rawfile):
        row, error = curate_record(r, assay_lookup.get(r.get("assay_chembl_id")), target_id, endpoint)
        if error:
            rejected.append({"activity_id": r.get("activity_id"), "assay_id": r.get("assay_chembl_id"), "reason": error})
            reasons[error] += 1
        else:
            good.append(row)
    extra_cols = ["activity_id", "molecule_chembl_id", "original_value_nM", "assay_description", "document_year"]
    full = pd.DataFrame(good, columns=UPLOAD_COLUMNS + extra_cols)
    pd.DataFrame(rejected, columns=["activity_id", "assay_id", "reason"]).to_csv(folder / "excluded_records.csv", index=False)
    full.to_csv(folder / "quality_filtered_full.csv", index=False)
    if full.empty:
        curated = full.copy()
        conflicts_n = 0
    else:
        # A single contradictory compound-assay group is not an independent replicate.
        spread = full.groupby(["smiles", "assay_id"]).response.agg(lambda x: x.max() - x.min())
        conflict_keys = set(spread.loc[spread >= 1.0].index)
        conflicts_n = len(conflict_keys)
        keys = list(zip(full.smiles, full.assay_id))
        conflict_mask = [pair in conflict_keys for pair in keys]
        if conflicts_n:
            full.loc[conflict_mask].to_csv(folder / "conflicting_compound_assay_records.csv", index=False)
        no_conflict = full.loc[[not x for x in conflict_mask]].copy()
        # Median within a compound + assay; the original data are preserved separately.
        curated = no_conflict.groupby(["smiles", "assay_id"], as_index=False).agg({
            "response": "median", "assay_type": "first", "assay_format": "first",
            "source_id": "first", "endpoint": "first", "target_id": "first",
            "standard_relation": "first", "data_validity_comment": "first",
        })
    if curated.empty:
        eligible = curated.copy()
    else:
        n_compounds = curated.groupby("assay_id").smiles.nunique()
        valid_assays = n_compounds[n_compounds >= C.MIN_UNIQUE_SMILES_PER_ASSAY].index
        eligible = curated[curated.assay_id.isin(valid_assays)].copy()
    eligible.to_csv(folder / "model_eligible_full.csv", index=False)
    upload, sampling_policy = prepare_upload(eligible)
    upload = upload.reindex(columns=UPLOAD_COLUMNS)
    upload.to_csv(folder / "assaylens_import.csv", index=False)
    assay_count = int(upload.assay_id.nunique()) if len(upload) else 0
    scaffold_count = int(upload.smiles.map(scaffold).nunique()) if len(upload) else 0
    stats = {
        "gene": gene, "accession": entry["accession"], "diseases": "|".join(entry["diseases"]),
        "research_context": entry["research_context"], "target_id": target_id,
        "target_name": target.get("pref_name"), "endpoint": endpoint,
        "raw_records": raw_n, "passed_quality": len(full), "conflict_groups_removed": conflicts_n,
        "eligible_rows": len(eligible), "upload_rows": len(upload),
        "unique_assays": assay_count, "unique_scaffolds": scaffold_count,
        "scaffold_split_candidate": len(upload) >= 50 and scaffold_count >= 7,
        "assay_split_candidate": len(upload) >= 50 and assay_count >= 7,
        "sampling_policy": sampling_policy, "pilot": bool(C.PILOT_RAW_LIMIT),
        "status": "ok" if len(upload) else "no_model_eligible_rows",
    }
    manifest = dict(stats,
        requested_at_utc=datetime.now(timezone.utc).isoformat(),
        database_status=C.DB_STATUS or api_get("status.json"),
        rdkit_version=rdBase.rdkitVersion,
        raw_sha256=sha256(rawfile),
        csv_sha256=sha256(folder / "assaylens_import.csv"),
        filter_configuration={
            "target": "human single protein verified by UniProt accession",
            "endpoints": "each downloaded and modeled separately",
            "standard_relation": "exact (=)", "standard_units": "nM",
            "quality_flags": "no potential duplicates or validity warnings",
            "assay_confidence": 9, "relationship": "D",
            "strict_binding_assays": C.STRICT_BINDING_ASSAYS,
            "min_unique_smiles_per_assay": C.MIN_UNIQUE_SMILES_PER_ASSAY,
            "compound_assay_conflict_cutoff_log_units": 1.0,
            "upload_row_limit": C.MAX_UPLOAD_ROWS,
        },
        scientific_notes=[
            "ChEMBL assay metadata and assay protocols remain heterogeneous even after filtering.",
            "AssayLabs web upload cap and top-assay selection may bias findings; benchmark full eligible cohort.",
            "High potency does not imply clinical benefit; some targets require restoration, not inhibition.",
            "These proteins form a research panel, not every enzyme implicated in either disease.",
            "Avoid interpreting any single-target activity result as an Alzheimer's/Parkinson's treatment.",
        ]
    )
    atomic_json(folder / "manifest.json", manifest)
    print(f"    {len(upload)} upload rows, {assay_count} assays, {scaffold_count} scaffolds; conflicts removed: {conflicts_n}")
    return stats


def pack_zip(disease, summary):
    zipfile_path = C.OUTPUT / f"{disease.replace(' ', '_')}_AssayLens_CSVs.zip"
    chosen = summary[(summary.status == "ok") & summary.diseases.str.contains(disease, regex=False)]
    with zipfile.ZipFile(zipfile_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for _, r in chosen.iterrows():
            folder = C.OUTPUT / r.gene / r.endpoint
            f = folder / "assaylens_import.csv"
            if f.exists():
                z.write(f, arcname=f"{r.gene}_{r.endpoint}_assaylens_import.csv")
            m = folder / "manifest.json"
            if m.exists():
                z.write(m, arcname=f"manifests/{r.gene}_{r.endpoint}.json")
        z.write(C.OUTPUT / "dataset_summary.csv", arcname="dataset_summary.csv")
    return zipfile_path


def run_pipeline():
    C.OUTPUT.mkdir(exist_ok=True, parents=True)
    C.DB_STATUS = api_get("status.json")
    print("ChEMBL database status:", C.DB_STATUS.get("chembl_db_version", "unknown"))
    choose = TARGETS if C.RUN_MODE == "all" else [t for t in TARGETS if t['gene'] in C.PRIORITY_GENES]
    print("Genes in selected panel:", len(choose), "| Endpoints:", C.ENDPOINTS)
    print("Output:", C.OUTPUT.resolve())
    print("Important: Each endpoint is a distinct ML regression dataset.")
    summary = []
    errors = []
    for ix, entry in enumerate(choose, start=1):
        print(f"\nTARGET {ix}/{len(choose)}: {entry['gene']} ({entry['accession']})")
        try:
            target = resolve_target(entry)
            for endpoint in C.ENDPOINTS:
                try:
                    summary.append(process_dataset(entry, target, endpoint))
                except (Exception) as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    print("    SKIPPED", endpoint, msg[:220])
                    errors.append({**entry, "endpoint": endpoint, "error": msg})
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print("    TARGET UNRESOLVED", msg[:220])
            errors.append({**entry, "endpoint": "*", "error": msg})
        pd.DataFrame(summary).to_csv(C.OUTPUT / "dataset_summary.csv", index=False)
        atomic_json(C.OUTPUT / "download_errors.json", errors)
    result = pd.DataFrame(summary)
    if not result.empty:
        result.to_csv(C.OUTPUT / "dataset_summary.csv", index=False)
        if C.MAKE_ZIPS:
            for disease in ["Alzheimer", "Parkinson"]:
                z = pack_zip(disease, result)
                print("Created:", z)
    print("\nSuccessful datasets:", sum(r.get('status') == 'ok' for r in summary))
    print("Failed target/endpoint steps:", len(errors))
    return result, errors

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download ChEMBL Alzheimer/Parkinson enzyme bioactivities")
    parser.add_argument("--mode", choices=["all", "priority"], default="all")
    parser.add_argument("--pilot-limit", type=int, default=0, help="0 means complete; positive values are incomplete pilots")
    parser.add_argument("--endpoint", action="append", choices=["IC50", "Ki"], help="Repeat to request both; defaults to both")
    parser.add_argument("--out", default=str(C.OUTPUT), help="Output dataset directory")
    args = parser.parse_args()
    C.RUN_MODE = args.mode
    C.PILOT_RAW_LIMIT = args.pilot_limit
    C.OUTPUT = Path(args.out)
    if args.endpoint:
        C.ENDPOINTS = tuple(dict.fromkeys(args.endpoint))
    result, errors = run_pipeline()
    print(result.to_string(index=False) if not result.empty else "No successful datasets")
