from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rdkit import Chem, rdBase

from .catalog import BY_GENE

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
BINDINGDB = "https://bindingdb.org/rest/getLigandsByUniprots"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
COLUMNS = ["smiles", "response", "assay_id", "assay_type", "assay_format", "source_id", "endpoint", "target_id", "standard_relation", "data_validity_comment"]
ENDPOINTS = ("IC50", "Ki", "Kd")


def session():
    s = requests.Session()
    policy = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=policy))
    s.headers.update({"User-Agent": "AssayLens-Research-Workbench/1.0", "Accept": "application/json"})
    return s


def fetch_json(s, url, params=None):
    response = s.get(url, params=params, timeout=(15, 90))
    response.raise_for_status()
    return response.json()


def chembl_pages(s, resource, params, collection, max_raw=0):
    count, offset = 0, 0
    seen = set()
    while True:
        data = fetch_json(s, CHEMBL + resource + ".json", {**params, "limit": 500, "offset": offset})
        rows = data.get(collection)
        if not isinstance(rows, list):
            raise ValueError("ChEMBL response did not contain " + collection)
        if not rows:
            return
        for row in rows:
            key = row.get("activity_id") if collection == "activities" else row.get("assay_chembl_id") if collection == "assays" else row.get("target_chembl_id")
            if key and key in seen:
                raise ValueError("Repeated identifier returned by ChEMBL pagination")
            if key:
                seen.add(key)
            yield row
            count += 1
            if max_raw and count >= max_raw:
                return
        offset += len(rows)
        if not data.get("page_meta", {}).get("next"):
            return


def chembl_target(s, gene):
    info = BY_GENE[gene]
    targets = list(chembl_pages(s, "target", {"target_components__accession": info["accession"], "order_by": "target_chembl_id"}, "targets"))
    matched = []
    for target in targets:
        components = [c.get("accession") for c in (target.get("target_components") or [])]
        if target.get("target_type") == "SINGLE PROTEIN" and target.get("organism") == "Homo sapiens" and components == [info["accession"]]:
            matched.append(target)
    if len(matched) != 1:
        raise ValueError(f"Could not unambiguously resolve the human target for {gene} ({info['accession']}); found {len(matched)}")
    return matched[0]


def positive(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def flag(v, expected):
    if isinstance(v, bool):
        return v == expected
    return str(v).strip().lower() in ({"1", "true", "1.0"} if expected else {"0", "false", "0.0"})


def check_chembl_record(row, assay, target_id, endpoint):
    if row.get("target_chembl_id") != target_id or row.get("target_organism") != "Homo sapiens":
        return None, "target_mismatch"
    if row.get("standard_type") != endpoint or row.get("standard_relation") not in ("=", "=="):
        return None, "endpoint_or_censoring"
    if row.get("standard_units") != "nM":
        return None, "units_not_nM"
    value = positive(row.get("standard_value"))
    if value is None:
        return None, "nonpositive_or_missing"
    if row.get("data_validity_comment") or not flag(row.get("standard_flag"), True) or not flag(row.get("potential_duplicate"), False):
        return None, "quality_flag"
    if not assay or assay.get("target_chembl_id") != target_id or int(assay.get("confidence_score") or 0) != 9 or assay.get("relationship_type") != "D":
        return None, "assay_target_confidence"
    if assay.get("assay_type") != "B":
        return None, "not_binding_assay"
    if not row.get("document_chembl_id"):
        return None, "missing_publication"
    structure = row.get("canonical_smiles")
    if not structure or not isinstance(structure, str):
        return None, "missing_structure"
    mol = Chem.MolFromSmiles(structure)
    if mol is None:
        return None, "invalid_structure"
    p = 9 - math.log10(value)
    try:
        if row.get("pchembl_value") is not None and abs(float(row["pchembl_value"]) - p) > 0.15:
            return None, "pchembl_discrepancy"
    except (ValueError, TypeError):
        return None, "pchembl_invalid"
    return {"smiles": Chem.MolToSmiles(mol, isomericSmiles=True), "response": round(p, 6), "assay_id": row.get("assay_chembl_id"), "assay_type": "B", "assay_format": str(assay.get("bao_label") or assay.get("bao_format") or "unknown"), "source_id": row["document_chembl_id"], "endpoint": endpoint, "target_id": target_id, "standard_relation": "=", "data_validity_comment": "", "original_nm": value, "activity_id": row.get("activity_id")}, None


def choose_upload(df, max_rows, seed=42):
    if len(df) <= max_rows:
        return df.copy(), "full_filtered_cohort"
    rng = __import__("numpy").random.default_rng(seed)
    counts = df.groupby("assay_id").size().sort_values(ascending=False)
    ids = counts.head(min(100, len(counts))).index
    subset = df[df.assay_id.isin(ids)].copy()
    if len(subset) > max_rows:
        group_indices = []
        for _, group in subset.groupby("assay_id", sort=True):
            n = min(8, len(group))
            group_indices.extend(rng.choice(group.index.to_numpy(), size=n, replace=False).tolist())
        group_indices = list(dict.fromkeys(group_indices))[:max_rows]
        remaining = subset.loc[~subset.index.isin(group_indices)].index.to_numpy()
        slots = max_rows - len(group_indices)
        if slots > 0:
            group_indices.extend(rng.choice(remaining, size=slots, replace=False).tolist())
        subset = subset.loc[group_indices]
    return subset.sort_values(["assay_id", "smiles"]).reset_index(drop=True), "assay_diversity_convenience_subsample"


def write_frame(records, columns, path):
    pd.DataFrame.from_records(records, columns=columns).to_csv(path, index=False)


def download_chembl(gene, endpoint, folder, progress, max_raw=0, max_rows=8000):
    if endpoint not in ENDPOINTS or gene not in BY_GENE:
        raise ValueError("Unrecognized target or endpoint")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    s = session()
    target = chembl_target(s, gene)
    target_id = target["target_chembl_id"]
    status = fetch_json(s, CHEMBL + "status.json")
    records = []
    with gzip.open(folder / "raw_activities.jsonl.gz", "wt", encoding="utf8") as stream:
        for row in chembl_pages(s, "activity", {"target_chembl_id": target_id, "standard_type": endpoint, "order_by": "activity_id"}, "activities", max_raw=max_raw):
            stream.write(json.dumps(row) + "\n")
            records.append(row)
            if len(records) % 500 == 0:
                progress(f"Retrieved {len(records):,} ChEMBL measurements for {gene}")
    if not records:
        raise ValueError("No measurements available for the selected target and endpoint")
    assay_ids = sorted({r.get("assay_chembl_id") for r in records if r.get("assay_chembl_id")})
    assay_map = {}
    for start in range(0, len(assay_ids), 25):
        block = assay_ids[start:start + 25]
        for a in chembl_pages(s, "assay", {"assay_chembl_id__in": ",".join(block), "order_by": "assay_chembl_id"}, "assays"):
            if a.get("assay_chembl_id") in block:
                assay_map[a["assay_chembl_id"]] = a
        for aid in block:
            if aid not in assay_map:
                try:
                    a = fetch_json(s, CHEMBL + "assay/" + aid + ".json")
                    assay_map[aid] = a
                except (requests.RequestException, ValueError):
                    pass
        if start % 250 == 0:
            progress(f"Assay metadata: {len(assay_map):,}/{len(assay_ids):,}")
    with gzip.open(folder / "assay_metadata.jsonl.gz", "wt", encoding="utf8") as stream:
        for aid in sorted(assay_map):
            stream.write(json.dumps(assay_map[aid]) + "\n")
    curated, excluded = [], []
    for row in records:
        item, problem = check_chembl_record(row, assay_map.get(row.get("assay_chembl_id")), target_id, endpoint)
        if problem:
            excluded.append({"activity_id": row.get("activity_id"), "assay_id": row.get("assay_chembl_id"), "reason": problem})
        else:
            curated.append(item)
    write_frame(excluded, ["activity_id", "assay_id", "reason"], folder / "excluded.csv")
    write_frame(curated, COLUMNS + ["original_nm", "activity_id"], folder / "filtered_full.csv")
    if not curated:
        raise ValueError("Measurements retrieved but all were excluded by strict quality criteria. See excluded.csv")
    df = pd.DataFrame(curated)
    upload, policy = choose_upload(df, max_rows)
    upload[COLUMNS].to_csv(folder / "assaylens_import.csv", index=False)
    manifest = {"provider": "ChEMBL", "gene": gene, "uniprot": BY_GENE[gene]["accession"], "target_id": target_id, "target_name": target.get("pref_name"), "endpoint": endpoint, "database_version": status.get("chembl_db_version"), "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "raw_count": len(records), "filtered_count": len(df), "upload_count": len(upload), "unique_assays": int(upload.assay_id.nunique()), "excluded": dict(Counter(r["reason"] for r in excluded)), "sampling_policy": policy, "pilot": bool(max_raw), "rdkit_version": rdBase.rdkitVersion, "raw_sha256": hashlib.sha256((folder / "raw_activities.jsonl.gz").read_bytes()).hexdigest(), "source_url": CHEMBL, "notes": ["An 8000-row upload subset is not an unbiased evaluation cohort", "Inhibitory activity is not synonymous with therapeutic efficacy", "Censored records and uncertain target assignments were excluded"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return folder / "assaylens_import.csv", manifest


def parse_bindingdb(payload):
    root = payload.get("getLindsByUniprotsResponse") or payload.get("getLindsByUniprotResponse") or {}
    rows = root.get("affinities") or root.get("bdb.affinities") or []
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError("Unrecognized BindingDB JSON structure; inspect the source file")
    return rows


def download_bindingdb(gene, endpoint, folder, progress, cutoff_nm=100000, max_rows=8000):
    if gene not in BY_GENE or endpoint not in ENDPOINTS:
        raise ValueError("Unrecognized gene/endpoint")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    s = session()
    acc = BY_GENE[gene]["accession"]
    progress(f"Requesting BindingDB records for UniProt {acc}")
    payload = fetch_json(s, BINDINGDB, {"uniprot": acc, "cutoff": cutoff_nm, "response": "application/json"})
    (folder / "raw_bindingdb.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    records = parse_bindingdb(payload)
    selected, excluded = [], []
    for idx, row in enumerate(records):
        name = row.get("affinity_type") or row.get("bdb.affinity_type")
        if str(name).lower() != endpoint.lower():
            continue
        value_text = str(row.get("affinity") or row.get("bdb.affinity") or "").strip()
        smi = row.get("smile") or row.get("bdb.smile") or row.get("bdb.smiles")
        value = positive(value_text) if not value_text.startswith(("<", ">", "~", "≤", "≥")) else None
        if value is None or not smi:
            excluded.append({"row": idx, "reason": "nonexact_or_missing_value_or_structure"})
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            excluded.append({"row": idx, "reason": "invalid_smiles"})
            continue
        source_id = row.get("doi") or row.get("pmid") or "UNSPECIFIED"
        selected.append({"smiles": Chem.MolToSmiles(mol, isomericSmiles=True), "response": round(9-math.log10(value), 6), "assay_id": "UNKNOWN", "assay_type": "B", "assay_format": "binding_affinity", "source_id": str(source_id), "endpoint": endpoint, "target_id": acc, "standard_relation": "=", "data_validity_comment": "", "original_nm": value, "record_target": row.get("query") or "", "monomerid": row.get("monomerid") or row.get("bdb.monomerid")})
    write_frame(excluded, ["row", "reason"], folder / "excluded.csv")
    write_frame(selected, COLUMNS+["original_nm", "record_target", "monomerid"], folder / "filtered_full.csv")
    if not selected:
        raise ValueError("No qualifying exact binding records returned for this endpoint and cutoff")
    df = pd.DataFrame(selected)
    upload, policy = choose_upload(df, max_rows)
    upload[COLUMNS].to_csv(folder / "assaylens_import.csv", index=False)
    manifest = {"provider": "BindingDB", "gene": gene, "uniprot_requested": acc, "endpoint": endpoint, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "raw_records": len(records), "filtered_count": len(selected), "upload_count": len(upload), "cutoff_nm": cutoff_nm, "sampling_policy": policy, "source_url": BINDINGDB, "scientific_limitations": ["Affinity cutoff selects a biased subset; do not treat it as an unbiased target activity dataset", "The REST endpoint may include related targets; review each record's 'record_target' before publication", "Per-assay identifiers are not exposed by this endpoint; assay-held-out evaluation is not supported", "Source publication and target linkage must be externally verified"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return folder / "assaylens_import.csv", manifest


def pubchem_assays(accession, max_aids=40):
    if not re.fullmatch(r"[A-Z0-9]{6,10}", accession):
        raise ValueError("Invalid UniProt accession")
    data = fetch_json(session(), PUBCHEM + f"assay/target/accession/{accession}/aids/JSON")
    values = data.get("IdentifierList", {}).get("AID", [])
    aids = [int(a) for a in values[:max_aids]]
    return [{"aid": aid, "download_url": PUBCHEM + f"assay/aid/{aid}/CSV", "source_page": "https://pubchem.ncbi.nlm.nih.gov/bioassay/" + str(aid)} for aid in aids]
