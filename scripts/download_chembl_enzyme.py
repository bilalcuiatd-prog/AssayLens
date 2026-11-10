#!/usr/bin/env python3
"""Reproducible ChEMBL -> AssayLens AI enzyme regression dataset exporter.

Downloads public *experimental records* (not synthetic labels), retains unfiltered
JSONL.GZ data, assesses explicit filtering rules, joins curated assay metadata,
checks RDKit molecular compatibility, and exports a 10-column AssayLens CSV.

ChEMBL API: https://www.ebi.ac.uk/chembl/api/data/docs
Example:
    python scripts/download_chembl_enzyme.py --target CHEMBL205 --endpoint Ki

Not designed for selecting compounds, clinical decisions, or chemical synthesis.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assaylens.chemistry import parse_molecule
from assaylens.quality import MAX_ROWS, audit_dataset

BASE = "https://www.ebi.ac.uk/chembl/api/data"
SCHEMA = [
    "smiles", "response", "assay_id", "assay_type", "assay_format",
    "source_id", "endpoint", "target_id", "standard_relation",
    "data_validity_comment",
]


def get_json(resource: str, params: dict | None = None, *, attempts: int = 5) -> dict:
    """JSON-only network request with retry on transient errors and 429."""
    url = BASE + "/" + resource.lstrip("/")
    if params:
        url += "?" + urlencode(params)
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": "AssayLensAI-Research/0.2 (+public ChEMBL API)",
                                        "Accept": "application/json"})
        try:
            with urlopen(request, timeout=45) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError(f"Unexpected JSON from {url}")
            return value
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise RuntimeError(f"ChEMBL returned HTTP {exc.code}: {url}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(45.0, float(retry_after)) if retry_after else 1.5 * 2**attempt
            except ValueError:
                delay = 1.5 * 2**attempt
            print(f"Transient HTTP {exc.code}; retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
        except (URLError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Cannot reach ChEMBL API: {url}: {exc}") from exc
            time.sleep(1.5 * 2**attempt)
    raise RuntimeError("Unreachable request retry state")


def iter_pages(resource: str, params: dict, collection_key: str, max_rows: int = 0,
               page_size: int = 500) -> Iterable[dict]:
    """Walk explicitly ordered API pages. 0 max_rows means no truncation."""
    offset = 0
    total = 0
    seen_ids: set[str] = set()
    while True:
        query = {**params, "limit": page_size, "offset": offset}
        payload = get_json(resource, query)
        records = payload.get(collection_key)
        if not isinstance(records, list):
            raise ValueError(f"ChEMBL response missing {collection_key!r}; keys={list(payload)}")
        if not records:
            break
        for record in records:
            # Checking activity_id prevents accidental duplicate pages; it DOES NOT
            # collapse repeated independent measurements of the same compound.
            unique_key = record.get("activity_id") if collection_key == "activities" else record.get("assay_chembl_id")
            if unique_key is not None:
                key = str(unique_key)
                if key in seen_ids:
                    raise RuntimeError(f"Duplicated API pagination record {key}; refusing partial unstable extraction")
                seen_ids.add(key)
            yield record
            total += 1
            if max_rows and total >= max_rows:
                return
        meta = payload.get("page_meta", {})
        if not meta.get("next"):
            break
        offset += len(records)
        print(f"  Retrieved {total} {collection_key}; next offset={offset}", flush=True)


def retrieve_activities(target: str, endpoint: str, path: Path, max_raw: int = 0) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in iter_pages(
            "activity.json",
            {"target_chembl_id": target, "standard_type": endpoint,
             "order_by": "activity_id"},
            "activities", max_rows=max_raw,
        ):
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows


def batches(items: list[str], n: int) -> Iterable[list[str]]:
    for start in range(0, len(items), n):
        yield items[start:start + n]


def retrieve_assays(assay_ids: list[str], path: Path) -> dict[str, dict]:
    """Join assay-level confidence + relationship + BAO metadata in API batches.

    Batched __in queries are supported by the official ChEMBL API. Refuse to
    silently assign unknown confidence scores when assay metadata is absent.
    """
    results: dict[str, dict] = {}
    for index, ids in enumerate(batches(sorted(set(assay_ids)), 40), start=1):
        for assay in iter_pages(
            "assay.json", {"assay_chembl_id__in": ",".join(ids),
                           "order_by": "assay_chembl_id"}, "assays", page_size=500,
        ):
            key = assay.get("assay_chembl_id")
            if key in ids:
                results[key] = assay
        if index % 5 == 0:
            print(f"  Assay metadata batches: {index}, assays loaded: {len(results)}", flush=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for key in sorted(results):
            stream.write(json.dumps(results[key], sort_keys=True, ensure_ascii=False) + "\n")
    return results


def number(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _integer_string(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        n = float(value)
        if n.is_integer() and math.isfinite(n):
            return str(int(n))
    except (TypeError, ValueError, OverflowError):
        pass
    return str(value)


def filter_activity(row: dict, assay: dict | None, *, target: str,
                    endpoint: str, organism: str, confidence: int = 9,
                    check_structure: bool = True) -> tuple[dict | None, str]:
    """Filter using a fixed a-priori order; first rejection reason is recorded."""
    if row.get("target_chembl_id") != target:
        return None, "wrong_target"
    if row.get("standard_type") != endpoint:
        return None, "wrong_endpoint"
    if row.get("target_organism") != organism:
        return None, "wrong_target_organism"
    if row.get("standard_relation") not in ("=", "=="):
        return None, "not_exact_measurement"
    if row.get("standard_units") != "nM":
        return None, "units_not_nM"
    measured = number(row.get("standard_value"))
    if measured is None or measured <= 0:
        return None, "missing_or_nonpositive_value"
    # Preserve scientific caution: even 'Manually validated' is filtered out,
    # because AssayLens quality.py rejects any non-empty validity comment.
    if str(row.get("data_validity_comment") or "").strip():
        return None, "activity_validity_comment"
    if _integer_string(row.get("potential_duplicate")) != "0":
        return None, "possible_duplicate_or_unknown_flag"
    if _integer_string(row.get("standard_flag")) != "1":
        return None, "unstandardized_or_unknown_flag"
    if row.get("assay_chembl_id") in (None, "") or assay is None:
        return None, "assay_metadata_missing"
    if _integer_string(assay.get("confidence_score")) != str(confidence):
        return None, "assay_confidence_not_9"
    if assay.get("target_chembl_id") != target:
        return None, "assay_target_mismatch"
    if assay.get("relationship_type") != "D":
        return None, "not_direct_target_association"
    if assay.get("assay_type") not in ("B", "F"):
        return None, "incompatible_assay_type"
    if not row.get("document_chembl_id"):
        return None, "missing_document_provenance"
    smiles = row.get("canonical_smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "missing_smiles"
    if check_structure:
        try:
            parse_molecule(smiles)
        except (ValueError, RuntimeError):
            return None, "invalid_or_unsupported_smiles"
    # One endpoint, a consistent scale, and exact values; do NOT mix Ki/IC50.
    response = 9.0 - math.log10(measured)
    reported = number(row.get("pchembl_value"))
    if reported is not None and abs(response - reported) > 0.15:
        return None, "pchembl_disagreement"
    result = {
        "smiles": smiles, "response": round(response, 6),
        "assay_id": str(row["assay_chembl_id"]),
        "assay_type": str(assay["assay_type"]),
        "assay_format": str(assay.get("bao_label") or assay.get("bao_format") or "UNKNOWN"),
        "source_id": str(row["document_chembl_id"]),
        "endpoint": endpoint, "target_id": target,
        "standard_relation": "=", "data_validity_comment": "",
        "activity_id": row.get("activity_id"),
        "molecule_chembl_id": row.get("molecule_chembl_id"),
        "original_standard_value_nM": measured,
        "reported_pchembl_value": reported,
        "assay_confidence_score": assay.get("confidence_score"),
        "assay_relationship_type": assay.get("relationship_type"),
        "assay_description": str(assay.get("description") or "")[:1000],
    }
    return result, "accepted"


def curate(raw: list[dict], assays: dict[str, dict], *, target: str, endpoint: str,
           organism: str = "Homo sapiens") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    accepted, excluded = [], []
    counts: Counter = Counter()
    for record in raw:
        a_id = record.get("assay_chembl_id")
        clean, reason = filter_activity(record, assays.get(a_id), target=target,
                                        endpoint=endpoint, organism=organism)
        counts[reason] += 1
        if clean:
            accepted.append(clean)
        else:
            excluded.append({
                "activity_id": record.get("activity_id"),
                "molecule_chembl_id": record.get("molecule_chembl_id"),
                "assay_chembl_id": a_id, "document_chembl_id": record.get("document_chembl_id"),
                "standard_value": record.get("standard_value"),
                "standard_relation": record.get("standard_relation"),
                "rejection_reason": reason,
            })
    cleaned = pd.DataFrame(accepted, columns=SCHEMA + [
        "activity_id", "molecule_chembl_id", "original_standard_value_nM",
        "reported_pchembl_value", "assay_confidence_score",
        "assay_relationship_type", "assay_description",
    ])
    if not cleaned.empty:
        cleaned = cleaned.sort_values("activity_id", kind="stable").reset_index(drop=True)
    return cleaned, pd.DataFrame(excluded, columns=[
        "activity_id", "molecule_chembl_id", "assay_chembl_id",
        "document_chembl_id", "standard_value", "standard_relation", "rejection_reason",
    ]), dict(sorted(counts.items()))


def prepare_import(cleaned: pd.DataFrame, *, min_assay_records: int = 10,
                   limit: int = MAX_ROWS, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if limit > MAX_ROWS:
        raise ValueError(f"AssayLens application hard limit is {MAX_ROWS} rows")
    if cleaned.empty:
        raise ValueError("No activity records passed stringent curation; inspect excluded_rows.csv")
    # Use unique molecular structures as a conservative estimate of assay size;
    # exact parent normalization and replicate resolution happen in AssayLens.
    sizes = cleaned.groupby("assay_id")["smiles"].nunique()
    allowed_assays = set(sizes[sizes >= min_assay_records].index)
    eligible = cleaned[cleaned.assay_id.isin(allowed_assays)].copy()
    if eligible.empty:
        raise ValueError(f"No assays have >= {min_assay_records} distinct SMILES; lower --min-assay-records")
    if len(eligible) > limit:
        # Reproducible random subset for a LOCAL 8000-row upload only; never
        # present it as a complete or representative external benchmark.
        imported = eligible.sample(n=limit, random_state=seed).sort_values("activity_id", kind="stable")
    else:
        imported = eligible
    return eligible.reset_index(drop=True), imported[SCHEMA].reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Download and curate a real enzyme activity dataset from ChEMBL for AssayLens AI")
    parser.add_argument("--target", default="CHEMBL205", help="Single-protein ChEMBL target (default: human CA2)")
    parser.add_argument("--endpoint", choices=["Ki", "IC50", "Kd"], default="Ki")
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--max-raw", type=int, default=0, help="0 = retrieve every matching record; otherwise truncated pilot")
    parser.add_argument("--min-assay-records", type=int, default=10)
    parser.add_argument("--max-upload-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_raw < 0 or args.min_assay_records < 1 or not 1 <= args.max_upload_rows <= MAX_ROWS:
        parser.error("Invalid limits. --max-raw>=0, --min-assay-records>=1, --max-upload-rows 1..8000")
    outdir = Path(args.outdir or f"datasets/{args.target}_{args.endpoint}")
    outdir.mkdir(parents=True, exist_ok=True)
    target_info = get_json(f"target/{args.target}.json")
    if target_info.get("target_type") != "SINGLE PROTEIN" or target_info.get("organism") != args.organism:
        raise ValueError("The target is not a matching single protein/organism. Choose a verified single-protein target.")
    status = get_json("status.json")
    print(f"Target: {args.target} {target_info.get('pref_name')}; ChEMBL version: {status.get('chembl_db_version')}")
    print("Downloading raw experimental activities (this may require several minutes) ...")
    raw = retrieve_activities(args.target, args.endpoint, outdir / "raw_activities.jsonl.gz", args.max_raw)
    if not raw:
        raise ValueError("No ChEMBL activities matched the target/endpoint")
    ids = sorted({str(a["assay_chembl_id"]) for a in raw if a.get("assay_chembl_id")})
    print(f"Retrieved {len(raw)} raw activities from {len(ids)} distinct assays. Loading assay metadata ...")
    assays = retrieve_assays(ids, outdir / "assays_metadata.jsonl.gz")
    print(f"Loaded metadata for {len(assays)}/{len(ids)} assay IDs. Filtering + RDKit checking ...")
    clean, excluded, reasons = curate(raw, assays, target=args.target,
                                       endpoint=args.endpoint, organism=args.organism)
    clean.to_csv(outdir / "cleaned_full.csv", index=False)
    excluded.to_csv(outdir / "excluded_rows.csv", index=False)
    eligible, uploaded = prepare_import(clean, min_assay_records=args.min_assay_records,
                                       limit=args.max_upload_rows, seed=args.seed)
    eligible.to_csv(outdir / "model_eligible_full.csv", index=False)
    uploaded.to_csv(outdir / "assaylens_import.csv", index=False)
    audit = audit_dataset(uploaded)
    (outdir / "assaylens_audit_summary.json").write_text(json.dumps(audit.summary, indent=2), encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": BASE,
        "api_status": status,
        "target_chembl_id": args.target,
        "target_name": target_info.get("pref_name"),
        "target_type": target_info.get("target_type"),
        "target_organism": args.organism,
        "endpoint": args.endpoint,
        "response_scale": f"p{args.endpoint} = 9 - log10({args.endpoint} in nM)",
        "filtering": {
            "standard_relation": "=", "standard_units": "nM", "positive_numeric": True,
            "standard_flag": "1", "potential_duplicate": "0",
            "data_validity_comment": "empty only, including omission of Manually validated",
            "assay_confidence_score": 9, "relationship_type": "D",
            "assay_types": ["B", "F"], "rdkit_mol_parse": True,
            "min_assay_distinct_smiles": args.min_assay_records,
        },
        "counts": {
            "downloaded_raw": len(raw),
            "assays_in_raw": len(ids), "assays_metadata_found": len(assays),
            "cleaned_full": len(clean), "model_eligible_full": len(eligible),
            "application_import": len(uploaded),
            "application_assays": int(uploaded.assay_id.nunique()),
            "application_unique_smiles": int(uploaded.smiles.nunique()),
        },
        "rejection_reasons": reasons,
        "source_truncated_by_max_raw": bool(args.max_raw and len(raw) >= args.max_raw),
        "application_random_subsampled": len(eligible) > args.max_upload_rows,
        "application_random_seed": args.seed,
        "caution": [
            "No results are claimed without downloaded raw real records and external validation.",
            "ChEMBL API is live; snapshot may change. Version, download time, checksums saved.",
            "Do not claim that an 8000-row sampled dataset is the complete ChEMBL cohort.",
            "Even confidence=9 does not prove assay protocol comparability.",
            "AssayLens also applies parent structure grouping, duplicate removal and conflict exclusion; consult its QC report.",
            "Repeated documents/compounds across groups can create dependencies; external evaluation needs source-level leakage checks.",
        ],
    }
    paths = ["raw_activities.jsonl.gz", "assays_metadata.jsonl.gz", "cleaned_full.csv",
             "excluded_rows.csv", "model_eligible_full.csv", "assaylens_import.csv",
             "assaylens_audit_summary.json"]
    manifest["file_sha256"] = {name: sha256(outdir / name) for name in paths}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("\nDONE. Main upload file:", (outdir / "assaylens_import.csv").resolve())
    print("Row counts:", manifest["counts"])
    print("AssayLens audit:", audit.summary)
    if manifest["source_truncated_by_max_raw"]:
        print("WARNING: --max-raw produced a truncated and potentially biased extraction")
    if audit.summary["unique_assays"] < 6:
        print("WARNING: fewer than six assay groups after audit; assay-held-out training will not run")
    if audit.summary["curated_assay_compound_pairs"] < 50:
        print("WARNING: fewer than 50 curated pairs; collect more data or relax justified filters")
    return outdir


if __name__ == "__main__":
    main()
