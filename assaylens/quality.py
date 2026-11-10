"""Scientific assay-data auditing. Decisions are preserved and exported."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd
from .chemistry import parse_molecule

REQUIRED = {"smiles", "response"}
OPTIONAL = ["assay_id", "assay_type", "assay_format", "source_id", "endpoint", "target_id"]
MEASUREMENT_FLAGS = ["standard_relation", "data_validity_comment"]
MAX_ROWS = 8000
CONFLICT_SPAN = 1.0  # response units; should be chosen a priori for a real study


@dataclass
class AuditResult:
    records: pd.DataFrame
    curated: pd.DataFrame
    summary: dict


def audit_dataset(frame: pd.DataFrame, conflict_span: float = CONFLICT_SPAN) -> AuditResult:
    """Audit a single-target, single-endpoint numeric response dataset.

    Training aggregates within compound+assay using median; unresolvable
    within-assay contradictions are excluded (not silently averaged away).
    """
    if not REQUIRED.issubset(frame.columns):
        raise ValueError("CSV requires columns: smiles,response")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Too many rows; maximum is {MAX_ROWS} to keep local processing safe")
    if len(frame) == 0:
        raise ValueError("CSV has no observations")
    df = frame.copy().reset_index(drop=True)
    df["source_row"] = np.arange(len(df)) + 1
    has_relation = "standard_relation" in df.columns
    has_validity_comment = "data_validity_comment" in df.columns
    for col in OPTIONAL + MEASUREMENT_FLAGS:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip().str.slice(0, 150)
    df["assay_id"] = df["assay_id"].replace("", "UNKNOWN")
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df["qc_reason"] = ""
    df["canonical_smiles"] = ""
    df["parent_smiles"] = ""
    df["scaffold"] = ""
    df["fragment_changed"] = False
    for idx, s in enumerate(df["smiles"]):
        try:
            c, p, scaf, changed = parse_molecule(s)
            df.at[idx, "canonical_smiles"] = c
            df.at[idx, "parent_smiles"] = p
            df.at[idx, "scaffold"] = scaf
            df.at[idx, "fragment_changed"] = changed
        except (ValueError, RuntimeError):
            df.at[idx, "qc_reason"] = "invalid_smiles"
    bad_response = ~np.isfinite(df["response"])
    df.loc[bad_response & (df.qc_reason == ""), "qc_reason"] = "invalid_response"
    # Respect explicit uncertainty/censoring metadata rather than pretending inequalities
    # or data-validity warnings are exact numeric measurements.
    if has_relation:
        inexact = ~df.standard_relation.isin(["=", "=="])
        df.loc[inexact & (df.qc_reason == ""), "qc_reason"] = "censored_or_inexact_measurement"
    if has_validity_comment:
        df.loc[(df.data_validity_comment != "") & (df.qc_reason == ""), "qc_reason"] = "data_validity_warning"

    # Mixing experimental readouts/targets is scientifically misleading.
    valid = df.loc[df.qc_reason == ""].copy()
    endpoints = [x for x in valid.endpoint.unique().tolist() if x]
    targets = [x for x in valid.target_id.unique().tolist() if x]
    if len(endpoints) > 1 or len(targets) > 1:
        raise ValueError("Multiple endpoints or targets detected. Filter to one endpoint and one target before training")
    if endpoints and (valid.endpoint == "").any():
        df.loc[valid.index[valid.endpoint == ""], "qc_reason"] = "missing_endpoint"
    if targets and (valid.target_id == "").any():
        df.loc[valid.index[(valid.target_id == "") & (df.loc[valid.index, "qc_reason"] == "")], "qc_reason"] = "missing_target"
    valid = df.loc[df.qc_reason == ""].copy()

    exact_keys = ["parent_smiles", "assay_id", "source_id", "response"]
    duplicate_mask = valid.duplicated(exact_keys, keep="first")
    df.loc[valid.index[duplicate_mask], "qc_reason"] = "exact_duplicate"
    valid = df.loc[df.qc_reason == ""].copy()
    group_cols = ["parent_smiles", "assay_id"]
    agg = valid.groupby(group_cols, dropna=False).agg(
        response=("response", "median"),
        within_assay_span=("response", lambda x: float(x.max() - x.min())),
        replicate_count=("response", "size"),
        scaffold=("scaffold", "first"),
        assay_type=("assay_type", "first"),
        assay_format=("assay_format", "first"),
        target_id=("target_id", "first"),
        endpoint=("endpoint", "first"),
        source_count=("source_id", "nunique"),
    ).reset_index()
    agg["high_conflict"] = agg.within_assay_span >= conflict_span
    high_conflict_keys = set(zip(agg.loc[agg.high_conflict, "parent_smiles"], agg.loc[agg.high_conflict, "assay_id"]))
    if high_conflict_keys:
        conflict_idx = valid.index[[
            (p, a) in high_conflict_keys
            for p, a in zip(valid.parent_smiles, valid.assay_id)
        ]]
        df.loc[conflict_idx, "qc_reason"] = "within_assay_conflict"
    curated = agg.loc[~agg.high_conflict].copy().reset_index(drop=True)
    if len(curated):
        ranges = curated.groupby("parent_smiles").response.agg(lambda x: float(x.max() - x.min()))
        curated["cross_assay_span"] = curated.parent_smiles.map(ranges)
        curated["cross_assay_conflict"] = curated.cross_assay_span >= conflict_span
        counts = curated.groupby("parent_smiles").assay_id.transform("count")
        # Downweight repeated records from the same molecule across assays;
        # downweight higher within-assay variance, without relying on test labels.
        curated["sample_weight"] = 1 / ((1 + curated.within_assay_span) * np.sqrt(counts))
    else:
        curated["cross_assay_span"] = pd.Series(dtype=float)
        curated["cross_assay_conflict"] = pd.Series(dtype=bool)
        curated["sample_weight"] = pd.Series(dtype=float)
    # Diagnostics are descriptive, not inferred assay-confidence ground truth.
    reasons = Counter(df.qc_reason.replace("", "accepted").tolist())
    summary = {
        "input_rows": int(len(df)),
        "valid_unique_measurements": int(len(valid)),
        "curated_assay_compound_pairs": int(len(curated)),
        "unique_parent_structures": int(curated.parent_smiles.nunique()),
        "unique_scaffolds": int(curated.scaffold.nunique()),
        "unique_assays": int(curated.assay_id.nunique()),
        "fragment_changes": int(df.fragment_changed.sum()),
        "cross_assay_conflict_pairs": int(curated.cross_assay_conflict.sum()),
        "within_assay_conflict_pairs_excluded": int(agg.high_conflict.sum()),
        "reason_counts": dict(reasons),
        "endpoint": endpoints[0] if endpoints else "UNSPECIFIED",
        "target": targets[0] if targets else "UNSPECIFIED",
        "warnings": [],
        "conflict_span_units": float(conflict_span),
    }
    if not endpoints or not targets:
        summary["warnings"].append("Endpoint/target unspecified: document biological comparability before publication")
    if not has_relation:
        summary["warnings"].append("No measurement relation column: assumes provided responses are exact; exclude censored values upstream")
    if not has_validity_comment:
        summary["warnings"].append("No source validity-comment column: experimental data validity must be assessed upstream")
    if curated.assay_id.nunique() < 3:
        summary["warnings"].append("Assay-held-out validation requires at least three assay groups")
    if (curated.assay_id == "UNKNOWN").any():
        summary["warnings"].append("Missing assay IDs prevent credible assay-held-out validation")
    if df.fragment_changed.any():
        summary["warnings"].append("Largest-fragment normalization changed some structures; inspect salt/mixture effects")
    if len(curated) < 50:
        summary["warnings"].append("Small curated dataset: model assessment will be unstable")
    return AuditResult(records=df, curated=curated, summary=summary)
