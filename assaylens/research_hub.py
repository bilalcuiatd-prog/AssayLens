"""Versioned scientific dataset registry and task-specific benchmark adapters."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Resource:
    key: str
    kind: str
    title: str
    provider: str
    loader_name: str
    data_type: str
    availability: str
    source_url: str
    license_note: str
    explanation: str
    endpoint: str = ""


RESOURCES = (
    Resource("dti_bindingdb_kd", "dti", "BindingDB · Kd", "TDC", "BindingDB_Kd", "affinity_regression", "automatic", "https://tdcommons.ai/multi_pred_tasks/dti/", "TDC documentation: BindingDB CC BY 3.0 US; check upstream conditions", "Measured dissociation affinity; one endpoint per dataset", "Kd"),
    Resource("dti_bindingdb_ki", "dti", "BindingDB · Ki", "TDC", "BindingDB_Ki", "affinity_regression", "automatic", "https://tdcommons.ai/multi_pred_tasks/dti/", "TDC documentation: BindingDB CC BY 3.0 US; check upstream conditions", "Measured inhibition constants; preserve input units", "Ki"),
    Resource("dti_bindingdb_ic50", "dti", "BindingDB · IC50", "TDC", "BindingDB_IC50", "affinity_regression", "automatic", "https://tdcommons.ai/multi_pred_tasks/dti/", "TDC documentation: BindingDB CC BY 3.0 US; check upstream conditions", "Large bioactivity dataset; sample limit configurable", "IC50"),
    Resource("dti_davis", "dti", "DAVIS kinase affinity", "TDC", "DAVIS", "affinity_regression", "automatic", "https://tdcommons.ai/multi_pred_tasks/dti/", "Check dataset origin and redistribution terms", "Kinase binding benchmark; retain native response values", "DAVIS_native"),
    Resource("dti_kiba", "dti", "KIBA kinase benchmark", "TDC", "KIBA", "affinity_regression", "automatic", "https://tdcommons.ai/multi_pred_tasks/dti/", "Check dataset origin and redistribution terms", "Integrated KIBA scores are not interchangeable with IC50/Ki/Kd", "KIBA_score"),
    Resource("ddi_twosides", "ddi", "TWOSIDES · polypharmacy adverse-event records", "TDC", "TWOSIDES", "multilabel_raw", "raw_only", "https://tdcommons.ai/multi_pred_tasks/ddi/", "Verify source licensing and provenance for your use", "Multi-label association events; no automatic binary-negative conversion", "adverse_event"),
    Resource("ddi_drugbank", "ddi", "DrugBank · interaction type records", "TDC", "DrugBank", "multiclass_licensed", "licensed_upload", "https://tdcommons.ai/multi_pred_tasks/ddi/", "DrugBank redistribution and use require an appropriate license", "Multi-class interaction type labels; not a binary interaction/noninteraction dataset", "interaction_type"),
    Resource("ppi_huri", "ppi", "HuRI · human reference interactome", "TDC", "HuRI", "positive_network", "automatic", "https://tdcommons.ai/multi_pred_tasks/ppi/", "TDC documentation: CC BY 4.0", "Positive human binary interaction network; reference negative sampling is optional", "binary_ppi"),
    Resource("ppi_string", "ppi", "STRING · functional / physical networks", "STRING", "", "association_network", "integrated_existing", "https://string-db.org/help/api/", "Cite STRING database and API version", "Association scores are not verified negatives or necessarily experimental physical binding", "association_score"),
    Resource("ppi_biogrid", "ppi", "BioGRID · curated networks", "BioGRID", "", "network_upload", "manual_upload", "https://downloads.thebiogrid.org/BioGRID", "Follow BioGRID use conditions", "Upload appropriately curated interaction CSV with protein sequences", "binary_ppi"),
    Resource("ppi_intact", "ppi", "IntAct · molecular interaction records", "IntAct", "", "network_upload", "manual_upload", "https://www.ebi.ac.uk/intact/", "Follow IntAct use conditions", "Upload appropriately curated interaction CSV with protein sequences", "physical_ppi"),
)
REGISTRY = {r.key:r for r in RESOURCES}

MODEL_PRESETS = {
    "dti": {
        "regression": ["extra_trees", "random_forest", "hist_gradient", "gradient_boosting", "ridge", "elastic_net", "svr", "ann_mlp", "xgboost", "lightgbm"],
        "classification": ["extra_trees", "random_forest", "hist_gradient", "logistic", "svm", "ann_mlp", "xgboost", "lightgbm"],
        "default": ["random_forest", "extra_trees"],
        "ablations": ["joint", "no_cross_features", "drug_only", "target_only"],
        "features": "Morgan molecular fingerprints and protein sequence composition/dipeptide features",
        "validation": ["warm", "cold_drug", "cold_target", "cold_both"],
    },
    "ddi": {
        "regression": ["extra_trees", "random_forest", "hist_gradient", "svr", "ann_mlp"],
        "classification": ["random_forest", "extra_trees", "hist_gradient", "logistic", "svm", "ann_mlp"],
        "default": ["random_forest", "extra_trees"],
        "ablations": ["joint", "no_cross_features", "similarity_only", "entity_descriptors"],
        "features": "Symmetric molecular fingerprint sum, absolute difference, product, similarity; undirected pairs only",
        "validation": ["warm", "cold_one", "cold_both"],
    },
    "ppi": {
        "regression": ["extra_trees", "random_forest", "svr", "ann_mlp"],
        "classification": ["logistic", "random_forest", "extra_trees", "svm", "ann_mlp"],
        "default": ["logistic", "random_forest"],
        "ablations": ["joint", "no_cross_features", "similarity_only", "entity_descriptors"],
        "features": "Symmetric protein sequence composition, reduced dipeptide descriptors, and pairwise interaction terms",
        "validation": ["warm", "cold_one", "cold_both"],
    },
}


def resource_catalog():
    return {"resources":[asdict(r) for r in RESOURCES], "models":MODEL_PRESETS,
            "notice":"External source datasets are not bundled. Do not use interaction predictions as medication guidance."}


def tdc_loader(kind:str, dataset_name:str, add_negatives=False):
    try:
        module = importlib.import_module("tdc.multi_pred")
    except ImportError:
        try:
            module = importlib.import_module("tdc_ml.multi_pred")
        except ImportError as exc:
            raise RuntimeError("PyTDC not installed. Run: py -3 -m pip install -r requirements-datasets.txt") from exc
    klass = getattr(module,kind.upper())
    dataset = klass(name=dataset_name)
    if add_negatives:
        if kind != "ppi" or dataset_name != "HuRI":
            raise ValueError("Reference negative sampling is available only for HuRI")
        dataset = dataset.neg_sample(frac=1)
    return dataset.get_data()


def _pick(frame, choices):
    cols = {str(col).lower():col for col in frame.columns}
    for candidate in choices:
        if candidate.lower() in cols:
            return frame[cols[candidate.lower()]]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def _take_balanced(df, limit, label_field="response", seed=42):
    if len(df) <= limit:
        return df.reset_index(drop=True)
    # For large datasets, deterministic random sampling only; split and class
    # balance must be evaluated after curation, not assumed from the sampling.
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def normalize_tdc(resource:Resource, raw:pd.DataFrame, limit=12000, sampling_seed=42, reference_negatives=False):
    if resource.key not in REGISTRY or resource != REGISTRY[resource.key]:
        raise ValueError("Dataset is not present in the curated registry")
    if not 100 <= limit <= 20000:
        raise ValueError("Choose 100–20,000 maximum import rows")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Source returned no dataset records")
    rows_original = len(raw)
    note = []
    if resource.kind == "dti":
        df = pd.DataFrame({
            "smiles": _pick(raw, ["Drug", "SMILES", "drug_smiles"]),
            "target_sequence": _pick(raw, ["Target", "Target_Sequence", "sequence"]),
            "target_id": _pick(raw, ["Target_ID", "target_id", "uniprot_id"]),
            "response": pd.to_numeric(_pick(raw, ["Y", "y", "label", "affinity"]), errors="coerce"),
            "endpoint": resource.endpoint,
            "assay_id": "TDC_NOT_PROVIDED",
            "source_id": "TDC:"+resource.loader_name,
            "standard_relation": "=",
            "data_validity_comment": "",
        })
        df = df[df.response.notna() & df.smiles.notna() & df.target_sequence.notna() & df.target_id.notna()].copy()
        # Avoid silent conversion from raw concentrations to pActivity. TDC and
        # original benchmark labels must be verified per dataset release.
        note.append("Response is kept on the original TDC numeric scale; no logarithmic transformation was assumed.")
        note.append("Assay identifiers are unavailable in TDC's standardized DTI table and are not fabricated.")
    elif resource.kind == "ppi":
        df = pd.DataFrame({
            "protein_a":_pick(raw,["Protein1","Protein_1","protein_a","Sequence1"]),
            "protein_b":_pick(raw,["Protein2","Protein_2","protein_b","Sequence2"]),
            "protein_a_id":_pick(raw,["Protein1_ID","protein1_id"]),
            "protein_b_id":_pick(raw,["Protein2_ID","protein2_id"]),
            "label":pd.to_numeric(_pick(raw,["Y","y","label"]),errors="coerce"),
            "interaction_type":"binary_physical_reference",
            "evidence_type":"TDC_HuRI_reference_generated_controls" if reference_negatives else "TDC_HuRI_positive_only",
            "source_id":"TDC:HuRI",
        })
        df = df[df.protein_a.notna() & df.protein_b.notna()].copy()
        df = df[df.label.notna()].copy()
        note.append("Unobserved pairs are not universally reliable negatives. Review HuRI and TDC negative-sampling methodology before training.")
        if reference_negatives:
            note.append("TDC generated reference negatives via neg_sample(frac=1); these are not manually verified noninteractions.")
        else:
            note.append("Positive-only PPI data support network analysis, not ordinary binary classification.")
    else:
        df = pd.DataFrame({
            "drug_a":_pick(raw,["Drug1","Drug_1","drug_a","SMILES1"]),
            "drug_b":_pick(raw,["Drug2","Drug_2","drug_b","SMILES2"]),
            "drug_a_id":_pick(raw,["Drug1_ID","drug1_id"]),
            "drug_b_id":_pick(raw,["Drug2_ID","drug2_id"]),
            "event_label":_pick(raw,["Y","y","label"]),
            "source_id":"TDC:TWOSIDES",
            "interpretation":"event_labels_only_no_binary_negative_assumption",
        })
        df = df[df.drug_a.notna() & df.drug_b.notna()].copy()
        note.append("TWOSIDES event labels are multi-label associations, NOT binary interaction/noninteraction labels. Data are provided only as an untrained research export.")
    excluded_before_limit = rows_original - len(df)
    df = _take_balanced(df, limit, seed=sampling_seed)
    audit = {
        "kind":resource.kind,
        "dataset_key":resource.key,
        "source":resource.provider+":"+resource.loader_name,
        "source_url":resource.source_url,
        "license_note":resource.license_note,
        "captured_utc":datetime.now(timezone.utc).isoformat(),
        "original_rows":int(rows_original),
        "dropped_missing_required":int(excluded_before_limit),
        "export_rows":int(len(df)),
        "sample_limit":int(limit),
        "sample_seed":int(sampling_seed),
        "trainable_as_is":resource.kind=="dti" or (resource.kind=="ppi" and reference_negatives),
        "notes":note,
    }
    return df, audit


def load_and_normalize(resource_key, limit=12000, sampling_seed=42, reference_negatives=False, loader=None):
    resource = REGISTRY.get(resource_key)
    if not resource:
        raise ValueError("Unsupported source dataset")
    if resource.availability not in ("automatic","raw_only"):
        raise ValueError("This dataset requires manual source-specific import or a data license; no automatic download is enabled")
    if resource.kind=="ddi" and resource.loader_name!="TWOSIDES":
        raise ValueError("A separate dataset permission check is required")
    call = loader or tdc_loader
    raw = call(resource.kind, resource.loader_name, reference_negatives)
    return normalize_tdc(resource, raw, limit=limit, sampling_seed=sampling_seed,
                         reference_negatives=reference_negatives)
