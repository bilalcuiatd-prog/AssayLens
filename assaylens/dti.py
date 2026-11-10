"""Pairwise molecular–protein modeling with entity-disjoint validation."""
from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold

from .advanced_experiments import build_estimator, REGRESSION_MODELS, CLASSIFICATION_MODELS

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {x: i for i, x in enumerate(AA)}
STRATEGIES = ("warm", "cold_drug", "cold_target", "cold_both")
VARIANTS = ("joint", "no_cross_features", "drug_only", "target_only")
REQUIRED = ("smiles", "target_id", "target_sequence", "response")
MAX_ROWS = 20000


def _sequence(s):
    if not isinstance(s, str):
        return ""
    clean = re.sub(r"\s+", "", s).upper()
    if len(clean) < 30 or len(clean) > 10000 or any(c not in AA_MAP for c in clean):
        return ""
    return clean


def _canonical(smiles):
    if not isinstance(smiles, str) or len(smiles) > 1500:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        parts = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        mol = max(parts, key=lambda m: m.GetNumHeavyAtoms())
        if mol.GetNumHeavyAtoms() < 2:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


def curate(frame: pd.DataFrame):
    """Return unique drug-target pairs, a row-level exclusion log, and an audit manifest."""
    if not set(REQUIRED).issubset(frame.columns):
        raise ValueError("DTI CSV must contain: " + ", ".join(REQUIRED))
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"DTI upload must have 1–{MAX_ROWS:,} rows")
    data = frame.copy().reset_index(drop=True)
    for column in ("endpoint", "source_id", "assay_id", "standard_relation", "data_validity_comment"):
        if column not in data.columns:
            data[column] = ""
    log = []
    good = []
    for i, row in enumerate(data.to_dict("records")):
        reason = ""
        smiles = _canonical(row["smiles"])
        seq = _sequence(row["target_sequence"])
        target = str(row["target_id"]).strip() if pd.notna(row["target_id"]) else ""
        endpoint = "" if pd.isna(row.get("endpoint")) else str(row.get("endpoint")).strip()
        relation = "" if pd.isna(row.get("standard_relation")) else str(row.get("standard_relation")).strip()
        validity = "" if pd.isna(row.get("data_validity_comment")) else str(row.get("data_validity_comment")).strip()
        try:
            y = float(row["response"])
        except (ValueError, TypeError):
            y = float("nan")
        if not smiles:
            reason = "invalid_smiles"
        elif not seq:
            reason = "missing_or_invalid_protein_sequence"
        elif not target or target.lower() in ("nan", "none"):
            reason = "missing_target_id"
        elif not np.isfinite(y):
            reason = "invalid_response"
        elif relation and relation not in ("=", "=="):
            reason = "censored_measurement"
        elif validity and validity.lower() not in ("nan", "none"):
            reason = "measurement_flag"
        if reason:
            log.append({"source_row": i + 1, "status": "excluded", "reason": reason})
        else:
            good.append({"source_row": i + 1, "smiles": smiles, "target_id": target,
                         "target_sequence": seq, "response": y, "endpoint": endpoint,
                         "source_id": str(row.get("source_id") or ""),
                         "assay_id": str(row.get("assay_id") or "")})
            log.append({"source_row": i + 1, "status": "eligible", "reason": ""})
    if not good:
        raise ValueError("No valid paired observations with target sequences")
    df = pd.DataFrame(good)
    target_sequences = df.groupby("target_id").target_sequence.nunique()
    if (target_sequences > 1).any():
        invalid = target_sequences[target_sequences > 1].index.tolist()
        raise ValueError("Inconsistent sequences for target identifier(s): " + ", ".join(invalid[:8]))
    distinct_endpoints = sorted(x for x in df.endpoint.unique() if x)
    if len(distinct_endpoints) > 1:
        raise ValueError("Multiple bioactivity endpoints detected. Separate IC50, Ki, Kd, or benchmark labels before DTI modeling")
    if distinct_endpoints and (df.endpoint == "").any():
        raise ValueError("Endpoint is missing in some records; do not mix known and unspecified response types")
    # Pair leakage from repeated assays is avoided by aggregating once, before splitting.
    records = []
    conflicted = 0
    for (drug, target), sub in df.groupby(["smiles", "target_id"], sort=True):
        vals = sub.response.to_numpy()
        if float(np.ptp(vals)) >= 1.0:
            conflicted += 1
            for source_row in sub.source_row.tolist():
                log[source_row - 1] = {"source_row": source_row, "status": "excluded", "reason": "discordant_pair_measurements"}
            continue
        first = sub.iloc[0]
        records.append({"smiles": drug, "target_id": target,
                        "target_sequence": first.target_sequence,
                        "sequence_group": hashlib.sha256(first.target_sequence.encode()).hexdigest()[:20],
                        "response": float(np.median(vals)),
                        "endpoint": first.endpoint, "source_ids": ";".join(sorted(set(sub.source_id))),
                        "source_rows": ";".join(str(int(x)) for x in sub.source_row),
                        "n_measurements": len(sub)})
    clean = pd.DataFrame.from_records(records)
    if clean.empty:
        raise ValueError("No consistent drug-target pairs remain after conflict audit")
    clean = clean.sort_values(["target_id", "smiles"]).reset_index(drop=True)
    manifest = {"raw_rows": len(data), "valid_rows_before_pair_reconciliation": len(df),
                "curated_pairs": len(clean), "unique_drugs": int(clean.smiles.nunique()),
                "unique_targets": int(clean.target_id.nunique()),
                "unique_protein_sequences": int(clean.sequence_group.nunique()),
                "discordant_pairs_excluded": conflicted,
                "endpoint": distinct_endpoints[0] if distinct_endpoints else "UNSPECIFIED",
                "response_interpretation": "Numeric response; classification requires a defensible threshold and harmonized units",
                "notes": ["Per-target assays may have different protocols even for the same endpoint.",
                          "Protein families may share high sequence identity, even when exact sequences differ.",
                          "Measured absence of activity is not equivalent to an untested drug-target pair."]}
    return clean, pd.DataFrame(log), manifest


def _features(df, radius=2, bits=256):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    drug_cache = {}
    protein_cache = {}
    for smi in df.smiles.unique():
        mol = Chem.MolFromSmiles(smi)
        fp = np.zeros(bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fp)
        desc = np.asarray([Descriptors.MolWt(mol) / 1000, Descriptors.MolLogP(mol) / 10,
                          rdMolDescriptors.CalcTPSA(mol) / 200,
                          rdMolDescriptors.CalcNumHBA(mol) / 20,
                          rdMolDescriptors.CalcNumHBD(mol) / 15,
                          rdMolDescriptors.CalcNumRotatableBonds(mol) / 30], dtype=np.float32)
        drug_cache[smi] = np.concatenate((fp, desc))
    for seq in df.target_sequence.unique():
        seq_len = len(seq)
        aa = np.zeros(len(AA), dtype=np.float32)
        dipeptides = np.zeros(64, dtype=np.float32)
        for a in seq:
            aa[AA_MAP[a]] += 1
        aa /= seq_len
        for a, b in zip(seq[:-1], seq[1:]):
            index = (AA_MAP[a] * 20 + AA_MAP[b]) % 64
            dipeptides[index] += 1
        dipeptides /= max(1, seq_len - 1)
        properties = np.asarray([math.log1p(seq_len) / 10,
            sum(aa[AA_MAP[c]] for c in "AILMFWV"),
            sum(aa[AA_MAP[c]] for c in "KRH"),
            sum(aa[AA_MAP[c]] for c in "DE")], dtype=np.float32)
        protein_cache[seq] = np.concatenate((aa, dipeptides, properties))
    drug = np.vstack([drug_cache[s] for s in df.smiles])
    protein = np.vstack([protein_cache[s] for s in df.target_sequence])
    # Interaction proxies are deterministic cross-products, not learned from test labels.
    cross = np.einsum('ni,nj->nij', drug[:, bits:bits+6], protein[:, :8]).reshape(len(df), -1)
    return {"joint": np.column_stack((drug, protein, cross)),
            "no_cross_features": np.column_stack((drug, protein)),
            "drug_only": drug,
            "target_only": protein}


def generate_splits(df, strategy, folds, seed):
    """Yield train/test plus excluded cross-block rows for one deterministic CV repeat."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown DTI validation strategy: {strategy}")
    if not 2 <= folds <= 10:
        raise ValueError("Use 2–10 folds")
    drugs = df.smiles.to_numpy()
    targets = df.sequence_group.to_numpy()
    pairs = list(zip(drugs, targets))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate drug–protein pairs detected before splitting")
    if strategy in ("cold_drug", "cold_both") and df.smiles.nunique() < folds:
        raise ValueError("Not enough distinct drugs for the selected number of folds")
    if strategy in ("cold_target", "cold_both") and df.sequence_group.nunique() < folds:
        raise ValueError("Cold-target evaluation requires at least as many distinct protein sequences as folds; a single-enzyme CSV is not suitable")
    ix = np.arange(len(df))
    if strategy == "warm":
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fit, candidate in splitter.split(ix):
            fit_drugs, fit_targets = set(drugs[fit]), set(targets[fit])
            present = np.asarray([drugs[i] in fit_drugs and targets[i] in fit_targets for i in candidate])
            test = candidate[present]
            skipped = candidate[~present]
            yield fit, test, skipped
        return
    if strategy in ("cold_drug", "cold_target"):
        group = drugs if strategy == "cold_drug" else targets
        # GroupKFold with shuffle provides seed-dependent, entity-disjoint folds.
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fit, test in splitter.split(ix, groups=group):
            yield fit, test, np.array([], dtype=int)
        return
    rng = np.random.default_rng(seed)
    unique_d = np.unique(drugs).copy()
    unique_t = np.unique(targets).copy()
    rng.shuffle(unique_d)
    rng.shuffle(unique_t)
    # Two independent entity partitions; cross-block observations are excluded from
    # that fold instead of contaminating train or test.
    fold_d = {g: i % folds for i, g in enumerate(unique_d)}
    fold_t = {g: i % folds for i, g in enumerate(unique_t)}
    bins_d = np.asarray([fold_d[x] for x in drugs])
    bins_t = np.asarray([fold_t[x] for x in targets])
    for fold in range(folds):
        test_mask = (bins_d == fold) & (bins_t == fold)
        fit_mask = (bins_d != fold) & (bins_t != fold)
        yield ix[fit_mask], ix[test_mask], ix[~(fit_mask | test_mask)]


def _check_split(df, strategy, train, test):
    if len(train) == 0 or len(test) == 0:
        raise ValueError("A fold has no training or test pairs; add more drug–target interactions or reduce folds")
    train_pairs = set(zip(df.smiles.iloc[train], df.sequence_group.iloc[train]))
    test_pairs = set(zip(df.smiles.iloc[test], df.sequence_group.iloc[test]))
    if train_pairs & test_pairs:
        raise RuntimeError("Drug–target pair leakage detected")
    drugs_a, drugs_b = set(df.smiles.iloc[train]), set(df.smiles.iloc[test])
    target_a, target_b = set(df.sequence_group.iloc[train]), set(df.sequence_group.iloc[test])
    drug_overlap = len(drugs_a & drugs_b)
    target_overlap = len(target_a & target_b)
    if strategy in ("cold_drug", "cold_both") and drug_overlap:
        raise RuntimeError("Cold-drug split contains shared chemical entities")
    if strategy in ("cold_target", "cold_both") and target_overlap:
        raise RuntimeError("Cold-target split contains shared protein sequences")
    if strategy == "warm" and (not drugs_b.issubset(drugs_a) or not target_b.issubset(target_a)):
        raise RuntimeError("Warm-start split contains an unseen drug or protein")
    return drug_overlap, target_overlap


def validate(cfg, df):
    task = cfg.get("task", "regression")
    strategy = cfg.get("strategy", "warm")
    folds = int(cfg.get("folds", 5))
    seeds = cfg.get("seeds", [17, 29, 42])
    models = cfg.get("models", ["random_forest"])
    variants = cfg.get("variants", ["joint"])
    if task not in ("regression", "classification") or strategy not in STRATEGIES:
        raise ValueError("Select a valid task and DTI evaluation setting")
    if not 2 <= folds <= 10 or not 1 <= len(seeds) <= 5 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 or s > 2_000_000_000 for s in seeds):
        raise ValueError("Use 2–10 folds and 1–5 different nonnegative random seeds")
    allowed = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    if not models or len(set(models)) != len(models) or any(m not in allowed for m in models):
        raise ValueError("One or more models are unsupported for the chosen task")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("Invalid feature ablation selections")
    if folds * len(seeds) * len(models) * len(variants) > 200:
        raise ValueError("Limit the experiment to 200 model fits")
    if strategy in ("cold_target", "cold_both") and df.sequence_group.nunique() < folds:
        raise ValueError("Cold-target / cold-both requires multiple independent protein sequences; upload or combine multi-target data")
    if strategy in ("cold_drug", "cold_both") and df.smiles.nunique() < folds:
        raise ValueError("Not enough distinct drugs for this protocol")
    if len(df) < max(50, folds * 8):
        raise ValueError("At least 50 drug–target pairs are required (more is recommended for cold-both)")
    if task == "classification":
        y = (df.response.to_numpy() >= float(cfg.get("activity_threshold", 6))).astype(int)
        if min(np.bincount(y, minlength=2)) < folds:
            raise ValueError("Both classification labels require at least one example per fold; select a justified activity threshold")
    fp_bits = int(cfg.get("fp_bits", 256))
    if fp_bits < 64 or fp_bits > 2048 or fp_bits & (fp_bits-1):
        raise ValueError("Fingerprint bit length must be a power of 2 between 64 and 2048")
    for key in ("n_estimators", "mlp_max_iter"):
        if key in cfg and not 10 <= int(cfg[key]) <= 500:
            raise ValueError(f"{key} out of allowed bounds")
    return task, strategy, folds, seeds, models, variants


def run(df, cfg, dest: Path, progress=lambda _: None):
    task, strategy, folds, seeds, models, variants = validate(cfg, df)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    x = _features(df, int(cfg.get("fp_radius",2)), int(cfg.get("fp_bits",256)))
    y = df.response.to_numpy(dtype=float)
    labels = (y >= float(cfg.get("activity_threshold", 6))).astype(int) if task == "classification" else y
    summary_rows, pred_rows, assignments, warnings_all = [], [], [], []
    count = 0
    for seed in seeds:
        for fold, (train, test, skipped) in enumerate(generate_splits(df, strategy, folds, seed), 1):
            if len(train) < 20 or len(test) < 3:
                raise ValueError(f"Seed {seed} fold {fold} has {len(train)} training and {len(test)} test pairs; use a denser multi-target dataset or fewer folds")
            drug_overlap, target_overlap = _check_split(df, strategy, train, test)
            if task == "classification" and len(np.unique(labels[train])) != 2:
                raise ValueError(f"Seed {seed} fold {fold} contains only one class in training data")
            for label, inds in (("train", train), ("test", test), ("excluded_cross_block", skipped)):
                assignments.extend({"seed": seed, "fold": fold, "source_row": int(idx), "partition": label,
                                    "drug_sha256": hashlib.sha256(df.smiles.iloc[idx].encode()).hexdigest()[:16],
                                    "protein_sequence_sha256": df.sequence_group.iloc[idx]} for idx in inds)
            for variant in variants:
                for model_name in models:
                    model = build_estimator(model_name, task, seed * 131 + fold, cfg)
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        model.fit(x[variant][train], labels[train])
                    warnings_all.extend(str(item.message)[:160] for item in captured[:2])
                    prediction = model.predict(x[variant][test])
                    row = {"task": task, "protocol": strategy, "seed": seed, "fold": fold,
                           "model": model_name, "ablation": variant, "train_n": len(train), "test_n": len(test),
                           "excluded_n": len(skipped), "test_fraction": len(test)/len(df),
                           "drug_overlap_n": drug_overlap, "target_overlap_n": target_overlap}
                    probability = None
                    if task == "regression":
                        row.update(mae=float(mean_absolute_error(labels[test], prediction)),
                                   rmse=float(np.sqrt(mean_squared_error(labels[test], prediction))),
                                   r2=float(r2_score(labels[test], prediction)) if len(test) >= 2 else np.nan)
                    else:
                        if hasattr(model, "predict_proba"):
                            scores = model.predict_proba(x[variant][test])
                            classes = model.classes_ if hasattr(model, "classes_") else model.steps[-1][1].classes_
                            probability = scores[:, list(classes).index(1)]
                        elif hasattr(model, "decision_function"):
                            raw_score = np.asarray(model.decision_function(x[variant][test]), dtype=float)
                            probability = 1/(1+np.exp(-np.clip(raw_score, -40, 40)))
                        row.update(accuracy=float(accuracy_score(labels[test], prediction)),
                                   balanced_accuracy=float(balanced_accuracy_score(labels[test], prediction)),
                                   f1=float(f1_score(labels[test], prediction, zero_division=0)),
                                   precision=float(precision_score(labels[test], prediction, zero_division=0)),
                                   recall=float(recall_score(labels[test], prediction, zero_division=0)),
                                   mcc=float(matthews_corrcoef(labels[test], prediction)),
                                   auroc=float(roc_auc_score(labels[test], probability)) if probability is not None and len(set(labels[test])) == 2 else np.nan,
                                   auprc=float(average_precision_score(labels[test], probability)) if probability is not None and len(set(labels[test])) == 2 else np.nan)
                    summary_rows.append(row)
                    for j, index in enumerate(test):
                        pred_rows.append({"seed": seed, "fold": fold, "model": model_name, "ablation": variant,
                                          "source_row": int(index), "drug": df.smiles.iloc[index], "target_id": df.target_id.iloc[index],
                                          "actual": float(labels[index]), "prediction": float(prediction[j]),
                                          "score": float(probability[j]) if probability is not None else np.nan})
                    count += 1
                    if count == 1 or count % 5 == 0:
                        progress(f"Completed {count}/{folds*len(seeds)*len(models)*len(variants)} model fits")
    fold_df = pd.DataFrame(summary_rows)
    pred_df = pd.DataFrame(pred_rows)
    assignment_df = pd.DataFrame(assignments)
    fold_df.to_csv(dest / "fold_metrics.csv", index=False)
    pred_df.to_csv(dest / "test_predictions.csv", index=False)
    assignment_df.to_csv(dest / "split_assignments.csv", index=False)
    metrics = ["mae", "rmse", "r2"] if task == "regression" else ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "mcc", "auroc", "auprc"]
    by_seed = fold_df.groupby(["model", "ablation", "seed"], as_index=False)[metrics].mean()
    by_seed.to_csv(dest / "per_seed_metrics.csv", index=False)
    result = by_seed.groupby(["model", "ablation"], as_index=False)[metrics].agg(["mean", "std"])
    result.columns = ["model", "ablation"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")]
    result.to_csv(dest / "summary.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": task,
                "protocol": strategy, "config": cfg, "model_fits": count,
                "rdkit_version": rdBase.rdkitVersion, "sklearn_version": sklearn_version,
                "test_coverage_mean": float(fold_df.groupby(["seed", "fold"]).test_fraction.first().mean()),
                "exclusion_mean": float(fold_df.groupby(["seed", "fold"]).excluded_n.first().mean()),
                "method": "Repeated K-fold entity-disjoint validation. Fixed hyperparameters, no supervised tuning on test folds.",
                "limitations": ["The cold-both diagonal-block procedure excludes cross-block pairs from each fold; not all interactions are tested.",
                   "Warm-start excludes test pairs involving entities absent in that fold's training set.",
                   "Identical sequences are grouped, but high protein homology is not explicitly controlled.",
                   "Multiple seeds on the same dataset do not represent biological replicates.",
                   "Assay heterogeneity and inter-study measurement differences remain potential confounders.",
                   "Classification treats thresholded measurements as labeled observations; unmeasured pairs are not assumed negative.",
                   "Metrics require external validation and independent benchmarking before publication."],
                "warnings": list(dict.fromkeys(warnings_all))[:15]}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary": result.replace({np.nan: None}).to_dict("records"), "n_fits": count,
            "task": task, "manifest": manifest}
