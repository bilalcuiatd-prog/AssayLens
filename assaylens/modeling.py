"""Leakage-aware comparative modeling, conformal calibration and diagnostics."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .chemistry import feature_matrix, closest_tanimoto, anonymized_id

ASSAY_FIELDS = ["assay_type", "assay_format"]


@dataclass
class TrainingResult:
    summary: dict
    residuals: pd.DataFrame
    fitted: dict


def partition_groups(curated: pd.DataFrame, split_type: str, seed: int):
    if split_type not in {"scaffold", "assay"}:
        raise ValueError("split_type must be scaffold or assay")
    column = "scaffold" if split_type == "scaffold" else "assay_id"
    groups = curated[column].astype(str).to_numpy()
    if split_type == "assay" and (groups == "UNKNOWN").any():
        raise ValueError("Assay-held-out split requires an assay_id for every record")
    if len(set(groups)) < 6:
        raise ValueError(f"Need >=6 distinct {column} groups for train/calibration/test")
    indices = np.arange(len(curated))
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_cal_local, test_local = next(split1.split(indices, groups=groups))
    remaining_groups = groups[train_cal_local]
    if len(set(remaining_groups)) < 3:
        raise ValueError("Insufficient training/calibration groups after test split")
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_idx_local, cal_idx_local = next(split2.split(train_cal_local, groups=remaining_groups))
    train = train_cal_local[train_idx_local]
    cal = train_cal_local[cal_idx_local]
    test = test_local
    if min(len(train), len(cal), len(test)) < 8:
        raise ValueError("Need >=8 observations in each train, calibration, test partition")
    # No leakage of any group between partitions.
    assert not (set(groups[train]) & set(groups[cal]))
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[cal]) & set(groups[test]))
    return train, cal, test


def conformal_radius(y_cal: np.ndarray, pred_cal: np.ndarray, alpha: float) -> float:
    residual = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
    n = len(residual)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(f"Calibration n={n} is insufficient for finite {1-alpha:.0%} conformal intervals; collect more calibration observations or increase alpha")
    return float(np.sort(residual)[rank - 1])


def summarize_metrics(y: np.ndarray, pred: np.ndarray, radius: float):
    error = np.abs(y - pred)
    return {
        "n": int(len(y)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "coverage": float(np.mean(error <= radius)),
        "mean_interval_width": float(2 * radius),
        "conformal_radius": float(radius),
    }


def train_evaluate(
    curated: pd.DataFrame, split_type: str = "scaffold", seed: int = 42,
    alpha: float = 0.10, trees: int = 64,
) -> TrainingResult:
    if not 0.01 <= alpha <= 0.3:
        raise ValueError("alpha must be in [0.01, 0.30]")
    if not 16 <= trees <= 250:
        raise ValueError("trees must be in [16, 250]")
    if len(curated) < 50:
        raise ValueError("Need >=50 curated compound-assay pairs; more are recommended")
    if curated.endpoint.nunique() > 1 or curated.target_id.nunique() > 1:
        raise ValueError("Multiple targets/endpoints cannot be pooled")
    train, cal, test = partition_groups(curated, split_type, seed)
    smiles = curated.parent_smiles.astype(str).tolist()
    X_mol = feature_matrix(smiles)
    meta = curated[ASSAY_FIELDS].fillna("").astype(str)
    # Vocabulary is fitted on training data only. Never use assay ID as a feature.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)
    encoded_train = encoder.fit_transform(meta.iloc[train])
    X_assay = np.zeros((len(curated), encoded_train.shape[1]), dtype=np.float32)
    for idx in (train, cal, test):
        X_assay[idx] = encoder.transform(meta.iloc[idx]).astype(np.float32)
    X_full = np.hstack([X_mol, X_assay])
    y = curated.response.to_numpy(dtype=float)
    weights = curated.sample_weight.to_numpy(dtype=float)
    cal_sim = closest_tanimoto([smiles[i] for i in cal], [smiles[i] for i in train])
    test_sim = closest_tanimoto([smiles[i] for i in test], [smiles[i] for i in train])
    # A dataset-relative threshold; NOT a universally validated chemical AD boundary.
    similarity_threshold = float(np.quantile(cal_sim, 0.10))
    model_configs = [
        ("chemistry_only", X_mol, False),
        ("chemistry_assay", X_full, False),
        ("chemistry_assay_weighted", X_full, True),
    ]
    metrics = {}
    predictions = pd.DataFrame({
        "sample_id": [anonymized_id(smiles[i]) for i in test],
        "test_row": np.arange(len(test)) + 1,
        "actual": y[test],
        "similarity_to_training": test_sim,
        "out_of_domain": test_sim < similarity_threshold,
    })
    fitted = {"metadata_encoder": encoder, "split": {"train": train, "cal": cal, "test": test}}
    for name, X, weighted in model_configs:
        model = ExtraTreesRegressor(
            n_estimators=trees, max_features=0.8,
            min_samples_leaf=2, n_jobs=-1, random_state=seed,
        )
        kw = {"sample_weight": weights[train]} if weighted else {}
        model.fit(X[train], y[train], **kw)
        pred_cal = model.predict(X[cal])
        pred_test = model.predict(X[test])
        radius = conformal_radius(y[cal], pred_cal, alpha)
        m = summarize_metrics(y[test], pred_test, radius)
        ood = test_sim < similarity_threshold
        m.update({
            "calibration_n": int(len(cal)),
            "ood_n": int(ood.sum()),
            "ood_coverage": float(np.mean(np.abs(y[test][ood] - pred_test[ood]) <= radius)) if ood.any() else None,
            "in_domain_coverage": float(np.mean(np.abs(y[test][~ood] - pred_test[~ood]) <= radius)) if (~ood).any() else None,
            "ood_mae": float(np.mean(np.abs(y[test][ood] - pred_test[ood]))) if ood.any() else None,
            "in_domain_mae": float(np.mean(np.abs(y[test][~ood] - pred_test[~ood]))) if (~ood).any() else None,
        })
        metrics[name] = m
        predictions[f"prediction_{name}"] = pred_test
        predictions[f"covered_{name}"] = np.abs(y[test] - pred_test) <= radius
        fitted[name] = model
    group_col = "scaffold" if split_type == "scaffold" else "assay_id"
    train_group_count = int(curated.iloc[train][group_col].nunique())
    test_group_count = int(curated.iloc[test][group_col].nunique())
    summary = {
        "method": "GroupShuffleSplit followed by disjoint group calibration split",
        "split_type": split_type,
        "seed": seed,
        "alpha": alpha,
        "trees_per_model": trees,
        "n_samples": len(curated),
        "train_n": len(train),
        "calibration_n": len(cal),
        "test_n": len(test),
        "train_groups": train_group_count,
        "test_groups": test_group_count,
        "similarity_threshold": similarity_threshold,
        "features": {"morgan_bits": 1024, "radius": 2, "physchem_descriptors": 8, "assay_feature_categories": int(X_assay.shape[1])},
        "results": metrics,
        "methodological_caveats": [
            "Split-conformal finite-sample marginal coverage assumes exchangeability; scaffold/assay distribution shift can violate this assumption",
            "The out-of-domain threshold is the calibration-set 10th percentile of nearest-training Tanimoto similarity, not a clinical or chemical validity boundary",
            "The quality weights are heuristic and require independent experimental justification",
            "No prospective or external wet-lab validation is supplied",
            "Model selection on the test set is forbidden: compare preregistered variants rather than select the highest test score",
            "Acyclic structures have no Murcko scaffold; each distinct parent structure forms its own group",
        ],
    }
    return TrainingResult(summary=summary, residuals=predictions, fitted=fitted)
