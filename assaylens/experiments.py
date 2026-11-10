from __future__ import annotations

import hashlib
import inspect
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

MODELS = ("extra_trees", "random_forest", "hist_gradient", "ridge", "svr", "ann_mlp")
VARIANTS = ("fingerprint_only", "descriptors_only", "chemistry", "chemistry_assay", "quality_weighted")


def fingerprints_and_descriptors(smiles, radius=2, bits=512):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fp = np.empty((len(smiles), bits), dtype=np.float32)
    desc = np.empty((len(smiles), 8), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Curated dataset contains an invalid molecular structure")
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), fp[i])
        desc[i] = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcFractionCSP3(mol)]
    return fp, desc


def select_features(fp, desc, meta, train_idx, variant):
    if variant == "fingerprint_only":
        return fp
    if variant == "descriptors_only":
        return desc
    base = np.column_stack([fp, desc])
    if variant == "chemistry":
        return base
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)
    encoder.fit(meta.iloc[train_idx].fillna("").astype(str))
    encoded = encoder.transform(meta.fillna("").astype(str))
    return np.column_stack([base, encoded]).astype(np.float32)


def build_model(kind, seed, cfg):
    leaf = int(cfg.get("min_samples_leaf", 2))
    depth = cfg.get("max_depth")
    depth = None if depth in (None, 0, "") else int(depth)
    trees = int(cfg.get("n_estimators", 64))
    if kind == "extra_trees":
        return ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=leaf, max_depth=depth, random_state=seed, n_jobs=1)
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=trees, min_samples_leaf=leaf, max_depth=depth, random_state=seed, n_jobs=1)
    if kind == "hist_gradient":
        return HistGradientBoostingRegressor(max_iter=trees, max_leaf_nodes=int(cfg.get("hgb_leaves", 15)), l2_regularization=float(cfg.get("regularization", 0.001)), random_state=seed)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(cfg.get("regularization", 1.0))))
    if kind == "svr":
        return make_pipeline(StandardScaler(), SVR(C=float(cfg.get("svr_c", 10.0)), epsilon=float(cfg.get("svr_epsilon", 0.1)), cache_size=400))
    if kind == "ann_mlp":
        layers = tuple(int(x) for x in cfg.get("mlp_layers", [128, 64]))
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=layers, learning_rate_init=float(cfg.get("mlp_lr", 0.001)), alpha=float(cfg.get("mlp_alpha", 0.001)), max_iter=int(cfg.get("mlp_max_iter", 160)), early_stopping=False, random_state=seed, solver="adam"))
    raise ValueError("Unsupported model")


def make_folds(df, strategy, folds, seed):
    if strategy == "scaffold":
        groups = df.scaffold.astype(str).to_numpy()
    elif strategy == "assay":
        groups = df.assay_id.astype(str).to_numpy()
        if any(x in ("UNKNOWN", "", "nan") for x in groups):
            raise ValueError("Assay-group CV needs genuine assay IDs on every record. BindingDB REST datasets cannot use assay-group CV")
    elif strategy == "molecule":
        groups = df.parent_smiles.astype(str).to_numpy()
    else:
        raise ValueError("Validation grouping must be scaffold, molecule, or assay")
    if len(set(groups)) < folds + 2:
        raise ValueError(f"At least {folds+2} independent {strategy} groups are required")
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    rng.shuffle(unique)
    mapping = {g: str(i) for i, g in enumerate(unique)}
    randomized = np.array([mapping[g] for g in groups])
    splitter = GroupKFold(n_splits=folds)
    for i, (outer, test) in enumerate(splitter.split(df, groups=randomized), 1):
        assert not (set(groups[outer]) & set(groups[test]))
        yield i, outer, test, groups


def split_calibration(outer, groups, seed, fold):
    groups_outer = groups[outer]
    if len(set(groups_outer)) < 3:
        raise ValueError("Too few independent groups for calibration")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + fold * 7919)
    fit_relative, calibration_relative = next(splitter.split(outer, groups=groups_outer))
    fit, cal = outer[fit_relative], outer[calibration_relative]
    assert not (set(groups[fit]) & set(groups[cal]))
    return fit, cal


def conformal(y_true, prediction, alpha):
    residuals = np.sort(np.abs(np.asarray(y_true) - np.asarray(prediction)))
    rank = math.ceil((len(residuals) + 1) * (1-alpha))
    if not len(residuals) or rank > len(residuals):
        return None
    return float(residuals[rank-1])


def sample_weights_train(df, train):
    sub = df.iloc[train]
    counts = sub.groupby("parent_smiles").size()
    spans = sub.get("within_assay_span", pd.Series(0.0, index=sub.index))
    return (1 / (1 + spans.fillna(0).to_numpy(dtype=float)) / np.sqrt(sub.parent_smiles.map(counts).to_numpy(dtype=float)))


def validate_config(config, df):
    models = config.get("models", ["extra_trees", "ann_mlp"])
    variants = config.get("variants", ["chemistry"])
    folds = int(config.get("folds", 5))
    seeds = config.get("seeds", [17, 29])
    if not isinstance(models, list) or not models or any(m not in MODELS for m in models):
        raise ValueError("Select at least one supported ML model")
    if not isinstance(variants, list) or not variants or any(v not in VARIANTS for v in variants):
        raise ValueError("Select at least one supported feature ablation")
    if len(models) != len(set(models)) or len(variants) != len(set(variants)):
        raise ValueError("Model and ablation selections must not repeat")
    if not 2 <= folds <= 10 or not isinstance(seeds, list) or not 1 <= len(seeds) <= 5 or any(type(s) is not int or not 0 <= s < 2_000_000_000 for s in seeds):
        raise ValueError("Choose 2–10 folds and 1–5 integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    if len(models) * len(variants) * len(seeds) * folds > 150:
        raise ValueError("This run exceeds 150 model fits. Reduce models, variants, folds, or seeds")
    if not 16 <= int(config.get("n_estimators", 64)) <= 350 or not 1 <= int(config.get("min_samples_leaf", 2)) <= 20:
        raise ValueError("Tree and leaf settings are out of range")
    if not 32 <= int(config.get("fp_bits", 512)) <= 2048 or int(config.get("fp_bits", 512)) & (int(config.get("fp_bits", 512))-1):
        raise ValueError("Morgan fingerprint size must be a power of two between 32 and 2048")
    if not 1 <= int(config.get("fp_radius", 2)) <= 3:
        raise ValueError("Fingerprint radius must be 1–3")
    layers = config.get("mlp_layers", [128, 64])
    if not isinstance(layers, list) or not 1 <= len(layers) <= 3 or any(type(n) is not int or not 8 <= n <= 512 for n in layers):
        raise ValueError("ANN hidden layers must contain 1–3 widths between 8 and 512")
    if not 25 <= int(config.get("mlp_max_iter", 160)) <= 500:
        raise ValueError("ANN max iterations must be 25–500")
    if not 0.05 <= float(config.get("alpha", 0.1)) <= 0.3:
        raise ValueError("Conformal alpha must be between 0.05 and 0.3")
    strategy = config.get("strategy", "scaffold")
    if strategy not in ("scaffold", "molecule", "assay"):
        raise ValueError("Choose scaffold, molecule or assay for validation grouping")
    column = {"scaffold": "scaffold", "molecule": "parent_smiles", "assay": "assay_id"}[strategy]
    labels = df[column].fillna("").astype(str)
    if strategy == "assay" and (labels.isin(["", "UNKNOWN", "nan"]).any()):
        raise ValueError("Assay-group CV requires genuine assay identifiers for all records")
    if labels.nunique() < folds + 2:
        raise ValueError(f"At least {folds+2} unique {strategy} groups are needed for grouped CV and independent calibration")
    if len(df) < 40:
        raise ValueError("At least 40 curated observations are needed for model evaluation")
    if df.target_id.nunique() > 1 or df.endpoint.nunique() > 1:
        raise ValueError("Use one target and one endpoint per experiment")
    return models, variants, folds, seeds


def run_experiment(df, config, outdir, report_progress=lambda m: None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models, variants, folds, seeds = validate_config(config, df)
    smiles = df.parent_smiles.astype(str).tolist()
    report_progress("Encoding molecular fingerprints and descriptors")
    fp, desc = fingerprints_and_descriptors(smiles, int(config.get("fp_radius", 2)), int(config.get("fp_bits", 512)))
    meta = df[["assay_type", "assay_format"]].fillna("").astype(str)
    y = df.response.to_numpy(dtype=float)
    detail = []
    predictions = []
    total = len(models) * len(variants) * folds * len(seeds)
    completed = 0
    training_warnings = []
    alpha = float(config.get("alpha", 0.10))
    for seed in seeds:
        for fold, outer, test, groups in make_folds(df, config.get("strategy", "scaffold"), folds, seed):
            fit, cal = split_calibration(outer, groups, seed, fold)
            if min(len(fit), len(cal), len(test)) < 8:
                raise ValueError(f"Fold {fold}, seed {seed}: insufficient train/calibration/test observations; reduce number of folds")
            cross_overlap = float(len(set(df.parent_smiles.iloc[fit]) & set(df.parent_smiles.iloc[test])) / max(1, df.parent_smiles.iloc[test].nunique()))
            for variant in variants:
                x = select_features(fp, desc, meta, fit, variant)
                weighting = variant == "quality_weighted"
                weight = sample_weights_train(df, fit) if weighting else None
                for kind in models:
                    model = build_model(kind, seed + fold, config)
                    fit_args = {}
                    if weight is not None:
                        if hasattr(model, "steps"):
                            final_name, final_estimator = model.steps[-1]
                            if "sample_weight" not in inspect.signature(final_estimator.fit).parameters:
                                raise ValueError(f"{kind} does not support sample weights in this scikit-learn version")
                            fit_args[f"{final_name}__sample_weight"] = weight
                        else:
                            fit_args["sample_weight"] = weight
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(x[fit], y[fit], **fit_args)
                    for w in caught:
                        if len(training_warnings) < 40:
                            training_warnings.append(f"{kind} / {variant}, seed {seed} fold {fold}: {str(w.message)[:180]}")
                    cal_pred = model.predict(x[cal])
                    test_pred = model.predict(x[test])
                    radius = conformal(y[cal], cal_pred, alpha)
                    absolute = np.abs(y[test] - test_pred)
                    row = {"seed": seed, "fold": fold, "model": kind, "ablation": variant, "train_n": len(fit), "cal_n": len(cal), "test_n": len(test), "mae": float(mean_absolute_error(y[test], test_pred)), "rmse": float(np.sqrt(mean_squared_error(y[test], test_pred))), "r2": float(r2_score(y[test], test_pred)) if len(test) > 1 else np.nan, "coverage": float(np.mean(absolute <= radius)) if radius is not None else np.nan, "interval_width": float(2*radius) if radius is not None else np.nan, "test_molecule_overlap_rate": cross_overlap}
                    detail.append(row)
                    for idx, pred in zip(test, test_pred):
                        predictions.append({"seed": seed, "fold": fold, "model": kind, "ablation": variant, "row": int(idx), "observed": float(y[idx]), "predicted": float(pred), "lower": float(pred-radius) if radius is not None else np.nan, "upper": float(pred+radius) if radius is not None else np.nan, "assay_id": str(df.iloc[idx].assay_id), "scaffold_id": hashlib.sha256(str(df.iloc[idx].scaffold).encode()).hexdigest()[:12]})
                    completed += 1
                    if completed == 1 or completed % 5 == 0 or completed == total:
                        report_progress(f"Completed {completed}/{total} model fits ({int(completed/total*100)}%)")
    table = pd.DataFrame(detail)
    preds = pd.DataFrame(predictions)
    table.to_csv(outdir / "fold_metrics.csv", index=False)
    preds.to_csv(outdir / "out_of_fold_predictions.csv", index=False)
    seed_means = table.groupby(["model", "ablation", "seed"], as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"), coverage=("coverage", "mean"), interval_width=("interval_width", "mean"))
    seed_means.to_csv(outdir / "per_seed_metrics.csv", index=False)
    summary = seed_means.groupby(["model", "ablation"], as_index=False).agg(mae_mean=("mae", "mean"), mae_sd=("mae", "std"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), r2_mean=("r2", "mean"), r2_sd=("r2", "std"), coverage_mean=("coverage", "mean"), interval_width_mean=("interval_width", "mean"))
    summary.to_csv(outdir / "summary.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "scikit_learn": sklearn_version, "rdkit": rdBase.rdkitVersion, "n_rows": len(df), "folds": folds, "seeds": seeds, "config": config, "train_only_assay_encoder": True, "calibration": "disjoint inner group holdout", "observations": "Group CV; assay split may reuse a molecular structure across assays; check overlap rate. Test-set model selection not justified.", "statistical_limits": "Standard deviation is across seed means; repeated folds are not independent replicates. Split conformal coverage guarantees need exchangeability, potentially violated by group shift.", "training_warnings": training_warnings}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return {"models": models, "ablations": variants, "folds": folds, "seeds": seeds, "n_fits": total, "n_rows": len(df), "summary": summary.astype(object).where(pd.notnull(summary), None).to_dict("records"), "warnings": training_warnings[:5], "manifest": manifest}
