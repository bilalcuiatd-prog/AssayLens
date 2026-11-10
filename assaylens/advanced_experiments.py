from __future__ import annotations

import hashlib
import inspect
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
    ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
    RandomForestRegressor)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, log_loss, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .experiments import fingerprints_and_descriptors, select_features, sample_weights_train, split_calibration, conformal

REGRESSION_MODELS = ('extra_trees', 'random_forest', 'hist_gradient', 'gradient_boosting',
    'ada_boost', 'ridge', 'elastic_net', 'svr', 'knn', 'ann_mlp', 'sgd',
    'xgboost', 'lightgbm', 'catboost')
CLASSIFICATION_MODELS = ('extra_trees', 'random_forest', 'hist_gradient', 'gradient_boosting',
    'ada_boost', 'logistic', 'ridge_classifier', 'svm', 'knn', 'naive_bayes', 'ann_mlp',
    'sgd', 'xgboost', 'lightgbm', 'catboost')
VARIANTS = ('fingerprint_only', 'descriptors_only', 'chemistry', 'chemistry_assay', 'quality_weighted')


def build_estimator(model, task, seed, config):
    classifier = task == 'classification'
    n = int(config.get('n_estimators', 100))
    leaf = int(config.get('min_samples_leaf', 2))
    depth = config.get('max_depth') or None
    if model == 'extra_trees':
        klass = ExtraTreesClassifier if classifier else ExtraTreesRegressor
        return klass(n_estimators=n, min_samples_leaf=leaf, max_depth=depth, random_state=seed, n_jobs=1)
    if model == 'random_forest':
        klass = RandomForestClassifier if classifier else RandomForestRegressor
        return klass(n_estimators=n, min_samples_leaf=leaf, max_depth=depth, random_state=seed, n_jobs=1)
    if model == 'hist_gradient':
        klass = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        return klass(max_iter=n, max_leaf_nodes=15, random_state=seed)
    if model == 'gradient_boosting':
        klass = GradientBoostingClassifier if classifier else GradientBoostingRegressor
        return klass(n_estimators=n, random_state=seed)
    if model == 'ada_boost':
        klass = AdaBoostClassifier if classifier else AdaBoostRegressor
        return klass(n_estimators=n, random_state=seed)
    if model == 'ann_mlp':
        klass = MLPClassifier if classifier else MLPRegressor
        return make_pipeline(StandardScaler(), klass(hidden_layer_sizes=tuple(config.get('mlp_layers',[128,64])),
            alpha=float(config.get('mlp_alpha', .001)), learning_rate_init=float(config.get('mlp_lr',.001)),
            max_iter=int(config.get('mlp_max_iter',150)), random_state=seed))
    if model == 'knn':
        klass = KNeighborsClassifier if classifier else KNeighborsRegressor
        return make_pipeline(StandardScaler(), klass(n_neighbors=int(config.get('knn_neighbors', 5))))
    if model == 'sgd':
        if classifier:
            return make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', alpha=.0001, random_state=seed, max_iter=1500))
        return make_pipeline(StandardScaler(), SGDRegressor(random_state=seed, max_iter=1500))
    if model == 'xgboost':
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as ex:
            raise ValueError('Install optional dependency: pip install xgboost') from ex
        klass = XGBClassifier if classifier else XGBRegressor
        return klass(n_estimators=n, max_depth=min(depth or 6, 12), learning_rate=float(config.get('boost_lr',.05)),
            tree_method='hist', n_jobs=1, random_state=seed, eval_metric='logloss' if classifier else 'rmse')
    if model == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as ex:
            raise ValueError('Install optional dependency: pip install lightgbm') from ex
        klass = LGBMClassifier if classifier else LGBMRegressor
        return klass(n_estimators=n, learning_rate=float(config.get('boost_lr',.05)), random_state=seed, n_jobs=1, verbosity=-1)
    if model == 'catboost':
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as ex:
            raise ValueError('Install optional dependency: pip install catboost') from ex
        klass = CatBoostClassifier if classifier else CatBoostRegressor
        return klass(iterations=n, learning_rate=float(config.get('boost_lr',.05)), random_seed=seed, verbose=False, thread_count=1)
    if classifier:
        if model == 'logistic':
            return make_pipeline(StandardScaler(), LogisticRegression(C=float(config.get('svr_c',10)),max_iter=1500,random_state=seed))
        if model == 'ridge_classifier':
            return make_pipeline(StandardScaler(), RidgeClassifier(alpha=float(config.get('regularization',1))))
        if model == 'svm':
            return make_pipeline(StandardScaler(), SVC(C=float(config.get('svr_c',10)),probability=True,random_state=seed))
        if model == 'naive_bayes':
            return GaussianNB()
    else:
        if model == 'ridge':
            return make_pipeline(StandardScaler(), Ridge(alpha=float(config.get('regularization',1))))
        if model == 'elastic_net':
            return make_pipeline(StandardScaler(), ElasticNet(alpha=float(config.get('regularization',1)),l1_ratio=.5,random_state=seed,max_iter=2000))
        if model == 'svr':
            return make_pipeline(StandardScaler(), SVR(C=float(config.get('svr_c',10)),epsilon=float(config.get('svr_epsilon',.1))))
    raise ValueError(f'Unsupported {task} model: {model}')


def validate(config, df):
    task = config.get('task','regression')
    if task not in ('regression','classification'):
        raise ValueError('Task must be regression or classification')
    models = config.get('models',['random_forest'])
    allowed = CLASSIFICATION_MODELS if task == 'classification' else REGRESSION_MODELS
    variants = config.get('variants',['chemistry'])
    seeds = config.get('seeds',[17])
    folds = int(config.get('folds',5))
    if not models or len(models)!=len(set(models)) or any(m not in allowed for m in models):
        raise ValueError('Select valid distinct models for the chosen task')
    if not variants or len(variants)!=len(set(variants)) or any(v not in VARIANTS for v in variants):
        raise ValueError('Select valid distinct ablations')
    if not (2 <= folds <= 10 and 1 <= len(seeds) <= 5 and len(seeds)==len(set(seeds))
            and all(type(s) is int and 0<=s<=2_000_000_000 for s in seeds)):
        raise ValueError('Use 2–10 folds and 1–5 distinct integer seeds')
    if folds*len(seeds)*len(models)*len(variants)>200:
        raise ValueError('Maximum 200 model fits per experiment')
    if not 32 <= config.get('fp_bits',512) <= 2048 or config.get('fp_bits',512)&(config.get('fp_bits',512)-1):
        raise ValueError('Fingerprint size must be a power of two between 32 and 2048')
    if not 1 <= int(config.get('fp_radius',2)) <= 3:
        raise ValueError('Fingerprint radius must be 1–3')
    if not 16 <= int(config.get('n_estimators',100)) <= 500:
        raise ValueError('Ensemble size out of range')
    if not 2<=int(config.get('knn_neighbors',5))<=100:
        raise ValueError('KNN neighbors must be 2–100')
    if not 1<=len(config.get('mlp_layers',[128,64]))<=3 or any(not 8<=v<=512 for v in config.get('mlp_layers',[128,64])):
        raise ValueError('ANN layer settings out of range')
    if config.get('strategy','scaffold') not in ('scaffold','molecule','assay'):
        raise ValueError('Choose scaffold, molecule, or assay grouping')
    group_col = {'scaffold':'scaffold','molecule':'parent_smiles','assay':'assay_id'}[config.get('strategy','scaffold')]
    group = df[group_col].fillna('').astype(str)
    if group.nunique()<folds or (group_col=='assay_id' and group.isin(['UNKNOWN','','nan']).any()):
        raise ValueError(f'Not enough genuine {group_col} groups for {folds}-fold evaluation')
    if len(df)<max(35, folds*8):
        raise ValueError('Too few curated rows for meaningful evaluation')
    if df.target_id.nunique()>1 or df.endpoint.nunique()>1:
        raise ValueError('Train one target and one endpoint at a time')
    if not .05 <= float(config.get('alpha',.10)) <= .30:
        raise ValueError('Regression interval alpha must be between .05 and .30')
    if task=='classification':
        threshold=float(config.get('activity_threshold',6.0))
        if not math.isfinite(threshold):
            raise ValueError('Activity threshold must be finite')
        y=(df.response.to_numpy(dtype=float)>=threshold).astype(int)
        if min(np.bincount(y,minlength=2))<folds:
            raise ValueError('Too few positive or negative classes for the requested fold count')
    return task,models,variants,folds,seeds


def fit_with_weights(estimator, x, y, weights=None):
    if weights is None:
        estimator.fit(x,y)
        return
    final = estimator.steps[-1][1] if hasattr(estimator,'steps') else estimator
    if 'sample_weight' not in inspect.signature(final.fit).parameters:
        raise ValueError(f'{type(final).__name__} does not support quality weighting; deselect weighted ablation')
    if hasattr(estimator,'steps'):
        estimator.fit(x,y,**{estimator.steps[-1][0]+'__sample_weight':weights})
    else:
        estimator.fit(x,y,sample_weight=weights)


def run(df, cfg, folder, progress=lambda msg:None):
    task, models, variants, folds, seeds=validate(cfg,df)
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    y_raw=df.response.to_numpy(dtype=float)
    y=(y_raw>=float(cfg.get('activity_threshold',6.0))).astype(int) if task=='classification' else y_raw
    fp,desc=fingerprints_and_descriptors(df.parent_smiles.astype(str).tolist(),int(cfg.get('fp_radius',2)),int(cfg.get('fp_bits',512)))
    meta=df[['assay_type','assay_format']].fillna('').astype(str)
    group_name={'scaffold':'scaffold','molecule':'parent_smiles','assay':'assay_id'}[cfg.get('strategy','scaffold')]
    groups=df[group_name].astype(str).to_numpy()
    records=[]; predictions=[]; warnings_log=[]
    total=folds*len(seeds)*len(models)*len(variants); completed=0
    for seed in seeds:
        if task=='classification':
            splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
            splits=splitter.split(np.zeros(len(df)),y,groups)
        else:
            splitter=GroupKFold(n_splits=folds,shuffle=True,random_state=seed)
            splits=splitter.split(np.zeros(len(df)),groups=groups)
        for fold,(train,test) in enumerate(splits,1):
            if set(groups[train])&set(groups[test]):
                raise RuntimeError('Group leakage detected')
            if min(len(train),len(test))<5:
                raise ValueError(f'Seed {seed}, fold {fold}: too few independent observations')
            if task=='classification' and len(set(y[train]))<2:
                raise ValueError(f'Seed {seed}, fold {fold}: training fold contains only one class')
            if task=='regression':
                fit_idx,cal_idx=split_calibration(train,groups,seed,fold)
                if min(len(fit_idx),len(cal_idx))<8:
                    raise ValueError('Too few independent calibration observations; reduce folds or use a larger dataset')
            else:
                fit_idx,cal_idx=train,None
            for variant in variants:
                features=select_features(fp,desc,meta,fit_idx,variant)
                weights=sample_weights_train(df,fit_idx) if variant=='quality_weighted' else None
                for model_name in models:
                    estimator=build_estimator(model_name,task,seed+fold,cfg)
                    with warnings.catch_warnings(record=True) as ww:
                        warnings.simplefilter('always')
                        fit_with_weights(estimator,features[fit_idx],y[fit_idx],weights)
                    warnings_log.extend(str(w.message)[:200] for w in ww[:2])
                    predicted=estimator.predict(features[test])
                    row={'task':task,'model':model_name,'ablation':variant,'seed':seed,'fold':fold,
                        'train_n':len(fit_idx),'calibration_n':len(cal_idx) if cal_idx is not None else 0,'test_n':len(test),'positive_rate_test':float(y[test].mean()) if task=='classification' else np.nan,
                        'molecule_overlap':float(len(set(df.parent_smiles.iloc[train])&set(df.parent_smiles.iloc[test]))/max(1,df.parent_smiles.iloc[test].nunique()))}
                    if task=='regression':
                        cal_pred=estimator.predict(features[cal_idx])
                        radius=conformal(y[cal_idx],cal_pred,float(cfg.get('alpha',.1)))
                        row.update(mae=float(mean_absolute_error(y[test],predicted)), rmse=float(np.sqrt(mean_squared_error(y[test],predicted))),
                            r2=float(r2_score(y[test],predicted)) if len(test)>1 else np.nan,
                            coverage=float(np.mean(np.abs(y[test]-predicted)<=radius)) if radius is not None else np.nan,
                            interval_width=2*radius if radius is not None else np.nan)
                        prob=None
                    else:
                        radius=None
                        prob=None
                        if hasattr(estimator,'predict_proba'):
                            pp=estimator.predict_proba(features[test])
                            prob=pp[:,list(estimator.classes_).index(1)] if hasattr(estimator,'classes_') and 1 in estimator.classes_ else pp[:,1]
                        row.update(accuracy=float(accuracy_score(y[test],predicted)),balanced_accuracy=float(balanced_accuracy_score(y[test],predicted)),
                            f1=float(f1_score(y[test],predicted,zero_division=0)),precision=float(precision_score(y[test],predicted,zero_division=0)),
                            recall=float(recall_score(y[test],predicted,zero_division=0)),mcc=float(matthews_corrcoef(y[test],predicted)),
                            roc_auc=float(roc_auc_score(y[test],prob)) if prob is not None and len(set(y[test]))==2 else np.nan,
                            auprc=float(average_precision_score(y[test],prob)) if prob is not None and len(set(y[test]))==2 else np.nan,
                            brier=float(brier_score_loss(y[test],prob)) if prob is not None else np.nan)
                    records.append(row)
                    for j,idx in enumerate(test):
                        predictions.append({'task':task,'model':model_name,'ablation':variant,'seed':seed,'fold':fold,
                            'source_row':int(idx),'true_response':float(y_raw[idx]),'observed':float(y[idx]),
                            'predicted':float(predicted[j]),'probability_active':float(prob[j]) if prob is not None else np.nan,
                            'lower':float(predicted[j]-radius) if radius is not None else np.nan,
                            'upper':float(predicted[j]+radius) if radius is not None else np.nan,
                            'assay_id':str(df.iloc[idx].assay_id),'scaffold_sha256':hashlib.sha256(str(df.iloc[idx].scaffold).encode()).hexdigest()[:12]})
                    completed+=1
                    if completed==1 or completed%5==0 or completed==total:
                        progress(f'{completed}/{total} model fits finished')
    detail=pd.DataFrame(records);preds=pd.DataFrame(predictions)
    detail.to_csv(out/'fold_metrics.csv',index=False)
    preds.to_csv(out/'out_of_fold_predictions.csv',index=False)
    metric_names=['mae','rmse','r2','coverage','interval_width'] if task=='regression' else ['accuracy','balanced_accuracy','f1','precision','recall','mcc','roc_auc','auprc','brier']
    per_seed=detail.groupby(['model','ablation','seed'],as_index=False)[metric_names].mean()
    per_seed.to_csv(out/'per_seed_metrics.csv',index=False)
    summary=per_seed.groupby(['model','ablation'],as_index=False)[metric_names].agg(['mean','std'])
    summary.columns=['model','ablation']+[f'{metric}_{stat}' for metric in metric_names for stat in ('mean','std')]
    summary.to_csv(out/'summary.csv',index=False)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'task':task,'config':cfg,'rdkit':rdBase.rdkitVersion,
        'scikit_learn':sklearn_version,'model_fits':total,'measurement_interpretation': 'pActivity >= threshold gives active class' if task=='classification' else 'continuous harmonized pActivity',
        'validation':'group-disjoint repeated k-fold; regression calibration uses disjoint inner groups; seeds are not independent biological replicates',
        'limitations':['Only binary classification is implemented','AUC/AUPRC are unavailable in a test fold with one class','Hyperparameters are not selected by nested CV','Assay split can reuse molecular identities; inspect overlap','Real database measurements and target assignments require source-level review'],
        'warnings':warnings_log[:30]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    clean_summary=summary.replace({np.nan:None}).to_dict('records')
    return {'summary':clean_summary,'n_fits':total,'metric': 'f1_mean' if task=='classification' else 'mae_mean','warnings':warnings_log[:5], 'task':task}
