"""Pairwise pharmacology and proteomics models with audited entity splits."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, matthews_corrcoef, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from .advanced_experiments import build_estimator, CLASSIFICATION_MODELS, REGRESSION_MODELS
from .dti import _canonical, _sequence, AA, AA_MAP

KINDS = ('ddi', 'ppi')
STRATEGIES = ('warm', 'cold_one', 'cold_both')
VARIANTS = ('joint', 'no_cross_features', 'similarity_only', 'entity_descriptors')
MAX_ROWS = 50000
MAX_BYTES = 32_000_000

SCHEMAS = {
    'ddi': ['drug_a', 'drug_b', 'label', 'drug_a_id', 'drug_b_id', 'interaction_type', 'source_id', 'evidence_type'],
    'ppi': ['protein_a', 'protein_b', 'label', 'protein_a_id', 'protein_b_id', 'protein_cluster_a', 'protein_cluster_b', 'interaction_type', 'source_id', 'evidence_type'],
}


def _val(v):
    if v is None or pd.isna(v):
        return ''
    return str(v).strip()


def _aa_vector(seq):
    a = np.zeros(20, np.float32)
    di = np.zeros(80, np.float32)
    if seq:
        for c in seq:
            a[AA_MAP[c]] += 1
        a /= len(seq)
        for x, y in zip(seq[:-1], seq[1:]):
            di[(AA_MAP[x] * 20 + AA_MAP[y]) % 80] += 1
        di /= max(1, len(seq) - 1)
    phys = np.array([
        math.log1p(len(seq)) / 12,
        float(sum(a[AA_MAP[x]] for x in 'AILMFWV')),
        float(sum(a[AA_MAP[x]] for x in 'KRH')),
        float(sum(a[AA_MAP[x]] for x in 'DE')),
    ], dtype=np.float32)
    return np.r_[a, di, phys]


def _smiles_vector(smiles, generator, bits):
    mol = Chem.MolFromSmiles(smiles)
    fp = np.zeros(bits, np.float32)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fp)
    desc = np.array([
        Descriptors.MolWt(mol) / 1000,
        Descriptors.MolLogP(mol) / 10,
        rdMolDescriptors.CalcTPSA(mol) / 200,
        rdMolDescriptors.CalcNumHBA(mol) / 20,
        rdMolDescriptors.CalcNumHBD(mol) / 15,
        rdMolDescriptors.CalcNumRotatableBonds(mol) / 30,
    ], dtype=np.float32)
    return np.r_[fp, desc]


def curate(frame, kind):
    if kind not in KINDS:
        raise ValueError('Interaction type must be DDI or PPI')
    a, b = ('drug_a', 'drug_b') if kind == 'ddi' else ('protein_a', 'protein_b')
    required = (a, b, 'label')
    if not set(required).issubset(frame):
        raise ValueError('Required CSV columns: ' + ', '.join(required))
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f'Upload 1–{MAX_ROWS:,} rows per dataset')
    for col in set(SCHEMAS[kind]) - set(frame):
        frame[col] = ''
    data = frame.copy().reset_index(drop=True)
    valid, excluded = [], []
    for n, r in enumerate(data.to_dict('records')):
        id1, id2 = _val(r.get(a+'_id')), _val(r.get(b+'_id'))
        p1, p2 = _val(r.get(a)), _val(r.get(b))
        if kind == 'ddi':
            p1, p2 = _canonical(p1), _canonical(p2)
            if not p1 or not p2:
                reason = 'missing_or_invalid_molecular_structure'
            else:
                reason = ''
            if id1 and id2 and id1 == id2 and p1 != p2:
                reason = 'contradictory_drug_identifier'
            id1, id2 = p1 or id1, p2 or id2
        else:
            # Real sequences are needed for predictive modeling, but a source-labeled
            # interaction graph may be explored without sequences.
            p1, p2 = _sequence(p1), _sequence(p2)
            if _val(r.get(a)) and not p1 or _val(r.get(b)) and not p2:
                reason = 'invalid_protein_sequence'
            elif not (p1 or id1) or not (p2 or id2):
                reason = 'missing_protein_identity'
            else:
                reason = ''
            id1, id2 = p1 or id1, p2 or id2
        if id1 == id2:
            reason = 'self_pair_not_supported'
        try:
            y = float(r['label'])
        except (TypeError, ValueError):
            y = float('nan')
        if not math.isfinite(y):
            reason = 'invalid_numeric_label'
        interaction = _val(r.get('interaction_type')) or 'unspecified'
        if reason:
            excluded.append({'row': n, 'reason': reason})
            continue
        # Both DDI and the modeled PPI networks are treated as undirected.
        # This prevents reversed-pair train/test leakage. Directional DDI effects
        # must be modeled in a separately specified directional dataset.
        if id1 > id2:
            id1, id2 = id2, id1
            p1, p2 = p2, p1
            cluster_a = _val(r.get('protein_cluster_b'))
            cluster_b = _val(r.get('protein_cluster_a'))
        else:
            cluster_a = _val(r.get('protein_cluster_a'))
            cluster_b = _val(r.get('protein_cluster_b'))
        valid.append({
            'entity_a': id1, 'entity_b': id2, 'repr_a': p1, 'repr_b': p2,
            'cluster_a': cluster_a or id1, 'cluster_b': cluster_b or id2,
            'label': y, 'interaction_type': interaction,
            'source_id': _val(r.get('source_id')),
            'evidence_type': _val(r.get('evidence_type')),
        })
    if not valid:
        raise ValueError('No valid observations remain after curation')
    records = []
    contradictions = 0
    ambiguous = 0
    for key, part in pd.DataFrame(valid).groupby(['entity_a','entity_b','interaction_type'], sort=True):
        ys = part.label.to_numpy(dtype=float)
        is_binary = np.isin(ys, [0,1]).all()
        if is_binary and len(np.unique(ys)) > 1:
            contradictions += 1
            excluded += [{'row': -1, 'reason': 'opposing_binary_labels'}]
            continue
        if not is_binary and len(ys) > 1 and (np.max(ys) - np.min(ys) > 1.0):
            ambiguous += 1
            excluded += [{'row': -1, 'reason': 'discordant_quantitative_measurements'}]
            continue
        row = part.iloc[0].to_dict()
        row['label'] = float(np.median(ys))
        row['n_sources'] = int(part.source_id.replace('', pd.NA).dropna().nunique())
        row['n_records'] = int(len(part))
        records.append(row)
    if not records:
        raise ValueError('All pairs excluded due to conflicting measurements')
    curated = pd.DataFrame.from_records(records).sort_values(['entity_a','entity_b','interaction_type']).reset_index(drop=True)
    nodes = set(curated.entity_a) | set(curated.entity_b)
    g = nx.Graph()
    g.add_edges_from(zip(curated.entity_a, curated.entity_b))
    sizes = sorted((len(x) for x in nx.connected_components(g)), reverse=True)
    valid_seqs = bool(kind == 'ddi' or ((curated.repr_a.astype(str).str.len() >= 30) & (curated.repr_b.astype(str).str.len() >= 30)).all())
    labeled = sorted(curated.interaction_type.unique().tolist())
    summary = {
        'kind': kind, 'raw_rows': int(len(data)), 'curated_pairs': int(len(curated)),
        'unique_entities': len(nodes), 'interaction_types': labeled[:30],
        'contradictory_pair_groups': contradictions, 'quantitative_disagreements': ambiguous,
        'excluded_rows': len(excluded), 'protein_sequence_complete': valid_seqs,
        'positive_only': bool((curated.label == 1).all()),
        'negative_only': bool((curated.label == 0).all()),
        'connected_components': len(sizes), 'largest_component_nodes': sizes[0] if sizes else 0,
        'assumptions': [
            'Pairs are treated as undirected. Directional DDI mechanisms cannot be represented.',
            'Missing PPI/DDI edges are unlabeled, not experimentally established negatives.',
            'A STRING functional association is not necessarily a physical interaction.',
            'Source, evidence and endpoint harmonization must be reviewed before publication.',
            'Cold-entity isolation uses exact structural/sequence identities or supplied cluster IDs.'
        ],
    }
    return curated, pd.DataFrame(excluded, columns=['row','reason']), summary


def features(frame, kind, bits=256, radius=2):
    if kind == 'ppi' and ((frame.repr_a.str.len() < 30) | (frame.repr_b.str.len() < 30)).any():
        raise ValueError('Protein sequences are required for PPI predictive modeling. Provide protein_a and protein_b amino-acid sequences; identifier-only networks support analysis, not model training.')
    gens = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits) if kind == 'ddi' else None
    unique = set(frame.repr_a) | set(frame.repr_b)
    cache = {x: (_smiles_vector(x, gens, bits) if kind == 'ddi' else _aa_vector(x)) for x in unique}
    A = np.vstack([cache[x] for x in frame.repr_a]); B = np.vstack([cache[x] for x in frame.repr_b])
    total = A + B
    diff = np.abs(A - B)
    prod = A * B
    cosine = ((A * B).sum(axis=1) / (np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-10)).reshape(-1,1)
    descriptors = 6 if kind == 'ddi' else 4
    return {
        'joint': np.column_stack([total,diff,prod,cosine]),
        'no_cross_features': np.column_stack([total,diff]),
        'similarity_only': np.column_stack([cosine, np.abs(A-B).mean(axis=1)]),
        'entity_descriptors': np.column_stack([A[:,-descriptors:]+B[:,-descriptors:],np.abs(A[:,-descriptors:]-B[:,-descriptors:])]),
    }


def make_splits(frame, strategy, folds, seed):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown evaluation strategy')
    if not 2 <= folds <= 10:
        raise ValueError('Choose 2–10 CV folds')
    a = frame.entity_a.to_numpy(); b = frame.entity_b.to_numpy()
    pairs = set(zip(a,b,frame.interaction_type))
    if len(pairs) != len(frame):
        raise ValueError('Duplicate pair and event type detected')
    # For multi-event datasets the same pair must always stay in the same partition.
    pair_ids = np.asarray([hashlib.sha256((x+'|'+y).encode()).hexdigest() for x,y in zip(a,b)])
    unique_pairs = np.unique(pair_ids)
    if strategy == 'warm':
        if len(unique_pairs) < folds:
            raise ValueError('Not enough distinct interaction pairs for warm CV')
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_pairs)
        part = {x: i % folds for i,x in enumerate(unique_pairs)}
        bins = np.asarray([part[x] for x in pair_ids])
        for fold in range(folds):
            fit = np.flatnonzero(bins != fold)
            candidate = np.flatnonzero(bins == fold)
            known = set(a[fit]) | set(b[fit])
            test = np.asarray([i for i in candidate if a[i] in known and b[i] in known], dtype=int)
            ignored = np.setdiff1d(candidate,test)
            yield fit,test,ignored
        return
    # User-supplied clusters are respected for protein homology-controlled partitioning.
    group_map = {}
    if 'cluster_a' in frame and 'cluster_b' in frame:
        for va, vb, ca, cb in zip(a,b,frame.cluster_a,frame.cluster_b):
            if va in group_map and group_map[va] != ca or vb in group_map and group_map[vb] != cb:
                raise ValueError('One protein has inconsistent cluster identifiers; fix cluster mapping')
            group_map[va], group_map[vb] = ca, cb
    else:
        group_map = {x: x for x in set(a)|set(b)}
    groups = np.unique([group_map[x] for x in set(a)|set(b)])
    if len(groups) < folds:
        raise ValueError('Too few independent entities or protein clusters for cold CV')
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    fold_map = {x: i%folds for i,x in enumerate(groups)}
    ga = np.array([fold_map[group_map[x]] for x in a]); gb=np.array([fold_map[group_map[x]] for x in b])
    ii=np.arange(len(frame))
    for fold in range(folds):
        new_a=ga==fold; new_b=gb==fold
        train=ii[~new_a & ~new_b]
        if strategy == 'cold_one':
            test=ii[new_a ^ new_b]
            ignored=ii[new_a & new_b]
        else:
            test=ii[new_a & new_b]
            ignored=ii[new_a ^ new_b]
        yield train,test,ignored


def check_split(frame, strategy, train,test, ignored):
    if len(train) < 12 or len(test) < 5:
        raise ValueError('Empty or underpopulated fold; choose fewer folds or a denser pairwise dataset')
    trainpairs=set(zip(frame.entity_a.iloc[train],frame.entity_b.iloc[train])); testpairs=set(zip(frame.entity_a.iloc[test],frame.entity_b.iloc[test]))
    if trainpairs & testpairs:
        raise RuntimeError('Pair leakage between train and test')
    tr=set(frame.entity_a.iloc[train]) | set(frame.entity_b.iloc[train])
    if strategy=='warm' and not (set(frame.entity_a.iloc[test])|set(frame.entity_b.iloc[test])).issubset(tr):
        raise RuntimeError('Warm-start test contains unseen entities')
    if strategy=='cold_one':
        for i in test:
            if (frame.entity_a.iloc[i] in tr)+(frame.entity_b.iloc[i] in tr) != 1:
                raise RuntimeError('Cold-one test pair does not have exactly one seen partner')
    if strategy=='cold_both' and any(frame.entity_a.iloc[i] in tr or frame.entity_b.iloc[i] in tr for i in test):
        raise RuntimeError('Cold-both test has training entity overlap')
    return {
        'train': len(train),'test':len(test),'excluded':len(ignored),
        'test_fraction_total':round(len(test)/len(frame),6),
        'shared_entities': len(tr & (set(frame.entity_a.iloc[test])|set(frame.entity_b.iloc[test]))),
    }


def validate(config, frame, kind):
    task=config.get('task','classification')
    if task not in ('classification','regression'):
        raise ValueError('Choose binary classification or regression')
    strategy=config.get('strategy','warm')
    if strategy not in STRATEGIES:
        raise ValueError('Invalid interaction validation protocol')
    folds=int(config.get('folds',5)); seeds=config.get('seeds',[17,29]); models=config.get('models',['random_forest']); variants=config.get('variants',['joint'])
    if not 2<=folds<=10 or not 1<=len(seeds)<=5 or len(set(seeds))!=len(seeds) or any(type(x)!=int or x<0 or x>2_000_000_000 for x in seeds):
        raise ValueError('Choose 2–10 folds and 1–5 distinct nonnegative integer seeds')
    allowed=CLASSIFICATION_MODELS if task=='classification' else REGRESSION_MODELS
    if not models or len(models)!=len(set(models)) or any(x not in allowed for x in models):
        raise ValueError('Unsupported estimator configuration')
    if not variants or len(variants)!=len(set(variants)) or any(x not in VARIANTS for x in variants):
        raise ValueError('Unsupported feature ablation')
    if folds*len(seeds)*len(models)*len(variants)>150:
        raise ValueError('Limit experiment to 150 outer-fold model fits')
    if len(frame)<max(50,folds*10):
        raise ValueError('At least 50 pairs required; larger datasets are preferred')
    if len(set(frame.interaction_type)) != 1:
        raise ValueError('Multiple interaction types detected. Filter one event type before model training; do not merge distinct biological outcomes.')
    if task=='classification' and (not np.isin(frame.label, [0,1]).all() or min((frame.label==0).sum(),(frame.label==1).sum()) < folds*2):
        raise ValueError('Binary classification requires both experimentally supported 0 and 1 labels, with at least two examples per class per fold. Unobserved pairs are not negatives.')
    if task=='regression' and frame.label.nunique() < 3:
        raise ValueError('Regression requires at least three distinct numeric values')
    if kind=='ppi' and not frame.repr_a.str.len().ge(30).all() or kind=='ppi' and not frame.repr_b.str.len().ge(30).all():
        raise ValueError('PPI predictive modeling requires protein sequences for both partners. Identifiers alone support network exploration.')
    if not 64 <= int(config.get('fp_bits',256)) <= 2048:
        raise ValueError('Fingerprint bits outside supported range')
    if not 1<=int(config.get('fp_radius',2))<=3:
        raise ValueError('Fingerprint radius outside supported range')
    if not 10<=int(config.get('n_estimators',80))<=500:
        raise ValueError('Ensemble size must be between 10 and 500')
    if not 1<=len(config.get('mlp_layers',[128,64]))<=3 or any(not 8<=v<=512 for v in config.get('mlp_layers',[128,64])):
        raise ValueError('Invalid ANN layers')
    if strategy!='warm':
        clusters=set(frame.cluster_a)|set(frame.cluster_b)
        if len(clusters)<folds:
            raise ValueError('Not enough independent entities/clusters for selected folds')
    return True


def _probability(model, x):
    if hasattr(model,'predict_proba'):
        p=model.predict_proba(x)
        return p[:,1] if p.shape[1]>1 else np.full(len(x),float(model.classes_[0]))
    if hasattr(model,'decision_function'):
        z=np.clip(model.decision_function(x),-20,20)
        return 1/(1+np.exp(-z))
    return None


def _metric(y, pred, task, prob=None):
    with np.errstate(all='ignore'):
        if task=='regression':
            return {'mae':float(mean_absolute_error(y,pred)), 'rmse':float(np.sqrt(mean_squared_error(y,pred))),
                    'r2':float(r2_score(y,pred)) if len(y)>1 else None}
        y=np.asarray(y).astype(int); pred=np.asarray(pred).astype(int)
        m={'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if len(np.unique(y))>1 else None,
           'f1':float(f1_score(y,pred,zero_division=0)),'mcc':float(matthews_corrcoef(y,pred)),
           'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0))}
        if prob is not None:
            m['brier']=float(brier_score_loss(y,prob))
            m['roc_auc']=float(roc_auc_score(y,prob)) if len(np.unique(y))>1 else None
            m['average_precision']=float(average_precision_score(y,prob)) if len(np.unique(y))>1 else None
        return m


def network_summary(frame, max_nodes=100):
    g=nx.Graph()
    for r in frame.itertuples(index=False):
        if g.has_edge(r.entity_a,r.entity_b):
            g[r.entity_a][r.entity_b]['count']+=1
        else:
            g.add_edge(r.entity_a,r.entity_b,count=1)
    deg=sorted(g.degree,key=lambda kv:kv[1],reverse=True)
    nodes=[{'id':str(node)[:80], 'degree':int(value)} for node,value in deg[:max_nodes]]
    chosen={n['id'] for n in nodes}
    # Display labels may be truncated, so use untruncated keys while selecting edges.
    original={node for node,_ in deg[:max_nodes]}
    edges=[{'source':str(a)[:80],'target':str(b)[:80]} for a,b in g.edges() if a in original and b in original][:300]
    comps=list(nx.connected_components(g))
    return {'nodes':int(g.number_of_nodes()), 'edges':int(g.number_of_edges()),
            'connected_components':len(comps), 'largest_component':max((len(c) for c in comps),default=0),
            'density':round(nx.density(g),6), 'top_hubs':nodes[:15], 'preview_nodes':nodes,'preview_edges':edges,
            'note':'Descriptive graph of recorded pairs only; absence of an edge is not verified non-interaction.'}


def run(frame, kind, config, output, progress=lambda msg:None):
    validate(config,frame,kind)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    cfg={**config,'kind':kind}
    variants=features(frame,kind,bits=int(cfg.get('fp_bits',256)),radius=int(cfg.get('fp_radius',2)))
    models=cfg.get('models',['random_forest']); selected=cfg.get('variants',['joint'])
    seeds=cfg.get('seeds',[17,29]); folds=int(cfg.get('folds',5)); strategy=cfg.get('strategy','warm'); task=cfg.get('task','classification')
    y=frame.label.to_numpy(dtype=float)
    if task=='classification': y=y.astype(int)
    fold_rows=[]; pred_rows=[]; split_rows=[]
    total=len(seeds)*folds*len(models)*len(selected);count=0
    for seed in seeds:
        for fid,(train,test,excluded) in enumerate(make_splits(frame,strategy,folds,seed),start=1):
            audit=check_split(frame,strategy,train,test,excluded)
            for category,ids in [('train',train),('test',test),('excluded',excluded)]:
                for j in ids:
                    split_rows.append({'seed':seed,'fold':fid,'role':category, 'row':int(j),
                                       'entity_a':frame.entity_a.iloc[j], 'entity_b':frame.entity_b.iloc[j]})
            if task=='classification' and len(set(y[train]))!=2:
                raise ValueError(f'Seed {seed}, fold {fid}: training data has only one class; choose fewer folds or a better-balanced dataset')
            for variant in selected:
                x=variants[variant]
                for name in models:
                    count+=1;progress(f'Fit {count}/{total}: {name} · {variant} · seed {seed} fold {fid}')
                    model=build_estimator(name,task,seed+fid,cfg)
                    model.fit(x[train],y[train])
                    predictions=model.predict(x[test]); prob=_probability(model,x[test]) if task=='classification' else None
                    scores=_metric(y[test],predictions,task,prob)
                    fold_rows.append({'seed':seed,'fold':fid,'model':name,'ablation':variant,'protocol':strategy,'n_train':len(train),'n_test':len(test),
                                      'n_excluded':len(excluded),'test_fraction_total':audit['test_fraction_total'],
                                      **scores})
                    for k,idx in enumerate(test):
                        pred_rows.append({'seed':seed,'fold':fid,'model':name,'ablation':variant,'row':int(idx),
                                          'entity_a':frame.entity_a.iloc[idx],'entity_b':frame.entity_b.iloc[idx],
                                          'observed':float(y[idx]),'predicted':float(predictions[k]),
                                          'prob_positive':float(prob[k]) if prob is not None else None})
    fold_data=pd.DataFrame(fold_rows)
    per_seed=fold_data.groupby(['seed','model','ablation'],as_index=False).mean(numeric_only=True)
    summary=per_seed.groupby(['model','ablation'],as_index=False).agg({
        k:['mean','std'] for k in ('f1','roc_auc','average_precision','mcc','mae','rmse','r2') if k in per_seed
    })
    summary.columns=['_'.join([str(z) for z in c if z]) for c in summary.columns.to_flat_index()]
    summary=summary.replace({np.nan:None})
    fold_data.to_csv(output/'fold_metrics.csv',index=False)
    per_seed.to_csv(output/'per_seed_metrics.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(output/'out_of_fold_predictions.csv',index=False)
    pd.DataFrame(split_rows).to_csv(output/'split_assignments.csv',index=False)
    net=network_summary(frame)
    safe_graph={k:v for k,v in net.items() if not k.startswith('preview_')}
    manifest={
        'kind':kind,'task':task,'config':cfg,'input_rows':len(frame), 'n_fits':count,
        'reproducibility':{'rdkit':rdBase.rdkitVersion,'sklearn':sklearn_version,
                           'run_utc':datetime.now(timezone.utc).isoformat()},
        'network':safe_graph,
        'scope':'Associations or labels from source data; not clinical interaction advice.',
        'limitations':['Only 2D fingerprints and sequence descriptors; no validated structure-based learned PPI embeddings.',
                        'Raw model hyperparameters fixed before CV; results are not from nested optimization.',
                        'Repeated folds/seeds are correlated, and seed SD is descriptive.',
                        'Cold-both cross-block pairs are omitted; evaluate retained test fraction.',
                        'The absence of an edge is never converted to a confirmed negative.',
                        'Biological duplicates and close analogs/homologs may remain beyond exact identity checks.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf8')
    graph_to_png(frame,output/'network.png')
    return {'summary':summary.to_dict('records'),'n_fits':count,'graph':net}


def graph_to_png(frame, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g=nx.Graph()
    for a,b in zip(frame.entity_a,frame.entity_b):g.add_edge(a,b)
    if not g.number_of_edges():return
    top=sorted(g.degree,key=lambda x:x[1],reverse=True)[:70]
    sub=g.subgraph([x for x,_ in top]).copy()
    p=nx.spring_layout(sub,seed=41,iterations=60)
    fig,ax=plt.subplots(figsize=(9,6))
    fig.patch.set_facecolor('#08131e');ax.set_facecolor('#08131e')
    nx.draw_networkx_edges(sub,p,ax=ax,alpha=.3,edge_color='#78a9b8',width=.8)
    nx.draw_networkx_nodes(sub,p,ax=ax,node_color='#44c9b8',node_size=[30+13*sub.degree(x) for x in sub],alpha=.9)
    ax.set_axis_off();ax.set_title('Observed interaction network · highest-degree entities',color='#e0eef5',fontsize=13,loc='left')
    fig.tight_layout();fig.savefig(dest,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
