"""Interaction-network import, inference benchmarking, and provenance endpoints."""
from __future__ import annotations

import hashlib
import io
import json
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from . import interactions

router = APIRouter(prefix='/api/v4', tags=['DDI / PPI Research'])
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='pair-research')
LOCK = threading.Lock()
JOBS = {}
STORAGE = None
PROJECT = None


class PairConfig(BaseModel):
    dataset_id: str
    task: Literal['classification', 'regression'] = 'classification'
    strategy: Literal['warm','cold_one','cold_both'] = 'warm'
    folds: int = Field(5, ge=2, le=10)
    seeds: list[int] = Field(default_factory=lambda:[17,29])
    models: list[str] = Field(default_factory=lambda:['random_forest'])
    variants: list[str] = Field(default_factory=lambda:['joint'])
    fp_bits: int = Field(256, ge=64, le=2048)
    fp_radius: int = Field(2, ge=1, le=3)
    n_estimators: int = Field(80, ge=10, le=500)
    min_samples_leaf: int = Field(2, ge=1, le=20)
    mlp_layers: list[int] = Field(default_factory=lambda:[128,64])
    mlp_max_iter: int = Field(125, ge=25, le=500)
    mlp_lr: float = Field(0.001, gt=0, le=.1)
    mlp_alpha: float = Field(0.001, gt=0, le=10)
    regularization: float = Field(1.0, gt=0, le=1000)
    svr_c: float = Field(10.0, gt=0, le=500)
    knn_neighbors: int = Field(5, ge=2, le=100)


class StringRequest(BaseModel):
    genes: list[str]
    species: int = Field(9606, ge=1)
    network_type: Literal['physical','functional'] = 'physical'
    required_score: int = Field(700, ge=0, le=1000)


def safe_dir(kind, folder, identifier):
    if kind not in interactions.KINDS or folder not in ('datasets','runs') or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
        raise FileNotFoundError('Invalid interaction record')
    p=STORAGE/'networks'/kind/folder/identifier
    if not p.is_dir():raise FileNotFoundError('Interaction record not found')
    return p


def save_data(kind, contents, provenance='CSV upload'):
    if kind not in interactions.KINDS:
        raise ValueError('Unsupported interaction type')
    if len(contents)>interactions.MAX_BYTES:
        raise ValueError('CSV exceeds the 32 MB size limit')
    frame=pd.read_csv(io.BytesIO(contents))
    clean, rejected, audit=interactions.curate(frame,kind)
    ident=uuid.uuid4().hex
    dest=STORAGE/'networks'/kind/'datasets'/ident
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'raw.csv').write_bytes(contents)
    clean.to_csv(dest/'curated.csv',index=False)
    rejected.to_csv(dest/'excluded.csv',index=False)
    report={**audit,'id':ident,'kind':kind,'origin':provenance,
            'sha256':hashlib.sha256(contents).hexdigest(),
            'imported_utc':datetime.now(timezone.utc).isoformat(),
            'synthetic_demo':bool(frame.get('source_id',pd.Series(dtype=str)).astype(str).str.startswith('SYNTHETIC_').any())}
    (dest/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def dataset(kind, identifier):
    dest=safe_dir(kind,'datasets',identifier)
    audit=json.loads((dest/'audit.json').read_text())
    if hashlib.sha256((dest/'raw.csv').read_bytes()).hexdigest()!=audit['sha256']:
        raise ValueError('Source checksum mismatch')
    return dest,audit,pd.read_csv(dest/'curated.csv').fillna('')


def make_job(kind, action):
    ident=uuid.uuid4().hex
    with LOCK:
        JOBS[ident]={'job_id':ident,'kind':kind,'action':action,'state':'queued','message':'Queued'}
    return ident


def job_update(ident, **fields):
    with LOCK:
        if ident in JOBS:JOBS[ident].update(fields)


def model_task(job, kind, req):
    job_update(job,state='running',message='Loading curated interaction records')
    try:
        original,audit,frame=dataset(kind,req.dataset_id)
        config=req.model_dump()
        interactions.validate(config,frame,kind)
        run_id=uuid.uuid4().hex
        output=STORAGE/'networks'/kind/'runs'/run_id
        result=interactions.run(frame,kind,config,output,lambda m:job_update(job,message=m))
        (output/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
        frame.to_csv(output/'curated_pairs.csv',index=False)
        index={'id':run_id,'kind':kind,'task':req.task,'strategy':req.strategy,'folds':req.folds,'seeds':req.seeds,
               'n_fits':result['n_fits'],'dataset_id':req.dataset_id,'summary':result['summary'],
               'synthetic_demo':audit.get('synthetic_demo',False)}
        (output/'result.json').write_text(json.dumps(index,indent=2),encoding='utf8')
        with zipfile.ZipFile(output/'research_package.zip','w',zipfile.ZIP_DEFLATED) as z:
            for child in output.iterdir():
                if child.is_file() and child.name!='research_package.zip':z.write(child,child.name)
        job_update(job,state='complete',message='Experiment finished',run_id=run_id,metrics=result['summary'],n_fits=result['n_fits'])
    except Exception as exc:
        job_update(job,state='error',message=f'{type(exc).__name__}: {str(exc)[:400]}')


def string_task(job, req):
    job_update(job,state='running',message='Requesting STRING interactions')
    try:
        genes=[x.strip() for x in req.genes if x.strip()]
        if not 2<=len(genes)<=30 or any(not x.replace('-','').replace('_','').isalnum() for x in genes):
            raise ValueError('Supply 2–30 unambiguous protein identifiers')
        payload={'identifiers':'\r'.join(genes),'species':req.species,'network_type':req.network_type,
                 'required_score':req.required_score,'caller_identity':'AssayLensResearchWorkbench'}
        response=requests.post('https://string-db.org/api/tsv/network',data=payload,timeout=75,headers={'User-Agent':'AssayLensResearch/4.0'})
        response.raise_for_status()
        table=pd.read_csv(io.StringIO(response.text),sep='\t')
        if not {'stringId_A','stringId_B','score'}.issubset(table):
            raise ValueError('STRING network API returned unexpected schema')
        records=[]
        for row in table.to_dict('records'):
            records.append({'protein_a_id':row['stringId_A'],'protein_b_id':row['stringId_B'],
                            'protein_a':'','protein_b':'','label':1,
                            'interaction_type':f'STRING_{req.network_type}_association',
                            'evidence_type':f'STRING score {row.get("score")}; mixed evidence, not experimentally confirmed',
                            'source_id':'STRING_API'})
        if not records:
            raise ValueError('No STRING interactions returned for requested protein identifiers')
        report=save_data('ppi',pd.DataFrame(records).to_csv(index=False).encode(),provenance={'source':'STRING API', 'network_type':req.network_type,'required_score':req.required_score,'species':req.species,'queried':genes})
        job_update(job,state='complete',message='STRING association network imported. Sequence and verified negatives are missing: predictive training is unavailable until supplied.',dataset_id=report['id'],audit=report)
    except Exception as exc:
        job_update(job,state='error',message=f'{type(exc).__name__}: {str(exc)[:400]}')


@router.get('/pairs/catalog')
def catalog():
    return {'kinds':list(interactions.KINDS),
            'protocols':[{'id':'warm','name':'Warm start','meaning':'Unseen interaction pairs; both entities represented in training'},
                         {'id':'cold_one','name':'Cold one partner','meaning':'Exactly one entity in each test pair is unseen in training'},
                         {'id':'cold_both','name':'Cold both partners','meaning':'Both entities unseen; cross-block interactions excluded and reported'}],
            'models':{'classification':list(interactions.CLASSIFICATION_MODELS),'regression':list(interactions.REGRESSION_MODELS)},
            'variants':list(interactions.VARIANTS),'schemas':interactions.SCHEMAS,
            'sources':[{'id':'string','kind':'ppi','availability':'API network import','limitations':'Association scores and positive-only labels; sequence mapping and verified negatives needed for classification'},
                       {'id':'biogrid','kind':'ppi','availability':'Upload curated CSV','homepage':'https://downloads.thebiogrid.org/BioGRID'},
                       {'id':'intact','kind':'ppi','availability':'Upload curated CSV','homepage':'https://www.ebi.ac.uk/intact/'},
                       {'id':'tdc','kind':'ddi','availability':'Import licensed/appropriately permitted dataset CSV','homepage':'https://tdcommons.ai/multi_pred_tasks/ddi/'}]}


@router.post('/pairs/{kind}/datasets',status_code=201)
async def upload(kind:Literal['ddi','ppi'],file:UploadFile=File(...)):
    if not (file.filename or '').lower().endswith('.csv'):
        raise HTTPException(400,'Please upload a CSV')
    try:return save_data(kind,await file.read(interactions.MAX_BYTES+1))
    except Exception as exc:raise HTTPException(422,str(exc)[:400])


@router.get('/pairs/{kind}/datasets')
def datasets(kind:Literal['ddi','ppi']):
    folder=STORAGE/'networks'/kind/'datasets'
    if not folder.exists():return []
    result=[]
    for child in sorted(folder.iterdir(),key=lambda x:x.stat().st_mtime,reverse=True):
        if child.is_dir() and (child/'audit.json').is_file():
            result.append(json.loads((child/'audit.json').read_text()))
    return result[:150]


@router.get('/pairs/{kind}/datasets/{ident}/csv')
def csv_export(kind:Literal['ddi','ppi'],ident:str):
    try:return FileResponse(safe_dir(kind,'datasets',ident)/'curated.csv',filename=f'assaylens_{kind}_curated.csv')
    except FileNotFoundError:raise HTTPException(404,'Dataset not found')


@router.get('/pairs/{kind}/datasets/{ident}/excluded.csv')
def excluded_export(kind:Literal['ddi','ppi'],ident:str):
    try:return FileResponse(safe_dir(kind,'datasets',ident)/'excluded.csv',filename=f'assaylens_{kind}_excluded.csv')
    except FileNotFoundError:raise HTTPException(404,'Dataset not found')


@router.get('/pairs/{kind}/datasets/{ident}/graph')
def graph_data(kind:Literal['ddi','ppi'],ident:str):
    try:
        _,audit,frame=dataset(kind,ident)
        return {'audit':audit,'network':interactions.network_summary(frame)}
    except (FileNotFoundError,ValueError) as exc:raise HTTPException(404,str(exc))


@router.post('/pairs/{kind}/experiments',status_code=202)
def create_experiment(kind:Literal['ddi','ppi'],req:PairConfig):
    try:
        _,_,frame=dataset(kind,req.dataset_id)
        interactions.validate(req.model_dump(),frame,kind)
        # Exercise partition generator before submitting potentially expensive training.
        for tr,te,sk in interactions.make_splits(frame,req.strategy,req.folds,req.seeds[0]):
            interactions.check_split(frame,req.strategy,tr,te,sk)
    except (FileNotFoundError,ValueError,RuntimeError) as exc:raise HTTPException(422,str(exc)[:400])
    ident=make_job(kind,'experiment')
    EXECUTOR.submit(model_task,ident,kind,req)
    return {'job_id':ident}


@router.get('/pairs/{kind}/runs')
def runs(kind:Literal['ddi','ppi']):
    base=STORAGE/'networks'/kind/'runs'
    if not base.exists():return []
    return [json.loads((p/'result.json').read_text()) for p in base.iterdir() if p.is_dir() and (p/'result.json').is_file()][:100]


@router.get('/pairs/{kind}/runs/{ident}/bundle')
def bundle(kind:Literal['ddi','ppi'],ident:str):
    try:return FileResponse(safe_dir(kind,'runs',ident)/'research_package.zip',filename=f'assaylens_{kind}_research_package.zip')
    except FileNotFoundError:raise HTTPException(404,'Experiment not found')


@router.get('/pairs/{kind}/runs/{ident}/network.png')
def image(kind:Literal['ddi','ppi'],ident:str):
    try:return FileResponse(safe_dir(kind,'runs',ident)/'network.png',media_type='image/png')
    except FileNotFoundError:raise HTTPException(404,'Experiment not found')


@router.get('/pairs/{kind}/example.csv')
def example(kind:Literal['ddi','ppi']):
    return FileResponse(PROJECT/'examples'/f'synthetic_{kind}_pairs.csv',filename=f'synthetic_{kind}_pairs.csv',media_type='text/csv')


@router.post('/ppi/string/import',status_code=202)
def download_string(req:StringRequest):
    if not 2<=len(req.genes)<=30:
        raise HTTPException(422,'Enter 2–30 protein symbols')
    ident=make_job('ppi','string')
    EXECUTOR.submit(string_task,ident,req)
    return {'job_id':ident}


@router.get('/jobs/{ident}')
def status(ident:str):
    with LOCK:
        job=JOBS.get(ident)
        if job is None:raise HTTPException(404,'Job not found')
        return dict(job)


def attach(app,root,project):
    global STORAGE,PROJECT
    STORAGE=Path(root);PROJECT=Path(project)
    app.include_router(router)
