"""HuRI data and audit routes for AssayLens local protein-interaction research."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .ppi_huri import load_huri, normalize_huri

router=APIRouter(prefix='/api/v5/ppi',tags=['Protein interactome datasets'])
EXECUTOR=ThreadPoolExecutor(max_workers=1,thread_name_prefix='huri-retrieval')
LOCK=threading.Lock()
JOBS={}
STORAGE=None
SAVE=None


class HuRIRequest(BaseModel):
    max_rows:int=Field(default=10000,ge=100,le=20000)
    seed:int=Field(default=42,ge=0,le=2000000000)
    include_reference_controls:bool=False


def _set(job,**update):
    with LOCK:
        if job in JOBS:JOBS[job].update(update)


def _work(job,request):
    try:
        _set(job,state='running',message='Retrieving human protein interaction records from TDC')
        raw=load_huri(request.include_reference_controls)
        df,audit=normalize_huri(raw,request.max_rows,request.seed,request.include_reference_controls)
        folder=STORAGE / job
        folder.mkdir(parents=True,exist_ok=False)
        csv=folder/'ppi_huri.csv'
        data=df.to_csv(index=False).encode('utf8')
        csv.write_bytes(data)
        audit['sha256']=hashlib.sha256(data).hexdigest()
        imported=SAVE('ppi',data,provenance={'source':'TDC HuRI','manifest':str(folder/'manifest.json'),'reference_controls':request.include_reference_controls})
        audit['workspace_dataset_id']=imported.get('id')
        (folder/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
        _set(job,state='complete',message='HuRI records imported',manifest=audit,dataset_id=imported.get('id'))
    except Exception as exc:
        _set(job,state='error',message=f'{type(exc).__name__}: {str(exc)[:380]}')


@router.post('/huri',status_code=202)
def download_huri(request:HuRIRequest):
    job=uuid.uuid4().hex
    with LOCK:JOBS[job]={'id':job,'state':'queued','message':'Queued'}
    EXECUTOR.submit(_work,job,request)
    return {'job_id':job}


@router.get('/jobs/{job}')
def get_job(job:str):
    if job not in JOBS:raise HTTPException(404,'Job not found')
    return JOBS[job]


@router.get('/jobs/{job}/csv')
def export_csv(job:str):
    if job not in JOBS or JOBS[job]['state']!='complete':raise HTTPException(404,'Dataset is not ready')
    path=STORAGE/job/'ppi_huri.csv'
    if not path.is_file():raise HTTPException(404,'Dataset not found')
    return FileResponse(path,filename='AssayLens_HuRI_PPI.csv',media_type='text/csv')


@router.get('/jobs/{job}/manifest')
def export_manifest(job:str):
    if job not in JOBS or JOBS[job]['state']!='complete':raise HTTPException(404,'Manifest not ready')
    return FileResponse(STORAGE/job/'manifest.json',filename='HuRI_dataset_manifest.json')


def attach(app,root,pair_save):
    global STORAGE,SAVE
    STORAGE=Path(root)/'huri_imports'
    STORAGE.mkdir(parents=True,exist_ok=True)
    SAVE=pair_save
    app.include_router(router)
