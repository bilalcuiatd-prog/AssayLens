from __future__ import annotations

import hashlib
import io
import json
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from . import advanced_experiments, custom_models, sources_dynamic, docking, diagnostics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import TARGETS, BY_GENE
from .quality import audit_dataset
from .sources import COLUMNS, download_chembl, download_bindingdb, pubchem_assays
from .experiments import run_experiment, validate_config
from .storage import ROOT, new_identifier, existing_path, path, dump_json, load_json, listing

PROJECT = Path(__file__).resolve().parent.parent
app = FastAPI(title="AssayLens Research Workbench", version="5.2.0", docs_url="/api/docs")
app.mount("/assets", StaticFiles(directory=str(PROJECT / "web")), name="assets")
JOBS = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="assaylens")


class BatchDownload(BaseModel):
    provider: Literal["chembl", "bindingdb"] = "chembl"
    genes: list[str] = Field(default_factory=lambda: ["ACHE"])
    endpoints: list[Literal["IC50", "Ki", "Kd"]] = Field(default_factory=lambda: ["IC50"])
    max_raw: int = Field(default=0, ge=0, le=100000)
    binding_cutoff_nm: int = Field(default=100000, ge=10, le=10000000)


class Experiment(BaseModel):
    dataset_id: str
    strategy: Literal["scaffold", "molecule", "assay"] = "scaffold"
    folds: int = Field(default=5, ge=2, le=10)
    seeds: list[int] = Field(default_factory=lambda: [17, 29])
    models: list[str] = Field(default_factory=lambda: ["extra_trees", "ann_mlp"])
    variants: list[str] = Field(default_factory=lambda: ["chemistry"])
    n_estimators: int = Field(default=64, ge=16, le=350)
    min_samples_leaf: int = Field(default=2, ge=1, le=20)
    max_depth: int = Field(default=0, ge=0, le=100)
    regularization: float = Field(default=1.0, gt=0, le=1000)
    hgb_leaves: int = Field(default=15, ge=2, le=80)
    fp_bits: int = Field(default=512, ge=32, le=2048)
    fp_radius: int = Field(default=2, ge=1, le=3)
    alpha: float = Field(default=0.1, ge=0.05, le=0.3)
    mlp_layers: list[int] = Field(default_factory=lambda: [128, 64])
    mlp_lr: float = Field(default=0.001, gt=0, le=0.1)
    mlp_alpha: float = Field(default=0.001, gt=0, le=10)
    mlp_max_iter: int = Field(default=160, ge=25, le=500)
    svr_c: float = Field(default=10.0, gt=0, le=500)
    svr_epsilon: float = Field(default=0.1, gt=0, le=5)


def update_job(identifier, **values):
    with JOBS_LOCK:
        if identifier in JOBS:
            JOBS[identifier].update(values)


def job_create(kind):
    identifier = uuid.uuid4().hex
    update = {"id": identifier, "kind": kind, "state": "queued", "message": "Queued", "items": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    with JOBS_LOCK:
        JOBS[identifier] = update
    return identifier


def store_csv(contents, origin=None):
    if len(contents) > 8_000_000:
        raise ValueError("Import CSV exceeds the 8 MB limit")
    raw = pd.read_csv(io.BytesIO(contents), on_bad_lines="error")
    audit = audit_dataset(raw)
    if len(audit.curated) == 0:
        raise ValueError("No records passed the assay integrity audit")
    ident = new_identifier()
    destination = path("datasets", ident)
    (destination / "raw.csv").write_bytes(contents)
    audit.records.to_csv(destination / "qc_rows.csv", index=False)
    audit.curated.to_csv(destination / "curated.csv", index=False)
    demo = audit.summary.get("target") == "SYNTHETIC_TARGET" or audit.summary.get("endpoint") == "SYNTHETIC_SIGNAL"
    payload = {"id": ident, "sha256": hashlib.sha256(contents).hexdigest(), "created_utc": datetime.now(timezone.utc).isoformat(), "origin": origin or {"provider": "Upload"}, "demo_data": demo, **audit.summary}
    if demo:
        payload["warnings"].insert(0, "Demonstration labels are synthetic; exclude these data and results from scientific publication")
    dump_json(destination / "audit.json", payload)
    return payload


def read_source_csv(dataset_id):
    location = existing_path("datasets", dataset_id)
    data = (location / "raw.csv").read_bytes()
    meta = load_json(location / "audit.json")
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError("Source dataset checksum mismatch")
    return location, meta, pd.read_csv(location / "curated.csv")


def download_task(job, data):
    update_job(job, state="running", message="Connecting to data provider")
    items = []
    jobs_dir = ROOT / "provider_downloads" / job
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for gene in data.genes:
        for endpoint in data.endpoints:
            message = f"{data.provider.upper()} · {gene} · {endpoint}"
            update_job(job, message=message)
            try:
                target_folder = jobs_dir / data.provider / gene / endpoint
                if data.provider == "chembl":
                    csv, provenance = download_chembl(gene, endpoint, target_folder, lambda m: update_job(job, message=f"{gene}: {m}"), max_raw=data.max_raw)
                else:
                    csv, provenance = download_bindingdb(gene, endpoint, target_folder, lambda m: update_job(job, message=f"{gene}: {m}"), cutoff_nm=data.binding_cutoff_nm)
                imported = store_csv(csv.read_bytes(), origin={"provider": data.provider, "gene": gene, "endpoint": endpoint, "provenance_path": str(target_folder / "manifest.json"), "pilot": bool(data.max_raw)})
                items.append({"gene": gene, "endpoint": endpoint, "state": "ready", "dataset_id": imported["id"], "records": imported["curated_assay_compound_pairs"], "assays": imported["unique_assays"], "provider": data.provider})
            except Exception as ex:
                items.append({"gene": gene, "endpoint": endpoint, "state": "error", "error": f"{type(ex).__name__}: {str(ex)[:350]}"})
            update_job(job, items=items.copy())
    successes = sum(x["state"] == "ready" for x in items)
    update_job(job, state="complete" if successes else "error", message=f"{successes} of {len(items)} datasets prepared", items=items)


def create_figure(folder):
    metrics = pd.read_csv(folder / "summary.csv")
    plot = metrics.head(12).copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    fig.patch.set_facecolor("#111c27")
    ax.set_facecolor("#111c27")
    ax.barh([f"{r.model} · {r.ablation}" for r in plot.itertuples()], plot.mae_mean, color="#53c6b1")
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error", color="white")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#364957")
    ax.set_title("Repeated grouped CV · mean of seed-level MAE", color="white", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(folder / "mae_comparison.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def experiment_task(job, req):
    update_job(job, state="running", message="Reading curated observations")
    try:
        directory, audit, df = read_source_csv(req.dataset_id)
        opts = req.model_dump()
        validate_config(opts, df)
        run_id = new_identifier()
        location = path("runs", run_id)
        result = run_experiment(df, opts, location, lambda m: update_job(job, message=m))
        shutil.copyfile(directory / "curated.csv", location / "curated.csv")
        shutil.copyfile(directory / "audit.json", location / "dataset_audit.json")
        dump_json(location / "metrics.json", {"id": run_id, "dataset_id": req.dataset_id, "strategy": req.strategy, "folds": req.folds, "seeds": req.seeds, "summary": result["summary"], "n_fits": result["n_fits"], "warnings": result["warnings"]})
        create_figure(location)
        (location / "METHODS.txt").write_text("Evaluation: repeated group-disjoint K-fold cross-validation. Calibration groups are separate from training and test groups inside each outer fold. Hyperparameters are fixed across outer folds; no nested tuning is performed. Repeated seeds and folds are correlated, so seed SD is descriptive rather than an independent uncertainty estimate. Split-conformal coverage assumes exchangeability and can fail under chemical or assay distribution shift. Assay-group CV may permit the same molecule across different assays. Dataset provenance and exclusions require independent review before publication.\n", encoding="utf8")
        with zipfile.ZipFile(location / "research_package.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ["fold_metrics.csv", "per_seed_metrics.csv", "summary.csv", "out_of_fold_predictions.csv", "manifest.json", "dataset_audit.json", "curated.csv", "metrics.json", "METHODS.txt", "mae_comparison.png"]:
                file = location / name
                if file.exists():
                    archive.write(file, arcname=name)
        update_job(job, state="complete", message="Experiment complete", run_id=run_id, summary=result["summary"], n_fits=result["n_fits"])
    except Exception as ex:
        update_job(job, state="error", message=f"{type(ex).__name__}: {str(ex)[:400]}")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((PROJECT / "web" / "suite.html").read_text(encoding="utf8"), headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "execution": "local-only"}


@app.get("/api/catalog")
def catalog():
    return {"targets": TARGETS, "providers": [
        {"id": "chembl", "name": "ChEMBL", "mode": "assay-rich", "description": "Target-resolved bioactivities; assay confidence and source documents"},
        {"id": "bindingdb", "name": "BindingDB", "mode": "affinity-limited", "description": "Affinity measurements; cutoff and target-linkage review required"},
        {"id": "pubchem", "name": "PubChem BioAssay", "mode": "raw-export", "description": "Search deposited assay IDs by accession; raw assay CSV requires endpoint harmonization"}
    ], "endpoints": ["IC50", "Ki", "Kd"]}


@app.get("/api/datasets")
def datasets():
    data = listing("datasets")
    return [{"id": r["id"], "target": r.get("target"), "endpoint": r.get("endpoint"), "rows": r.get("curated_assay_compound_pairs"), "scaffolds": r.get("unique_scaffolds"), "assays": r.get("unique_assays"), "provider": (r.get("origin") or {}).get("provider", "Upload"), "gene": (r.get("origin") or {}).get("gene", "") } for r in data]


@app.post("/api/datasets", status_code=201)
async def add_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Choose a CSV file")
    try:
        return store_csv(await file.read(8_000_001))
    except Exception as ex:
        raise HTTPException(422, str(ex)[:350])


@app.get("/api/datasets/{dataset_id}/csv")
def download_dataset_csv(dataset_id: str):
    try:
        return FileResponse(existing_path("datasets", dataset_id) / "raw.csv", filename=f"assaylens_{dataset_id[:8]}.csv", media_type="text/csv")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Dataset not found")


@app.get("/api/datasets/{dataset_id}/qc.csv")
def quality_rows(dataset_id: str):
    try:
        return FileResponse(existing_path("datasets", dataset_id) / "qc_rows.csv", filename="assaylens_quality_audit.csv", media_type="text/csv")
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Quality audit not found")


@app.get("/api/datasets/{dataset_id}/source-manifest")
def source_manifest(dataset_id: str):
    try:
        location = existing_path("datasets", dataset_id)
        audit_data = load_json(location / "audit.json")
        manifest = (audit_data.get("origin") or {}).get("provenance_path")
        if not manifest:
            raise FileNotFoundError("No external source manifest for uploaded files")
        source = Path(manifest).resolve()
        allowed = (ROOT / "provider_downloads").resolve()
        if not source.is_relative_to(allowed) or source.name != "manifest.json" or not source.is_file():
            raise FileNotFoundError("Source manifest unavailable")
        return FileResponse(source, filename="source_manifest.json", media_type="application/json")
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Source manifest unavailable")


@app.get("/api/datasets/{dataset_id}/audit")
def audit(dataset_id: str):
    try:
        return load_json(existing_path("datasets", dataset_id) / "audit.json")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Dataset not found")


@app.post("/api/downloads", status_code=202)
def download_datasets(req: BatchDownload):
    if not req.genes or len(req.genes) > len(BY_GENE) or len(req.endpoints) * len(req.genes) > 60:
        raise HTTPException(422, "Select valid catalog targets, with a maximum of 60 target-endpoint datasets per batch")
    if len(set(req.genes)) != len(req.genes) or len(set(req.endpoints)) != len(req.endpoints) or not all(gene in BY_GENE for gene in req.genes):
        raise HTTPException(422, "Unknown or repeated target/endpoint")
    if req.provider == "bindingdb" and req.max_raw:
        raise HTTPException(422, "Raw ChEMBL record limit does not apply to BindingDB")
    ident = job_create("download")
    EXECUTOR.submit(download_task, ident, req)
    return {"job_id": ident}


@app.get("/api/jobs/{identifier}")
def job_status(identifier: str):
    with JOBS_LOCK:
        status = JOBS.get(identifier)
        if status is None:
            raise HTTPException(404, "Job not found")
        return dict(status)


@app.get("/api/pubchem/{gene}/assays")
def pubchem_search(gene: str):
    if gene not in BY_GENE:
        raise HTTPException(404, "Target not in catalog")
    try:
        return {"gene": gene, "accession": BY_GENE[gene]["accession"], "assays": pubchem_assays(BY_GENE[gene]["accession"]), "notice": "Raw assay CSV exports; not automatically harmonized to one quantitative endpoint. Curate manually before importing."}
    except Exception as ex:
        raise HTTPException(502, f"PubChem lookup failed: {str(ex)[:220]}")


@app.post("/api/experiments", status_code=202)
def start_experiment(req: Experiment):
    try:
        _, _, df = read_source_csv(req.dataset_id)
        validate_config(req.model_dump(), df)
    except (ValueError, FileNotFoundError) as ex:
        raise HTTPException(422, str(ex)[:320])
    ident = job_create("experiment")
    EXECUTOR.submit(experiment_task, ident, req)
    return {"job_id": ident}


@app.get("/api/experiments")
def list_experiments():
    results = listing("runs")
    return [{"id": r["id"], "dataset_id": r.get("dataset_id"), "folds": r.get("folds"), "seeds": r.get("seeds"), "strategy": r.get("strategy"), "n_fits": r.get("n_fits")} for r in results if "summary" in r]


@app.get("/api/experiments/{run_id}")
def get_experiment(run_id: str):
    try:
        return load_json(existing_path("runs", run_id) / "metrics.json")
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Experiment not found")


@app.get("/api/experiments/{run_id}/bundle")
def run_bundle(run_id: str):
    try:
        return FileResponse(existing_path("runs", run_id) / "research_package.zip", filename="assaylens_research_package.zip")
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Report not found")


@app.get("/api/experiments/{run_id}/figure")
def run_figure(run_id: str):
    try:
        return FileResponse(existing_path("runs", run_id) / "mae_comparison.png", media_type="image/png")
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Figure not found")


@app.get("/api/example.csv")
def example_csv():
    return FileResponse(PROJECT / "examples" / "synthetic_assay_demo.csv", filename="synthetic_assay_demo.csv", media_type="text/csv")


@app.get('/classic', response_class=HTMLResponse)
def classic_workbench():
    return HTMLResponse((PROJECT/'web'/'index.html').read_text(encoding='utf8'), headers={'Cache-Control':'no-store'})


class ExtendedExperiment(BaseModel):
    dataset_id: str
    task: Literal['regression','classification']='regression'
    activity_threshold: float=Field(default=6.0,ge=-5,le=20)
    strategy: Literal['scaffold','molecule','assay']='scaffold'
    folds: int=Field(default=5,ge=2,le=10)
    seeds: list[int]=Field(default_factory=lambda:[17,29,42])
    models: list[str]=Field(default_factory=lambda:['random_forest','ann_mlp'])
    variants: list[str]=Field(default_factory=lambda:['chemistry'])
    n_estimators: int=Field(default=100,ge=16,le=500)
    min_samples_leaf: int=Field(default=2,ge=1,le=20)
    max_depth: int=Field(default=0,ge=0,le=100)
    regularization: float=Field(default=1.,gt=0,le=1000)
    fp_bits: int=Field(default=512,ge=32,le=2048)
    fp_radius: int=Field(default=2,ge=1,le=3)
    mlp_layers: list[int]=Field(default_factory=lambda:[128,64])
    mlp_lr: float=Field(default=.001,gt=0,le=.1)
    mlp_alpha: float=Field(default=.001,gt=0,le=10)
    mlp_max_iter: int=Field(default=150,ge=25,le=500)
    svr_c: float=Field(default=10.,gt=0,le=500)
    svr_epsilon: float=Field(default=.1,gt=0,le=5)
    knn_neighbors: int=Field(default=5,ge=2,le=100)
    boost_lr: float=Field(default=.05,gt=0,le=.5)
    alpha: float=Field(default=.1,ge=.05,le=.3)


class DynamicTargetRequest(BaseModel):
    target_id: str
    endpoint: Literal['IC50','Ki','Kd']='IC50'
    max_raw: int=Field(default=0,ge=0,le=100000)


@app.get('/api/v2/catalog')
def v2_catalog():
    return {'targets':TARGETS,'models':{'regression':advanced_experiments.REGRESSION_MODELS,
        'classification':advanced_experiments.CLASSIFICATION_MODELS},
        'ablations':advanced_experiments.VARIANTS,'available_sources':['ChEMBL','BindingDB','PubChem BioAssay'],
        'custom_model_execution':custom_models.sandbox_status(),
        'docking':docking.capabilities()}


@app.get('/api/v2/targets/search')
def v2_search(query:str):
    try:
        return {'targets':sources_dynamic.search_targets(query)}
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    except Exception as exc:
        raise HTTPException(502,'ChEMBL target search unavailable: '+str(exc)[:170])


def dynamic_download_task(job,req):
    update_job(job,state='running',message='Fetching ChEMBL experimental activity data')
    try:
        output=path('provider_downloads',job)
        filename,manifest=sources_dynamic.download_target(req.target_id,req.endpoint,output,
            lambda msg:update_job(job,message=msg),max_raw=req.max_raw)
        registered=store_csv(filename.read_bytes(),origin={'provider':'ChEMBL','gene':req.target_id,
            'endpoint':req.endpoint,'provenance_path':str(output/'manifest.json'),'pilot':bool(req.max_raw)})
        update_job(job,state='complete',message='ChEMBL target dataset ready',dataset_id=registered['id'],
            rows=registered.get('curated_assay_compound_pairs'),manifest=manifest)
    except Exception as exc:
        update_job(job,state='error',message=str(exc)[:500])


@app.post('/api/v2/targets/download',status_code=202)
def v2_dynamic_download(req:DynamicTargetRequest):
    identifier=job_create('target-download')
    EXECUTOR.submit(dynamic_download_task,identifier,req)
    return {'job_id':identifier}


def advanced_figure(folder,task):
    import numpy as np
    metrics=pd.read_csv(folder/'summary.csv')
    field='f1_mean' if task=='classification' else 'mae_mean'
    title='Held-out F1 score' if task=='classification' else 'Held-out mean absolute error'
    top=metrics.sort_values(field,ascending=(task=='regression')).head(14).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9.5,max(3.5,len(top)*.39)))
    fig.patch.set_facecolor('#101e29');ax.set_facecolor('#101e29')
    ax.barh([f"{r.model} / {r.ablation}" for r in top.itertuples()],top[field],color='#30c0af')
    ax.tick_params(colors='#d4e3ed',labelsize=8);ax.set_xlabel(title,color='white')
    ax.set_title('Repeated grouped CV · seed-level means',color='white',loc='left')
    for sp in ax.spines.values():sp.set_color('#38505f')
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=165,facecolor=fig.get_facecolor());plt.close(fig)


def v2_experiment_task(job,req):
    update_job(job,state='running',message='Preparing molecular features and validation folds')
    try:
        dataset_folder,audit,df=read_source_csv(req.dataset_id)
        cfg=req.model_dump()
        advanced_experiments.validate(cfg,df)
        run_id=new_identifier();out=path('runs_v2',run_id)
        result=advanced_experiments.run(df,cfg,out,lambda msg:update_job(job,message=msg))
        shutil.copyfile(dataset_folder/'audit.json',out/'dataset_audit.json')
        shutil.copyfile(dataset_folder/'curated.csv',out/'curated.csv')
        advanced_figure(out,req.task)
        diagnostics.create_diagnostics(out,req.task)
        metadata={'id':run_id,'dataset_id':req.dataset_id,'task':req.task,'strategy':req.strategy,
            'folds':req.folds,'seeds':req.seeds,'n_fits':result['n_fits'],'summary':result['summary'],
            'model_settings':cfg,'limitations':result['warnings']}
        dump_json(out/'metrics.json',metadata)
        with zipfile.ZipFile(out/'research_package.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in out.iterdir():
                if f.is_file() and f.suffix in ('.csv','.json','.png'):
                    archive.write(f,f.name)
        update_job(job,state='complete',message='Experiment complete',run_id=run_id,
            summary=result['summary'],n_fits=result['n_fits'])
    except Exception as exc:
        update_job(job,state='error',message=f'{type(exc).__name__}: {str(exc)[:450]}')


@app.post('/api/v2/experiments',status_code=202)
def v2_start_experiment(req:ExtendedExperiment):
    try:
        _,_,df=read_source_csv(req.dataset_id)
        advanced_experiments.validate(req.model_dump(),df)
    except (ValueError,FileNotFoundError) as exc:
        raise HTTPException(422,str(exc)[:350])
    identifier=job_create('evaluation')
    EXECUTOR.submit(v2_experiment_task,identifier,req)
    return {'job_id':identifier}


@app.get('/api/v2/experiments')
def v2_runs():
    return [r for r in listing('runs_v2') if 'summary' in r]


@app.get('/api/v2/experiments/{run_id}')
def v2_result(run_id:str):
    try:
        return load_json(existing_path('runs_v2',run_id)/'metrics.json')
    except (ValueError,FileNotFoundError):
        raise HTTPException(404,'Run not found')


@app.get('/api/v2/experiments/{run_id}/bundle')
def v2_bundle(run_id:str):
    try:
        return FileResponse(existing_path('runs_v2',run_id)/'research_package.zip',filename='assaylens_v2_results.zip')
    except (ValueError,FileNotFoundError):
        raise HTTPException(404,'Run not found')


@app.get('/api/v2/experiments/{run_id}/figure')
def v2_figure(run_id:str):
    try:
        return FileResponse(existing_path('runs_v2',run_id)/'comparison.png',media_type='image/png')
    except (ValueError,FileNotFoundError):
        raise HTTPException(404,'Figure not found')


@app.get('/api/v2/custom-models')
def model_list():
    return {'models':custom_models.list_plugins(),'runtime':custom_models.sandbox_status()}


@app.post('/api/v2/custom-models',status_code=201)
async def upload_custom_model(file:UploadFile=File(...)):
    try:
        return custom_models.save_plugin(file.filename or '',await file.read(300001))
    except ValueError as exc:
        raise HTTPException(422,str(exc))


class CustomExperimentRequest(BaseModel):
    model_id: str
    dataset_id: str
    task: Literal['regression','classification']='regression'
    strategy: Literal['scaffold','molecule','assay']='scaffold'
    folds: int=Field(default=5,ge=2,le=10)
    seeds: list[int]=Field(default_factory=lambda:[42])
    variants: list[str]=Field(default_factory=lambda:['chemistry'])
    activity_threshold: float=Field(default=6.,ge=-5,le=20)
    fp_bits: int=512
    fp_radius: int=2


def custom_task(job,req):
    update_job(job,state='running',message='Launching isolated custom-model runtime')
    try:
        dataset_folder,audit,df=read_source_csv(req.dataset_id)
        opts={**req.model_dump(),'models':['random_forest'],'n_estimators':32}
        advanced_experiments.validate(opts,df)
        opts['models']=['custom']
        run_id=new_identifier()
        out=path('custom_runs',run_id)
        result=custom_models.docker_run(req.model_id,dataset_folder/'curated.csv',opts,out,
            lambda msg:update_job(job,message=msg))
        with zipfile.ZipFile(out/'research_package.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in out.iterdir():
                if f.is_file() and f.suffix in ('.csv','.json'):
                    archive.write(f,f.name)
        update_job(job,state='complete',message='Isolated custom model evaluation finished',run_id=run_id,summary=result['summary'],n_fits=result['n_fits'])
    except Exception as exc:
        update_job(job,state='error',message=str(exc)[:450])


@app.post('/api/v2/custom-experiments',status_code=202)
def start_custom(req:CustomExperimentRequest):
    if req.model_id not in {x['id'] for x in custom_models.list_plugins()}:
        raise HTTPException(404,'Custom model not found')
    identifier=job_create('custom-model')
    EXECUTOR.submit(custom_task,identifier,req)
    return {'job_id':identifier}


@app.get('/api/v2/custom-experiments/{run_id}/bundle')
def custom_bundle(run_id:str):
    try:
        return FileResponse(existing_path('custom_runs',run_id)/'research_package.zip',filename='assaylens_custom_model_results.zip')
    except (ValueError,FileNotFoundError):
        raise HTTPException(404,'Custom experiment not found')


class ConformerInput(BaseModel):
    smiles:str=Field(min_length=1,max_length=500)


@app.get('/api/v2/docking/capabilities')
def docking_status():
    return docking.capabilities()


@app.post('/api/v2/docking/conformer')
def conformer(req:ConformerInput):
    try:
        return {'sdf':docking.generate_conformer(req.smiles),'type':'3D SDF conformer'}
    except ValueError as exc:
        raise HTTPException(422,str(exc))


@app.get('/api/v2/structures/{pdb_id}')
def protein_structure(pdb_id:str):
    try:
        return Response(docking.get_pdb_structure(pdb_id),media_type='text/plain')
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    except Exception as exc:
        raise HTTPException(502,'RCSB structure unavailable: '+str(exc)[:130])


@app.post('/api/v2/docking/prepare-ligand')
def prepare_ligand(req:ConformerInput):
    try:
        return {'id':docking.prepare_ligand_pdbqt(req.smiles),'notice':'Check protonation and torsion parameters before docking'}
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(422,str(exc))


@app.get('/api/v2/docking/prepared/{ident}')
def prepared_ligand(ident:str):
    if not __import__('re').fullmatch('[a-f0-9]{32}',ident):
        raise HTTPException(404,'Ligand not found')
    candidate=ROOT/'docking'/ident/'ligand.pdbqt'
    if not candidate.is_file():
        raise HTTPException(404,'Ligand not found')
    return FileResponse(candidate,filename='ligand.pdbqt')


class DockSettings(BaseModel):
    center_x:float;center_y:float;center_z:float
    size_x:float=Field(default=20.,ge=8,le=35)
    size_y:float=Field(default=20.,ge=8,le=35)
    size_z:float=Field(default=20.,ge=8,le=35)
    exhaustiveness:int=Field(default=8,ge=1,le=32)
    seed:int=Field(default=42,ge=0,le=2000000000)


@app.post('/api/v2/docking/run',status_code=201)
async def run_docking(receptor:UploadFile=File(...),ligand:UploadFile=File(...),
    center_x:float=Form(0),center_y:float=Form(0),center_z:float=Form(0),
    size_x:float=Form(20),size_y:float=Form(20),size_z:float=Form(20),
    exhaustiveness:int=Form(8),seed:int=Form(42)):
    if not (receptor.filename or '').lower().endswith('.pdbqt') or not (ligand.filename or '').lower().endswith('.pdbqt'):
        raise HTTPException(422,'Upload prepared .pdbqt receptor and ligand files')
    try:
        settings=DockSettings(center_x=center_x,center_y=center_y,center_z=center_z,
            size_x=size_x,size_y=size_y,size_z=size_z,exhaustiveness=exhaustiveness,seed=seed)
        return docking.docking_run(await receptor.read(12_000_001),await ligand.read(1_000_001),
            [settings.center_x,settings.center_y,settings.center_z],
            [settings.size_x,settings.size_y,settings.size_z],settings.exhaustiveness,settings.seed)
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(422,str(exc)[:350])


@app.get('/api/v2/docking/{ident}/poses')
def docking_poses(ident:str):
    if not __import__('re').fullmatch('[a-f0-9]{32}',ident):
        raise HTTPException(404,'Docking output not found')
    candidate=ROOT/'docking'/ident/'poses.pdbqt'
    if not candidate.is_file():
        raise HTTPException(404,'Docking output not found')
    return FileResponse(candidate,filename='docked_poses.pdbqt')


@app.get('/api/v2/experiments/{run_id}/diagnostic/{figure_name}')
def v2_diagnostic(run_id:str,figure_name:str):
    allowed={'observed_vs_predicted.png','error_by_activity.png','roc_curve.png','pr_curve.png','confusion_matrix.png'}
    if figure_name not in allowed:
        raise HTTPException(404,'Figure not found')
    try:
        figure=existing_path('runs_v2',run_id)/figure_name
        if not figure.is_file():
            raise FileNotFoundError('No figure')
        return FileResponse(figure,media_type='image/png')
    except (FileNotFoundError,ValueError):
        raise HTTPException(404,'Figure not found')


from . import dti
from .storage import new_identifier as dti_new_identifier


class DTIExperiment(BaseModel):
    dataset_id: str
    task: Literal['regression', 'classification'] = 'regression'
    strategy: Literal['warm', 'cold_drug', 'cold_target', 'cold_both'] = 'warm'
    folds: int = Field(default=5, ge=2, le=10)
    seeds: list[int] = Field(default_factory=lambda: [17, 29, 42])
    models: list[str] = Field(default_factory=lambda: ['extra_trees', 'random_forest'])
    variants: list[str] = Field(default_factory=lambda: ['joint'])
    activity_threshold: float = Field(default=6.0, ge=-5, le=20)
    fp_bits: int = Field(default=256, ge=64, le=2048)
    fp_radius: int = Field(default=2, ge=1, le=3)
    n_estimators: int = Field(default=80, ge=10, le=500)
    min_samples_leaf: int = Field(default=2, ge=1, le=20)
    mlp_layers: list[int] = Field(default_factory=lambda: [128, 64])
    mlp_lr: float = Field(default=.001, gt=0, le=.1)
    mlp_alpha: float = Field(default=.001, gt=0, le=10)
    mlp_max_iter: int = Field(default=125, ge=25, le=500)
    regularization: float = Field(default=1.0, gt=0, le=1000)
    svr_c: float = Field(default=10.0, gt=0, le=500)
    knn_neighbors: int = Field(default=5, ge=2, le=100)


class DTIMerge(BaseModel):
    dataset_ids: list[str]
    sequences: dict[str, str] = Field(default_factory=dict)


def dti_dir(kind, identifier):
    if kind not in ('dti_datasets', 'dti_runs') or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise FileNotFoundError('Invalid DTI dataset or experiment identifier')
    directory = ROOT / kind / identifier
    if not directory.is_dir():
        raise FileNotFoundError('DTI record not found')
    return directory


def save_dti_csv(data: bytes, origin=None):
    if len(data) > 22_000_000:
        raise ValueError('DTI CSV exceeds the 22 MB upload limit')
    frame = pd.read_csv(io.BytesIO(data))
    curated, exclusions, manifest = dti.curate(frame)
    identifier = dti_new_identifier()
    directory = ROOT / 'dti_datasets' / identifier
    directory.mkdir(parents=True)
    (directory / 'raw.csv').write_bytes(data)
    curated.to_csv(directory / 'curated.csv', index=False)
    exclusions.to_csv(directory / 'exclusions.csv', index=False)
    audit = {'id': identifier, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_sha256': hashlib.sha256(data).hexdigest(), 'origin': origin or 'CSV upload',
             'synthetic_example': bool(curated.target_id.astype(str).str.startswith('DEMO_').any()), **manifest}
    dump_json(directory / 'audit.json', audit)
    return audit


def load_dti_dataset(identifier):
    directory = dti_dir('dti_datasets', identifier)
    audit = load_json(directory / 'audit.json')
    if hashlib.sha256((directory / 'raw.csv').read_bytes()).hexdigest() != audit['source_sha256']:
        raise ValueError('DTI source checksum failed verification')
    return directory, audit, pd.read_csv(directory / 'curated.csv')


@app.get('/api/v3/dti/catalog')
def dti_catalog():
    return {'protocols': [{'id': 'warm', 'title': 'Warm start', 'description': 'Unseen pairs; each test drug and protein is also represented in training.'},
                          {'id': 'cold_drug', 'title': 'Cold drug', 'description': 'Molecular identities are disjoint between training and testing.'},
                          {'id': 'cold_target', 'title': 'Cold target', 'description': 'Exact protein sequences are disjoint between training and testing.'},
                          {'id': 'cold_both', 'title': 'Cold drug–target', 'description': 'Both molecular identities and protein sequences are unseen; cross-block rows are excluded.'}],
            'models': {'regression': dti.REGRESSION_MODELS, 'classification': dti.CLASSIFICATION_MODELS},
            'ablations': dti.VARIANTS, 'required_columns': dti.REQUIRED}


@app.post('/api/v3/dti/datasets', status_code=201)
async def dti_import(file: UploadFile = File(...)):
    try:
        if not (file.filename or '').lower().endswith('.csv'):
            raise ValueError('Upload a CSV file')
        return save_dti_csv(await file.read(22_000_001))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(422, str(exc)[:350])


@app.get('/api/v3/dti/datasets')
def dti_list_datasets():
    root = ROOT / 'dti_datasets'
    if not root.is_dir():
        return []
    data = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if child.is_dir() and (child / 'audit.json').is_file():
            data.append(load_json(child / 'audit.json'))
    return data[:100]


@app.get('/api/v3/dti/datasets/{identifier}/csv')
def dti_export(identifier: str):
    try:
        return FileResponse(dti_dir('dti_datasets', identifier) / 'curated.csv', filename='assaylens_dti_curated.csv')
    except FileNotFoundError:
        raise HTTPException(404, 'DTI dataset not found')


@app.get('/api/v3/dti/example.csv')
def dti_example():
    return FileResponse(PROJECT / 'examples' / 'synthetic_multi_target_dti.csv', filename='synthetic_multi_target_dti.csv', media_type='text/csv')


_UNIPROT_SEQUENCES = {}


def uniprot_sequence(accession):
    if accession in _UNIPROT_SEQUENCES:
        return _UNIPROT_SEQUENCES[accession]
    import requests
    if not __import__('re').fullmatch(r'[A-Z0-9]{6,10}', accession):
        raise ValueError('Invalid UniProt accession')
    url = f'https://rest.uniprot.org/uniprotkb/{accession}.fasta'
    response = requests.get(url, timeout=25, headers={'Accept': 'text/plain'})
    response.raise_for_status()
    text_fasta = response.text.strip().splitlines()
    if not text_fasta or not text_fasta[0].startswith('>'):
        raise ValueError(f'No FASTA sequence returned for {accession}')
    protein = ''.join(text_fasta[1:]).upper()
    if not dti._sequence(protein):
        raise ValueError(f'UniProt {accession}: invalid or unsupported amino-acid sequence')
    _UNIPROT_SEQUENCES[accession] = protein
    return protein


@app.post('/api/v3/dti/datasets/merge', status_code=201)
def dti_merge(req: DTIMerge):
    if not 2 <= len(req.dataset_ids) <= 16 or len(set(req.dataset_ids)) != len(req.dataset_ids):
        raise HTTPException(422, 'Choose 2–16 distinct single-target datasets')
    try:
        combined = []
        for ident in req.dataset_ids:
            _, meta, frame = read_source_csv(ident)
            if frame.target_id.astype(str).nunique() != 1:
                raise ValueError('Only audited single-target datasets can be merged')
            target_id = str(frame.target_id.iloc[0])
            gene = str((meta.get('origin') or {}).get('gene') or '')
            supplied = (req.sequences.get(target_id) or req.sequences.get(gene) or '').strip()
            if supplied:
                sequence = dti._sequence(supplied)
                if not sequence:
                    raise ValueError(f'Invalid manually supplied protein sequence for {target_id}')
            elif gene in BY_GENE:
                sequence = uniprot_sequence(BY_GENE[gene]['accession'])
            else:
                raise ValueError(f'Protein sequence needed for {target_id}. Supply a real FASTA sequence in the sequence mapping.')
            combined.append(pd.DataFrame({
                'smiles': frame.parent_smiles, 'response': frame.response,
                'target_id': target_id, 'target_sequence': sequence,
                'endpoint': frame.endpoint, 'assay_id': frame.assay_id,
                'source_id': frame.source_id, 'standard_relation': '=',
                'data_validity_comment': ''
            }))
        data = pd.concat(combined, ignore_index=True)
        if len(data) > dti.MAX_ROWS:
            raise ValueError(f'Merged dataset has {len(data)} rows; maximum is {dti.MAX_ROWS}. Select fewer individual datasets or build a curated subset.')
        result = save_dti_csv(data.to_csv(index=False).encode(), origin={'merged_single_target_dataset_ids': req.dataset_ids})
        if result['unique_protein_sequences'] < 2:
            raise ValueError('Merged dataset still has only one unique protein sequence')
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)[:350])
    except Exception as exc:
        raise HTTPException(502, f'Protein sequence retrieval or merge failed: {str(exc)[:250]}')


def dti_task(job, req):
    update_job(job, state='running', message='Preparing paired molecular and sequence features')
    try:
        directory, audit, frame = load_dti_dataset(req.dataset_id)
        config = req.model_dump()
        dti.validate(config, frame)
        runid = dti_new_identifier()
        output = ROOT / 'dti_runs' / runid
        outcome = dti.run(frame, config, output, lambda msg: update_job(job, message=msg))
        shutil.copyfile(directory / 'audit.json', output / 'dataset_audit.json')
        shutil.copyfile(directory / 'curated.csv', output / 'dataset_pairs.csv')
        audit_meta = {'id': runid, 'dataset_id': req.dataset_id, 'task': req.task,
                      'strategy': req.strategy, 'folds': req.folds, 'seeds': req.seeds,
                      'n_fits': outcome['n_fits'], 'summary': outcome['summary'],
                      'synthetic_example': audit.get('synthetic_example', False)}
        dump_json(output / 'metrics.json', audit_meta)
        with zipfile.ZipFile(output / 'research_package.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for f in output.iterdir():
                if f.is_file() and f.suffix in ('.csv', '.json'):
                    archive.write(f, arcname=f.name)
        update_job(job, state='complete', message='Pairwise evaluation finished',
                   run_id=runid, summary=outcome['summary'], n_fits=outcome['n_fits'])
    except Exception as exc:
        update_job(job, state='error', message=f'{type(exc).__name__}: {str(exc)[:480]}')


@app.post('/api/v3/dti/experiments', status_code=202)
def start_dti_experiment(req: DTIExperiment):
    try:
        _, _, frame = load_dti_dataset(req.dataset_id)
        dti.validate(req.model_dump(), frame)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)[:350])
    identifier = job_create('dti-experiment')
    EXECUTOR.submit(dti_task, identifier, req)
    return {'job_id': identifier}


@app.get('/api/v3/dti/experiments')
def dti_results():
    root = ROOT / 'dti_runs'
    if not root.is_dir():
        return []
    return [load_json(p / 'metrics.json') for p in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if p.is_dir() and (p / 'metrics.json').exists()][:100]


@app.get('/api/v3/dti/experiments/{identifier}/bundle')
def dti_results_package(identifier: str):
    try:
        return FileResponse(dti_dir('dti_runs', identifier) / 'research_package.zip', filename='assaylens_dti_research_package.zip')
    except FileNotFoundError:
        raise HTTPException(404, 'DTI experiment not found')

from .interaction_api import attach as attach_interaction_routes
attach_interaction_routes(app, ROOT, PROJECT)
app.version = '4.0.0'

from .research_hub_api import attach as attach_research_hub
from .interaction_api import save_data as import_pair_csv
attach_research_hub(app, ROOT, save_dti_csv, import_pair_csv)
app.version = '5.0.0'


# ASSAYLENS_PPI_HURI_V5
from .ppi_huri_api import attach as attach_huri_dataset_routes
from .interaction_api import save_data as store_ppi_evidence
attach_huri_dataset_routes(app, ROOT, store_ppi_evidence)
