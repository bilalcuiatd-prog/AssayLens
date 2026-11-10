"""Separate DTI, DDI and PPI data sources and algorithm metadata."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from .research_hub import REGISTRY, resource_catalog, load_and_normalize

router = APIRouter(prefix="/api/v5", tags=["Research datasets and model catalogs"])
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="source-dataset")
LOCK = threading.Lock()
JOBS = {}
DATA_ROOT = None
DTI_SAVE = None
PAIR_SAVE = None


class DownloadRequest(BaseModel):
    resource_key: str
    max_rows: int = Field(default=12000, ge=100, le=20000)
    random_seed: int = Field(default=42, ge=0, le=2_000_000_000)
    ppi_reference_negatives: bool = False


def _update(job_id, **props):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(props)


def _download_worker(job_id, settings:DownloadRequest):
    try:
        res = REGISTRY[settings.resource_key]
        _update(job_id, state="running", message=f"Downloading {res.title} from {res.provider}")
        data, manifest = load_and_normalize(
            res.key, limit=settings.max_rows, sampling_seed=settings.random_seed,
            reference_negatives=settings.ppi_reference_negatives
        )
        folder = DATA_ROOT / job_id
        folder.mkdir(parents=True, exist_ok=False)
        file = folder / "research_data.csv"
        data.to_csv(file, index=False)
        manifest["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        manifest["dataset_content_class"] = "scientific pairwise observations"
        manifest["end_use"] = "model benchmarking only; not individual medication guidance"
        import_result = None
        error = None
        if manifest.get("trainable_as_is"):
            try:
                payload = file.read_bytes()
                origin = {"provider":res.provider, "resource":res.key,"manifest":str(folder/'manifest.json')}
                if res.kind == "dti":
                    import_result = DTI_SAVE(payload, origin=origin)
                elif res.kind == "ppi":
                    import_result = PAIR_SAVE("ppi", payload, provenance=origin)
            except Exception as exc:
                # Keep a raw, reproducible export even when a strict audit
                # prevents training on the underlying data.
                error = f"Model import requires additional curation: {type(exc).__name__}: {str(exc)[:280]}"
        manifest["import_result"] = import_result
        if error:
            manifest["import_warning"] = error
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        _update(job_id, state="complete", message="Dataset export prepared", manifest=manifest,
                imported=import_result, warning=error)
    except Exception as exc:
        _update(job_id, state="error", message=f"{type(exc).__name__}: {str(exc)[:450]}")


@router.get("/catalog")
def catalog():
    return resource_catalog()


@router.post("/datasets/download", status_code=202)
def begin_download(settings: DownloadRequest):
    res = REGISTRY.get(settings.resource_key)
    if not res:
        raise HTTPException(404, "Unknown curated dataset")
    if res.availability not in ("automatic", "raw_only"):
        raise HTTPException(422, "This source requires an existing license or manual import; read its source terms")
    if settings.ppi_reference_negatives and res.key != "ppi_huri":
        raise HTTPException(422, "Reference negative sampling is HuRI-specific")
    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = {"id":job_id, "state":"queued", "resource_key":res.key,
                        "created_utc":datetime.now(timezone.utc).isoformat(), "message":"Queued"}
    EXECUTOR.submit(_download_worker, job_id, settings)
    return {"job_id":job_id}


@router.get("/datasets/jobs/{job_id}")
def job(job_id:str):
    if job_id not in JOBS:
        raise HTTPException(404, "Dataset job not found")
    return JOBS[job_id]


@router.get("/datasets/jobs/{job_id}/csv")
def export(job_id:str):
    if job_id not in JOBS or JOBS[job_id]["state"] != "complete":
        raise HTTPException(404, "Dataset not ready")
    p = DATA_ROOT / job_id / "research_data.csv"
    if not p.is_file():
        raise HTTPException(404, "Export not found")
    return FileResponse(p, filename=f"assaylens_{JOBS[job_id]['resource_key']}.csv", media_type="text/csv")


@router.get("/datasets/jobs/{job_id}/manifest")
def export_manifest(job_id:str):
    if job_id not in JOBS or JOBS[job_id]["state"] != "complete":
        raise HTTPException(404, "Dataset not ready")
    p = DATA_ROOT / job_id / "manifest.json"
    if not p.is_file():
        raise HTTPException(404, "Manifest not found")
    return FileResponse(p, filename="dataset_provenance.json")


def attach(app, root, dti_save, pair_save):
    global DATA_ROOT, DTI_SAVE, PAIR_SAVE
    DATA_ROOT = root / "research_sources"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DTI_SAVE = dti_save
    PAIR_SAVE = pair_save
    app.include_router(router)
