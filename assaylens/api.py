"""Single-user, local-first API and static research dashboard."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import shutil
from typing import Literal
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .quality import audit_dataset
from .modeling import train_evaluate
from .reporting import generate_report
from .storage import dump_json, load_json, path, existing_path, new_identifier, listing

APP_ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="AssayLens AI", version="0.1.0", docs_url="/api/docs")
app.mount("/assets", StaticFiles(directory=str(APP_ROOT / "web")), name="assets")


class TrainRequest(BaseModel):
    dataset_id: str
    split_type: Literal["scaffold", "assay"] = "scaffold"
    seed: int = Field(default=42, ge=0, le=2_147_483_000)
    alpha: float = Field(default=0.10, ge=0.01, le=0.30)
    trees: int = Field(default=64, ge=16, le=250)


@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "deployment": "single-user local"}


@app.get("/api/datasets")
def datasets_list():
    return [{"id": x["id"], "input_rows": x["input_rows"],
             "curated_assay_compound_pairs": x["curated_assay_compound_pairs"]} for x in listing("datasets")]


@app.post("/api/datasets", status_code=201)
async def create_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")
    content = await file.read(8_000_001)
    if len(content) > 8_000_000:
        raise HTTPException(413, "CSV too large: 8MB limit")
    try:
        raw = pd.read_csv(io.BytesIO(content), on_bad_lines="error")
        result = audit_dataset(raw)
    except Exception as exc:
        raise HTTPException(422, f"Dataset audit failed: {str(exc)[:300]}") from exc
    identifier = new_identifier()
    directory = path("datasets", identifier)
    (directory / "raw.csv").write_bytes(content)
    result.records.to_csv(directory / "qc_rows.csv", index=False)
    result.curated.to_csv(directory / "curated.csv", index=False)
    is_demo = (result.summary.get("target") == "SYNTHETIC_TARGET"
               and result.summary.get("endpoint") == "SYNTHETIC_SIGNAL")
    payload = {**result.summary, "sha256": hashlib.sha256(content).hexdigest(),
               "id": identifier, "demo_data": is_demo}
    if is_demo:
        payload["warnings"].insert(0, "SYNTHETIC EXAMPLE: response values are invented; exclude all figures and metrics from research publications")
    dump_json(directory / "audit.json", payload)
    return payload


@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str):
    try:
        return load_json(existing_path("datasets", dataset_id) / "audit.json")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Dataset not found")


@app.get("/api/datasets/{dataset_id}/audit.csv")
def download_audit(dataset_id: str):
    try:
        directory = existing_path("datasets", dataset_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Dataset not found")
    return FileResponse(directory / "qc_rows.csv", filename="assaylens_row_audit.csv")


@app.post("/api/runs", status_code=201)
def train(request: TrainRequest):
    try:
        dataset_dir = existing_path("datasets", request.dataset_id)
        curated = pd.read_csv(dataset_dir / "curated.csv").fillna({"assay_id": "UNKNOWN", "assay_type": "", "assay_format": "", "target_id": "", "endpoint": ""})
        # Verify data provenance and prevent stale file substitution.
        audit = load_json(dataset_dir / "audit.json")
        if hashlib.sha256((dataset_dir / "raw.csv").read_bytes()).hexdigest() != audit["sha256"]:
            raise ValueError("Source CSV checksum changed")
        result = train_evaluate(curated, request.split_type, request.seed, request.alpha, request.trees)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)[:300]) from exc
    except Exception as exc:
        # Keep client errors useful while avoiding dumping local stack traces in browser.
        raise HTTPException(500, f"Training failed: {type(exc).__name__}: {str(exc)[:160]}") from exc
    run_id = new_identifier()
    directory = path("runs", run_id)
    for export_file in ("curated.csv", "qc_rows.csv", "raw.csv"):
        shutil.copyfile(dataset_dir / export_file, directory / export_file)
    dump_json(directory / "audit.json", audit)
    split = result.fitted["split"]
    dump_json(directory / "split_indices.json", {k: v.tolist() for k, v in split.items()})
    manifest = {"run_id": run_id, "dataset_id": request.dataset_id,
                "source_sha256": audit["sha256"], "demo_data": bool(audit.get("demo_data", False)),
                "version": app.version, "split_type": request.split_type, "seed": request.seed,
                "alpha": request.alpha, "trees": request.trees}
    generate_report(directory, result.summary, result.residuals, audit, manifest)
    return {"id": run_id, "dataset_id": request.dataset_id, **result.summary}


@app.get("/api/runs")
def runs_list():
    return [{"id": x["id"], "split_type": x["split_type"], "test_n": x["test_n"]} for x in listing("runs")]


@app.get("/api/runs/{run_id}")
def run_results(run_id: str):
    try:
        return load_json(existing_path("runs", run_id) / "metrics.json")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Run not found")


@app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
def html_report(run_id: str):
    try:
        return (existing_path("runs", run_id) / "report.html").read_text(encoding="utf8")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Run not found")


@app.get("/api/runs/{run_id}/figures/{name}")
def figure(run_id: str, name: Literal["observed_vs_predicted.png", "coverage.png"]):
    try:
        return FileResponse(existing_path("runs", run_id) / name, media_type="image/png")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Figure not found")


@app.get("/api/runs/{run_id}/bundle")
def download_bundle(run_id: str):
    try:
        return FileResponse(existing_path("runs", run_id) / "research_bundle.zip", filename="assaylens_research_bundle.zip")
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Run not found")


@app.get("/api/demo.csv")
def demo_csv():
    return FileResponse(APP_ROOT / "examples" / "synthetic_assay_demo.csv", filename="synthetic_assay_demo.csv")


@app.get("/api/runs/{run_id}/{name}")
def report_relative_figure(run_id: str, name: Literal["observed_vs_predicted.png", "coverage.png"]):
    # Allows the HTML report's relative image URLs to render in browser and ZIP.
    return figure(run_id, name)
