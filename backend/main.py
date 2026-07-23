from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .models import RunCreateRequest
from .pipeline_registry import load_all_pipelines, scan_model_path
from . import storage, job_manager

app = FastAPI(title="LTX/WAN Pipeline Runner")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
app.mount("/runs-files", StaticFiles(directory=RUNS_DIR), name="runs-files")


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/new")
def new_run_page():
    return FileResponse(FRONTEND_DIR / "new_run.html")


@app.get("/settings")
def settings_page():
    return FileResponse(FRONTEND_DIR / "settings.html")


@app.get("/run/{run_id}")
def run_detail_page(run_id: str):
    return FileResponse(FRONTEND_DIR / "run_detail.html")


@app.get("/api/pipelines")
def api_pipelines():
    """Returns all pipeline definitions, grouped by model, for the New Run form."""
    return load_all_pipelines()


@app.get("/api/nodes")
def api_nodes():
    from .pipeline_registry import load_nodes_config
    cfg = load_nodes_config()
    return [{"id": n["id"], "gpu_indices": n["gpu_indices"]} for n in cfg["nodes"]]


@app.post("/api/scan-model-path")
def api_scan_model_path(req: dict):
    model = req.get("model")
    path = req.get("path")
    if not model or not path:
        raise HTTPException(422, "Both model and path are required")
    try:
        return scan_model_path(model, path)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Scan failed: {exc}")


@app.get("/api/runs")
def api_list_runs():
    return storage.list_runs()


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str):
    try:
        return storage.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.post("/api/runs")
async def api_create_run(req: RunCreateRequest):
    """
    Kicks off a comparison run. Executes synchronously for now (the caller's
    request stays open until all jobs finish) -- fine for a personal tool.
    For long multi-pipeline runs, poll GET /api/runs/{run_id} from another
    tab instead of waiting on this call; run_id is written to storage
    before jobs start so it's visible on the home page immediately.
    """
    record = await job_manager.execute_run(req)
    return record


@app.get("/runs-files/{run_id}/report.html")
def report_html(run_id: str):
    return FileResponse(RUNS_DIR / run_id / "report.html")
