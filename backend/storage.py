from __future__ import annotations
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from .models import RunRecord

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(record: RunRecord):
    d = run_dir(record.run_id)
    with open(d / "report.json", "w") as f:
        f.write(record.model_dump_json(indent=2))


def load_run(run_id: str) -> RunRecord:
    d = run_dir(run_id)
    with open(d / "report.json") as f:
        return RunRecord.model_validate_json(f.read())


def list_runs() -> list[dict]:
    """Lightweight summaries for the home page, newest first."""
    summaries = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        report = d / "report.json"
        if not report.exists():
            continue
        try:
            r = RunRecord.model_validate_json(report.read_text())
        except Exception:
            continue
        summaries.append({
            "run_id": r.run_id,
            "label": r.label,
            "model": r.model,
            "prompt": r.prompt,
            "status": r.status,
            "created_at": r.created_at,
            "pipelines": [j.display_name for j in r.jobs],
        })
    return summaries
