from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


class ParamOverride(BaseModel):
    """A user-supplied value for one param of one pipeline in a run."""
    name: str
    value: Optional[str] = None       # for path/string/int
    weight: Optional[float] = None    # for path_and_weight
    enabled: Optional[bool] = None    # for bool_flag


class PipelineJobRequest(BaseModel):
    pipeline_name: str                # e.g. "ltx_two_stage"
    node_id: Optional[str] = None     # if None, orchestrator picks one
    overrides: list[ParamOverride] = []


class RunCreateRequest(BaseModel):
    model: str                        # "ltx" | "wan"
    prompt: str
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    jobs: list[PipelineJobRequest]
    label: Optional[str] = None       # optional tag for this run


class GpuSample(BaseModel):
    t: float                          # seconds since job start
    gpu_index: int
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float] = None


class PipelineJobResult(BaseModel):
    pipeline_name: str
    display_name: str
    node_id: str
    command: str
    status: Literal["pending", "running", "succeeded", "failed"] = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    output_video_local_path: Optional[str] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    error: Optional[str] = None
    gpu_samples: list[GpuSample] = []
    peak_memory_mb: Optional[float] = None
    avg_utilization_pct: Optional[float] = None
    params_used: dict = {}


class RunRecord(BaseModel):
    run_id: str
    label: Optional[str] = None
    model: str
    prompt: str
    execution_mode: str
    created_at: str
    status: Literal["pending", "running", "succeeded", "failed", "partial"] = "pending"
    jobs: list[PipelineJobResult] = []
