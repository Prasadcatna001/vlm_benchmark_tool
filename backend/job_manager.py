from __future__ import annotations
import asyncio
import itertools
from datetime import datetime, timezone
from pathlib import Path

from .models import RunCreateRequest, RunRecord, PipelineJobResult, GpuSample
from .pipeline_registry import load_pipeline, load_nodes_config, build_command
from .ssh_executor import run_job_over_ssh, scp_download
from . import storage

_node_cycle_state: dict[str, itertools.cycle] = {}


def _pick_node(nodes_cfg: dict, requested_node_id: str | None) -> dict:
    nodes = nodes_cfg["nodes"]
    if requested_node_id:
        for n in nodes:
            if n["id"] == requested_node_id:
                return n
        raise ValueError(f"Unknown node_id '{requested_node_id}'")
    # simple round robin across all configured nodes
    if "default" not in _node_cycle_state:
        _node_cycle_state["default"] = itertools.cycle(nodes)
    return next(_node_cycle_state["default"])


async def _execute_one_job(req_job, nodes_cfg: dict, prompt: str, run_id: str) -> PipelineJobResult:
    pipeline_def = load_pipeline(req_job.pipeline_name)
    node = _pick_node(nodes_cfg, req_job.node_id)

    remote_output = f"{node['workdir']}/runs_output/{run_id}_{req_job.pipeline_name}.mp4"
    overrides = [o.model_dump() for o in req_job.overrides]
    command, params_used = build_command(pipeline_def, prompt, overrides, remote_output)

    result = PipelineJobResult(
        pipeline_name=req_job.pipeline_name,
        display_name=pipeline_def["display_name"],
        node_id=node["id"],
        command=command,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        params_used=params_used,
    )

    try:
        outcome = await run_job_over_ssh(
            node=node,
            remote_env_activate=nodes_cfg.get("remote_env_activate", "true"),
            command=command,
            gpu_poll_interval=nodes_cfg.get("gpu_poll_interval_seconds", 2),
        )
        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = outcome["duration_seconds"]
        result.stdout_tail = outcome["stdout_tail"]
        result.stderr_tail = outcome["stderr_tail"]
        result.gpu_samples = outcome["gpu_samples"]
        if result.gpu_samples:
            result.peak_memory_mb = max(s.memory_used_mb for s in result.gpu_samples)
            result.avg_utilization_pct = sum(s.utilization_pct for s in result.gpu_samples) / len(result.gpu_samples)

        if outcome["exit_code"] != 0:
            result.status = "failed"
            result.error = f"Remote command exited with code {outcome['exit_code']}"
            return result

        # pull the generated video back to local runs dir for the report
        local_dir = storage.run_dir(run_id)
        local_video = local_dir / f"{req_job.pipeline_name}.mp4"
        await scp_download(node, remote_output, str(local_video))
        result.output_video_local_path = local_video.name
        result.status = "succeeded"

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.finished_at = datetime.now(timezone.utc).isoformat()

    return result


async def execute_run(req: RunCreateRequest) -> RunRecord:
    nodes_cfg = load_nodes_config()
    run_id = storage.new_run_id()

    record = RunRecord(
        run_id=run_id,
        label=req.label,
        model=req.model,
        prompt=req.prompt,
        execution_mode=req.execution_mode,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        jobs=[],
    )
    storage.save_run(record)  # so it shows up on the home page immediately

    if req.execution_mode == "parallel":
        results = await asyncio.gather(
            *[_execute_one_job(j, nodes_cfg, req.prompt, run_id) for j in req.jobs]
        )
    else:
        results = []
        for j in req.jobs:
            results.append(await _execute_one_job(j, nodes_cfg, req.prompt, run_id))
            record.jobs = results
            storage.save_run(record)  # incremental save after each pipeline

    record.jobs = list(results)
    statuses = {r.status for r in record.jobs}
    if statuses == {"succeeded"}:
        record.status = "succeeded"
    elif "succeeded" in statuses:
        record.status = "partial"
    else:
        record.status = "failed"

    storage.save_run(record)

    from .report_generator import generate_html_report, generate_csv_report
    generate_html_report(record)
    generate_csv_report(record)

    return record
