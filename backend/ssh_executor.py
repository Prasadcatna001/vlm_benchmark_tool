from __future__ import annotations
import asyncio
import time
import paramiko
from pathlib import Path
from .models import GpuSample


def _connect(node: dict) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=node["host"],
        port=node.get("port", 22),
        username=node["user"],
        key_filename=str(Path(node["key_path"]).expanduser()),
        timeout=15,
    )
    return client


def _parse_nvidia_smi_csv(raw: str, gpu_indices: list[int], t: float) -> list[GpuSample]:
    samples = []
    for line in raw.strip().splitlines():
        # index, utilization.gpu, memory.used, memory.total, temperature.gpu
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5:
            continue
        idx = int(cols[0])
        if idx not in gpu_indices:
            continue
        samples.append(GpuSample(
            t=t,
            gpu_index=idx,
            utilization_pct=float(cols[1]),
            memory_used_mb=float(cols[2]),
            memory_total_mb=float(cols[3]),
            temperature_c=float(cols[4]) if cols[4] not in ("", "N/A") else None,
        ))
    return samples


async def run_job_over_ssh(node: dict, remote_env_activate: str, command: str,
                            gpu_poll_interval: float, on_gpu_sample=None,
                            on_output_line=None) -> dict:
    """
    Runs `command` on the remote node inside its workdir, polling nvidia-smi
    concurrently. Returns dict with stdout/stderr tails, exit_code, duration,
    and all gpu samples collected.

    on_gpu_sample(GpuSample) and on_output_line(str) are optional callbacks
    for streaming updates (e.g. to a websocket) as the job runs.
    """
    loop = asyncio.get_event_loop()
    client = await loop.run_in_executor(None, _connect, node)

    workdir = node.get("workdir", "~")
    full_cmd = f"cd {workdir} && {remote_env_activate} && {command}"

    start = time.time()
    stdin, stdout, stderr = await loop.run_in_executor(
        None, lambda: client.exec_command(full_cmd, get_pty=True)
    )
    channel = stdout.channel

    all_gpu_samples: list[GpuSample] = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def poll_gpu():
        gpu_client = await loop.run_in_executor(None, _connect, node)
        try:
            while not channel.exit_status_ready():
                elapsed = time.time() - start
                query = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,"
                          "memory.total,temperature.gpu --format=csv,noheader,nounits")
                _, o, _ = await loop.run_in_executor(None, lambda: gpu_client.exec_command(query))
                raw = await loop.run_in_executor(None, o.read)
                samples = _parse_nvidia_smi_csv(raw.decode(errors="ignore"),
                                                 node.get("gpu_indices", [0]), elapsed)
                for s in samples:
                    all_gpu_samples.append(s)
                    if on_gpu_sample:
                        on_gpu_sample(s)
                await asyncio.sleep(gpu_poll_interval)
        finally:
            gpu_client.close()

    async def stream_output():
        while not channel.exit_status_ready() or channel.recv_ready():
            if channel.recv_ready():
                chunk = channel.recv(4096).decode(errors="ignore")
                stdout_lines.append(chunk)
                if on_output_line:
                    on_output_line(chunk)
            else:
                await asyncio.sleep(0.5)

    await asyncio.gather(poll_gpu(), stream_output())

    exit_code = channel.recv_exit_status()
    stderr_lines.append(await loop.run_in_executor(None, stderr.read))
    duration = time.time() - start

    client.close()

    return {
        "exit_code": exit_code,
        "duration_seconds": duration,
        "stdout_tail": "".join(stdout_lines)[-4000:],
        "stderr_tail": (stderr_lines[-1].decode(errors="ignore") if stderr_lines and
                         isinstance(stderr_lines[-1], bytes) else "")[-4000:],
        "gpu_samples": all_gpu_samples,
    }


async def scp_download(node: dict, remote_path: str, local_path: str):
    loop = asyncio.get_event_loop()
    client = await loop.run_in_executor(None, _connect, node)
    try:
        sftp = client.open_sftp()
        await loop.run_in_executor(None, sftp.get, remote_path, local_path)
        sftp.close()
    finally:
        client.close()
