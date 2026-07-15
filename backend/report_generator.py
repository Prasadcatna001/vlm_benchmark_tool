from __future__ import annotations
import csv
from pathlib import Path
from .models import RunRecord
from . import storage

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Run {run_id} - Pipeline Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; background:#0f1115; color:#e6e6e6; margin:0; padding:32px; }}
  h1 {{ font-size:22px; margin-bottom:4px; }}
  .meta {{ color:#9aa0a6; margin-bottom:24px; font-size:14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:20px; margin-bottom:32px; }}
  .card {{ background:#181b21; border:1px solid #262b33; border-radius:10px; padding:16px; }}
  .card h2 {{ font-size:16px; margin:0 0 8px 0; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }}
  .badge.succeeded {{ background:#12351f; color:#4ade80; }}
  .badge.failed {{ background:#3a1414; color:#f87171; }}
  .badge.running {{ background:#332a10; color:#facc15; }}
  video {{ width:100%; border-radius:8px; margin:10px 0; background:#000; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
  td, th {{ text-align:left; padding:6px 4px; border-bottom:1px solid #262b33; }}
  th {{ color:#9aa0a6; font-weight:500; }}
  pre {{ background:#0b0d11; padding:8px; border-radius:6px; overflow-x:auto; font-size:11px; max-height:150px; }}
  .compare-table {{ width:100%; border-collapse:collapse; margin-bottom:32px; }}
  .compare-table th, .compare-table td {{ border:1px solid #262b33; padding:8px; font-size:13px; }}
  .compare-table th {{ background:#181b21; }}
  canvas {{ max-height:220px; }}
</style>
</head>
<body>
  <h1>Pipeline Comparison — {run_id}</h1>
  <div class="meta">
    Model: <b>{model}</b> &nbsp;|&nbsp; Mode: <b>{execution_mode}</b> &nbsp;|&nbsp; Created: {created_at}<br>
    Prompt: <i>"{prompt}"</i>
  </div>

  <table class="compare-table">
    <tr><th>Pipeline</th><th>Node</th><th>Status</th><th>Duration (s)</th><th>Peak GPU Mem (MB)</th><th>Avg GPU Util (%)</th></tr>
    {compare_rows}
  </table>

  <div class="grid">
    {cards}
  </div>

<script>
{chart_scripts}
</script>
</body>
</html>
"""

CARD_TEMPLATE = """
<div class="card">
  <h2>{display_name} <span class="badge {status}">{status}</span></h2>
  <div style="font-size:12px;color:#9aa0a6;">node: {node_id} · duration: {duration}s</div>
  {video_html}
  <canvas id="chart-{chart_id}"></canvas>
  <table>
    {param_rows}
  </table>
  {error_html}
  <details><summary style="cursor:pointer;color:#9aa0a6;font-size:12px;">stdout tail</summary>
  <pre>{stdout_tail}</pre></details>
</div>
"""


def _fmt(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f}"


def generate_html_report(record: RunRecord) -> Path:
    run_dir = storage.run_dir(record.run_id)

    compare_rows = ""
    cards = ""
    chart_scripts = ""

    for i, job in enumerate(record.jobs):
        compare_rows += (
            f"<tr><td>{job.display_name}</td><td>{job.node_id}</td>"
            f"<td>{job.status}</td><td>{_fmt(job.duration_seconds)}</td>"
            f"<td>{_fmt(job.peak_memory_mb, 0)}</td><td>{_fmt(job.avg_utilization_pct)}</td></tr>\n"
        )

        video_html = ""
        if job.output_video_local_path:
            video_html = f'<video controls src="{job.output_video_local_path}"></video>'

        error_html = f'<div style="color:#f87171;font-size:12px;">{job.error}</div>' if job.error else ""

        param_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in job.params_used.items()
        )

        cards += CARD_TEMPLATE.format(
            display_name=job.display_name,
            status=job.status,
            node_id=job.node_id,
            duration=_fmt(job.duration_seconds),
            video_html=video_html,
            chart_id=i,
            param_rows=param_rows,
            error_html=error_html,
            stdout_tail=(job.stdout_tail or "")[-1500:],
        )

        labels = [f"{s.t:.0f}s" for s in job.gpu_samples]
        util = [s.utilization_pct for s in job.gpu_samples]
        mem = [s.memory_used_mb for s in job.gpu_samples]
        chart_scripts += f"""
        new Chart(document.getElementById('chart-{i}'), {{
          type: 'line',
          data: {{
            labels: {labels},
            datasets: [
              {{ label: 'GPU Util %', data: {util}, borderColor:'#60a5fa', yAxisID:'y' }},
              {{ label: 'Mem MB', data: {mem}, borderColor:'#facc15', yAxisID:'y1' }}
            ]
          }},
          options: {{
            responsive: true,
            scales: {{
              y: {{ type:'linear', position:'left', title:{{display:true,text:'Util %'}} }},
              y1: {{ type:'linear', position:'right', grid:{{drawOnChartArea:false}}, title:{{display:true,text:'MB'}} }}
            }}
          }}
        }});
        """

    html = HTML_TEMPLATE.format(
        run_id=record.run_id,
        model=record.model,
        execution_mode=record.execution_mode,
        created_at=record.created_at,
        prompt=record.prompt,
        compare_rows=compare_rows,
        cards=cards,
        chart_scripts=chart_scripts,
    )

    out = run_dir / "report.html"
    out.write_text(html)
    return out


def generate_csv_report(record: RunRecord) -> Path:
    run_dir = storage.run_dir(record.run_id)
    out = run_dir / "report.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "node", "status", "duration_seconds",
                          "peak_memory_mb", "avg_utilization_pct", "error"])
        for job in record.jobs:
            writer.writerow([job.pipeline_name, job.node_id, job.status,
                              job.duration_seconds, job.peak_memory_mb,
                              job.avg_utilization_pct, job.error or ""])
    return out
