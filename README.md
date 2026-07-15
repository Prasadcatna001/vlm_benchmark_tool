# LTX / WAN Pipeline Runner

A small self-hosted "Jenkins for video-gen pipelines" dashboard. Pick a model
(LTX/WAN), pick N pipeline configs, give one prompt, run them across your 4
HPC nodes over SSH, and get back a shareable HTML report with GPU
utilization charts, the generated videos, and JSON/CSV comparison data.
Run history lives on the home page.

## What's real vs. stubbed

- `config/pipelines/ltx_two_stage.yaml` is built from the exact command you
  shared and should work as-is (once paths are real).
- `config/pipelines/ltx_one_stage.yaml` and `ltx_hq_two_stage.yaml` are
  **guesses** — I don't have your real commands for those. Paste them to me
  and I'll fix the `module` and flag names, or edit the YAML yourself; the
  app reads pipeline definitions dynamically, no code changes needed.
- `config/nodes.yaml` has placeholder IPs/paths for your 4 nodes — fill in
  real host/user/key_path/workdir and the conda/venv activate command.
- There's no WAN pipeline yaml yet since you haven't shared that command.
  Same process: drop a `config/pipelines/wan_*.yaml` with `model: wan`.

## How it works

1. **Pipeline registry** (`backend/pipeline_registry.py`) reads the YAML
   files in `config/pipelines/` and builds the `python -m module --flags`
   command from your param overrides. Add a new pipeline = add a new YAML
   file, nothing else changes.
2. **Job manager** (`backend/job_manager.py`) picks a node per pipeline
   (round-robin, or one you chose in the UI), builds the command, and runs
   it — sequential or parallel per-run, as you choose in the New Run form.
3. **SSH executor** (`backend/ssh_executor.py`) opens an SSH session to the
   node, runs the command in its workdir, and concurrently polls
   `nvidia-smi` every N seconds (`gpu_poll_interval_seconds` in
   `nodes.yaml`) to capture utilization/memory/temperature over time. When
   the job finishes it SFTPs the output `.mp4` back locally.
4. **Report generator** builds `runs/<run_id>/report.html` (video previews,
   GPU charts via Chart.js, param diff table) plus `report.json` and
   `report.csv`.
5. **Frontend** is plain HTML/CSS/JS (matches what you already had) served
   by FastAPI: home page (run history), new-run form (dynamically generated
   from the pipeline YAMLs — only shows the params each pipeline actually
   needs), and a run-detail page that polls status while jobs run and links
   out to the full report.

## Running it

```bash
cd ltx-pipeline-runner
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## Known gaps / things to decide next

- **SSH auth**: assumes key-based auth (`key_path` in `nodes.yaml`). If you
  use a jump host or passphrase-protected keys, `ssh_executor.py` needs a
  small update (paramiko supports both, just needs config).
- **Concurrent runs**: right now one run's HTTP request blocks until all its
  jobs finish. Fine solo, but if you ever kick off two runs at once from two
  tabs they won't interfere (each writes to its own `runs/<run_id>/`), just
  know the second request just waits its turn on the event loop with
  parallel jobs, or runs concurrently — worth testing once real.
- **GPU polling granularity**: currently node-level `nvidia-smi`, filtered
  to `gpu_indices` in `nodes.yaml`. If a node exposes multiple GPUs and only
  one is used per job, this already isolates by index — but if two jobs ever
  share a node, their GPU curves will look identical (both see the whole
  node). Not an issue with round-robin single-job-per-node scheduling.
- **Auth/exposure**: no login on the dashboard — fine on localhost, but
  don't expose port 8000 publicly as-is.
- Once your one-stage and HQ commands are in hand, worth double-checking the
  `unquantized` bool-flag mechanic actually matches how that CLI flag works
  (flag-with-no-value vs `--unquantized true`).
