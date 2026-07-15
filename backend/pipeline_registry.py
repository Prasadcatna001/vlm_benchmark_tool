from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
PIPELINES_DIR = CONFIG_DIR / "pipelines"


def load_all_pipelines() -> dict[str, dict]:
    """Returns {pipeline_name: pipeline_def} for every yaml file in config/pipelines."""
    pipelines = {}
    for f in sorted(PIPELINES_DIR.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        pipelines[data["name"]] = data
    return pipelines


def load_pipeline(name: str) -> dict:
    pipelines = load_all_pipelines()
    if name not in pipelines:
        raise KeyError(f"Unknown pipeline '{name}'. Known: {list(pipelines)}")
    return pipelines[name]


def load_nodes_config() -> dict:
    with open(CONFIG_DIR / "nodes.yaml") as fh:
        return yaml.safe_load(fh)


def build_command(pipeline_def: dict, shared_prompt: str, overrides: list,
                   output_path: str) -> tuple[str, dict]:
    """
    Builds the `python -m module --flag value ...` command string for one job.

    overrides: list of ParamOverride-like dicts with keys name/value/weight/enabled
    Returns (command_string, params_used_dict) for logging/report purposes.
    """
    override_by_name = {o["name"]: o for o in overrides} if overrides else {}
    parts = [f"python -m {pipeline_def['module']}"]
    params_used: dict[str, Any] = {}

    for p in pipeline_def.get("params", []):
        name = p["name"]
        ptype = p["type"]
        ov = override_by_name.get(name)

        if p.get("shared") and name == "prompt":
            value = shared_prompt
            parts.append(f'--prompt "{value}"')
            params_used["prompt"] = value
            continue

        if p.get("auto") and name == "output_path":
            parts.append(f'{p["flag"]} "{output_path}"')
            params_used["output_path"] = output_path
            continue

        if ptype == "path_and_weight":
            path_val = ov["value"] if ov and ov.get("value") else p.get("default_path")
            weight_val = ov["weight"] if ov and ov.get("weight") is not None else p.get("default_weight")
            if path_val:
                parts.append(f'{p["flag"]} "{path_val}" {weight_val}')
                params_used[name] = {"path": path_val, "weight": weight_val}
            elif p.get("required"):
                raise ValueError(f"Missing required param '{name}' for {pipeline_def['name']}")
            continue

        if ptype == "bool_flag":
            enabled = ov["enabled"] if ov and ov.get("enabled") is not None else p.get("default", False)
            if enabled:
                parts.append(p["flag"])
            params_used[name] = bool(enabled)
            continue

        # plain path / string / int
        value = ov["value"] if ov and ov.get("value") not in (None, "") else p.get("default")
        if value in (None, "null"):
            if p.get("required"):
                raise ValueError(f"Missing required param '{name}' for {pipeline_def['name']}")
            continue
        if ptype == "path" or ptype == "string":
            parts.append(f'{p["flag"]} "{value}"')
        else:
            parts.append(f'{p["flag"]} {value}')
        params_used[name] = value

    return " \\\n    ".join(parts), params_used
