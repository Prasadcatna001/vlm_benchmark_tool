from __future__ import annotations
import fnmatch
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


def _is_directory_param(param: dict) -> bool:
    name = param["name"].lower()
    flag = param.get("flag", "").lower()
    return name.endswith("_root") or name.endswith("_dir") or "root" in name or "dir" in name or flag.endswith("root") or flag.endswith("dir")


def _infer_search_patterns(param: dict) -> list[str]:
    patterns: list[str] = []
    default_path = param.get("default_path") or param.get("default") or ""
    default_name = Path(default_path).name
    if default_name and not default_name.startswith("path"):
        patterns.append(default_name)
        patterns.append(f"*{default_name}*")

    name = param["name"].lower()
    if "checkpoint" in name:
        patterns.extend(["*checkpoint*.safetensors", "*.safetensors"])
    elif "lora" in name:
        patterns.extend(["*lora*.safetensors", "*lora*", "*.safetensors"])
    elif "upsampler" in name:
        patterns.extend(["*upsampler*", "*upsampler*.safetensors"])
    elif "gemma" in name:
        patterns.extend(["*gemma*", "*gemma_root*"])
    elif "prompt" in name:
        patterns.append("*prompt*")
    else:
        patterns.append(f"*{name}*")

    return list(dict.fromkeys(patterns))


def _scan_file_param(root: Path, param: dict) -> str | None:
    default_path = param.get("default_path") or param.get("default") or ""
    if root.is_file():
        if default_path and Path(default_path).name.lower() == root.name.lower():
            return str(root.resolve())
    if root.is_dir():
        if default_path:
            exact = root / Path(default_path).name
            if exact.exists():
                return str(exact.resolve())
        patterns = _infer_search_patterns(param)
        for pat in patterns:
            for candidate in root.rglob(pat):
                if candidate.is_file():
                    return str(candidate.resolve())
    return None


def _scan_dir_param(root: Path, param: dict) -> str | None:
    if root.is_dir():
        patterns = _infer_search_patterns(param)
        if any(fnmatch.fnmatch(root.name.lower(), pat.lower()) for pat in patterns):
            return str(root.resolve())
        for candidate in root.rglob("*"):
            if candidate.is_dir() and any(fnmatch.fnmatch(candidate.name.lower(), pat.lower()) for pat in patterns):
                return str(candidate.resolve())
        return str(root.resolve())
    return None


def scan_model_path(model: str, root_path: str) -> dict:
    pipelines = [p for p in load_all_pipelines().values() if p["model"] == model]
    if not pipelines:
        raise KeyError(f"Unknown model '{model}'")

    root = Path(root_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path '{root}' does not exist")

    path_params: dict[str, dict] = {}
    for pipeline in pipelines:
        for param in pipeline.get("params", []):
            if param.get("auto"):
                continue
            if param["type"] in ("path", "path_and_weight"):
                path_params[param["name"]] = param

    matches: dict[str, str | None] = {}
    for name, param in path_params.items():
        if _is_directory_param(param):
            matches[name] = _scan_dir_param(root, param)
        else:
            matches[name] = _scan_file_param(root, param)

    return {"model": model, "root_path": str(root.resolve()), "matches": matches}


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
