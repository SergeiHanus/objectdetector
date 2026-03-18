#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


THIS_FILE = Path(__file__).resolve()
TRAIN_YOLOX = THIS_FILE.parents[3]
REPO_ROOT = TRAIN_YOLOX.parent
STATE_FILE = TRAIN_YOLOX / "ralph_state.json"
DEFAULT_CONFIG_PATHS = [
    TRAIN_YOLOX / "config" / "remote.yaml",
    TRAIN_YOLOX / "config" / "remote.example.yaml",
]


def parse_stage_args(stage_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Ralph stage runner: {stage_name}")
    parser.add_argument("--config", default="", help="Path to config yaml")
    parser.add_argument("--state", default=str(STATE_FILE), help="Path to state json")
    return parser.parse_args()


def load_config(config_path: str = "") -> dict:
    if not yaml:
        raise RuntimeError("PyYAML required for ralph stage scripts. pip install pyyaml")
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _fill_config_defaults(data)
    for p in DEFAULT_CONFIG_PATHS:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _fill_config_defaults(data)
    raise FileNotFoundError("No config found. Copy config/remote.example.yaml to config/remote.yaml and edit.")


def _fill_config_defaults(data: dict) -> dict:
    local = data.setdefault("local", {})
    if local.get("repo_root") is None:
        local["repo_root"] = str(REPO_ROOT)
    return data


def load_state(state_path: str) -> dict:
    p = Path(state_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_git_revision() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip()
    except Exception:
        return None
    return None


def _ssh_opts(cfg: dict) -> str:
    key = cfg.get("ssh_key_path")
    if key and os.path.isfile(key):
        return f"-o ConnectTimeout=10 -o BatchMode=yes -i {key}"
    return "-o ConnectTimeout=10 -o BatchMode=yes"


def ssh_run(cfg: dict, command: str, timeout: int | None = 60) -> tuple[bool, str, str]:
    target = cfg["ssh_target"]
    try:
        # Avoid `shell=True` and manual quoting: pass the remote command as a single SSH argument.
        # This is more reliable for commands that contain quotes, pipes, or multi-line Python snippets.
        key = cfg.get("ssh_key_path")
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if key and os.path.isfile(key):
            cmd.extend(["-i", key])
        cmd.append(target)
        cmd.append(command)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as exc:
        return False, "", str(exc)


def rsync_to_remote(cfg: dict, local_path: str, remote_path: str, excludes: list[str] | None = None) -> tuple[bool, str, str]:
    target = cfg["ssh_target"]
    dest = f"{target}:{remote_path}"
    key = cfg.get("ssh_key_path")
    ssh_cmd = f"ssh -i {key}" if key and os.path.isfile(key) else "ssh"
    cmd = ["rsync", "-az", "--timeout=1800", "-e", ssh_cmd]
    for ex in excludes or []:
        cmd.append(f"--exclude={ex}")
    cmd.append(local_path.rstrip("/") + "/")
    cmd.append(dest)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, r.stdout or "", r.stderr or ""
    except Exception as exc:
        return False, "", str(exc)


def rsync_from_remote(cfg: dict, remote_path: str, local_path: str) -> tuple[bool, str, str]:
    target = cfg["ssh_target"]
    src = f"{target}:{remote_path}"
    key = cfg.get("ssh_key_path")
    ssh_cmd = f"ssh -i {key}" if key and os.path.isfile(key) else "ssh"
    Path(local_path).mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["rsync", "-az", "--timeout=1800", "-e", ssh_cmd, src, local_path.rstrip("/") + "/"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return r.returncode == 0, r.stdout or "", r.stderr or ""
    except Exception as exc:
        return False, "", str(exc)


def raw_data_subdir(cfg: dict) -> str:
    """Subdir under data/ where raw images and YOLO labels live (config sync.raw_data_subdir)."""
    return cfg.get("sync", {}).get("raw_data_subdir") or "Markers01.20026"


def dataset_path(cfg: dict) -> Path:
    raw = cfg.get("sync", {}).get("dataset_source") or "data"
    if os.path.isabs(raw):
        return Path(raw)
    repo_root = Path(cfg.get("local", {}).get("repo_root") or REPO_ROOT)
    if repo_root.name == "train_yolox" and raw.startswith("train_yolox/"):
        raw = raw[len("train_yolox/"):]
    return repo_root / raw


def pending_feedback(state: dict, stage_name: str) -> dict:
    fb = state.get("pending_feedback") or {}
    if fb.get("stage") == stage_name:
        return fb
    return {}


def stage_ok(message: str, details: dict | None = None) -> int:
    payload = {"ok": True, "message": message}
    if details:
        payload["details"] = details
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def stage_fail(message: str, details: dict | None = None) -> int:
    payload = {"ok": False, "message": message}
    if details:
        payload["details"] = details
    print(json.dumps(payload, ensure_ascii=True))
    return 1


def ensure_uv(cfg: dict, remote_root: str) -> str:
    """Ensure uv is available on remote. Returns a shell fragment exporting PATH so uv is resolvable."""
    uv_dir = cfg.get("remote", {}).get("uv_install_dir") or f"{remote_root}/.local"
    uv_bin = f"{uv_dir}/bin"
    path_export = f"export PATH=\"$PATH:{uv_dir}:{uv_bin}\""
    check = f"cd {remote_root} && ({path_export}; command -v uv >/dev/null 2>&1 && uv --version) || true"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and out and "uv" in (out or ""):
        return path_export

    install = (
        f"cd {remote_root} && mkdir -p {uv_dir} && "
        "curl -LsSf https://astral.sh/uv/install.sh | "
        f"env UV_NO_MODIFY_PATH=1 UV_UNMANAGED_INSTALL={uv_dir} sh"
    )
    ok_install, _, err_install = ssh_run(cfg, install, timeout=120)
    if not ok_install:
        raise RuntimeError(f"uv install failed: {err_install or 'unknown'}")
    return path_export
