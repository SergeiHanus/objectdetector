#!/usr/bin/env python3
"""
Ralph loop: stage-transition orchestrator.
Each stage is a standalone script under scripts/ralph/stages/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_YOLOX = SCRIPT_DIR.parent
STATE_FILE = TRAIN_YOLOX / "ralph_state.json"
CONFIG_PATHS = [TRAIN_YOLOX / "config" / "remote.yaml", TRAIN_YOLOX / "config" / "remote.example.yaml"]
STAGE_SCRIPTS_DIR = SCRIPT_DIR / "ralph" / "stages"

STAGES: list[tuple[str, str]] = [
    ("preflight_local", "01_preflight_local.py"),
    ("ssh_connect", "02_ssh_connect.py"),
    ("sync_code", "03_sync_code.py"),
    ("ensure_env", "04_ensure_env.py"),
    ("sync_data", "05_sync_data.py"),
    ("prepare_data", "05b_prepare_data.py"),
    ("smoke_device", "06_smoke_device.py"),
    ("smoke_train", "07_smoke_train.py"),
    ("full_train", "08_full_train.py"),
    ("verify_outputs", "09_verify_outputs.py"),
    ("pull_artifacts", "10_pull_artifacts.py"),
]
STAGE_ORDER = [name for name, _ in STAGES]


def load_config(config_path: str = "") -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for ralph_loop. pip install pyyaml") from exc

    paths = [Path(config_path)] if config_path else CONFIG_PATHS
    for p in paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError("No config found. Copy config/remote.example.yaml to config/remote.yaml and edit.")


def resolve_config_path(config_path: str = "") -> Path:
    if config_path:
        return Path(config_path)
    for p in CONFIG_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("No config found")


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "iteration": 0,
        "stage_results": {},
        "last_stage": None,
        "last_error": None,
        "pending_feedback": None,
        "agent_last_action": None,
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_stage_output(stdout: str, stderr: str) -> tuple[bool, str, dict]:
    lines = [x.strip() for x in (stdout or "").splitlines() if x.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict) and "ok" in payload and "message" in payload:
            return bool(payload["ok"]), str(payload["message"]), payload.get("details") or {}
    message = (stderr or stdout or "stage returned unparseable output").strip()
    return False, message, {}


def run_stage_script(stage_name: str, script_path: Path, config_path: Path, state_path: Path) -> tuple[bool, str, dict, str, str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--state",
        str(state_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return False, f"{stage_name} timed out", {}, "", "timeout"
    ok, message, details = parse_stage_output(r.stdout or "", r.stderr or "")
    if r.returncode != 0 and ok:
        ok = False
    return ok, message, details, r.stdout or "", r.stderr or ""


def run_agent_repair(cfg: dict, state: dict, stage_name: str, stage_script: Path, config_path: Path) -> tuple[bool, str, str, str]:
    command = (cfg.get("loop", {}) or {}).get("agent_repair_command", "")
    if not command:
        return False, "agent_repair_command not configured", "", ""

    failure = {
        "stage": stage_name,
        "error": state.get("last_error"),
        "iteration": state.get("iteration"),
        "stage_script": str(stage_script),
        "state_file": str(STATE_FILE),
        "config_file": str(config_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    env = dict(os.environ)
    env["RALPH_FAILURE_JSON"] = json.dumps(failure, ensure_ascii=True)
    env["RALPH_FAILED_STAGE"] = stage_name
    env["RALPH_STAGE_SCRIPT"] = str(stage_script)
    env["RALPH_STAGE_ERROR"] = str(state.get("last_error") or "")
    env["RALPH_STATE_FILE"] = str(STATE_FILE)
    env["RALPH_CONFIG_FILE"] = str(config_path)

    timeout = int((cfg.get("loop", {}) or {}).get("agent_repair_timeout_sec", 300))
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        state["agent_last_action"] = {
            "stage": stage_name,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
        return False, f"agent repair command timed out after {timeout}s", "", f"timeout after {timeout}s"
    state["agent_last_action"] = {
        "stage": stage_name,
        "command": command,
        "exit_code": r.returncode,
        "stdout": (r.stdout or "").strip(),
        "stderr": (r.stderr or "").strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)
    if r.returncode == 0:
        return True, "agent repair command succeeded", r.stdout or "", r.stderr or ""
    return False, f"agent repair command failed (exit={r.returncode})", r.stdout or "", r.stderr or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Ralph loop: staged remote training orchestration")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run only preflight stage and print config")
    parser.add_argument("--from-stage", type=str, default=None, help="Resume from this stage for next iteration")
    parser.add_argument("--config", type=str, default="", help="Path to config yaml")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        config_path = resolve_config_path(args.config)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        stage_name, script_name = STAGES[0]
        ok, msg, _details, _stdout, _stderr = run_stage_script(
            stage_name,
            STAGE_SCRIPTS_DIR / script_name,
            config_path,
            STATE_FILE,
        )
        print(f"{stage_name}: {msg}")
        print("config:", json.dumps(cfg, indent=2, default=str))
        sys.exit(0 if ok else 1)

    state = load_state()
    max_retries = int((cfg.get("loop", {}) or {}).get("max_stage_retries", 2))
    sleep_after = int((cfg.get("loop", {}) or {}).get("sleep_after_success", 60))

    while True:
        state["iteration"] = int(state.get("iteration", 0)) + 1
        state["stage_results"] = {}
        save_state(state)

        start_at = 0
        if args.from_stage and state.get("iteration") == 1:
            try:
                start_at = STAGE_ORDER.index(args.from_stage)
            except ValueError:
                pass
            args.from_stage = None

        failed = False
        for idx, (stage_name, script_name) in enumerate(STAGES):
            if idx < start_at:
                continue
            script_path = STAGE_SCRIPTS_DIR / script_name
            state["last_stage"] = stage_name
            for attempt in range(max_retries + 1):
                ok, msg, details, stdout, stderr = run_stage_script(stage_name, script_path, config_path, STATE_FILE)
                state["stage_results"][stage_name] = {
                    "ok": ok,
                    "message": msg,
                    "details": details,
                    "script": str(script_path),
                }
                save_state(state)
                if ok:
                    if (state.get("pending_feedback") or {}).get("stage") == stage_name:
                        state["pending_feedback"] = None
                    state["last_error"] = None
                    print(f"[iter {state['iteration']}] {stage_name}: {msg}")
                    save_state(state)
                    break

                print(f"[iter {state['iteration']}] {stage_name} (attempt {attempt + 1}): FAIL - {msg}", file=sys.stderr)
                state["last_error"] = msg
                state["pending_feedback"] = {
                    "stage": stage_name,
                    "error": msg,
                    "attempt": attempt + 1,
                    "iteration": state["iteration"],
                    "script": str(script_path),
                    "stdout_tail": (stdout or "").splitlines()[-15:],
                    "stderr_tail": (stderr or "").splitlines()[-15:],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                save_state(state)
                if attempt < max_retries:
                    time.sleep(5)
            else:
                repaired, repair_msg, repair_stdout, repair_stderr = run_agent_repair(cfg, state, stage_name, script_path, config_path)
                if repaired:
                    print(f"[iter {state['iteration']}] agent_repair: {repair_msg}")
                else:
                    print(f"[iter {state['iteration']}] agent_repair: {repair_msg}", file=sys.stderr)
                if repair_stdout.strip():
                    print(f"[iter {state['iteration']}] agent_repair stdout BEGIN")
                    print(repair_stdout.rstrip("\n"))
                    print(f"[iter {state['iteration']}] agent_repair stdout END")
                if repair_stderr.strip():
                    print(f"[iter {state['iteration']}] agent_repair stderr BEGIN", file=sys.stderr)
                    print(repair_stderr.rstrip("\n"), file=sys.stderr)
                    print(f"[iter {state['iteration']}] agent_repair stderr END", file=sys.stderr)
                print(f"[iter {state['iteration']}] Stage failed, will retry next iteration: {stage_name}", file=sys.stderr)
                save_state(state)
                failed = True
                break

        if failed:
            if args.once:
                break
            time.sleep(sleep_after)
            continue

        print(f"[iter {state['iteration']}] All stages passed.")
        state["last_error"] = None
        state["last_stage"] = None
        state["pending_feedback"] = None
        save_state(state)
        break


if __name__ == "__main__":
    main()
