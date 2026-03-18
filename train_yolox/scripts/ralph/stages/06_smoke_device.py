#!/usr/bin/env python3
from __future__ import annotations

import json

from common import load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok


def main() -> int:
    args = parse_stage_args("smoke_device")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    cmd = f"cd {remote_root} && . {venv}/bin/activate && python scripts/smoke_device.py --json"
    ok, out, err = ssh_run(cfg, cmd, timeout=60)
    if not ok:
        return stage_fail(f"smoke_device failed: {err or out}")
    try:
        data = json.loads((out or "").strip().split("\n")[-1])
    except Exception as exc:
        return stage_fail(f"smoke_device parse error: {exc}")
    if not data.get("ok"):
        return stage_fail(f"smoke_device not ok: {data.get('errors', [])}")
    return stage_ok(f"smoke_device ok (device={data.get('device', '?')})")


if __name__ == "__main__":
    raise SystemExit(main())
