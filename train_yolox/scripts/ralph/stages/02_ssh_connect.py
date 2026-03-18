#!/usr/bin/env python3
from __future__ import annotations

from common import load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok


def main() -> int:
    args = parse_stage_args("ssh_connect")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "").rstrip("/")
    if not remote_root:
        return stage_fail("remote.repo_root not set")

    if remote_root.startswith("~"):
        mkdir_cmd = f"mkdir -p {remote_root}"
    else:
        mkdir_cmd = f"test -d {remote_root} || mkdir -p {remote_root}"
    ok, out, err = ssh_run(cfg, mkdir_cmd)
    if not ok:
        return stage_fail(f"SSH or path failed: {err or out}. (Tip: use ~/path on remote)")

    ok2, out2, _ = ssh_run(cfg, f"test -w {remote_root} && echo ok")
    if not ok2 or "ok" not in out2:
        return stage_fail(f"remote path not writable: {remote_root}")
    return stage_ok("ssh_connect ok")


if __name__ == "__main__":
    raise SystemExit(main())
