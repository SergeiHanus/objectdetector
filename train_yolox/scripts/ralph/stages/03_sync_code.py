#!/usr/bin/env python3
from __future__ import annotations

import time

from common import REPO_ROOT, TRAIN_YOLOX, get_git_revision, load_config, load_state, parse_stage_args, rsync_to_remote, ssh_run, stage_fail, stage_ok

MAX_RSYNC_ATTEMPTS = 3
RSYNC_RETRY_DELAY_SEC = 15


def main() -> int:
    args = parse_stage_args("sync_code")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    excludes = cfg.get("sync", {}).get("code_excludes", [".git", ".venv", "__pycache__"])
    err = ""
    for attempt in range(1, MAX_RSYNC_ATTEMPTS + 1):
        ok, _out, err = rsync_to_remote(cfg, str(TRAIN_YOLOX), remote_root, excludes=excludes)
        if ok:
            break
        is_timeout = "timed out" in (err or "").lower()
        if not is_timeout or attempt == MAX_RSYNC_ATTEMPTS:
            return stage_fail(f"rsync code failed: {err}")
        time.sleep(RSYNC_RETRY_DELAY_SEC)

    # Also sync top-level src/ so ralph can run shared scripts from remote_root/src/.
    # This keeps "manual code" in repo root src/, while ralph only executes staged scripts.
    ok_src, _out_src, err_src = rsync_to_remote(
        cfg,
        str(REPO_ROOT / "src"),
        f"{remote_root}/src",
        excludes=["__pycache__", "*.pyc"],
    )
    if not ok_src:
        return stage_fail(f"rsync src failed: {err_src}")

    rev = get_git_revision() or "unknown"
    ok2, _out2, err2 = ssh_run(cfg, f"printf %s {rev!r} > {remote_root}/.ralph_revision", timeout=10)
    if not ok2:
        return stage_fail(f"revision marker write failed: {err2}")

    ok3, out3, err3 = ssh_run(cfg, f"cat {remote_root}/.ralph_revision", timeout=10)
    if not ok3:
        return stage_fail(f"revision marker read failed: {err3}")
    if (out3 or "").strip() != rev:
        return stage_fail(f"revision marker mismatch: expected {rev[:8]} got {(out3 or '').strip()[:8]}")

    return stage_ok(f"sync_code ok (rev={rev[:8]})")


if __name__ == "__main__":
    raise SystemExit(main())
