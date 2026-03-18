#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from common import (
    TRAIN_YOLOX,
    load_config,
    load_state,
    parse_stage_args,
    rsync_from_remote,
    ssh_run,
    stage_fail,
    stage_ok,
)


def _has_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.suffix == ".pth" for p in path.rglob("*.pth"))


def _has_train_log(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.name == "train_log.txt" for p in path.rglob("train_log.txt"))


def _first_local_pth(path: Path) -> Path | None:
    if not path.exists():
        return None
    for p in sorted(path.rglob("*.pth")):
        if p.is_file():
            return p
    return None


def _remote_first_pth(cfg: dict, remote_root: str, exp: str) -> str | None:
    """Find a remote .pth in common fallback locations (best-effort)."""
    base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    base_parent = f"{remote_root}/../YOLOX/YOLOX_outputs/{exp}"
    models_dir = f"{remote_root}/models"
    models_parent = f"{remote_root}/../models"

    # Prefer fast checks (globs) before running find.
    globs = [
        f"{base}/checkpoint/*.pth",
        f"{base}/*.pth",
        f"{models_dir}/yolox_{exp}_*.pth",
        f"{models_dir}/*{exp}*.pth",
        f"{models_dir}/*.pth",
        f"{base_parent}/checkpoint/*.pth",
        f"{base_parent}/*.pth",
        f"{models_parent}/*{exp}*.pth",
        f"{models_parent}/*.pth",
    ]
    for pat in globs:
        ok, out, _ = ssh_run(cfg, f"sh -lc 'ls -1 {pat} 2>/dev/null | head -n 1'", timeout=12)
        p = (out or "").strip()
        if ok and p.endswith(".pth"):
            return p

    # Bounded find as a last resort.
    roots = [base, f"{remote_root}/YOLOX/YOLOX_outputs", models_dir, base_parent, f"{remote_root}/.."]
    for root in roots:
        ok, out, _ = ssh_run(
            cfg,
            f"sh -lc 'test -e {root} && find {root} -maxdepth 8 -name \"*.pth\" -print -quit 2>/dev/null'",
            timeout=20,
        )
        p = (out or "").strip()
        if ok and p.endswith(".pth"):
            return p
    return None


def main() -> int:
    args = parse_stage_args("pull_artifacts")
    cfg = load_config(args.config)
    state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    art_dir = cfg.get("local", {}).get("artifacts_dir") or str(TRAIN_YOLOX / "ralph_artifacts")
    iter_id = state.get("iteration", 0)
    local_iter = os.path.join(art_dir, f"iter_{iter_id}")
    os.makedirs(local_iter, exist_ok=True)
    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")
    remote_out = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    local_out_root = os.path.join(local_iter, "YOLOX_outputs", exp)
    local_ckpt_root = os.path.join(local_out_root, "checkpoint")
    os.makedirs(local_ckpt_root, exist_ok=True)

    # Pull only essential artifacts to avoid long/blocking rsync of full YOLOX_outputs tree.
    # This keeps pull_artifacts fast while preserving checkpoints and train logs for inspection.
    ok_ckpt, _out_ckpt, err_ckpt = rsync_from_remote(
        cfg, f"{remote_out}/checkpoint", local_out_root
    )
    if not ok_ckpt:
        return stage_fail(f"pull_artifacts failed (checkpoint): {err_ckpt}")

    # Best-effort pulls for logs/metadata; keep stage success based on checkpoint/train_log presence.
    rsync_from_remote(cfg, f"{remote_out}/train_log.txt", local_out_root)
    rsync_from_remote(cfg, f"{remote_out}/best_ckpt.pth", local_out_root)

    remote_models = f"{remote_root}/models"
    rsync_from_remote(cfg, remote_models, os.path.join(local_iter, "models"))
    run_dir = cfg.get("remote", {}).get("run_dir") or f"{remote_root}/ralph_runs"
    rsync_from_remote(cfg, f"{run_dir}/iter_{iter_id}.log", local_iter)

    yolox_out = Path(local_iter) / "YOLOX_outputs"
    if _has_checkpoint(yolox_out):
        return stage_ok(f"pull_artifacts ok -> {local_iter}")

    # Self-heal: training may have placed checkpoints under models/ or parent dirs; normalize into checkpoint/.
    models_root = Path(local_iter) / "models"
    local_models_pth = _first_local_pth(models_root)
    if local_models_pth:
        try:
            dst = Path(local_ckpt_root) / local_models_pth.name
            if not dst.exists():
                dst.write_bytes(local_models_pth.read_bytes())
        except Exception:
            pass
        if _has_checkpoint(yolox_out):
            return stage_ok(
                f"pull_artifacts ok (from models/) -> {local_iter}",
                details={"self_heal": "normalized_models_pth"},
            )

    remote_pth = _remote_first_pth(cfg, remote_root, exp)
    if remote_pth:
        ok_pth, _out_pth, err_pth = rsync_from_remote(cfg, remote_pth, local_ckpt_root)
        if not ok_pth:
            return stage_fail(f"pull_artifacts failed (remote pth): {err_pth}")
        if _has_checkpoint(yolox_out):
            return stage_ok(
                f"pull_artifacts ok (remote fallback) -> {local_iter}",
                details={"self_heal": "pulled_remote_pth", "remote_pth": remote_pth},
            )

    # Self-heal: allow explicit log-only success only if verify_outputs said so.
    verify_details = (state.get("stage_results") or {}).get("verify_outputs") or {}
    has_log = _has_train_log(yolox_out) or Path(local_iter, f"iter_{iter_id}.log").exists()
    if verify_details.get("details", {}).get("train_log_only") is True and has_log:
        return stage_ok(f"pull_artifacts ok (train_log only) -> {local_iter}", details={"train_log_only": True})
    return stage_fail(f"pull_artifacts validation failed: no .pth in {local_iter}/YOLOX_outputs")


if __name__ == "__main__":
    raise SystemExit(main())
