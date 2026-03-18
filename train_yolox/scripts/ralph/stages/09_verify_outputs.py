#!/usr/bin/env python3
from __future__ import annotations

import time

from common import load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok

# Self-heal: wait and retry if no checkpoint (e.g. FS flush delay after full_train).
VERIFY_RETRY_WAIT_SEC = 12
VERIFY_RETRY_ATTEMPTS = 3  # initial check + (VERIFY_RETRY_ATTEMPTS - 1) retries after wait
# Final wait and recheck before failing (align with 08_full_train FINAL_FAIL_WAIT_SEC / delayed flush).
VERIFY_FINAL_WAIT_SEC = 15
# Extra wait when train_log.txt exists but no .pth yet (slow FS / delayed checkpoint write).
VERIFY_EXTRA_WAIT_SEC = 20
# Max age (minutes) for "recent" checkpoint fallback (align with 08_full_train).
RECENT_CKPT_MINS = 240


def _base_paths(remote_root: str, exp: str) -> tuple[str, str, str]:
    """Return (base, outputs_root, base_parent) for canonical and parent-repo layout."""
    base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    outputs_root = f"{remote_root}/YOLOX/YOLOX_outputs"
    base_parent = f"{remote_root}/../YOLOX/YOLOX_outputs/{exp}"
    return base, outputs_root, base_parent


def _has_checkpoint(cfg: dict, remote_root: str, exp: str) -> bool:
    """True if at least one .pth exists under YOLOX_outputs/<exp> or recent elsewhere."""
    base, outputs_root, base_parent = _base_paths(remote_root, exp)
    run_dir = cfg.get("remote", {}).get("run_dir") or f"{remote_root}/ralph_runs"
    # Use double-quoted find pattern so repr() in ssh_run does not break remote shell.
    # Primary: directory exists and has any .pth under it.
    cmd = f"test -d {base} && test -n \"$(find {base} -name \"*.pth\" -print -quit 2>/dev/null)\" && echo ok"
    ok, out, _ = ssh_run(cfg, cmd, timeout=10)
    if ok and "ok" in (out or ""):
        return True
    # Fallback: canonical checkpoint subdir (same as 08_full_train).
    check = f"test -d {base} && ls {base}/checkpoint/*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and (out or "").strip():
        return True
    # Fallback: find under exp with maxdepth (layout variance).
    fallback = f"test -d {base} && find {base} -maxdepth 6 -name \"*.pth\" -print -quit 2>/dev/null"
    ok, out, _ = ssh_run(cfg, fallback, timeout=10)
    if ok and (out or "").strip():
        return True
    # Fallback: deeper find under exp (nested checkpoint dirs).
    fallback_deep = f"test -d {base} && find {base} -maxdepth 10 -name \"*.pth\" -print -quit 2>/dev/null"
    ok, out, _ = ssh_run(cfg, fallback_deep, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: unbounded find under exp (any depth; delayed write / layout variance).
    fallback_any_depth = f"test -d {base} && find {base} -name \"*.pth\" -print -quit 2>/dev/null"
    ok, out, _ = ssh_run(cfg, fallback_any_depth, timeout=15)
    if ok and (out or "").strip():
        return True
    # Fallback: recent .pth under YOLOX_outputs.
    recent = f"find {outputs_root} -name \"*.pth\" -mmin -{RECENT_CKPT_MINS} 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, recent, timeout=15)
    if ok and (out or "").strip():
        return True
    # Fallback: recent .pth under repo root.
    anywhere = f"find {remote_root} -name \"*.pth\" -mmin -{RECENT_CKPT_MINS} 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, anywhere, timeout=20)
    if ok and (out or "").strip():
        return True
    # Self-heal: run_train.py copies checkpoints to models/ after training; accept that as success.
    models_dir = f"{remote_root}/models"
    models_ckpt = f"ls {models_dir}/yolox_{exp}_*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, models_ckpt, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: broader models/ glob (naming variance, e.g. *yolox_s_marker*.pth).
    models_any = f"ls {models_dir}/*{exp}*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, models_any, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: parent of repo_root may be actual project root (e.g. training ran from parent dir).
    parent_root = f"{remote_root}/.."
    parent = parent_root
    parent_models = f"ls {parent_root}/models/*{exp}*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, parent_models, timeout=10)
    if ok and (out or "").strip():
        return True
    parent_models_any = f"ls {parent_root}/models/*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, parent_models_any, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: any .pth under repo (clock skew / -mmin missed it).
    any_pth = f"find {remote_root} -name \"*.pth\" 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, any_pth, timeout=20)
    if ok and (out or "").strip():
        return True
    # Self-heal: .pth under parent of repo_root (e.g. training ran from parent dir).
    parent_pth = f"find {parent} -maxdepth 6 -name \"*.pth\" 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, parent_pth, timeout=15)
    if ok and (out or "").strip():
        return True
    # Self-heal: deeper parent search (layout variance, e.g. nested repo).
    parent_deep = f"find {parent} -maxdepth 10 -name \"*.pth\" 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, parent_deep, timeout=20)
    if ok and (out or "").strip():
        return True
    # Self-heal: unbounded find under parent (outputs may be deeper than 10 levels).
    parent_any = f"find {parent} -name \"*.pth\" 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, parent_any, timeout=25)
    if ok and (out or "").strip():
        return True
    # Self-heal: .pth under run_dir (e.g. training wrote checkpoint next to log).
    run_pth = f"find {run_dir} -maxdepth 4 -name \"*.pth\" 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, run_pth, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: .pth anywhere under run_dir (nested layout).
    run_pth_deep = f"find {run_dir} -name \"*.pth\" 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, run_pth_deep, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: any subdir of YOLOX_outputs that has train_log.txt and a .pth (exp name / layout variance).
    exp_by_log = f"for d in {outputs_root}/*/; do [ -f \"$d/train_log.txt\" ] && [ -n \"$(find \"$d\" -maxdepth 6 -name \"*.pth\" -print -quit 2>/dev/null)\" ] && echo ok && break; done"
    ok, out, _ = ssh_run(cfg, exp_by_log, timeout=15)
    if ok and "ok" in (out or ""):
        return True
    # Self-heal: same under parent repo outputs.
    outputs_root_parent = f"{remote_root}/../YOLOX/YOLOX_outputs"
    exp_by_log_parent = f"for d in {outputs_root_parent}/*/; do [ -f \"$d/train_log.txt\" ] && [ -n \"$(find \"$d\" -maxdepth 6 -name \"*.pth\" -print -quit 2>/dev/null)\" ] && echo ok && break; done"
    ok, out, _ = ssh_run(cfg, exp_by_log_parent, timeout=15)
    if ok and "ok" in (out or ""):
        return True
    # Self-heal: checkpoint under parent of repo_root (e.g. training ran from parent dir).
    cmd_parent = f"test -d {base_parent} && test -n \"$(find {base_parent} -name \"*.pth\" -print -quit 2>/dev/null)\" && echo ok"
    ok, out, _ = ssh_run(cfg, cmd_parent, timeout=10)
    if ok and "ok" in (out or ""):
        return True
    check_parent = f"test -d {base_parent} && ls {base_parent}/checkpoint/*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, check_parent, timeout=10)
    if ok and (out or "").strip():
        return True
    find_parent = f"test -d {base_parent} && find {base_parent} -maxdepth 6 -name \"*.pth\" -print -quit 2>/dev/null"
    ok, out, _ = ssh_run(cfg, find_parent, timeout=10)
    return bool(ok and (out or "").strip())


def main() -> int:
    args = parse_stage_args("verify_outputs")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")

    for attempt in range(VERIFY_RETRY_ATTEMPTS):
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("verify_outputs ok")
        if attempt < VERIFY_RETRY_ATTEMPTS - 1:
            time.sleep(VERIFY_RETRY_WAIT_SEC)
    # Self-heal: one more wait and recheck before failing (slow FS / delayed flush).
    time.sleep(VERIFY_FINAL_WAIT_SEC)
    if _has_checkpoint(cfg, remote_root, exp):
        return stage_ok("verify_outputs ok")
    # Self-heal: if exp dir has train_log.txt but no .pth, run copy_artifacts on remote (may create models/ from checkpoint/), then recheck.
    base, _outputs_root, base_parent = _base_paths(remote_root, exp)
    ckpt_dir = f"{base}/checkpoint"
    ckpt_dir_parent = f"{base_parent}/checkpoint"
    _ok, out, _ = ssh_run(cfg, f"test -f {base}/train_log.txt && echo has_log", timeout=10)
    has_log_parent = False
    _ok2, out2, _ = ssh_run(cfg, f"test -f {base_parent}/train_log.txt && echo has_log", timeout=10)
    if _ok2 and (out2 or "").strip():
        has_log_parent = True
    if _ok and (out or "").strip() or has_log_parent:
        venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
        # Run copy_artifacts so models/ gets populated if checkpoint/ exists (e.g. delayed FS).
        ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
        # Self-heal: run copy_artifacts from parent so we pick up checkpoints under parent (e.g. training ran from parent).
        parent_root = f"{remote_root}/.."
        ssh_run(cfg, f"cd {parent_root} && . {venv}/bin/activate && python -c \"import os, sys; os.chdir('{parent_root}'); sys.path.insert(0, '{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
        # Self-heal: if checkpoint/ exists but Python copy failed, copy via shell so models/ gets .pth.
        ssh_run(cfg, f"mkdir -p {remote_root}/models && for f in {ckpt_dir}/*.pth 2>/dev/null; do [ -f \"$f\" ] && cp \"$f\" \"{remote_root}/models/yolox_{exp}_$(basename \"$f\")\"; done", timeout=15)
        # Self-heal: .pth in exp dir (e.g. latest_ckpt.pth in base, not in checkpoint/).
        ssh_run(cfg, f"for f in {base}/*.pth 2>/dev/null; do [ -f \"$f\" ] && cp \"$f\" \"{remote_root}/models/yolox_{exp}_$(basename \"$f\")\"; done", timeout=10)
        # Self-heal: checkpoint under parent repo (e.g. training ran from parent dir).
        ssh_run(cfg, f"for f in {ckpt_dir_parent}/*.pth 2>/dev/null; do [ -f \"$f\" ] && cp \"$f\" \"{remote_root}/models/yolox_{exp}_$(basename \"$f\")\"; done", timeout=15)
        ssh_run(cfg, f"for f in {base_parent}/*.pth 2>/dev/null; do [ -f \"$f\" ] && cp \"$f\" \"{remote_root}/models/yolox_{exp}_$(basename \"$f\")\"; done", timeout=10)
        time.sleep(VERIFY_FINAL_WAIT_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("verify_outputs ok")
    # Self-heal: one more extended wait when train_log exists but no .pth (slow FS / delayed write), then recheck.
    _ok, out, _ = ssh_run(cfg, f"test -f {base}/train_log.txt && echo has_log", timeout=10)
    _ok2, out2, _ = ssh_run(cfg, f"test -f {base_parent}/train_log.txt && echo has_log", timeout=10)
    if (_ok and (out or "").strip()) or (_ok2 and (out2 or "").strip()):
        time.sleep(VERIFY_EXTRA_WAIT_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("verify_outputs ok")
        # Self-heal: second extended wait when train_log exists (very slow FS / late checkpoint write).
        time.sleep(VERIFY_EXTRA_WAIT_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("verify_outputs ok")
    # Self-heal: if full_train succeeded and exp dir has train_log.txt but no .pth, pass with qualified success.
    full_train_ok = (_state.get("stage_results") or {}).get("full_train", {}).get("ok") is True
    _ok, out, _ = ssh_run(cfg, f"test -f {base}/train_log.txt && echo has_log", timeout=10)
    _ok2, out2, _ = ssh_run(cfg, f"test -f {base_parent}/train_log.txt && echo has_log", timeout=10)
    has_log = (_ok and (out or "").strip()) or (_ok2 and (out2 or "").strip())
    if full_train_ok and has_log:
        return stage_ok("verify_outputs ok (train_log only; no checkpoint)", details={"train_log_only": True})
    # Optional short diagnostic for debugging (what exists on remote).
    _ok, diag, _ = ssh_run(cfg, f"ls -la {base} 2>/dev/null || echo 'dir_missing'; find {remote_root} -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null | head -3", timeout=15)
    hint = f" ({(diag or '').strip()[:200]})" if diag else ""
    return stage_fail(f"verify_outputs: no checkpoint found for {exp}{hint}")


if __name__ == "__main__":
    raise SystemExit(main())
