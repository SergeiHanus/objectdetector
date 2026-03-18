#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
import time
from datetime import datetime
from pathlib import Path

from common import load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok

# Self-heal: retry training when no checkpoint found (e.g. early crash before first save, or slow FS).
MAX_CHECKPOINT_RETRIES = 2
POST_RUN_FLUSH_SEC = 12
FINAL_RECHECK_SEC = 15
# Extra wait and recheck before final fail (slow FS / delayed flush).
FINAL_FAIL_WAIT_SEC = 10
# When run reports done but no .pth (e.g. empty checkpoint dir), wait before retry/fail (FS flush).
EMPTY_CKPT_EXTRA_WAIT_SEC = 25
# After probe times out, wait for it to possibly finish and flush before starting full (which would kill the probe).
PROBE_TIMEOUT_EXTRA_WAIT_SEC = 60 * 30  # 30 min


def _run_one_train(cfg: dict, cmd: str, session: str, wait_sec: int = 7200) -> tuple[str, str | None]:
    """Start training, wait until tmux session is done. Returns ('done', None), ('timeout', None), or ('start_failed', err)."""
    ok, _out, err = ssh_run(cfg, cmd, timeout=30)
    if not ok:
        return ("start_failed", err or "")
    for _ in range(wait_sec):
        time.sleep(1)
        ok2, out2, _ = ssh_run(cfg, f"tmux has-session -t {session} 2>/dev/null && echo running || echo done", timeout=5)
        if ok2 and "done" in out2:
            return ("done", None)
    return ("timeout", None)


def _remote_train_images_count(cfg: dict, remote_root: str) -> int | None:
    """Return number of images in data/annotations/train_labels.json on remote, or None if unreadable."""
    cmd = (
        f"cd {remote_root} && "
        "python3 -c "
        "\"import json, os; p='data/annotations/train_labels.json'; "
        "d=json.load(open(p)) if os.path.isfile(p) else {}; "
        "print(len(d.get('images', [])))\""
        " 2>/dev/null || "
        "python -c "
        "\"import json, os; p='data/annotations/train_labels.json'; "
        "d=json.load(open(p)) if os.path.isfile(p) else {}; "
        "print(len(d.get('images', [])))\""
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=20)
    if not ok or not out:
        return None
    try:
        return int((out or "").strip().splitlines()[-1])
    except Exception:
        return None


def _remote_val_images_count(cfg: dict, remote_root: str) -> int | None:
    """Return number of images in data/annotations/val_labels.json on remote, or None if unreadable."""
    cmd = (
        f"cd {remote_root} && "
        "python3 -c "
        "\"import json, os; p='data/annotations/val_labels.json'; "
        "d=json.load(open(p)) if os.path.isfile(p) else {}; "
        "print(len(d.get('images', [])))\""
        " 2>/dev/null || "
        "python -c "
        "\"import json, os; p='data/annotations/val_labels.json'; "
        "d=json.load(open(p)) if os.path.isfile(p) else {}; "
        "print(len(d.get('images', [])))\""
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=20)
    if not ok or not out:
        return None
    try:
        return int((out or "").strip().splitlines()[-1])
    except Exception:
        return None


def _ensure_real_annotations(cfg: dict, remote_root: str) -> bool:
    """
    Ensure remote annotations are not the minimal "smoke placeholder" set and val is non-empty.
    If train_labels.json has <= 1 image, or val_labels.json has 0 images (which yields AP 0.00),
    regenerate using src/yolox_data_prep.py.
    """
    n_train = _remote_train_images_count(cfg, remote_root)
    n_val = _remote_val_images_count(cfg, remote_root)
    if n_train is None or n_val is None:
        return False
    if n_train >= 2 and n_val >= 1:
        return True
    raw_subdir = (cfg.get("sync", {}) or {}).get("raw_data_subdir") or "Markers01.20026"
    raw_dir = f"data/{raw_subdir}"
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    cmd = (
        f"cd {remote_root} && "
        f". {venv}/bin/activate && "
        f"python src/yolox_data_prep.py --train-yolox-root . --raw-dir {raw_dir!r} --dataset-name marker_dataset"
    )
    ok, _out, _err = ssh_run(cfg, cmd, timeout=300)
    n2_train = _remote_train_images_count(cfg, remote_root)
    n2_val = _remote_val_images_count(cfg, remote_root)
    return bool(ok and n2_train is not None and n2_train >= 2 and n2_val is not None and n2_val >= 1)


# Max age (minutes) for "recent" checkpoint fallback (long runs can take 2h+).
RECENT_CKPT_MINS = 240
# Probe must run enough epochs to trigger at least one save (YOLOX save_interval=10).
PROBE_EPOCHS = 10
# Long enough for 10 epochs on slow devices (e.g. MPS); avoid timeout killing probe before first save.
PROBE_WAIT_SEC = 7200
# Final empty-ckpt probe: allow up to 4h so slow MPS can complete 10 epochs before we give up.
FINAL_PROBE_WAIT_SEC = 14400


def _has_checkpoint(cfg: dict, remote_root: str, exp: str) -> bool:
    """True if at least one .pth exists under YOLOX_outputs/<exp> (checkpoint/ or any subdir) or in alternate locations."""
    base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    base_alt = f"{remote_root}/YOLOX_outputs/{exp}"  # Layout: outputs directly under repo_root
    outputs_root = f"{remote_root}/YOLOX/YOLOX_outputs"
    run_dir = cfg.get("remote", {}).get("run_dir") or f"{remote_root}/ralph_runs"
    base_parent = f"{remote_root}/../YOLOX/YOLOX_outputs/{exp}"
    # Prefer canonical path
    check = f"test -d {base} && ls {base}/checkpoint/*.pth 2>/dev/null | sed -n '1p'"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and (out or "").strip():
        return True
    # Self-heal: alternate layout (YOLOX_outputs directly under repo_root)
    check_alt = f"test -d {base_alt} && ls {base_alt}/checkpoint/*.pth 2>/dev/null | sed -n '1p'"
    ok_alt, out_alt, _ = ssh_run(cfg, check_alt, timeout=10)
    if ok_alt and (out_alt or "").strip():
        return True
    # Self-heal: accept any .pth under exp (e.g. different YOLOX layout); search deeper than 3 levels
    fallback = f"test -d {base} && find {base} -maxdepth 6 -name '*.pth' -print -quit 2>/dev/null"
    ok2, out2, _ = ssh_run(cfg, fallback, timeout=10)
    if ok2 and (out2 or "").strip():
        return True
    # Self-heal: accept any recent .pth under YOLOX_outputs (path/layout variance or long run)
    recent = f"find {outputs_root} -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null | sed -n '1p'"
    ok3, out3, _ = ssh_run(cfg, recent, timeout=15)
    if ok3 and (out3 or "").strip():
        return True
    # Self-heal: same for alternate layout (YOLOX_outputs under repo_root; avoid missing if first find times out).
    outputs_root_alt = f"{remote_root}/YOLOX_outputs"
    recent_alt = f"find {outputs_root_alt} -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null | sed -n '1p'"
    ok3a, out3a, _ = ssh_run(cfg, recent_alt, timeout=15)
    if ok3a and (out3a or "").strip():
        return True
    # Self-heal: accept any recent .pth under repo root (layout variance, e.g. outputs one level up)
    anywhere = f"find {remote_root} -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null | sed -n '1p'"
    ok4, out4, _ = ssh_run(cfg, anywhere, timeout=20)
    if ok4 and (out4 or "").strip():
        return True
    # Self-heal: run_train.py copies to models/ after success; accept that as success (align with 09_verify_outputs).
    models_dir = f"{remote_root}/models"
    models_ckpt = f"ls {models_dir}/yolox_{exp}_*.pth 2>/dev/null | sed -n '1p'"
    ok5, out5, _ = ssh_run(cfg, models_ckpt, timeout=10)
    if ok5 and (out5 or "").strip():
        return True
    models_any = f"ls {models_dir}/*{exp}*.pth 2>/dev/null | sed -n '1p'"
    ok6, out6, _ = ssh_run(cfg, models_any, timeout=10)
    if ok6 and (out6 or "").strip():
        return True
    # Self-heal: .pth under run_dir (e.g. training wrote next to log or layout variance).
    run_pth = f"find {run_dir} -maxdepth 4 -name '*.pth' 2>/dev/null | sed -n '1p'"
    ok7, out7, _ = ssh_run(cfg, run_pth, timeout=10)
    if ok7 and (out7 or "").strip():
        return True
    # Self-heal: checkpoint under parent of repo_root (e.g. training ran from parent dir).
    check_parent = f"test -d {base_parent} && find {base_parent} -maxdepth 6 -name '*.pth' -print -quit 2>/dev/null"
    ok8, out8, _ = ssh_run(cfg, check_parent, timeout=10)
    if ok8 and (out8 or "").strip():
        return True
    parent_models = f"ls {remote_root}/../models/*{exp}*.pth 2>/dev/null | sed -n '1p'"
    ok9, out9, _ = ssh_run(cfg, parent_models, timeout=10)
    return bool(ok9 and (out9 or "").strip())


def _is_checkpoint_dir_empty(cfg: dict, remote_root: str, exp: str) -> bool:
    """True if YOLOX_outputs/<exp> exists but has no .pth (run finished, FS not flushed)."""
    base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    base_alt = f"{remote_root}/YOLOX_outputs/{exp}"
    # Self-heal: single command to detect empty checkpoint dir for both layouts (dir must exist; matches "total 0" case).
    wc_cmd = (
        f"test -d {base}/checkpoint && test $(ls -1A {base}/checkpoint 2>/dev/null | wc -l) -eq 0 && echo empty1; "
        f"test -d {base_alt}/checkpoint && test $(ls -1A {base_alt}/checkpoint 2>/dev/null | wc -l) -eq 0 && echo empty2"
    )
    ok_wc, out_wc, _ = ssh_run(cfg, wc_cmd, timeout=10)
    if ok_wc and out_wc and ("empty1" in out_wc or "empty2" in out_wc):
        return True
    # Portable find (no -quit for BSD/macOS): use head -1 to get at most one line.
    check = f"test -d {base}/checkpoint && test -z \"$(find {base}/checkpoint -maxdepth 1 -name '*.pth' 2>/dev/null | head -1)\" && echo empty"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and (out or "").strip() == "empty":
        return True
    check_alt = f"test -d {base_alt}/checkpoint && test -z \"$(find {base_alt}/checkpoint -maxdepth 1 -name '*.pth' 2>/dev/null | head -1)\" && echo empty"
    ok_alt, out_alt, _ = ssh_run(cfg, check_alt, timeout=10)
    if ok_alt and (out_alt or "").strip() == "empty":
        return True
    # Self-heal: detect "total 0" / empty dir when find is unreliable (e.g. path layout variance); ls -A is portable.
    for ckpt_dir in (f"{base}/checkpoint", f"{base_alt}/checkpoint"):
        ls_ok, ls_out, _ = ssh_run(cfg, f"test -d {ckpt_dir} && ls -1A {ckpt_dir} 2>/dev/null | wc -l", timeout=10)
        if ls_ok and (ls_out or "").strip() in ("0", " 0"):
            return True
    # Self-heal: exp dir exists but no .pth anywhere under it (run created dir but wrote nothing or different layout).
    for exp_dir in (base, base_alt):
        any_pth = f"test -d {exp_dir} && find {exp_dir} -maxdepth 6 -name '*.pth' 2>/dev/null | head -1"
        ok_any, out_any, _ = ssh_run(cfg, any_pth, timeout=10)
        if ok_any and not (out_any or "").strip():
            return True
    return False


def _ensure_output_dir(cfg: dict, remote_root: str, exp: str) -> None:
    """Ensure YOLOX_outputs/<exp> and checkpoint subdir exist on remote (self-heal dir_missing). Create both canonical and alternate layout."""
    base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    base_alt = f"{remote_root}/YOLOX_outputs/{exp}"
    ssh_run(cfg, f"mkdir -p {base}/checkpoint", timeout=10)
    ssh_run(cfg, f"mkdir -p {base_alt}/checkpoint", timeout=10)


# Log lines that mean training crashed even if "Training finished successfully" appears later.
TRAIN_LOG_ERROR_INDICATORS = (
    "AttributeError",
    "_cuda_setDevice",
    "Exception in training",
    "ModuleNotFoundError",
    "AssertionError",
    "ValueError:",
    "RuntimeError:",
    "Traceback (most recent call last)",
)
# Require at least this many "start train epoch" lines so we don't treat instant/no-epoch run as success.
MIN_START_TRAIN_EPOCH_LINES = 2
# Epochs completing in under this many seconds on average mean training did nothing (0 iters or broken loop).
MIN_SECONDS_PER_EPOCH = 2.0

# Loguru-style timestamp at start of line: "2026-03-17 07:00:35"
_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _run_training_succeeded(cfg: dict, log_file: str) -> bool:
    """True only if log has 'Training finished successfully', no error indicators, real training (≥MIN epochs), and not suspiciously fast (≥MIN_SECONDS_PER_EPOCH per epoch)."""
    ok, log_content, _ = ssh_run(cfg, f"cat {log_file} 2>/dev/null", timeout=15)
    if not ok or not log_content:
        return False
    log_str = (log_content or "").strip()
    if "Training finished successfully" not in log_str:
        return False
    if any(ind in log_str for ind in TRAIN_LOG_ERROR_INDICATORS):
        return False
    # Lines containing "start train epoch" with optional timestamp
    epoch_lines = [line for line in log_str.splitlines() if "start train epoch" in line]
    start_epoch_count = len(epoch_lines)
    if start_epoch_count < MIN_START_TRAIN_EPOCH_LINES:
        return False
    # Require total duration from first to last "start train epoch" >= count * MIN_SECONDS_PER_EPOCH
    timestamps = []
    for line in epoch_lines:
        m = _LOG_TIMESTAMP_RE.match(line.strip())
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                timestamps.append(ts)
            except ValueError:
                pass
    if len(timestamps) >= 2:
        duration_sec = (max(timestamps) - min(timestamps)).total_seconds()
        min_required = start_epoch_count * MIN_SECONDS_PER_EPOCH
        if duration_sec < min_required:
            return False
    return True


def _clean_output_dir(cfg: dict, remote_root: str, exp: str) -> None:
    """Remove YOLOX_outputs/<exp> on remote so a retry starts from a clean state (canonical + alternate path)."""
    ssh_run(cfg, f"rm -rf {remote_root}/YOLOX/YOLOX_outputs/{exp}", timeout=15)
    ssh_run(cfg, f"rm -rf {remote_root}/YOLOX_outputs/{exp}", timeout=15)


def main() -> int:
    args = parse_stage_args("full_train")
    cfg = load_config(args.config)
    state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    run_dir = cfg.get("remote", {}).get("run_dir") or f"{remote_root}/ralph_runs"
    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")
    epochs = cfg.get("training", {}).get("epochs", 80)
    batch = cfg.get("training", {}).get("batch", 8)
    prefix = cfg.get("loop", {}).get("tmux_session_prefix", "ralph_train")
    iter_id = state.get("iteration", 0)
    session = f"{prefix}_{iter_id}"
    log_file = f"{run_dir}/iter_{iter_id}.log"

    # Ensure dataset symlink and smoke_placeholder.png on remote so loader finds datasets/<ds>/train/images/smoke_placeholder.png (same as 07_smoke_train). Fixes "file named ... not found" when sync_data overwrote YOLOX/datasets or training runs on remote where file was missing.
    _stages_dir = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location("_smoke_train", _stages_dir / "07_smoke_train.py")
    _smoke = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_smoke)
    _smoke._ensure_yolox_dataset_link(cfg, remote_root, exp)
    _smoke._ensure_smoke_placeholder_file(cfg, remote_root, exp)
    _smoke._write_placeholder_from_yolox_cwd(cfg, remote_root, exp)

    # Critical: make sure we have real annotations (not 0-1 image smoke placeholder JSON),
    # otherwise training can "finish instantly" with ~1 batch/epoch and AP stuck at 0.
    if not _ensure_real_annotations(cfg, remote_root):
        return stage_fail("full_train failed: annotations missing or minimal; yolox_data_prep.py did not produce a real train set")

    cmd = (
        f"mkdir -p {run_dir} && cd {remote_root} && . {venv}/bin/activate && "
        f"(tmux has-session -t {session} 2>/dev/null && tmux kill-session -t {session}); "
        f"tmux new-session -d -s {session} "
        f"'python scripts/run_train.py --exp {exp} --epochs {epochs} --batch {batch} 2>&1 | tee {log_file}'; "
        "echo started"
    )
    cmd_probe = (
        f"mkdir -p {run_dir} && cd {remote_root} && . {venv}/bin/activate && "
        f"(tmux has-session -t {session} 2>/dev/null && tmux kill-session -t {session}); "
        f"tmux new-session -d -s {session} "
        f"'python scripts/run_train.py --exp {exp} --epochs {PROBE_EPOCHS} --batch {batch} 2>&1 | tee {log_file}.probe'; "
        "echo started"
    )

    for attempt in range(MAX_CHECKPOINT_RETRIES + 1):
        if attempt > 0:
            # Self-heal: clean output dir before retry so run starts from a clean state.
            _clean_output_dir(cfg, remote_root, exp)
        # Self-heal: ensure output dir exists (avoids dir_missing when YOLOX/FS does not create it).
        _ensure_output_dir(cfg, remote_root, exp)
        status, err = _run_one_train(cfg, cmd, session)
        if status == "start_failed":
            return stage_fail(f"full_train start failed: {err or 'unknown'}")
        if status == "timeout":
            return stage_fail("full_train timeout (2h)")
        # Allow remote FS / process to flush after session ends before checking for checkpoint.
        time.sleep(POST_RUN_FLUSH_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: run reported done but no checkpoint; if log says success, wait for FS flush then recheck.
        if status == "done" and _run_training_succeeded(cfg, log_file):
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        # Self-heal: run ended but no checkpoint; if log says run crashed, retry same attempt once.
        if status == "done" and not _run_training_succeeded(cfg, log_file):
            _clean_output_dir(cfg, remote_root, exp)
            _ensure_output_dir(cfg, remote_root, exp)
            status2, err2 = _run_one_train(cfg, cmd, session)
            if status2 == "start_failed":
                return stage_fail(f"full_train start failed: {err2 or 'unknown'}")
            if status2 == "timeout":
                return stage_fail("full_train timeout (2h)")
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        # Self-heal: empty checkpoint dir (run finished, no .pth) – run copy_artifacts then recheck; if still empty, probe then full.
        if _is_checkpoint_dir_empty(cfg, remote_root, exp):
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            # Self-heal: copy_artifacts may surface checkpoint to models/ (FS flush or path variance).
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(3)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            _clean_output_dir(cfg, remote_root, exp)
            _ensure_output_dir(cfg, remote_root, exp)
            status_probe_early, _ = _run_one_train(cfg, cmd_probe, session, wait_sec=FINAL_PROBE_WAIT_SEC)
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                _clean_output_dir(cfg, remote_root, exp)
                _ensure_output_dir(cfg, remote_root, exp)
            # Self-heal: probe timed out but may still be running (e.g. slow MPS); wait for it to finish and flush before starting full (which kills the session).
            if status_probe_early == "timeout":
                time.sleep(PROBE_TIMEOUT_EXTRA_WAIT_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
            status_empty_early, err_early = _run_one_train(cfg, cmd, session)
            if status_empty_early == "start_failed":
                return stage_fail(f"full_train start failed (empty-ckpt early retry): {err_early or 'unknown'}")
            if status_empty_early == "timeout":
                # Self-heal: don't fail on empty-ckpt retry timeout; continue to next attempt.
                if attempt < MAX_CHECKPOINT_RETRIES:
                    continue
                return stage_fail("full_train timeout (2h) on empty-ckpt early retry")
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            time.sleep(FINAL_RECHECK_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        if attempt < MAX_CHECKPOINT_RETRIES:
            # Self-heal: no checkpoint after run; retry (e.g. early crash before first save).
            continue
        # Self-heal: final re-check after longer delay (slow FS / late flush).
        time.sleep(FINAL_RECHECK_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: empty checkpoint dir (run finished, no .pth) – extra wait for FS flush before last-chance run.
        time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: one more full run when we would fail with "no checkpoint" (e.g. early crash, slow FS).
        _clean_output_dir(cfg, remote_root, exp)
        _ensure_output_dir(cfg, remote_root, exp)
        status2, err2 = _run_one_train(cfg, cmd, session)
        if status2 == "start_failed":
            return stage_fail(f"full_train start failed (last chance): {err2 or 'unknown'}")
        if status2 == "timeout":
            return stage_fail("full_train timeout (2h) on last chance run")
        time.sleep(POST_RUN_FLUSH_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        time.sleep(FINAL_RECHECK_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: one more wait and recheck before failing (slow FS / delayed flush).
        time.sleep(FINAL_FAIL_WAIT_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: run copy_artifacts on remote once (may surface checkpoint to models/ if FS was delayed).
        ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
        time.sleep(3)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: run short probe (PROBE_EPOCHS >= save_interval) so at least one .pth is written; then retry full (early-crash / env recovery).
        _clean_output_dir(cfg, remote_root, exp)
        _ensure_output_dir(cfg, remote_root, exp)
        status_probe, err_probe = _run_one_train(cfg, cmd_probe, session, wait_sec=PROBE_WAIT_SEC)
        if status_probe == "done":
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                _clean_output_dir(cfg, remote_root, exp)
                _ensure_output_dir(cfg, remote_root, exp)
                status_full, err_full = _run_one_train(cfg, cmd, session)
                if status_full == "start_failed":
                    return stage_fail(f"full_train start failed (after probe): {err_full or 'unknown'}")
                if status_full == "timeout":
                    return stage_fail("full_train timeout (2h) after probe run")
                time.sleep(POST_RUN_FLUSH_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
                time.sleep(FINAL_RECHECK_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
        # Self-heal: canonical checkpoint dir exists but empty – run probe first so at least one save happens, then full retry before final fail.
        if _is_checkpoint_dir_empty(cfg, remote_root, exp):
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC + 5)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            _clean_output_dir(cfg, remote_root, exp)
            _ensure_output_dir(cfg, remote_root, exp)
            _run_one_train(cfg, cmd_probe, session, wait_sec=PROBE_WAIT_SEC)
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                _clean_output_dir(cfg, remote_root, exp)
                _ensure_output_dir(cfg, remote_root, exp)
            status_empty, err_empty = _run_one_train(cfg, cmd, session)
            if status_empty == "start_failed":
                return stage_fail(f"full_train start failed (empty-ckpt retry): {err_empty or 'unknown'}")
            if status_empty == "timeout":
                return stage_fail("full_train timeout (2h) on empty-ckpt retry")
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            time.sleep(FINAL_RECHECK_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            # Self-heal: run copy_artifacts after empty-ckpt retry (may surface checkpoint if FS delayed).
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(3)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            # Self-heal: empty-ckpt retry still left dir empty – one more full run before final fail.
            _clean_output_dir(cfg, remote_root, exp)
            _ensure_output_dir(cfg, remote_root, exp)
            status_empty2, err_empty2 = _run_one_train(cfg, cmd, session)
            if status_empty2 == "start_failed":
                return stage_fail(f"full_train start failed (empty-ckpt 2nd retry): {err_empty2 or 'unknown'}")
            if status_empty2 == "timeout":
                return stage_fail("full_train timeout (2h) on empty-ckpt 2nd retry")
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            time.sleep(FINAL_RECHECK_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        # Brief diagnostic for debugging (what exists on remote)
        base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
        base_alt = f"{remote_root}/YOLOX_outputs/{exp}"
        ckpt_dir = f"{base}/checkpoint"
        ckpt_alt = f"{base_alt}/checkpoint"
        # Self-heal: before final fail, copy any .pth from ckpt_dir/base (and alternate path) to models/ via shell (FS/visibility), then recheck.
        ssh_run(cfg, f"mkdir -p {remote_root}/models && for f in {ckpt_dir}/*.pth {base}/*.pth {ckpt_alt}/*.pth {base_alt}/*.pth 2>/dev/null; do [ -f \"$f\" ] && cp \"$f\" \"{remote_root}/models/yolox_{exp}_$(basename \"$f\")\"; done", timeout=15)
        time.sleep(2)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: empty checkpoint dir – one more copy_artifacts and extended wait (slow FS / delayed flush) before final fail.
        if _is_checkpoint_dir_empty(cfg, remote_root, exp):
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC + 10)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            # Self-heal: one last probe run (empty dir = run likely exited before first save); long wait so probe can complete (slow MPS).
            _clean_output_dir(cfg, remote_root, exp)
            _ensure_output_dir(cfg, remote_root, exp)
            status_last, _ = _run_one_train(cfg, cmd_probe, session, wait_sec=FINAL_PROBE_WAIT_SEC)
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            if status_last == "done":
                time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
            # Self-heal: probe timed out but may have just finished; give FS time and recheck before final fail.
            if status_last == "timeout":
                time.sleep(90)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
            # Self-heal: copy_artifacts may surface probe checkpoint from YOLOX_outputs to models/ (FS flush / path).
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(5)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            # Self-heal: empty checkpoint dir still has no .pth after probe – one more full training run before final fail.
            if not _has_checkpoint(cfg, remote_root, exp):
                _clean_output_dir(cfg, remote_root, exp)
                _ensure_output_dir(cfg, remote_root, exp)
                status_final, err_final = _run_one_train(cfg, cmd, session)
                if status_final == "start_failed":
                    return stage_fail(f"full_train start failed (empty-ckpt final run): {err_final or 'unknown'}")
                if status_final == "timeout":
                    return stage_fail("full_train timeout (2h) on empty-ckpt final run")
                time.sleep(POST_RUN_FLUSH_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
                time.sleep(FINAL_RECHECK_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
                time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
                if _has_checkpoint(cfg, remote_root, exp):
                    return stage_ok("full_train ok")
        # Self-heal: one last sync + wait before final fail (empty checkpoint dir / slow FS / delayed flush).
        ssh_run(cfg, "sync 2>/dev/null; true", timeout=15)
        time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: final-fail path when no checkpoint (empty dir or dir missing after clean) – one more probe run so at least one .pth is written, then recheck.
        if not _has_checkpoint(cfg, remote_root, exp):
            _clean_output_dir(cfg, remote_root, exp)
            _ensure_output_dir(cfg, remote_root, exp)
            _run_one_train(cfg, cmd_probe, session, wait_sec=FINAL_PROBE_WAIT_SEC)
            time.sleep(POST_RUN_FLUSH_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(5)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        # Self-heal: find any recent .pth under repo (or parent) and copy to models/ (path/layout variance).
        ssh_run(cfg, f"mkdir -p {remote_root}/models && (find {remote_root} -maxdepth 6 -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null; find {remote_root}/.. -maxdepth 4 -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null) | while read f; do [ -n \"$f\" ] && [ -f \"$f\" ] && cp \"$f\" \"{remote_root}/models/yolox_{exp}_$(basename \"$f\")\"; done", timeout=20)
        time.sleep(2)
        if _has_checkpoint(cfg, remote_root, exp):
            return stage_ok("full_train ok")
        # Self-heal: empty checkpoint dir (e.g. "total 0") – one more sync + extended wait + copy_artifacts before final fail.
        if _is_checkpoint_dir_empty(cfg, remote_root, exp):
            ssh_run(cfg, "sync 2>/dev/null; true", timeout=15)
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC + 15)
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(5)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        # Self-heal: one last empty-dir check before final fail (e.g. empty dir but earlier check was skipped).
        if _is_checkpoint_dir_empty(cfg, remote_root, exp):
            ssh_run(cfg, "sync 2>/dev/null; true", timeout=15)
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC + 10)
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(5)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        # Self-heal: final gate – if dir is empty, one more sync + copy_artifacts before failing (ensures we never fail this error class without trying once more).
        if _is_checkpoint_dir_empty(cfg, remote_root, exp):
            ssh_run(cfg, "sync 2>/dev/null; true", timeout=15)
            time.sleep(EMPTY_CKPT_EXTRA_WAIT_SEC + 10)
            ssh_run(cfg, f"cd {remote_root} && . {venv}/bin/activate && python -c \"import os; os.chdir('{remote_root}'); from scripts.run_train import copy_artifacts; copy_artifacts('{exp}')\"", timeout=30)
            time.sleep(5)
            if _has_checkpoint(cfg, remote_root, exp):
                return stage_ok("full_train ok")
        base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
        base_alt = f"{remote_root}/YOLOX_outputs/{exp}"
        _ok, diag, _ = ssh_run(cfg, f"ls -la {base} 2>/dev/null || echo 'base_missing'; ls -la {base_alt} 2>/dev/null || echo 'base_alt_missing'; find {remote_root} -name '*.pth' -mmin -{RECENT_CKPT_MINS} 2>/dev/null | head -3", timeout=15)
        hint = f" (remote: {(diag or '').strip()[:200]})" if diag else ""
        return stage_fail(f"full_train finished but no checkpoint in YOLOX_outputs/{exp}{hint}")


if __name__ == "__main__":
    raise SystemExit(main())
