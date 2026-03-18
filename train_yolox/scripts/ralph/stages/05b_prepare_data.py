#!/usr/bin/env python3
"""
Ralph stage: run YOLOX data preparation on remote after sync_data.
Assumes sync_data has run and data/ (e.g. data/<raw_data_subdir>) is on remote.
Runs top-level src/yolox_data_prep.py on remote to produce data/train, data/val, data/annotations.
Raw path comes from config sync.raw_data_subdir.
"""

from __future__ import annotations

from common import load_config, load_state, parse_stage_args, raw_data_subdir, ssh_run, stage_fail, stage_ok


def _annotations_look_minimal(cfg: dict, remote_root: str) -> bool:
    """
    True if remote data/annotations exist but are the "smoke placeholder" minimal set
    (e.g. 0-1 images). In that case we must run src/yolox_data_prep.py so full training
    doesn't "finish epochs instantly" due to ~1 batch/epoch.
    """
    cmd = (
        f"cd {remote_root} && "
        "python3 -c "
        "\"import json, os; "
        "p='data/annotations/train_labels.json'; "
        "d=json.load(open(p)) if os.path.isfile(p) else {}; "
        "print(len(d.get('images', [])))\""
        " 2>/dev/null || "
        "python -c "
        "\"import json, os; "
        "p='data/annotations/train_labels.json'; "
        "d=json.load(open(p)) if os.path.isfile(p) else {}; "
        "print(len(d.get('images', [])))\""
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=15)
    if not ok or not out:
        return False
    try:
        n = int((out or "").strip().splitlines()[-1])
    except Exception:
        return False
    return n <= 1


def _prepared_images_exist(cfg: dict, remote_root: str) -> bool:
    """
    True if data/train/images and data/val/images contain at least one .jpg/.png each.
    This is the layout YOLOX training expects (via YOLOX/datasets/<ds> -> data symlink).
    """
    cmd = (
        f"cd {remote_root} && "
        "test -d data/train/images && "
        "test -d data/val/images && "
        "test -n \"$(ls -1 data/train/images/*.jpg data/train/images/*.png 2>/dev/null | head -n 1)\" && "
        "test -n \"$(ls -1 data/val/images/*.jpg data/val/images/*.png 2>/dev/null | head -n 1)\" && "
        "echo ok"
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=15)
    return bool(ok and out and "ok" in out)


def main() -> int:
    args = parse_stage_args("prepare_data")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    if not cfg.get("sync", {}).get("sync_dataset", True):
        return stage_ok("prepare_data skipped (sync_dataset=false)")

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    raw_subdir = raw_data_subdir(cfg)
    raw_dir = f"data/{raw_subdir}"

    # If annotations already exist (e.g. pre-built data synced), skip prep only if images are prepared too.
    check_ann = (
        f"test -f {remote_root}/data/annotations/train_labels.json && "
        f"test -f {remote_root}/data/annotations/val_labels.json && echo ok"
    )
    ok_ann, out_ann, _ = ssh_run(cfg, check_ann, timeout=10)
    if (
        ok_ann
        and out_ann
        and "ok" in out_ann
        and not _annotations_look_minimal(cfg, remote_root)
        and _prepared_images_exist(cfg, remote_root)
    ):
        return stage_ok("prepare_data skipped (annotations already present)")

    cmd = (
        f"cd {remote_root} && "
        f". {venv}/bin/activate && "
        f"python src/yolox_data_prep.py --train-yolox-root . --raw-dir {raw_dir!r} --dataset-name marker_dataset"
    )
    ok, out, err = ssh_run(cfg, cmd, timeout=300)
    if not ok:
        return stage_fail(f"prepare_data failed: {err or out or 'unknown'}")

    # Quick check: annotations exist so smoke_train can pass
    check = (
        f"test -f {remote_root}/data/annotations/train_labels.json && "
        f"test -f {remote_root}/data/annotations/val_labels.json && "
        "echo ok"
    )
    ok2, out2, _ = ssh_run(cfg, check, timeout=10)
    if not ok2 or "ok" not in (out2 or ""):
        return stage_fail(f"prepare_data ran but annotations not found; check {raw_dir} on remote (config: sync.raw_data_subdir)")

    return stage_ok("prepare_data ok")


if __name__ == "__main__":
    raise SystemExit(main())
