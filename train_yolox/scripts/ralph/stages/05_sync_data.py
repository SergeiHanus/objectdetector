#!/usr/bin/env python3
from __future__ import annotations

from common import dataset_path, load_config, load_state, parse_stage_args, raw_data_subdir, rsync_to_remote, ssh_run, stage_fail, stage_ok


def main() -> int:
    args = parse_stage_args("sync_data")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    if not cfg.get("sync", {}).get("sync_dataset", True):
        return stage_ok("sync_data skipped (sync_dataset=false)")

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    local_data = dataset_path(cfg)
    if not local_data.exists():
        return stage_fail(f"dataset_source {local_data} not found")

    remote_data = f"{remote_root}/data"
    ok, _out, err = rsync_to_remote(cfg, str(local_data), remote_data, excludes=["*.pyc", ".DS_Store"])
    if not ok:
        return stage_fail(f"rsync data failed: {err}")

    # Do not rsync local YOLOX/datasets to remote: 07_smoke_train and 08_full_train create
    # YOLOX/datasets/<ds> -> data symlink on remote. Overwriting it here caused "file named
    # datasets/.../smoke_placeholder.png not found" when the loader uses that path (cwd=YOLOX).

    raw_subdir = raw_data_subdir(cfg)
    raw_path = f"{remote_data}/{raw_subdir}"
    # Accept either: (1) annotations already present, or (2) data/<raw_data_subdir> with .jpg/.png (prepare_data will create annotations)
    validate_cmd = (
        f"test -d {remote_data} && "
        f"(test -f {remote_data}/annotations/train_labels.json 2>/dev/null || "
        f" (test -d {raw_path} && (find {raw_path} -maxdepth 1 \\( -name '*.jpg' -o -name '*.png' \\) 2>/dev/null | head -1 | grep -q .))) && "
        "echo ok"
    )
    ok3, out3, err3 = ssh_run(cfg, validate_cmd, timeout=20)
    if not ok3 or "ok" not in out3:
        return stage_fail(f"sync_data validation failed: need data/annotations/*.json or data/{raw_subdir}/*.jpg|*.png; {err3 or out3}")
    return stage_ok("sync_data ok")


if __name__ == "__main__":
    raise SystemExit(main())
