#!/usr/bin/env python3
from __future__ import annotations

from common import TRAIN_YOLOX, dataset_path, load_config, load_state, parse_stage_args, stage_fail, stage_ok


def main() -> int:
    args = parse_stage_args("preflight_local")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")
    if not (TRAIN_YOLOX / "exps").is_dir():
        return stage_fail("exps/ not found")
    if not (TRAIN_YOLOX / "scripts" / "run_train.py").exists():
        return stage_fail("scripts/run_train.py not found")
    if not (TRAIN_YOLOX / "exps" / f"{exp}.py").exists():
        return stage_fail(f"exps/{exp}.py not found")
    if cfg.get("sync", {}).get("sync_dataset", True):
        ds = dataset_path(cfg)
        if not ds.exists():
            return stage_fail(f"dataset_source {ds} not found")
    return stage_ok("preflight_local ok")


if __name__ == "__main__":
    raise SystemExit(main())
