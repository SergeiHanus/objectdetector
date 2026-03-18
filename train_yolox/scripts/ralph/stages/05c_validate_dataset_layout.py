#!/usr/bin/env python3
"""
Ralph stage: validate that prepared images/annotations exist where YOLOX training expects them.

We require:
- data/annotations/{train_labels.json,val_labels.json} exist
- data/train/images and data/val/images contain at least one .jpg/.png
- YOLOX/datasets/<ds_name> is present (directory or symlink) and contains those same subpaths
- a small sample of images referenced by val_labels.json exist on disk
"""
from __future__ import annotations

import shlex

from common import load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok


def _exp_to_dataset_name(exp_name: str) -> str:
    if exp_name.startswith("yolox_s_"):
        return exp_name.replace("yolox_s_", "", 1) + "_dataset"
    return exp_name + "_dataset"


def main() -> int:
    args = parse_stage_args("validate_dataset_layout")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")
    ds_name = _exp_to_dataset_name(exp)

    cmd = (
        f"cd {shlex.quote(remote_root)} && "
        # Self-heal (1): pick a working remote python (some macOS hosts have python3 only; others use a venv)
        f"PYTHON=; "
        f"test -x {shlex.quote(venv)}/bin/python && PYTHON={shlex.quote(venv)}/bin/python; "
        f"test -z \"$PYTHON\" && command -v python3 >/dev/null 2>&1 && PYTHON=python3; "
        f"test -z \"$PYTHON\" && command -v python >/dev/null 2>&1 && PYTHON=python; "
        f"test -n \"$PYTHON\" || (echo 'PROBLEMS:'; echo '- no python interpreter found on remote (expected python3 or venv python)'; exit 2); "
        # Self-heal (2): ensure YOLOX/datasets/<ds> (and repo-root datasets/<ds>) resolve to data/ when safe
        "mkdir -p YOLOX/datasets datasets && "
        f"(test -d YOLOX/datasets/{shlex.quote(ds_name)} && test ! -L YOLOX/datasets/{shlex.quote(ds_name)} && echo ok) || "
        f"(ln -sfn {shlex.quote(remote_root)}/data YOLOX/datasets/{shlex.quote(ds_name)} && echo ok) >/dev/null 2>&1 || true; "
        f"(test -d datasets/{shlex.quote(ds_name)} && test ! -L datasets/{shlex.quote(ds_name)} && echo ok) || "
        f"(ln -sfn {shlex.quote(remote_root)}/data datasets/{shlex.quote(ds_name)} && echo ok) >/dev/null 2>&1 || true; "
        "\"$PYTHON\" - <<'PY'\n"
        "import json, os, sys\n"
        f"ds_name = {ds_name!r}\n"
        "problems = []\n"
        "def require(path, kind='path'):\n"
        "    if kind == 'file' and not os.path.isfile(path):\n"
        "        problems.append(f'missing file: {path}')\n"
        "    elif kind == 'dir' and not os.path.isdir(path):\n"
        "        problems.append(f'missing dir: {path}')\n"
        "    elif kind == 'path' and not os.path.exists(path):\n"
        "        problems.append(f'missing: {path}')\n"
        "\n"
        "require('data/annotations/train_labels.json', 'file')\n"
        "require('data/annotations/val_labels.json', 'file')\n"
        "require('data/train/images', 'dir')\n"
        "require('data/val/images', 'dir')\n"
        "\n"
        "def has_images(d):\n"
        "    try:\n"
        "        for name in os.listdir(d):\n"
        "            if name.lower().endswith(('.jpg', '.jpeg', '.png')):\n"
        "                return True\n"
        "    except Exception:\n"
        "        return False\n"
        "    return False\n"
        "\n"
        "if os.path.isdir('data/train/images') and not has_images('data/train/images'):\n"
        "    problems.append('no images in data/train/images')\n"
        "if os.path.isdir('data/val/images') and not has_images('data/val/images'):\n"
        "    problems.append('no images in data/val/images')\n"
        "\n"
        "ds_root = os.path.join('YOLOX', 'datasets', ds_name)\n"
        "require(ds_root, 'path')\n"
        "require(os.path.join(ds_root, 'train', 'images'), 'dir')\n"
        "require(os.path.join(ds_root, 'val', 'images'), 'dir')\n"
        "require(os.path.join(ds_root, 'annotations', 'train_labels.json'), 'file')\n"
        "require(os.path.join(ds_root, 'annotations', 'val_labels.json'), 'file')\n"
        "\n"
        "# Validate a small sample of referenced val images exists\n"
        "sample_limit = 10\n"
        "try:\n"
        "    with open('data/annotations/val_labels.json', 'r', encoding='utf-8') as f:\n"
        "        val = json.load(f) or {}\n"
        "except Exception as e:\n"
        "    problems.append(f'could not read val_labels.json: {e}')\n"
        "    val = {}\n"
        "\n"
        "images = val.get('images', []) or []\n"
        "missing_sample = 0\n"
        "checked = 0\n"
        "for img in images:\n"
        "    if checked >= sample_limit:\n"
        "        break\n"
        "    file_name = img.get('file_name')\n"
        "    if not file_name:\n"
        "        continue\n"
        "    checked += 1\n"
        "    base = os.path.basename(file_name)\n"
        "    # Some exporters include subpaths in file_name; accept either full rel or basename.\n"
        "    candidates = [\n"
        "        os.path.join('data', 'val', 'images', file_name),\n"
        "        os.path.join('data', 'val', 'images', base),\n"
        "        os.path.join(ds_root, 'val', 'images', file_name),\n"
        "        os.path.join(ds_root, 'val', 'images', base),\n"
        "    ]\n"
        "    if not any(os.path.exists(p) for p in candidates):\n"
        "        missing_sample += 1\n"
        "\n"
        "if checked and missing_sample:\n"
        "    problems.append(f'missing {missing_sample}/{checked} sampled val images referenced by annotations')\n"
        "\n"
        "if problems:\n"
        "    print('PROBLEMS:')\n"
        "    for p in problems:\n"
        "        print('-', p)\n"
        "    sys.exit(2)\n"
        "\n"
        "print('ok')\n"
        "PY"
    )
    ok, out, err = ssh_run(cfg, cmd, timeout=60)
    if ok and (out or "").strip().endswith("ok"):
        return stage_ok("validate_dataset_layout ok")
    details = {"stdout": (out or "")[:1500], "stderr": (err or "")[:1500], "dataset": ds_name}
    return stage_fail("validate_dataset_layout failed: prepared images/annotations missing or not where YOLOX expects", details=details)


if __name__ == "__main__":
    raise SystemExit(main())

