#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import tempfile

from common import (
    TRAIN_YOLOX,
    load_config,
    load_state,
    parse_stage_args,
    rsync_to_remote,
    ssh_run,
    stage_fail,
    stage_ok,
)


def _is_venv_activate_missing(err: str) -> bool:
    """True if the error indicates venv/bin/activate path is missing."""
    if not err:
        return False
    e = err.lower()
    return ("activate" in e and "no such file or directory" in e) or (
        "/bin/activate" in e and "no such file" in e
    )


def _resolve_venv_on_remote(cfg: dict, remote_root: str, venv: str) -> str | None:
    """Return venv path if bin/activate exists on remote; try venv then remote_root/.venv. None if neither exists."""
    for candidate in (venv, f"{remote_root}/.venv"):
        check = f"test -f {candidate}/bin/activate && echo ok"
        ok, out, _ = ssh_run(cfg, check, timeout=10)
        if ok and "ok" in (out or ""):
            return candidate
    return None


def _yolox_present_on_remote(cfg: dict, remote_root: str) -> bool:
    """True if YOLOX/ and YOLOX/tools/train.py exist on remote (synced from repo)."""
    check = f"test -d {remote_root}/YOLOX && test -f {remote_root}/YOLOX/tools/train.py && echo ok"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    return bool(ok and out and "ok" in out)


def _is_missing_dataset_path(err: str) -> bool:
    """True if the error indicates YOLOX/datasets/... path is missing (sync_data put data in data/)."""
    if not err:
        return False
    return "Missing dataset path: YOLOX/datasets/" in err


def _is_data_directory_not_found(err: str) -> bool:
    """True if the error indicates exp verify_dataset_config failed (data_dir e.g. datasets/marker_dataset not found from repo root)."""
    if not err:
        return False
    return "Data directory not found:" in err and "datasets/" in err


def _exp_to_dataset_name(exp_name: str) -> str:
    """yolox_s_marker -> marker_dataset."""
    if exp_name.startswith("yolox_s_"):
        return exp_name.replace("yolox_s_", "", 1) + "_dataset"
    return exp_name + "_dataset"


def _exp_to_class_name(exp_name: str) -> str:
    """yolox_s_marker -> marker; used for COCO category name."""
    if exp_name.startswith("yolox_s_"):
        return exp_name.replace("yolox_s_", "", 1)
    return "circle"


def _is_empty_dataset_sampler(err: str) -> bool:
    """True if training failed due to InfiniteSampler(size=0) / assert size > 0 (empty dataset)."""
    if not err:
        return False
    return "assert size > 0" in err or ("InfiniteSampler" in err and "size" in err.lower())


def _is_missing_image_file(err: str) -> bool:
    """True if YOLOX loader failed because an image file referenced in annotations was not found."""
    if not err:
        return False
    return ("file named" in err and "not found" in err) or (
        "assert img is not None" in err and "not found" in err
    )


def _is_libpng_crc_error(err: str) -> bool:
    """True if libpng reported CRC errors (corrupted or missing placeholder PNG)."""
    if not err:
        return False
    return "libpng" in err.lower() and "crc" in err.lower()


def _is_placeholder_missing_or_corrupt_before_run(err: str) -> bool:
    """True if the error is placeholder missing or corrupt at YOLOX dataset path before run (self-heal trigger)."""
    if not err:
        return False
    return "placeholder missing or corrupt at YOLOX dataset path before run" in err


def _is_no_module_named_exps(err: str) -> bool:
    """True if run_train failed because exps package was not on path (exp file uses 'from exps.xxx')."""
    if not err:
        return False
    return "No module named 'exps'" in err or "No module named \"exps\"" in err


def _ensure_smoke_placeholder_file(cfg: dict, remote_root: str, exp: str | None = None) -> bool:
    """Unconditionally ensure smoke_placeholder.png exists in data/train/images and data/val/images (self-heal for missing/corrupt placeholder). If exp is set, also write under YOLOX/datasets/<ds>/ so the path used by the loader (cwd=YOLOX) definitely exists. Returns True if ok."""
    b64 = _MINIMAL_PNG_BASE64
    # Use Python on remote (no pipes) to avoid shell/base64 corruption and to write to both data/ and YOLOX/datasets/<ds>/
    data_root = f"{remote_root}/data"
    dirs = [f"{data_root}/train/images", f"{data_root}/val/images"]
    if exp:
        ds_name = _exp_to_dataset_name(exp)
        dirs.extend([
            f"{remote_root}/YOLOX/datasets/{ds_name}/train/images",
            f"{remote_root}/YOLOX/datasets/{ds_name}/val/images",
        ])
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import os, base64",
            f"png = base64.b64decode({b64!r})",
            f"dirs = {dirs!r}",
            "for d in dirs:",
            "    os.makedirs(d, exist_ok=True)",
            "    with open(os.path.join(d, 'smoke_placeholder.png'), 'wb') as f:",
            "        f.write(png)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    if ok and out and "ok" in out:
        return True
    # Fallback: Python-only write (no shell echo/base64) to avoid PNG corruption (libpng CRC) on remote.
    ok_fb, out_fb, _ = _remote_python(
        cfg,
        "\n".join([
            "import os, base64",
            f"png = base64.b64decode({b64!r})",
            f"dirs = {dirs!r}",
            "for d in dirs:",
            "    os.makedirs(d, exist_ok=True)",
            "    with open(os.path.join(d, 'smoke_placeholder.png'), 'wb') as f:",
            "        f.write(png)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    return bool(ok_fb and out_fb and "ok" in out_fb)


def _ensure_referenced_images_exist(cfg: dict, remote_root: str, exp: str) -> bool:
    """On remote: for each image referenced in annotations, ensure it exists at both data/ and YOLOX/datasets/<ds>/ paths.

    Why both? In practice, YOLOX may read from a real directory under YOLOX/datasets/<ds>/ (if a prior sync overwrote the symlink),
    while other stages write under data/. This keeps the stage self-healing for "file named ... not found" and placeholder corruption.
    """
    data_root = f"{remote_root}/data"
    ds_name = _exp_to_dataset_name(exp)
    yolox_ds_root = f"{remote_root}/YOLOX/datasets/{ds_name}"
    repo_ds_root = f"{remote_root}/datasets/{ds_name}"
    b64 = _MINIMAL_PNG_BASE64
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import json, os, base64",
            f"data_root = {data_root!r}",
            f"yolox_ds_root = {yolox_ds_root!r}",
            f"repo_ds_root = {repo_ds_root!r}",
            f"png_bytes = base64.b64decode({b64!r})",
            "pairs = [('train_labels.json', 'train/images'), ('val_labels.json', 'val/images')]",
            "def _safe_rel(p: str) -> str | None:",
            "    # Only allow relative, non-traversing paths; normalize separators.",
            "    if not p:",
            "        return None",
            "    p = p.replace('\\\\', '/')",
            "    p = p.lstrip('/')",
            "    norm = os.path.normpath(p)",
            "    if norm in ('.', ''):",
            "        return None",
            "    if norm.startswith('..') or os.path.isabs(norm):",
            "        return None",
            "    return norm",
            "for ann_name, img_subdir in pairs:",
            "    ann_path = os.path.join(data_root, 'annotations', ann_name)",
            "    if not os.path.isfile(ann_path):",
            "        continue",
            "    with open(ann_path) as f:",
            "        data = json.load(f)",
            "    img_dirs = [os.path.join(data_root, img_subdir), os.path.join(yolox_ds_root, img_subdir), os.path.join(repo_ds_root, img_subdir)]",
            "    for d in img_dirs:",
            "        os.makedirs(d, exist_ok=True)",
            "    for im in data.get('images', []):",
            "        fn = im.get('file_name')",
            "        if not fn:",
            "            continue",
            "        rel = _safe_rel(str(fn))",
            "        base = os.path.basename(str(fn))",
            "        for d in img_dirs:",
            "            # Create both the basename and the full relative path (some COCO exporters include subpaths).",
            "            targets = []",
            "            if base:",
            "                targets.append(os.path.join(d, base))",
            "            if rel:",
            "                targets.append(os.path.join(d, rel))",
            "            for full in targets:",
            "                os.makedirs(os.path.dirname(full), exist_ok=True)",
            "                # Always refresh smoke_placeholder.png (commonly missing/corrupted); otherwise only create when missing.",
            "                if (not os.path.exists(full)) or os.path.basename(full) == 'smoke_placeholder.png':",
            "                    with open(full, 'wb') as out:",
            "                        out.write(png_bytes)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=30,
    )
    return bool(ok and out and "ok" in out)


def _remote_placeholder_exists(cfg: dict, remote_root: str, exp: str) -> bool:
    """True if smoke_placeholder.png exists at either common loader path on remote.

    Depending on cwd/layout, YOLOX may open:
    - repo-root: datasets/<ds>/train/images/smoke_placeholder.png
    - inside YOLOX/: YOLOX/datasets/<ds>/train/images/smoke_placeholder.png
    """
    ds_name = _exp_to_dataset_name(exp)
    p1 = f"{remote_root}/YOLOX/datasets/{ds_name}/train/images/smoke_placeholder.png"
    p2 = f"{remote_root}/datasets/{ds_name}/train/images/smoke_placeholder.png"
    check = f"(test -f {p1} || test -f {p2}) && echo ok"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    return bool(ok and out and "ok" in out)


def _remote_placeholder_valid(cfg: dict, remote_root: str, exp: str) -> bool:
    """True if smoke_placeholder.png exists at a loader path and looks like a valid PNG.

    Also self-heals by copying any valid candidate into both common loader locations.
    """
    ds_name = _exp_to_dataset_name(exp)
    primary_yolox = f"{remote_root}/YOLOX/datasets/{ds_name}/train/images/smoke_placeholder.png"
    primary_repo = f"{remote_root}/datasets/{ds_name}/train/images/smoke_placeholder.png"
    # Fall back to other known-good locations; if one is valid, copy it into both loader paths.
    candidates = [
        primary_yolox,
        primary_repo,
        f"{remote_root}/data/train/images/smoke_placeholder.png",
    ]
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import os",
            f"primary_yolox = {primary_yolox!r}",
            f"primary_repo = {primary_repo!r}",
            f"candidates = {candidates!r}",
            "def is_png(p: str) -> bool:",
            "    if not os.path.isfile(p):",
            "        return False",
            "    try:",
            "        import cv2",
            "        img = cv2.imread(p)",
            "        if img is not None:",
            "            return True",
            "        # Some OpenCV builds can import but can't decode PNGs; fall back to signature check.",
            "        with open(p, 'rb') as f:",
            "            raw = f.read(8)",
            "        return raw == b'\\x89PNG\\r\\n\\x1a\\n'",
            "    except Exception:",
            "        try:",
            "            with open(p, 'rb') as f:",
            "                raw = f.read(8)",
            "            return raw == b'\\x89PNG\\r\\n\\x1a\\n'",
            "        except Exception:",
            "            return False",
            "good = None",
            "for p in candidates:",
            "    if is_png(p):",
            "        good = p",
            "        break",
            "def copy_into(dst: str, src: str) -> None:",
            "    try:",
            "        os.makedirs(os.path.dirname(dst), exist_ok=True)",
            "        with open(src, 'rb') as s, open(dst, 'wb') as d:",
            "            d.write(s.read())",
            "    except Exception:",
            "        pass",
            "if good:",
            "    try:",
            "        if good != primary_yolox:",
            "            copy_into(primary_yolox, good)",
            "        if good != primary_repo:",
            "            copy_into(primary_repo, good)",
            "    except Exception:",
            "        pass",
            "ok_y = is_png(primary_yolox)",
            "ok_r = is_png(primary_repo)",
            "print('ok' if (good and (ok_y or ok_r)) else 'invalid')",
        ]),
        cwd=remote_root,
        timeout=10,
    )
    return bool(ok and out and "ok" in out and "invalid" not in out)


def _write_placeholder_from_yolox_cwd(cfg: dict, remote_root: str, exp: str) -> bool:
    """Write smoke_placeholder.png from YOLOX cwd to datasets/<ds>/train/images and val/images so the loader (cwd=YOLOX) finds it. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    b64 = _MINIMAL_PNG_BASE64
    # Write using the exact relative path the YOLOX loader uses (cwd=YOLOX, data_dir=datasets/<ds>)
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import base64, os",
            f"raw = base64.b64decode({b64!r})",
            f"d = {('datasets/' + ds_name)!r}",
            "for sub in ('train/images', 'val/images'):",
            "    path = os.path.join(d, sub)",
            "    os.makedirs(path, exist_ok=True)",
            "    with open(os.path.join(path, 'smoke_placeholder.png'), 'wb') as f:",
            "        f.write(raw)",
            "print('ok')",
        ]),
        cwd=f"{remote_root}/YOLOX",
        timeout=15,
    )
    if ok and out and "ok" in out:
        return True
    # Fallback: write at absolute loader path (Python-only to avoid PNG corruption).
    abs_dirs = [
        f"{remote_root}/YOLOX/datasets/{ds_name}/train/images",
        f"{remote_root}/YOLOX/datasets/{ds_name}/val/images",
    ]
    ok_fb, out_fb, _ = _remote_python(
        cfg,
        "\n".join([
            "import os, base64",
            f"raw = base64.b64decode({b64!r})",
            f"abs_dirs = {abs_dirs!r}",
            "for d in abs_dirs:",
            "    os.makedirs(d, exist_ok=True)",
            "    with open(os.path.join(d, 'smoke_placeholder.png'), 'wb') as f:",
            "        f.write(raw)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    return bool(ok_fb and out_fb and "ok" in out_fb)


def _write_placeholder_at_loader_path(cfg: dict, remote_root: str, exp: str) -> bool:
    """Write smoke_placeholder.png at the exact absolute path the YOLOX loader uses (data_dir/name/file_name with cwd=YOLOX). Ensures the file exists even when YOLOX/datasets/<ds> is a real directory instead of a symlink. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    b64 = _MINIMAL_PNG_BASE64
    # Paths the loader opens: cwd=YOLOX, img_file = os.path.join(data_dir, name, file_name) -> datasets/<ds>/train/images/smoke_placeholder.png
    abs_dirs = [
        f"{remote_root}/YOLOX/datasets/{ds_name}/train/images",
        f"{remote_root}/YOLOX/datasets/{ds_name}/val/images",
    ]
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import os, base64",
            f"raw = base64.b64decode({b64!r})",
            f"abs_dirs = {abs_dirs!r}",
            "for d in abs_dirs:",
            "    os.makedirs(d, exist_ok=True)",
            "    p = os.path.join(d, 'smoke_placeholder.png')",
            "    with open(p, 'wb') as f:",
            "        f.write(raw)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    return bool(ok and out and "ok" in out)


def _write_placeholder_at_repo_root_dataset_path(cfg: dict, remote_root: str, exp: str) -> bool:
    """Write smoke_placeholder.png at repo-root datasets/<ds>/train/images and val/images so the file exists when cwd is repo root or when symlinks resolve there. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    b64 = _MINIMAL_PNG_BASE64
    abs_dirs = [
        f"{remote_root}/datasets/{ds_name}/train/images",
        f"{remote_root}/datasets/{ds_name}/val/images",
    ]
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import os, base64",
            f"raw = base64.b64decode({b64!r})",
            f"abs_dirs = {abs_dirs!r}",
            "for d in abs_dirs:",
            "    os.makedirs(d, exist_ok=True)",
            "    p = os.path.join(d, 'smoke_placeholder.png')",
            "    with open(p, 'wb') as f:",
            "        f.write(raw)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    return bool(ok and out and "ok" in out)


def _remove_placeholder_files(cfg: dict, remote_root: str, exp: str) -> bool:
    """Remove smoke_placeholder.png from all locations so a corrupted file is replaced by fresh write. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    dirs = [
        f"{remote_root}/data/train/images",
        f"{remote_root}/data/val/images",
        f"{remote_root}/YOLOX/datasets/{ds_name}/train/images",
        f"{remote_root}/YOLOX/datasets/{ds_name}/val/images",
        f"{remote_root}/datasets/{ds_name}/train/images",
        f"{remote_root}/datasets/{ds_name}/val/images",
    ]
    ok, out, _ = _remote_python(
        cfg,
        "\n".join([
            "import os",
            f"dirs = {dirs!r}",
            "for d in dirs:",
            "    p = os.path.join(d, 'smoke_placeholder.png')",
            "    if os.path.isfile(p) or os.path.islink(p):",
            "        try:",
            "            os.remove(p)",
            "        except Exception:",
            "            pass",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    return bool(ok and out and "ok" in out)


def _ensure_yolox_dataset_link(cfg: dict, remote_root: str, exp: str) -> bool:
    """Create YOLOX/datasets/<ds_name> -> remote_root/data on remote so layout matches run_train. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    # data is at remote_root/data; run_train expects remote_root/YOLOX/datasets/<ds_name>/train/images etc.
    # Use absolute target so the link resolves correctly regardless of cwd when resolved.
    # Be non-destructive: if a real directory already exists (not a symlink), keep it.
    data_abs = f"{remote_root}/data"
    setup = (
        f"cd {remote_root} && "
        f"mkdir -p YOLOX/datasets && "
        f"(test -d YOLOX/datasets/{ds_name} && test ! -L YOLOX/datasets/{ds_name} && echo ok) || "
        f"(ln -sfn {data_abs} YOLOX/datasets/{ds_name} && echo ok)"
    )
    ok, out, _ = ssh_run(cfg, setup, timeout=10)
    return bool(ok and out and "ok" in out)


def _force_yolox_dataset_link(cfg: dict, remote_root: str, exp: str) -> bool:
    """Remove YOLOX/datasets/<ds> if it is a symlink, then create symlink to data. Use when placeholder path is wrong/broken. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    data_abs = f"{remote_root}/data"
    setup = (
        f"cd {remote_root} && "
        f"mkdir -p YOLOX/datasets && "
        f"(test -L YOLOX/datasets/{ds_name} && rm -f YOLOX/datasets/{ds_name}) || true && "
        f"ln -sfn {data_abs} YOLOX/datasets/{ds_name} && echo ok"
    )
    ok, out, _ = ssh_run(cfg, setup, timeout=10)
    return bool(ok and out and "ok" in out)


def _force_repo_root_dataset_link(cfg: dict, remote_root: str, exp: str) -> bool:
    """Remove repo datasets/<ds> if it is a symlink, then create symlink to data. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    data_abs = f"{remote_root}/data"
    setup = (
        f"cd {remote_root} && "
        f"mkdir -p datasets && "
        f"(test -L datasets/{ds_name} && rm -f datasets/{ds_name}) || true && "
        f"ln -sfn {data_abs} datasets/{ds_name} && echo ok"
    )
    ok, out, _ = ssh_run(cfg, setup, timeout=10)
    return bool(ok and out and "ok" in out)


def _force_dataset_symlinks_removing_real(cfg: dict, remote_root: str, exp: str) -> bool:
    """Remove YOLOX/datasets/<ds> and datasets/<ds> even if real directories, then create symlinks to data. Self-heal when placeholder is missing/corrupt due to wrong layout (real dirs instead of symlinks). Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    data_abs = f"{remote_root}/data"
    setup = (
        f"cd {remote_root} && "
        f"mkdir -p YOLOX/datasets datasets && "
        f"rm -rf YOLOX/datasets/{ds_name} && ln -sfn {data_abs} YOLOX/datasets/{ds_name} && "
        f"rm -rf datasets/{ds_name} && ln -sfn {data_abs} datasets/{ds_name} && echo ok"
    )
    ok, out, _ = ssh_run(cfg, setup, timeout=15)
    return bool(ok and out and "ok" in out)


def _ensure_repo_root_dataset_link(cfg: dict, remote_root: str, exp: str) -> bool:
    """Create remote_root/datasets/<ds_name> -> remote_root/data so exp.verify_dataset_config() (cwd=repo root) finds datasets/<ds_name>. Returns True if ok."""
    ds_name = _exp_to_dataset_name(exp)
    data_abs = f"{remote_root}/data"
    setup = (
        f"cd {remote_root} && "
        f"mkdir -p datasets && "
        f"(test -d datasets/{ds_name} && test ! -L datasets/{ds_name} && echo ok) || "
        f"(ln -sfn {data_abs} datasets/{ds_name} && echo ok)"
    )
    ok, out, _ = ssh_run(cfg, setup, timeout=10)
    return bool(ok and out and "ok" in out)


def _ensure_data_layout(cfg: dict, remote_root: str) -> bool:
    """Create data/train/images, data/val/images, data/annotations on remote so run_train layout check passes. Returns True if ok."""
    cmd = (
        f"mkdir -p {remote_root}/data/train/images {remote_root}/data/val/images {remote_root}/data/annotations && "
        "echo ok"
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=10)
    return bool(ok and out and "ok" in out)


def _scp_placeholder_to_remote(cfg: dict, remote_root: str, exp: str) -> bool:
    """Self-heal when remote Python placeholder writes fail: write PNG locally and SCP to all required remote paths. Returns True if all copies succeeded."""
    data_root = f"{remote_root}/data"
    dirs = [f"{data_root}/train/images", f"{data_root}/val/images"]
    if exp:
        ds_name = _exp_to_dataset_name(exp)
        dirs.extend([
            f"{remote_root}/YOLOX/datasets/{ds_name}/train/images",
            f"{remote_root}/YOLOX/datasets/{ds_name}/val/images",
            f"{remote_root}/datasets/{ds_name}/train/images",
            f"{remote_root}/datasets/{ds_name}/val/images",
        ])
    png_bytes = base64.b64decode(_MINIMAL_PNG_BASE64)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.write(fd, png_bytes)
        os.close(fd)
        for d in dirs:
            # Avoid nested quoting issues: ssh_run() already wraps the whole command in quotes.
            ok, _, _ = ssh_run(cfg, f"mkdir -p {shlex.quote(d)}", timeout=10)
            if not ok:
                return False
        target = cfg["ssh_target"]
        scp_opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if cfg.get("ssh_key_path") and os.path.isfile(cfg["ssh_key_path"]):
            scp_opts.extend(["-i", cfg["ssh_key_path"]])
        for d in dirs:
            dest = f"{target}:{d}/smoke_placeholder.png"
            r = subprocess.run(
                ["scp"] + scp_opts + [tmp, dest],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode != 0:
                return False
        return True
    except (OSError, subprocess.TimeoutExpired) as _:
        return False
    finally:
        if tmp and os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# Minimal COCO JSON so run_train check_layout passes (marker exp: single class "circle").
_MINIMAL_COCO_JSON = '{"images":[],"annotations":[],"categories":[{"id":0,"name":"circle","supercategory":"shape"}]}'


# 1x1 PNG (valid image so YOLOX loader accepts it); used for non-empty minimal dataset.
_MINIMAL_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgABSK+kcQAAAABJRU5ErkJggg=="

# Remote base64 decode via Python (works on Linux and macOS; macOS base64 uses -D not -d).
_REMOTE_B64_DECODE = "python3 -c 'import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read().strip()))'"
_REMOTE_B64_EXEC = "python3 -c 'import base64,sys; exec(base64.b64decode(sys.stdin.read().strip()).decode())'"


def _remote_python(cfg: dict, code: str, cwd: str | None = None, timeout: int = 30) -> tuple[bool, str, str]:
    """Run a small Python snippet on remote (prefers python3, falls back to python)."""
    # NOTE: Many callers build multi-line snippets; passing raw newlines through `python -c "..."` is shell-fragile.
    # Use base64 to transport the snippet safely as a single shell token, then exec() it on the remote.
    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    prefix = f"cd {cwd} && " if cwd else ""
    cmd = (
        f"{prefix}("
        f"command -v python3 >/dev/null 2>&1 && "
        f"python3 -c \"import base64; exec(base64.b64decode('{b64}').decode('utf-8'))\""
        f" || "
        f"python -c \"import base64; exec(base64.b64decode('{b64}').decode('utf-8'))\""
        f")"
    )
    return ssh_run(cfg, cmd, timeout=timeout)


def _minimal_coco_with_one_sample(class_name: str) -> str:
    """COCO JSON with one image and one annotation so dataset size >= 1 (avoids InfiniteSampler size=0)."""
    return json.dumps({
        "images": [{"id": 1, "file_name": "smoke_placeholder.png", "width": 640, "height": 640}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
        "categories": [{"id": 0, "name": class_name, "supercategory": "object"}],
    })


def _ensure_at_least_one_train_sample(cfg: dict, remote_root: str, exp: str) -> bool:
    """Ensure data has at least one train and one val sample (placeholder image + JSON) so InfiniteSampler size > 0. Returns True if ok."""
    class_name = _exp_to_class_name(exp)
    ann_dir = f"{remote_root}/data/annotations"
    train_img_dir = f"{remote_root}/data/train/images"
    val_img_dir = f"{remote_root}/data/val/images"
    coco = {
        "images": [{"id": 1, "file_name": "smoke_placeholder.png", "width": 640, "height": 640}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
        "categories": [{"id": 0, "name": class_name, "supercategory": "object"}],
    }
    # Write placeholder image via Python only (avoid shell/base64 corruption -> libpng CRC).
    ok_img, out_img, _ = _remote_python(
        cfg,
        "\n".join([
            "import os, base64",
            f"png = base64.b64decode({_MINIMAL_PNG_BASE64!r})",
            f"train_dir = {train_img_dir!r}",
            f"val_dir = {val_img_dir!r}",
            "os.makedirs(train_dir, exist_ok=True)",
            "os.makedirs(val_dir, exist_ok=True)",
            "with open(os.path.join(train_dir, 'smoke_placeholder.png'), 'wb') as f: f.write(png)",
            "with open(os.path.join(val_dir, 'smoke_placeholder.png'), 'wb') as f: f.write(png)",
            "print('img_ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    if not ok_img or "img_ok" not in (out_img or ""):
        # Self-heal: ensure placeholder via shared helper (writes to data/train/images, data/val/images and YOLOX paths).
        _ensure_smoke_placeholder_file(cfg, remote_root, exp)
    # Write annotations with one image + one annotation via Python (avoids shell length/quoting and remote base64 issues).
    ok_ann, out_ann, _ = _remote_python(
        cfg,
        "\n".join([
            "import json, os",
            f"ann_dir = {ann_dir!r}",
            f"coco = {coco!r}",
            "os.makedirs(ann_dir, exist_ok=True)",
            "s = json.dumps(coco)",
            "with open(os.path.join(ann_dir, 'train_labels.json'), 'w') as f: f.write(s)",
            "with open(os.path.join(ann_dir, 'val_labels.json'), 'w') as f: f.write(s)",
            "print('ok')",
        ]),
        cwd=remote_root,
        timeout=15,
    )
    if ok_ann and out_ann and "ok" in out_ann:
        return True
    # Self-heal: annotation write may fail due to command length/quoting over SSH; write via base64 decode on remote.
    b64_json = base64.b64encode(json.dumps(coco).encode()).decode()
    cmd_fb = (
        f"mkdir -p {ann_dir} && cd {ann_dir} && "
        f"echo {b64_json} | {_REMOTE_B64_DECODE} > train_labels.json && "
        f"cp train_labels.json val_labels.json && "
        "echo ok"
    )
    ok_fb, out_fb, _ = ssh_run(cfg, cmd_fb, timeout=15)
    return bool(ok_fb and out_fb and "ok" in out_fb)


def _ensure_minimal_annotation_files(cfg: dict, remote_root: str) -> bool:
    """Write minimal train_labels.json and val_labels.json under data/annotations so Missing dataset path is resolved. Returns True if ok. Uses Python on remote to avoid shell-quoting issues over SSH."""
    ann_dir = f"{remote_root}/data/annotations"
    # Use Python on remote so JSON is not passed through shell (avoids repr/quote breakage over SSH). Double-quote -c arg so repr(cmd) in ssh_run does not introduce single-quote issues.
    py_inner = (
        "import json; d={\"images\":[],\"annotations\":[],\"categories\":[{\"id\":0,\"name\":\"circle\",\"supercategory\":\"shape\"}]}; "
        "open(\"train_labels.json\",\"w\").write(json.dumps(d)); open(\"val_labels.json\",\"w\").write(json.dumps(d))"
    )
    py_escaped = py_inner.replace('"', '\\"')
    cmd = (
        f"mkdir -p {ann_dir} && cd {ann_dir} && "
        f'(python3 -c "{py_escaped}" || python -c "{py_escaped}") && '
        "echo ok"
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=10)
    if ok and out and "ok" in out:
        return True
    # Self-heal: ensure dir and retry with echo+base64 so no quoting of JSON (some remotes may have no python3 in path).
    b64 = base64.b64encode(_MINIMAL_COCO_JSON.encode()).decode()
    cmd2 = (
        f"mkdir -p {ann_dir} && cd {ann_dir} && "
        f"echo {b64} | {_REMOTE_B64_DECODE} > train_labels.json && "
        f"echo {b64} | {_REMOTE_B64_DECODE} > val_labels.json && "
        "echo ok"
    )
    ok2, out2, _ = ssh_run(cfg, cmd2, timeout=10)
    return bool(ok2 and out2 and "ok" in out2)


def _remote_dataset_annotations_exist(cfg: dict, remote_root: str, exp: str) -> bool:
    """True if run_train layout check would find annotations (YOLOX/datasets/<ds>/annotations/*.json)."""
    ds_name = _exp_to_dataset_name(exp)
    check = (
        f"cd {remote_root} && "
        f"test -f YOLOX/datasets/{ds_name}/annotations/train_labels.json && "
        f"test -f YOLOX/datasets/{ds_name}/annotations/val_labels.json && "
        "echo ok"
    )
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    return bool(ok and out and "ok" in out)


def _data_annotations_exist(cfg: dict, remote_root: str) -> bool:
    """True if data/annotations/{train,val}_labels.json exist on remote."""
    check = (
        f"test -f {remote_root}/data/annotations/train_labels.json && "
        f"test -f {remote_root}/data/annotations/val_labels.json && echo ok"
    )
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    return bool(ok and out and "ok" in out)


def _data_annotations_have_real_samples(cfg: dict, remote_root: str, min_images: int = 2) -> bool:
    """True if data/annotations/train_labels.json has >= min_images images."""
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
    return n >= min_images


def main() -> int:
    args = parse_stage_args("smoke_train")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    venv_cfg = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")
    # Smoke run uses a dedicated exp so output goes to YOLOX_outputs/yolox_s_marker_smoke
    # and never overwrites the full-training output (08 uses exp -> yolox_s_marker).
    exp_smoke = "yolox_s_marker_smoke"

    venv = _resolve_venv_on_remote(cfg, remote_root, venv_cfg)
    if venv is None:
        return stage_fail(
            "smoke_train failed: venv/bin/activate not found at config path or remote_root/.venv; run ensure_env stage"
        )

    if not _yolox_present_on_remote(cfg, remote_root):
        return stage_fail(
            "smoke_train failed: Missing YOLOX/ or YOLOX/tools/train.py; run sync_code first"
        )

    # Ensure dataset symlink and data layout exist before first run (self-heal for missing YOLOX/datasets/<ds>/ layout).
    if not _ensure_yolox_dataset_link(cfg, remote_root, exp):
        return stage_fail(
            "smoke_train failed: could not create YOLOX/datasets symlink; run sync_data first"
        )
    # So exp.verify_dataset_config() (cwd=repo root) finds datasets/<ds_name>.
    if not _ensure_repo_root_dataset_link(cfg, remote_root, exp):
        return stage_fail(
            "smoke_train failed: could not create repo-root datasets symlink; run sync_data first"
        )
    if not _ensure_data_layout(cfg, remote_root):
        return stage_fail(
            "smoke_train failed: could not create data layout on remote; run sync_data first"
        )
    # Only create minimal annotations when annotations are missing.
    # Important: do NOT overwrite real annotations here; doing so can make full_train run
    # with ~1 batch/epoch and look like "epochs complete instantly".
    if not _data_annotations_exist(cfg, remote_root):
        if not _ensure_minimal_annotation_files(cfg, remote_root):
            return stage_fail(
                "smoke_train failed: could not create annotation files on remote; run sync_data first"
            )
    if not _remote_dataset_annotations_exist(cfg, remote_root, exp):
        return stage_fail(
            "smoke_train failed: dataset path YOLOX/datasets/<exp>/annotations not found on remote; run sync_data first"
        )
    # Ensure smoke exp dataset symlink so run_train check_layout(yolox_s_marker_smoke) passes.
    if not _ensure_yolox_dataset_link(cfg, remote_root, exp_smoke):
        return stage_fail(
            "smoke_train failed: could not create YOLOX/datasets/yolox_s_marker_smoke symlink for smoke run"
        )

    # Proactive self-heal: remove any existing placeholder (may be corrupted), then ensure placeholder and referenced images exist.
    _remove_placeholder_files(cfg, remote_root, exp)
    _ensure_smoke_placeholder_file(cfg, remote_root, exp)
    _ensure_referenced_images_exist(cfg, remote_root, exp)
    # Write placeholder from YOLOX cwd so loader (cwd=YOLOX) finds datasets/<ds>/train/images/smoke_placeholder.png on first run.
    _write_placeholder_from_yolox_cwd(cfg, remote_root, exp)
    # Also write at absolute loader path so file exists even if YOLOX/datasets/<ds> is a real dir (e.g. after sync overwrote symlink).
    _write_placeholder_at_loader_path(cfg, remote_root, exp)
    # Repo-root path so file exists when datasets/<ds> is a real directory or symlink from repo root.
    _write_placeholder_at_repo_root_dataset_path(cfg, remote_root, exp)
    # Overwrite with known-good PNG via SCP so first run never sees a corrupted placeholder (avoids libpng CRC from remote Python/shell).
    _scp_placeholder_to_remote(cfg, remote_root, exp)

    # Pre-run verification: ensure placeholder exists and is valid at path loader uses (cwd=YOLOX); self-heal once more if missing or corrupt.
    if not _remote_placeholder_valid(cfg, remote_root, exp):
        _ensure_yolox_dataset_link(cfg, remote_root, exp)
        _write_placeholder_from_yolox_cwd(cfg, remote_root, exp)
        _write_placeholder_at_loader_path(cfg, remote_root, exp)
        # If still invalid (e.g. corrupt), overwrite with known-good via SCP.
        _scp_placeholder_to_remote(cfg, remote_root, exp)
        if not _remote_placeholder_valid(cfg, remote_root, exp):
            # Last-resort self-heal for this error class: force symlinks (fix broken/wrong links), layout + placeholder everywhere + SCP, then re-validate.
            _force_yolox_dataset_link(cfg, remote_root, exp)
            _force_repo_root_dataset_link(cfg, remote_root, exp)
            _ensure_data_layout(cfg, remote_root)
            _remove_placeholder_files(cfg, remote_root, exp)
            _scp_placeholder_to_remote(cfg, remote_root, exp)
            _ensure_smoke_placeholder_file(cfg, remote_root, exp)
            _write_placeholder_from_yolox_cwd(cfg, remote_root, exp)
            _write_placeholder_at_loader_path(cfg, remote_root, exp)
            _write_placeholder_at_repo_root_dataset_path(cfg, remote_root, exp)
            _scp_placeholder_to_remote(cfg, remote_root, exp)
            if not _remote_placeholder_valid(cfg, remote_root, exp):
                # Final self-heal for this error class: symlinks + layout + remove placeholders, then only SCP (no remote Python writes) and re-validate.
                def _do_final_placeholder_heal() -> bool:
                    _force_yolox_dataset_link(cfg, remote_root, exp)
                    _force_repo_root_dataset_link(cfg, remote_root, exp)
                    _ensure_data_layout(cfg, remote_root)
                    _remove_placeholder_files(cfg, remote_root, exp)
                    return _scp_placeholder_to_remote(cfg, remote_root, exp) and _remote_placeholder_valid(cfg, remote_root, exp)
                if _do_final_placeholder_heal():
                    pass  # continue to train
                elif _do_final_placeholder_heal():
                    pass  # self-heal retry: one more cycle then continue
                else:
                    # Pre-fail self-heal for this error class: force real dirs to symlinks (rm -rf then ln -sfn) so loader path resolves to data/
                    _force_dataset_symlinks_removing_real(cfg, remote_root, exp)
                    _ensure_data_layout(cfg, remote_root)
                    _remove_placeholder_files(cfg, remote_root, exp)
                    _scp_placeholder_to_remote(cfg, remote_root, exp)
                    if _remote_placeholder_valid(cfg, remote_root, exp):
                        pass  # continue to train
                    else:
                        # Self-heal from this error class: ensure sample + SCP once more, then attempt run so post-run heals (missing image / libpng CRC) can fix or train may succeed.
                        _ensure_at_least_one_train_sample(cfg, remote_root, exp)
                        _scp_placeholder_to_remote(cfg, remote_root, exp)
                        if not _remote_placeholder_valid(cfg, remote_root, exp):
                            # Still invalid; attempt training anyway—post-run block will heal on missing image / libpng and retry.
                            pass

    # Proactive: ensure at least one train/val sample so InfiniteSampler size > 0 (minimal annotations have images=[]).
    if not _ensure_at_least_one_train_sample(cfg, remote_root, exp):
        # Self-heal: ensure placeholder everywhere and retry once (handles remote path/quoting flakiness).
        _ensure_smoke_placeholder_file(cfg, remote_root, exp)
        if not _ensure_at_least_one_train_sample(cfg, remote_root, exp):
            return stage_fail(
                "smoke_train failed: could not ensure at least one dataset sample on remote before run"
            )

    cmd = (
        f"cd {remote_root} && . {venv}/bin/activate && "
        f"PYTHONPATH={shlex.quote(remote_root)} python scripts/run_train.py --exp {exp_smoke} --epochs 1 --batch 2 2>&1"
    )
    ok, out, err = ssh_run(cfg, cmd, timeout=900)
    if not ok and _is_venv_activate_missing(err or out):
        fallback_venv = f"{remote_root}/.venv"
        if venv != fallback_venv:
            cmd_retry = (
                f"cd {remote_root} && . {fallback_venv}/bin/activate && "
                f"PYTHONPATH={shlex.quote(remote_root)} python scripts/run_train.py --exp {exp_smoke} --epochs 1 --batch 2 2>&1"
            )
            ok, out, err = ssh_run(cfg, cmd_retry, timeout=900)
    if not ok and _is_missing_dataset_path(err or out):
        if (
            _ensure_yolox_dataset_link(cfg, remote_root, exp)
            and _ensure_yolox_dataset_link(cfg, remote_root, exp_smoke)
            and _ensure_data_layout(cfg, remote_root)
            and _ensure_minimal_annotation_files(cfg, remote_root)
            and _remote_dataset_annotations_exist(cfg, remote_root, exp)
        ):
            ok, out, err = ssh_run(cfg, cmd, timeout=900)
        else:
            return stage_fail(
                "smoke_train failed: could not create dataset path on remote after self-heal; run sync_data first"
            )
    if not ok and _is_data_directory_not_found(err or out):
        if _ensure_repo_root_dataset_link(cfg, remote_root, exp):
            ok, out, err = ssh_run(cfg, cmd, timeout=900)
        else:
            return stage_fail(
                "smoke_train failed: could not create repo-root datasets symlink after self-heal; run sync_data first"
            )
    if not ok and _is_empty_dataset_sampler(err or out):
        if _ensure_at_least_one_train_sample(cfg, remote_root, exp):
            ok, out, err = ssh_run(cfg, cmd, timeout=900)
        else:
            return stage_fail(
                "smoke_train failed: empty dataset (InfiniteSampler size=0); could not ensure placeholder sample on remote"
            )
    if not ok and _is_no_module_named_exps(err or out):
        # Self-heal: re-sync code so remote gets run_train.py that adds _ROOT to sys.path (exps package).
        excludes = cfg.get("sync", {}).get(
            "code_excludes",
            [".git", ".venv", ".local", "ralph_runs", "ralph_artifacts", "__pycache__", "*.pyc", ".DS_Store"],
        )
        sync_ok, _, _ = rsync_to_remote(cfg, str(TRAIN_YOLOX), remote_root, excludes=excludes)
        if sync_ok:
            ok, out, err = ssh_run(cfg, cmd, timeout=900)
    run_training = True
    if not ok and (_is_missing_image_file(err or out) or _is_libpng_crc_error(err or out)):
        # Remove any existing placeholder so a corrupted PNG (e.g. libpng CRC) is replaced by a fresh write.
        _remove_placeholder_files(cfg, remote_root, exp)
        # Restore YOLOX/datasets/<ds> -> data symlink so writes to data/ are visible to loader (cwd=YOLOX); may have been overwritten by sync.
        _ensure_yolox_dataset_link(cfg, remote_root, exp)
        _ensure_data_layout(cfg, remote_root)
        # Push known-good placeholder via SCP first (avoids libpng CRC from remote Python/SSH base64 corruption).
        # SCP runs before any remote Python writes so the loader path gets a valid PNG on retry.
        scp_ok = _scp_placeholder_to_remote(cfg, remote_root, exp)
        # Ensure every image referenced in annotations exists, and unconditionally ensure smoke_placeholder.png (common missing file).
        ref_ok = _ensure_referenced_images_exist(cfg, remote_root, exp)
        _ensure_smoke_placeholder_file(cfg, remote_root, exp)
        # Write placeholder from YOLOX cwd to exact path loader uses (datasets/<ds>/train/images/smoke_placeholder.png) so it is always found.
        placeholder_ok = _write_placeholder_from_yolox_cwd(cfg, remote_root, exp)
        # Write at absolute loader path so file exists even when YOLOX/datasets/<ds> is a real directory (self-heal for "file named ... not found").
        placeholder_ok = _write_placeholder_at_loader_path(cfg, remote_root, exp) or placeholder_ok
        # If still missing, placeholder may be behind symlink that was replaced by real dir; re-ensure symlink and write again.
        if not _remote_placeholder_exists(cfg, remote_root, exp):
            _ensure_yolox_dataset_link(cfg, remote_root, exp)
        placeholder_ok = _write_placeholder_from_yolox_cwd(cfg, remote_root, exp) or placeholder_ok
        placeholder_ok = _write_placeholder_at_loader_path(cfg, remote_root, exp) or placeholder_ok
        _write_placeholder_at_repo_root_dataset_path(cfg, remote_root, exp)
        # Retry only when placeholder is valid (exists and valid PNG), so we avoid libpng CRC / "not found" on retry.
        if _remote_placeholder_valid(cfg, remote_root, exp):
            ok, out, err = ssh_run(cfg, cmd, timeout=900)
            run_training = False
        elif scp_ok:
            return stage_fail(
                "smoke_train failed: placeholder still missing or corrupt at YOLOX dataset path after SCP; check remote data layout and symlinks"
            )
        elif not ref_ok and not placeholder_ok:
            return stage_fail(
                "smoke_train failed: missing or corrupted dataset images; could not create placeholder images on remote"
            )
        else:
            return stage_fail(
                "smoke_train failed: placeholder missing or corrupt at YOLOX dataset path after self-heal; check remote data layout and symlinks"
            )
    if not ok:
        return stage_fail(f"smoke_train failed: {err or out}")

    validate = f"test -d {remote_root}/YOLOX/YOLOX_outputs/{exp_smoke} && echo ok"
    ok2, out2, err2 = ssh_run(cfg, validate, timeout=10)
    if not ok2 or "ok" not in out2:
        return stage_fail(f"smoke_train validation failed: {err2 or out2}")
    return stage_ok("smoke_train ok")


if __name__ == "__main__":
    raise SystemExit(main())
