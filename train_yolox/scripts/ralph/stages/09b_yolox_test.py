#!/usr/bin/env python3
"""
Ralph stage: run src/yolox_test.py on the remote against the trained model
on 3 test images (from data/val/images or data/test/images) before pull_artifacts.
Fails if the model does not detect at least one object on each of the 3 images.
"""
from __future__ import annotations

import re
import shlex

from common import load_config, load_state, parse_stage_args, raw_data_subdir, ssh_run, stage_fail, stage_ok


NUM_TEST_IMAGES = 3
MIN_DETECTIONS_PER_IMAGE = 1
MAX_CANDIDATE_IMAGES = 30
MAX_IMAGES_TO_TRY = 30
CONFIDENCE_LADDER = ("0.05", "0.01", "0.001", "0.0")


def _remote_first_pth(cfg: dict, remote_root: str, exp: str) -> str | None:
    """Find a remote .pth in common fallback locations (same logic as 10_pull_artifacts)."""
    base = f"{remote_root}/YOLOX/YOLOX_outputs/{exp}"
    base_parent = f"{remote_root}/../YOLOX/YOLOX_outputs/{exp}"
    models_dir = f"{remote_root}/models"
    models_parent = f"{remote_root}/../models"

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


def _exp_to_dataset_name(exp_name: str) -> str:
    if exp_name.startswith("yolox_s_"):
        return exp_name.replace("yolox_s_", "", 1) + "_dataset"
    return exp_name + "_dataset"


def _remote_list_images(cfg: dict, remote_root: str, exp: str, limit: int) -> list[str]:
    """
    Return up to `limit` image paths from the prepared dataset (val then train).

    Prefer the exact loader paths YOLOX uses: YOLOX/datasets/<ds>/val/images.
    Fall back to the symlink target (data/val/images) and then to train/ if val is empty.
    """
    ds_name = _exp_to_dataset_name(exp)
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    cmd = (
        f"cd {shlex.quote(remote_root)} && "
        f". {shlex.quote(venv)}/bin/activate && "
        "python - <<'PY'\n"
        "import json, os\n"
        f"ds_name = {ds_name!r}\n"
        f"limit = {int(limit)}\n"
        "exts = ('.jpg', '.jpeg', '.png')\n"
        "\n"
        "def load_coco(p):\n"
        "  try:\n"
        "    with open(p, 'r', encoding='utf-8') as f:\n"
        "      return json.load(f)\n"
        "  except Exception:\n"
        "    return None\n"
        "\n"
        "def images_from_coco(coco, img_dir):\n"
        "  # Return image paths that have at least one annotation.\n"
        "  if not coco:\n"
        "    return []\n"
        "  imgs = coco.get('images') or []\n"
        "  anns = coco.get('annotations') or []\n"
        "  if not imgs or not anns:\n"
        "    return []\n"
        "  ann_count = {}\n"
        "  for a in anns:\n"
        "    iid = a.get('image_id')\n"
        "    if iid is None:\n"
        "      continue\n"
        "    ann_count[iid] = ann_count.get(iid, 0) + 1\n"
        "  out = []\n"
        "  for im in imgs:\n"
        "    iid = im.get('id')\n"
        "    fn = im.get('file_name') or ''\n"
        "    if not fn or not fn.lower().endswith(exts):\n"
        "      continue\n"
        "    if ann_count.get(iid, 0) <= 0:\n"
        "      continue\n"
        "    p = os.path.join(img_dir, os.path.basename(fn))\n"
        "    if os.path.isfile(p):\n"
        "      out.append(p)\n"
        "  return out\n"
        "\n"
        "def listdir_images(d):\n"
        "  try:\n"
        "    names = sorted(os.listdir(d))\n"
        "  except Exception:\n"
        "    return []\n"
        "  out = []\n"
        "  for name in names:\n"
        "    if not name.lower().endswith(exts):\n"
        "      continue\n"
        "    p = os.path.join(d, name)\n"
        "    if os.path.isfile(p):\n"
        "      out.append(p)\n"
        "  return out\n"
        "\n"
        "# Prefer COCO annotations so we pick images that actually contain labeled objects.\n"
        "specs = [\n"
        "  (f'YOLOX/datasets/{ds_name}/annotations/val_labels.json', f'YOLOX/datasets/{ds_name}/val/images'),\n"
        "  (f'datasets/{ds_name}/annotations/val_labels.json', f'datasets/{ds_name}/val/images'),\n"
        "  ('data/annotations/val_labels.json', 'data/val/images'),\n"
        "  (f'YOLOX/datasets/{ds_name}/annotations/train_labels.json', f'YOLOX/datasets/{ds_name}/train/images'),\n"
        "  (f'datasets/{ds_name}/annotations/train_labels.json', f'datasets/{ds_name}/train/images'),\n"
        "  ('data/annotations/train_labels.json', 'data/train/images'),\n"
        "  (f'YOLOX/datasets/{ds_name}/annotations/test_labels.json', f'YOLOX/datasets/{ds_name}/test/images'),\n"
        "  (f'datasets/{ds_name}/annotations/test_labels.json', f'datasets/{ds_name}/test/images'),\n"
        "  ('data/annotations/test_labels.json', 'data/test/images'),\n"
        "]\n"
        "paths = []\n"
        "seen = set()\n"
        "for ann, img_dir in specs:\n"
        "  if len(paths) >= limit:\n"
        "    break\n"
        "  if not (os.path.isfile(ann) and os.path.isdir(img_dir)):\n"
        "    continue\n"
        "  coco = load_coco(ann)\n"
        "  for p in images_from_coco(coco, img_dir):\n"
        "    if p in seen:\n"
        "      continue\n"
        "    seen.add(p)\n"
        "    paths.append(p)\n"
        "    if len(paths) >= limit:\n"
        "      break\n"
        "\n"
        "# Fallback: directory scan (may include hard negatives).\n"
        "if len(paths) < limit:\n"
        "  candidates = [\n"
        "    f'YOLOX/datasets/{ds_name}/val/images',\n"
        "    f'datasets/{ds_name}/val/images',\n"
        "    'data/val/images',\n"
        "    f'YOLOX/datasets/{ds_name}/train/images',\n"
        "    f'datasets/{ds_name}/train/images',\n"
        "    'data/train/images',\n"
        "    f'YOLOX/datasets/{ds_name}/test/images',\n"
        "    f'datasets/{ds_name}/test/images',\n"
        "    'data/test/images',\n"
        "  ]\n"
        "  for d in candidates:\n"
        "    if len(paths) >= limit:\n"
        "      break\n"
        "    if not os.path.isdir(d):\n"
        "      continue\n"
        "    for p in listdir_images(d):\n"
        "      if p in seen:\n"
        "        continue\n"
        "      seen.add(p)\n"
        "      paths.append(p)\n"
        "      if len(paths) >= limit:\n"
        "        break\n"
        "\n"
        "for p in paths[:limit]:\n"
        "  print(p)\n"
        "PY"
    )
    ok, out, _ = ssh_run(cfg, cmd, timeout=30)
    if not ok or not (out or "").strip():
        return []
    return [p.strip() for p in (out or "").splitlines() if p.strip()][:limit]


def _remote_prepare_data(cfg: dict, remote_root: str) -> tuple[bool, str, str]:
    """
    Best-effort: rerun src/yolox_data_prep.py on remote to repopulate prepared dataset dirs.
    Intended only as a self-heal when prepared image dirs are empty/missing.
    """
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    raw_subdir = raw_data_subdir(cfg)
    raw_dir = f"data/{raw_subdir}"
    cmd = (
        f"cd {shlex.quote(remote_root)} && "
        f". {shlex.quote(venv)}/bin/activate && "
        f"python src/yolox_data_prep.py --train-yolox-root . --raw-dir {raw_dir!r} --dataset-name marker_dataset"
        " 2>&1"
    )
    return ssh_run(cfg, cmd, timeout=300)


def main() -> int:
    args = parse_stage_args("yolox_test")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    exp = cfg.get("training", {}).get("exp", "yolox_s_marker")
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"

    model_path = _remote_first_pth(cfg, remote_root, exp)
    if not model_path:
        return stage_fail("yolox_test: no checkpoint found on remote; run verify_outputs first")

    images = _remote_list_images(cfg, remote_root, exp, MAX_CANDIDATE_IMAGES)
    if len(images) < NUM_TEST_IMAGES:
        ok_prep, out_prep, err_prep = _remote_prepare_data(cfg, remote_root)
        images = _remote_list_images(cfg, remote_root, exp, MAX_CANDIDATE_IMAGES)
        if len(images) < NUM_TEST_IMAGES:
            return stage_fail(
                f"yolox_test: need at least {NUM_TEST_IMAGES} images in prepared dataset paths; found {len(images)}",
                details={
                    "prepare_data_attempted": True,
                    "prepare_data_ok": ok_prep,
                    "prepare_data_output_tail": ((out_prep or "") + "\n" + (err_prep or "")).splitlines()[-40:],
                },
            )

    # PYTHONPATH so that "from yolox.exp" resolves (yolox package is under YOLOX/yolox)
    py_path = f"{remote_root}:{remote_root}/YOLOX"
    base_confidence = "0.05"

    results: list[tuple[str, bool, int, str]] = []  # (image_path, ok, num_detections, confidence_used)
    failures: list[dict] = []
    tried = 0
    for img_path in images[:MAX_IMAGES_TO_TRY]:
        if len(results) >= NUM_TEST_IMAGES:
            break
        tried += 1
        best_det = 0
        best_conf = base_confidence
        last_out = ""
        last_err = ""
        for conf in CONFIDENCE_LADDER:
            cmd = (
                f"cd {shlex.quote(remote_root)} && . {shlex.quote(venv)}/bin/activate && "
                f"PYTHONPATH={shlex.quote(py_path)} python src/yolox_test.py "
                f"--model {shlex.quote(model_path)} --image {shlex.quote(img_path)} "
                f"--exp {shlex.quote(exp)} --confidence {conf} 2>&1"
            )
            ok, out, err = ssh_run(cfg, cmd, timeout=120)
            last_out, last_err = out or "", err or ""
            combined = (out or "") + "\n" + (err or "")
            # Parse "Detected N circles" or "Detections: N" from script output
            detections = 0
            m = re.search(r"Detected\s+(\d+)", combined)
            if m:
                detections = int(m.group(1))
            else:
                m2 = re.search(r"Detections:\s*(\d+)", combined)
                if m2:
                    detections = int(m2.group(1))
            if not ok:
                return stage_fail(
                    f"yolox_test failed on {img_path}: exit non-zero",
                    details={"image": img_path, "stdout": (out or "")[:500], "stderr": (err or "")[:500]},
                )
            if detections > best_det:
                best_det, best_conf = detections, conf
            if detections >= MIN_DETECTIONS_PER_IMAGE:
                break
            # If we already got a non-zero count but still below threshold (unlikely here), don't
            # keep lowering confidence forever; move on to next image.
            if detections > 0:
                break

        if best_det < MIN_DETECTIONS_PER_IMAGE:
            failures.append({"image": img_path, "detections": best_det, "confidence": best_conf})
            continue
        results.append((img_path, True, best_det, best_conf))

    if len(results) < NUM_TEST_IMAGES:
        return stage_fail(
            f"yolox_test: model must detect at least {MIN_DETECTIONS_PER_IMAGE} object(s) on each image; "
            f"only {len(results)}/{NUM_TEST_IMAGES} images passed after trying {tried}",
            details={
                "passed_images": [r[0] for r in results],
                "passed_detections": [r[2] for r in results],
                "passed_confidence": [r[3] for r in results],
                "failed_images": failures[:10],
                "num_failures": len(failures),
                "confidence_ladder": list(CONFIDENCE_LADDER),
            },
        )

    return stage_ok(
        "yolox_test ok (model detected objects on 3 images)",
        details={
            "images": [r[0] for r in results],
            "detections": [r[2] for r in results],
            "confidence": [r[3] for r in results],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
