# YOLOX Training Contract (Canonical Runtime Root)

**Canonical runtime root**: `train_yolox/`. All training and orchestration commands must be executed with this directory as the current working directory.

## Required layout (under `train_yolox/`)

| Path | Purpose |
|------|---------|
| `YOLOX/` | Cloned YOLOX repo (Megvii-BaseDetection/YOLOX). Contains `tools/train.py`, `datasets/`. |
| `exps/` | Experiment configs (e.g. `yolox_s_marker.py`, `yolox_s_base.py`). |
| `data/` | Local raw or prepared data sources. |
| `YOLOX/datasets/<name>_dataset/` | Prepared dataset for an experiment: `train/images`, `val/images`, `test/images`, `annotations/*.json`. |
| `.venv/` or venv | Optional; Python virtualenv for YOLOX (use .venv on macOS to avoid clashing with YOLOX). |
| `models/` | Output directory for copied checkpoints and ONNX (created if missing). |
| `scripts/` | Canonical entry points: `run_train.py`, `ralph_loop.py`, etc. |

Paths in experiment files (e.g. `data_dir = "datasets/marker_dataset"`) are relative to `YOLOX/` when training runs inside `YOLOX/`.

## Single train command contract

- **Entry point**: `python scripts/run_train.py` (from `train_yolox/`).
- **Inputs**:
  - `--exp`: Experiment name (e.g. `yolox_s_marker`). Determines `exps/<exp>.py` and dataset name.
  - `--epochs`, `--batch`, `--workers`: Training hyperparameters.
  - Dataset path is implied by the experiment (e.g. `YOLOX/datasets/marker_dataset` for `yolox_s_marker`).
- **Outputs**:
  - Checkpoints and logs: `YOLOX/YOLOX_outputs/<exp>/`.
  - Copied artifacts: `models/` (best/latest checkpoint, ONNX if exported).
- **Success**: Training process exits 0 and `YOLOX/YOLOX_outputs/<exp>/` contains checkpoint files.

## Verification (preflight)

A valid environment satisfies:

1. `YOLOX/` exists and contains `tools/train.py`.
2. `exps/<exp>.py` exists for the chosen experiment.
3. `YOLOX/datasets/<dataset_name>/train/images` and `.../val/images` exist with at least one image each.
4. `YOLOX/datasets/<dataset_name>/annotations/train_labels.json` and `val_labels.json` exist.

Run layout check: `python scripts/ensure_layout.py --exp yolox_s_marker` (from `train_yolox/`).
