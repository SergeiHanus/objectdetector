#!/usr/bin/env python3
from __future__ import annotations

from common import ensure_uv, load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok

# Pin for reproducible remote clone (match src/yolox_setup.py)
YOLOX_REVISION = "0.3.0"


def _ensure_cmake(cfg: dict, venv_path: str | None = None, path_export: str | None = None) -> bool:
    """Ensure cmake is available on remote (needed to build onnxsim, a YOLOX dependency). Returns True if cmake is available.
    If venv_path and path_export are set, falls back to pip-installing cmake into the venv when system install fails."""
    # Include Homebrew paths so we find cmake after install on macOS (non-interactive SSH often has minimal PATH)
    path_prefix = "PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\""
    check = f"{path_prefix} command -v cmake >/dev/null 2>&1 && cmake --version | head -1"
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and out:
        return True
    # Try to install: macOS (Homebrew) or Linux (apt). On Darwin try login shell first so brew is on PATH.
    install_cmd = (
        "case \"$(uname -s)\" in "
        "Darwin) "
        "PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\" CI=1 "
        "(bash -lc 'brew install cmake' 2>/dev/null) || "
        "((/opt/homebrew/bin/brew install cmake 2>/dev/null) || (/usr/local/bin/brew install cmake 2>/dev/null)) ;; "
        "Linux) (command -v sudo >/dev/null && sudo apt-get update && sudo apt-get install -y cmake) || (apt-get update && apt-get install -y cmake) ;; "
        "*) exit 1 ;; "
        "esac"
    )
    ssh_run(cfg, install_cmd, timeout=120)
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and out:
        return True
    # Retry once: Darwin often needs brew update first; Linux may have transient failures
    retry_cmd = (
        "case \"$(uname -s)\" in "
        "Darwin) "
        "PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\" CI=1 "
        "((bash -lc 'brew update && brew install cmake' 2>/dev/null) || "
        "((/opt/homebrew/bin/brew update 2>/dev/null && /opt/homebrew/bin/brew install cmake) || "
        "(/usr/local/bin/brew update 2>/dev/null && /usr/local/bin/brew install cmake))) ;; "
        "Linux) (command -v sudo >/dev/null && sudo apt-get update && sudo apt-get install -y cmake) || (apt-get update && apt-get install -y cmake) ;; "
        "*) exit 1 ;; "
        "esac"
    )
    ssh_run(cfg, retry_cmd, timeout=180)
    ok, out, _ = ssh_run(cfg, check, timeout=10)
    if ok and out:
        return True
    # Fallback: install cmake via pip into venv (works when brew/apt unavailable or failing)
    if venv_path and path_export:
        remote_root = cfg.get("remote", {}).get("repo_root", "")
        pip_cmake = (
            f"cd {remote_root} && {path_export} && "
            f"uv pip install -p {venv_path}/bin/python -q cmake"
        )
        ok_pip, _, _ = ssh_run(cfg, pip_cmake, timeout=120)
        if ok_pip:
            check_venv = f"test -x {venv_path}/bin/cmake && {venv_path}/bin/cmake --version | head -1"
            ok_venv, out_venv, _ = ssh_run(cfg, check_venv, timeout=10)
            if ok_venv and out_venv:
                return True
    return False


def main() -> int:
    args = parse_stage_args("ensure_env")
    cfg = load_config(args.config)
    state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    python_version = cfg.get("remote", {}).get("python_version") or "3.12"

    path_export = ensure_uv(cfg, remote_root)

    check_cmd = (
        f"cd {remote_root} && "
        f"(test -f {venv}/bin/activate && . {venv}/bin/activate && "
        "python -c 'import torch; import yolox' 2>/dev/null) && echo ok || echo need_setup"
    )
    ok, out, err = ssh_run(cfg, check_cmd, timeout=30)
    if ok and out and "ok" in out:
        return stage_ok("ensure_env ok", {"python": f"uv-managed {python_version}"})
    if not ok:
        return stage_fail(f"ensure_env check failed: {err}")

    # Create venv with uv-managed Python (no system Python)
    uv_venv_cmd = (
        f"cd {remote_root} && {path_export} && "
        f"uv python install {python_version} && "
        f"uv venv {venv} --python {python_version} --clear"
    )
    ok1, _out1, err1 = ssh_run(cfg, uv_venv_cmd, timeout=300)
    if not ok1:
        return stage_fail(f"ensure_env uv venv failed: {err1 or _out1 or 'unknown'}")

    # Ensure pip is importable (avoid pip-less venvs that break downstream tooling).
    pip_check = f"test -x {venv}/bin/python && {venv}/bin/python -c 'import pip; print(1)'"
    ok_pip_check, out_pip_check, _err_pip_check = ssh_run(cfg, pip_check, timeout=20)
    if not ok_pip_check or "1" not in (out_pip_check or ""):
        pip_bootstrap = f"cd {remote_root} && {path_export} && uv pip install -p {venv}/bin/python -q pip"
        ok_boot, _out_boot, err_boot = ssh_run(cfg, pip_bootstrap, timeout=120)
        if not ok_boot:
            return stage_fail(f"ensure_env pip bootstrap failed: {err_boot or _out_boot or 'unknown'}")

    pip_cmd = (
        f"cd {remote_root} && {path_export} && "
        f"uv pip install -p {venv}/bin/python -q torch torchvision pyyaml"
    )
    ok_pip, _out_pip, err_pip = ssh_run(cfg, pip_cmd, timeout=600)
    if not ok_pip:
        return stage_fail(f"ensure_env pip install failed: {err_pip or _out_pip or 'unknown'}")

    yolox_check = (
        f"cd {remote_root} && test -d YOLOX && test -f YOLOX/tools/train.py && echo ok"
    )
    ok2, out2, err2 = ssh_run(cfg, yolox_check, timeout=10)
    if not ok2 or "ok" not in (out2 or ""):
        # Self-heal: clone YOLOX on remote if missing or incomplete
        clone_cmd = (
            f"cd {remote_root} && rm -rf YOLOX && "
            f'git clone --depth 1 --branch "{YOLOX_REVISION}" '
            "https://github.com/Megvii-BaseDetection/YOLOX.git"
        )
        ok_clone, _out_c, err_c = ssh_run(cfg, clone_cmd, timeout=120)
        if not ok_clone:
            # Fallback: full clone then checkout (e.g. shallow clone may not have tag)
            fallback = (
                f"cd {remote_root} && rm -rf YOLOX && git clone https://github.com/Megvii-BaseDetection/YOLOX.git && "
                f'cd YOLOX && git checkout "{YOLOX_REVISION}"'
            )
            ok_clone, _out_c, err_c = ssh_run(cfg, fallback, timeout=180)
        if not ok_clone:
            return stage_fail(
                "ensure_env failed: YOLOX/ not found or missing tools/train.py; "
                "clone on remote failed: " + (err_c or "unknown")
            )
        ok2, out2, err2 = ssh_run(cfg, yolox_check, timeout=10)
        if not ok2 or "ok" not in (out2 or ""):
            return stage_fail(
                "ensure_env failed: YOLOX/ not found or missing tools/train.py after clone; "
                "run sync_code first (do not exclude YOLOX from code_excludes)"
            )

    # onnxsim (YOLOX dependency) needs cmake to build; ensure it is available (system or venv fallback)
    if not _ensure_cmake(cfg, venv_path=venv, path_export=path_export):
        if not _ensure_cmake(cfg, venv_path=venv, path_export=path_export):
            return stage_fail(
                "ensure_env: cmake required to build YOLOX dependency onnxsim; "
                "install cmake on remote (e.g. brew install cmake or apt-get install cmake)"
            )

    # YOLOX setup.py requires torch at build time (get_ext_modules); use --no-build-isolation
    # so the editable build runs in the venv where torch is already installed.
    # Prepend venv/bin to PATH so build finds cmake when installed via pip fallback.
    install_yolox_cmd = (
        f"cd {remote_root} && export PATH=\"{venv}/bin:$PATH\" && {path_export} && "
        f"uv pip install -p {venv}/bin/python -q --no-build-isolation -e YOLOX"
    )
    ok3, _out3, err3 = ssh_run(cfg, install_yolox_cmd, timeout=600)
    if not ok3:
        # Self-heal: if failure was due to missing cmake, ensure cmake and retry once
        err_lower = (err3 or "").lower()
        if "cmake" in err_lower and ("could not find" in err_lower or "executable" in err_lower):
            _ensure_cmake(cfg, venv_path=venv, path_export=path_export)
            ok3, _out3, err3 = ssh_run(cfg, install_yolox_cmd, timeout=600)
        # Self-heal: onnxsim (YOLOX dep) build can fail with "Compatibility with CMake < 3.5 has been removed";
        # onnxsim setup.py respects CMAKE_ARGS; pass policy flag so bundled ONNX CMake config succeeds.
        out_lower = (_out3 or "").lower()
        combined = f"{err_lower} {out_lower}"
        if not ok3 and "onnxsim" in combined and "failed to build" in combined:
            if (
                "compatibility with cmake" in combined
                or "cmake_policy_version_minimum" in combined
                or "cmake_minimum_required" in combined
            ):
                install_yolox_cmake_cmd = (
                    f"cd {remote_root} && export PATH=\"{venv}/bin:$PATH\" && {path_export} && "
                    "export CMAKE_ARGS=\"-DCMAKE_POLICY_VERSION_MINIMUM=3.5\" && "
                    f"uv pip install -p {venv}/bin/python -q --no-build-isolation -e YOLOX"
                )
                ok3, _out3, err3 = ssh_run(cfg, install_yolox_cmake_cmd, timeout=600)
        if not ok3:
            return stage_fail(f"ensure_env pip install -e YOLOX failed: {err3}")

    validate = f"cd {remote_root} && {venv}/bin/python -c 'import torch; import yolox; print(1)'"
    ok4, out4, err4 = ssh_run(cfg, validate, timeout=20)
    if not ok4 or "1" not in (out4 or ""):
        return stage_fail(f"ensure_env validation failed: {err4 or out4}")

    # Ensure ONNX runtime deps are installed in the venv so onnxsim doesn't try to pip --user install at import time.
    onnx_cmd = (
        f"cd {remote_root} && {path_export} && "
        f"uv pip install -p {venv}/bin/python -q onnx onnxruntime onnxsim"
    )
    ok_onnx, _out_onnx, err_onnx = ssh_run(cfg, onnx_cmd, timeout=600)
    if not ok_onnx:
        return stage_fail(f"ensure_env ONNX deps install failed: {err_onnx or _out_onnx or 'unknown'}")

    return stage_ok("ensure_env ok", {"python": f"uv-managed {python_version}"})


if __name__ == "__main__":
    raise SystemExit(main())
