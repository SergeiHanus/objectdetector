#!/usr/bin/env python3
from __future__ import annotations

from common import ensure_uv, load_config, load_state, parse_stage_args, ssh_run, stage_fail, stage_ok


def main() -> int:
    args = parse_stage_args("ensure_uv_python")
    cfg = load_config(args.config)
    _state = load_state(args.state)

    remote_root = cfg.get("remote", {}).get("repo_root", "")
    if not remote_root:
        return stage_fail("remote.repo_root not set")

    venv = cfg.get("remote", {}).get("venv_path") or f"{remote_root}/.venv"
    python_version = cfg.get("remote", {}).get("python_version") or "3.12"

    try:
        path_export = ensure_uv(cfg, remote_root)
    except Exception as exc:
        return stage_fail(f"ensure_uv_python: uv install failed: {exc}")

    # Ensure uv-managed Python and venv exist.
    uv_venv_cmd = (
        f"cd {remote_root} && {path_export} && "
        f"uv python install {python_version} && "
        f"uv venv {venv} --python {python_version}"
    )
    ok1, _out1, err1 = ssh_run(cfg, uv_venv_cmd, timeout=300)
    if not ok1:
        return stage_fail(f"ensure_uv_python: uv venv failed: {err1 or _out1 or 'unknown'}")

    # Ensure pip is importable in the venv (some venvs may be created without pip).
    pip_check = f"test -x {venv}/bin/python && {venv}/bin/python -c 'import pip; print(1)'"
    ok2, out2, err2 = ssh_run(cfg, pip_check, timeout=20)
    if not ok2 or "1" not in (out2 or ""):
        pip_bootstrap = (
            f"cd {remote_root} && {path_export} && "
            f"uv pip install -p {venv}/bin/python -q pip"
        )
        ok3, _out3, err3 = ssh_run(cfg, pip_bootstrap, timeout=120)
        if not ok3:
            return stage_fail(f"ensure_uv_python: pip bootstrap failed: {err3 or _out3 or 'unknown'}")

        ok4, out4, err4 = ssh_run(cfg, pip_check, timeout=20)
        if not ok4 or "1" not in (out4 or ""):
            return stage_fail(f"ensure_uv_python: pip still not importable: {err4 or out4 or 'unknown'}")

    return stage_ok("ensure_uv_python ok", {"python": f"uv-managed {python_version}", "venv": venv})


if __name__ == "__main__":
    raise SystemExit(main())

