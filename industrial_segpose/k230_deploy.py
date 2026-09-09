"""Build a local SD-card-ready K230 deployment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .k230_export import export_k230_bundle
from .calibration import PlanarCalibration


def _copy_python_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "selftest.py", "simulation.py", "deploy_main.py",
            "autostart_main.py", "install_autostart.py", "restore_original.py",
        ),
    )


def build_k230_deployment(
    project_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Deployment directory is not empty: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    app_root = staging / "industrial_vision"
    app_root.mkdir(parents=True)
    (app_root / "logs").mkdir()
    (app_root / "captures").mkdir()
    (app_root / "recovery").mkdir()
    (app_root / "calibration").mkdir()

    _copy_python_tree(project / "k230_runtime", app_root / "k230_runtime")
    _copy_python_tree(project / "shared_protocol", app_root / "shared_protocol")
    shutil.copy2(project / "k230_runtime" / "deploy_main.py", app_root / "main.py")
    shutil.copy2(project / "k230_runtime" / "autostart_main.py", app_root / "autostart_main.py")
    shutil.copy2(project / "k230_runtime" / "install_autostart.py", app_root / "install_autostart.py")
    shutil.copy2(project / "k230_runtime" / "restore_original.py", app_root / "restore_original.py")
    shutil.copy2(project / "k230_runtime" / "device_config.example.json", app_root / "device_config.json")
    export_k230_bundle(project / "templates", app_root / "templates", max_edge=384, overwrite=True)
    calibration_source = project / "calibration" / "calibration.json"
    calibration_included = False
    if calibration_source.is_file():
        PlanarCalibration.load(calibration_source)
        shutil.copy2(calibration_source, app_root / "calibration" / "calibration.json")
        calibration_included = True

    # Keep the generated device configuration consistent with the assets that
    # are actually present in this deployment.  Enabling calibration without a
    # calibration file makes the board application fail during startup and the
    # safe launcher then falls back to the original camera program.
    device_config_path = app_root / "device_config.json"
    device_config = json.loads(device_config_path.read_text(encoding="utf-8"))
    device_config.setdefault("calibration", {})["enabled"] = calibration_included
    device_config_path.write_text(
        json.dumps(device_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "format_version": 1,
        "deploy_directory": "/sdcard/industrial_vision",
        "entry_point": "/sdcard/industrial_vision/main.py",
        "runtime_mode": "dry_run_console_only",
        "actuator_enabled": False,
        "template_authoring_enabled": True,
        "autostart_installer": "/sdcard/industrial_vision/install_autostart.py",
        "autostart_restore": "/sdcard/industrial_vision/restore_original.py",
        "safe_mode_recovery": "hold_board_key_or_create_disable_autostart_file",
        "calibration_included": calibration_included,
    }
    (app_root / "deployment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if output.exists():
        shutil.rmtree(output)
    # ``Path.replace`` can fail for directories on Windows even when the
    # destination was just removed.  ``shutil.move`` handles that platform
    # detail while preserving the same atomic-ish staging workflow.
    shutil.move(str(staging), str(output))
    return output / "industrial_vision"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local K230 SD-card deployment directory")
    parser.add_argument("--project", default=".")
    parser.add_argument("--output", default="build/k230_sdcard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    path = build_k230_deployment(args.project, args.output, args.overwrite)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
