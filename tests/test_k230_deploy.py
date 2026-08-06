import json
from pathlib import Path

from industrial_segpose.k230_deploy import build_k230_deployment


def test_build_k230_deployment_contains_safe_entrypoint(tmp_path):
    project = Path(__file__).parents[1]
    app_root = build_k230_deployment(project, tmp_path / "sdcard")

    assert (app_root / "main.py").is_file()
    assert (app_root / "autostart_main.py").is_file()
    assert (app_root / "install_autostart.py").is_file()
    assert (app_root / "restore_original.py").is_file()
    assert (app_root / "k230_runtime" / "detector.py").is_file()
    assert (app_root / "shared_protocol" / "messages.py").is_file()
    assert (app_root / "k230_runtime" / "image_quality.py").is_file()
    assert (app_root / "k230_runtime" / "calibration.py").is_file()
    assert (app_root / "calibration").is_dir()
    assert (app_root / "templates" / "template_library.json").is_file()
    deployment = json.loads((app_root / "deployment.json").read_text(encoding="utf-8"))
    device_config = json.loads((app_root / "device_config.json").read_text(encoding="utf-8"))
    assert deployment["actuator_enabled"] is False
    assert deployment["template_authoring_enabled"] is True
    assert device_config["template_authoring"]["enabled"] is True
    assert deployment["entry_point"] == "/sdcard/industrial_vision/main.py"
    assert deployment["autostart_installer"].endswith("install_autostart.py")
    assert deployment["calibration_included"] is False
    assert device_config["calibration"]["enabled"] is False


def test_build_k230_deployment_requires_overwrite(tmp_path):
    output = tmp_path / "sdcard"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    try:
        build_k230_deployment(Path(__file__).parents[1], output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected FileExistsError")
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
