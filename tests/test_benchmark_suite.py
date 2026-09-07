import json

from industrial_segpose.evaluation.suite import run_synthetic_suite


def test_suite_runs_all_profiles_and_algorithms(tmp_path, monkeypatch):
    import industrial_segpose.evaluation.suite as suite

    monkeypatch.setattr(suite, "DEFAULT_PROFILES", ("clean", "touching"))
    result = run_synthetic_suite(tmp_path, images_per_profile=1, random_seed=5)
    assert set(result) == {"threshold", "adaptive_threshold", "color_range", "watershed"}
    assert all(item["image_count"] == 2 for item in result.values())
    report = tmp_path / "reports" / "offline_benchmark_production"
    payload = json.loads((report / "suite_summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["profiles"] == ["clean", "touching"]
    assert (report / "suite_summary.csv").is_file()
    assert (report / "suite_summary.md").is_file()
