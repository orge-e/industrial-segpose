"""模板匹配：检查模板库健康状态并执行有界参数搜索。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..evaluation.benchmark import SyntheticSceneGenerator, SyntheticStressConfig, circular_angle_error
from . import (
    MatchParameters,
    TemplateLibrary,
    TemplateMatcher,
    analyze_template_mask,
    recommend_feature_mode,
)
from .library import LoadedTemplateEntry


@dataclass(frozen=True)
class TemplateAudit:
    template_id: str
    template_name: str
    enabled: bool
    valid: bool
    status: str
    health_score: int
    current_mode: str
    recommended_mode: str | None
    recommendation_confidence: float | None
    recommendation_reason: str | None
    mask_coverage: float | None
    mask_component_count: int | None
    mask_main_component_ratio: float | None
    mask_touches_border: bool | None
    foreground_lightness: float | None
    foreground_chroma: float | None
    foreground_texture: float | None
    issues: tuple[str, ...]
    actions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParameterSearchConfig:
    scene_count: int = 3
    seed: int = 42
    profile: str = "balanced"
    max_trials: int = 6
    center_tolerance_px: float = 60.0
    angle_tolerance_deg: float = 10.0
    score_thresholds: tuple[float, ...] = (0.50, 0.60, 0.70, 0.78)

    def validate(self) -> None:
        if not 1 <= self.scene_count <= 20:
            raise ValueError("Parameter search scene_count must be in [1, 20]")
        if not 1 <= self.max_trials <= 20:
            raise ValueError("Parameter search max_trials must be in [1, 20]")
        if self.profile not in {"clean", "balanced", "harsh"}:
            raise ValueError("Unsupported parameter search profile")
        if self.center_tolerance_px <= 0 or not 0 < self.angle_tolerance_deg <= 180:
            raise ValueError("Invalid parameter search tolerances")
        if not self.score_thresholds or any(not 0 < item <= 1 for item in self.score_thresholds):
            raise ValueError("Invalid score threshold candidates")


@dataclass(frozen=True)
class ParameterTrial:
    feature_mode: str
    score_threshold: float
    angle_step: float
    scale_min: float
    scale_max: float
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_center_error_px: float | None
    mean_angle_error_deg: float | None
    mean_elapsed_ms: float
    objective: float


@dataclass(frozen=True)
class ParameterSearchResult:
    template_id: str
    template_name: str
    scene_count: int
    evaluated_configurations: int
    best_parameters: MatchParameters
    best_trial: ParameterTrial
    trials: tuple[ParameterTrial, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "scene_count": self.scene_count,
            "evaluated_configurations": self.evaluated_configurations,
            "best_parameters": asdict(self.best_parameters),
            "best_trial": asdict(self.best_trial),
            "trials": [asdict(item) for item in self.trials],
            "notes": list(self.notes),
        }


def audit_loaded_template(loaded: LoadedTemplateEntry) -> TemplateAudit:
    entry = loaded.entry
    if not loaded.valid or loaded.model is None:
        issue = loaded.error or "模板文件无法读取"
        return TemplateAudit(
            entry.template_id, entry.name, entry.enabled, False, "error", 0,
            entry.parameters.feature_mode, None, None, None,
            None, None, None, None, None, None, None,
            (issue,), ("修复或重新导入模板文件",),
        )
    quality = analyze_template_mask(loaded.model.mask)
    recommendation = recommend_feature_mode(loaded.model)
    current = entry.parameters.feature_mode
    effective_current = recommendation.mode if current == "auto" else current
    issues: list[str] = []
    actions: list[str] = []
    score = 100
    if not quality.valid:
        issues.append(quality.message)
        actions.append("重新检查并修正模板Mask")
        score -= 35
    elif quality.touches_border:
        issues.append("Mask接触ROI边界，旋转后可能丢失轮廓")
        actions.append("重新紧裁剪并保留少量背景边距")
        score -= 15
    if effective_current != recommendation.mode:
        issues.append(
            f"当前模式与外观推荐不一致：{current} → {recommendation.mode}"
        )
        actions.append(f"建议改用 {recommendation.mode} 并执行压力测试")
        score -= int(round(15 + 20 * recommendation.confidence))
    if entry.parameters.angle_step > 8.0:
        issues.append("角度步长较大，可能降低角度输出精度")
        actions.append("在满足实时性的前提下减小角度步长")
        score -= 8
    if entry.parameters.scale_max / entry.parameters.scale_min > 3.0:
        issues.append("尺度搜索范围过宽，可能增加耗时和误检")
        actions.append("按固定相机工况收窄尺度范围")
        score -= 8
    score = max(0, min(100, score))
    status = "good" if score >= 85 and not issues else "warning" if score >= 45 else "error"
    if not issues:
        actions.append("配置与模板外观一致，可进入批量回放验证")
    return TemplateAudit(
        entry.template_id,
        entry.name,
        entry.enabled,
        True,
        status,
        score,
        current,
        recommendation.mode,
        recommendation.confidence,
        recommendation.reason,
        quality.coverage,
        quality.component_count,
        quality.main_component_ratio,
        quality.touches_border,
        recommendation.foreground_lightness,
        recommendation.foreground_chroma,
        recommendation.foreground_texture,
        tuple(issues),
        tuple(actions),
    )


def audit_template_library(library: TemplateLibrary) -> list[TemplateAudit]:
    return [audit_loaded_template(item) for item in library.load_entries()]


def write_template_audit_report(output: str | Path, audits: list[TemplateAudit]) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "template_count": len(audits),
        "good_count": sum(item.status == "good" for item in audits),
        "warning_count": sum(item.status == "warning" for item in audits),
        "error_count": sum(item.status == "error" for item in audits),
        "templates": [item.to_dict() for item in audits],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _candidate_parameter_sets(
    loaded: LoadedTemplateEntry,
    recommended_mode: str,
    max_trials: int,
) -> list[MatchParameters]:
    base = loaded.entry.parameters
    modes = list(dict.fromkeys((recommended_mode, base.feature_mode, "edges")))
    angle_steps = list(dict.fromkeys((base.angle_step, max(4.0, base.angle_step), max(8.0, base.angle_step))))
    candidates: list[MatchParameters] = []
    for mode in modes:
        candidates.append(replace(base, feature_mode=mode, score_threshold=0.45))
    for step in angle_steps[1:]:
        candidates.append(replace(base, feature_mode=recommended_mode, angle_step=step, score_threshold=0.45))
    span = base.scale_max - base.scale_min
    if span >= 0.20:
        candidates.append(replace(
            base,
            feature_mode=recommended_mode,
            scale_min=base.scale_min + span * 0.10,
            scale_max=base.scale_max - span * 0.10,
            score_threshold=0.45,
        ))
    unique: list[MatchParameters] = []
    for candidate in candidates:
        if candidate not in unique:
            candidate.validate()
            unique.append(candidate)
    return unique[:max_trials]


def _score_matches(matches, truth, threshold: float, config: ParameterSearchConfig) -> tuple[int, int, int, float | None, float | None]:
    filtered = [item for item in matches if item.score >= threshold]
    valid: list[tuple[float, float, object]] = []
    for item in filtered:
        center_error = float(np.hypot(item.center_x - truth.center_x, item.center_y - truth.center_y))
        angle_error = circular_angle_error(item.angle_deg, truth.angle_deg)
        if center_error <= config.center_tolerance_px and angle_error <= config.angle_tolerance_deg:
            valid.append((center_error, angle_error, item))
    if not valid:
        return 0, len(filtered), 1, None, None
    center_error, angle_error, _ = min(valid, key=lambda value: (value[0], value[1]))
    return 1, max(0, len(filtered) - 1), 0, center_error, angle_error


def search_template_parameters(
    library: TemplateLibrary,
    template_id: str,
    config: ParameterSearchConfig | None = None,
) -> ParameterSearchResult:
    settings = config or ParameterSearchConfig()
    settings.validate()
    loaded = next((item for item in library.load_entries() if item.entry.template_id == template_id), None)
    if loaded is None:
        raise KeyError(f"Unknown template ID: {template_id}")
    if not loaded.valid or loaded.model is None:
        raise ValueError(loaded.error or "Template is invalid")
    recommendation = recommend_feature_mode(loaded.model)
    parameter_sets = _candidate_parameter_sets(loaded, recommendation.mode, settings.max_trials)
    # Generate one fixed corpus and reuse it for every trial.  This is critical
    # for a fair comparison and makes the search reproducible.
    diagonal = float(np.hypot(loaded.model.image.shape[1], loaded.model.image.shape[0]))
    min_scale = loaded.entry.parameters.scale_min
    height = max(480, int(np.ceil(diagonal * min_scale * 1.55)))
    width = max(720, int(np.ceil(height * 1.6)))
    generator = SyntheticSceneGenerator(
        library,
        SyntheticStressConfig(
            width=width,
            height=height,
            min_objects=1,
            max_objects=1,
            profile=settings.profile,
        ),
        settings.seed,
        loaded_entries=[loaded],
    )
    corpus = [generator.generate(index) for index in range(settings.scene_count)]
    trials: list[ParameterTrial] = []
    threshold_candidates = tuple(sorted(set(settings.score_thresholds + (loaded.entry.parameters.score_threshold,))))
    for parameters in parameter_sets:
        matcher = TemplateMatcher(loaded.model, parameters)
        raw: list[tuple[list, object, float]] = []
        for image, truth_items in corpus:
            started = perf_counter()
            matches = matcher.match(image)
            raw.append((matches, truth_items[0], (perf_counter() - started) * 1000.0))
        for threshold in threshold_candidates:
            tp = fp = fn = 0
            centers: list[float] = []
            angles: list[float] = []
            for matches, truth, _elapsed in raw:
                item_tp, item_fp, item_fn, center, angle = _score_matches(matches, truth, threshold, settings)
                tp += item_tp
                fp += item_fp
                fn += item_fn
                if center is not None:
                    centers.append(center)
                if angle is not None:
                    angles.append(angle)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            mean_elapsed = float(np.mean([item[2] for item in raw]))
            center_mean = float(np.mean(centers)) if centers else None
            angle_mean = float(np.mean(angles)) if angles else None
            error_quality = 0.0
            if center_mean is not None and angle_mean is not None:
                error_quality = max(0.0, 1.0 - center_mean / settings.center_tolerance_px) + max(
                    0.0, 1.0 - angle_mean / settings.angle_tolerance_deg
                )
            # Keep model selection deterministic. Runtime is reported for the
            # operator but is not mixed into the accuracy objective because
            # desktop scheduling jitter would otherwise change the winner.
            objective = 3.0 * f1 + 0.4 * error_quality
            trials.append(ParameterTrial(
                parameters.feature_mode,
                threshold,
                parameters.angle_step,
                parameters.scale_min,
                parameters.scale_max,
                tp, fp, fn, precision, recall, f1,
                center_mean,
                angle_mean,
                mean_elapsed,
                objective,
            ))
    best = max(
        trials,
        key=lambda item: (item.objective, item.f1, item.recall, item.score_threshold),
    )
    best_parameters = replace(
        loaded.entry.parameters,
        feature_mode=best.feature_mode,
        score_threshold=best.score_threshold,
        angle_step=best.angle_step,
        scale_min=best.scale_min,
        scale_max=best.scale_max,
    )
    notes = (
        "搜索结果仅基于合成扰动，用于初始参数推荐，不替代现场独立测试集。",
        f"共复用{settings.scene_count}个固定场景，评估{len(parameter_sets)}组模式/搜索范围。",
    )
    return ParameterSearchResult(
        loaded.entry.template_id,
        loaded.entry.name,
        settings.scene_count,
        len(parameter_sets),
        best_parameters,
        best,
        tuple(sorted(trials, key=lambda item: item.objective, reverse=True)),
        notes,
    )


def write_parameter_search_report(output: str | Path, result: ParameterSearchResult) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
