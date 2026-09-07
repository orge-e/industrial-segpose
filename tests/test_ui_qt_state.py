from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.template_matching.matcher import MatchParameters
from industrial_segpose.ui_qt.models import TemplatePageState, TemplateStep
from industrial_segpose.ui_qt.services import TemplateService


def _scene() -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    image = np.full((180, 240, 3), (42, 42, 42), np.uint8)
    roi = (45, 35, 150, 110)
    mask = np.zeros((roi[3], roi[2]), np.uint8)
    cv2.ellipse(mask, (75, 55), (55, 34), 0, 0, 360, 255, -1)
    crop = image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    crop[mask > 0] = (160, 175, 205)
    return image, roi, mask


def test_template_page_state_enables_save_only_when_complete() -> None:
    image, roi, mask = _scene()
    state = TemplatePageState(name="工件_A")
    assert not state.can_save
    state = state.evolve(image=image, source_path=Path("sample.png"), roi_xywh=roi, mask=mask)
    assert state.can_save
    assert state.completed_steps() >= {TemplateStep.SOURCE, TemplateStep.ROI, TemplateStep.MASK}
    assert state.image_size_text == "240 × 180"


def test_template_page_state_disables_actions_while_busy() -> None:
    image, roi, mask = _scene()
    ready = TemplatePageState(name="工件_A", image=image, roi_xywh=roi, mask=mask)
    assert ready.can_save
    busy = ready.evolve(busy=True, busy_message="处理中")
    assert not busy.can_save
    assert not busy.can_segment


def test_template_service_writes_existing_library_format(tmp_path: Path) -> None:
    image, roi, mask = _scene()
    service = TemplateService(tmp_path)
    entry = service.save_template(
        name="工件_A",
        image=image,
        source_path=tmp_path / "source.png",
        roi_xywh=roi,
        mask=mask,
        parameters=MatchParameters(),
    )
    assert service.template_count() == 1
    assert entry.name == "工件_A"
    assert (tmp_path / "templates" / "template_library.json").is_file()
    loaded = service.library.load_entries()[0]
    assert loaded.valid
    assert loaded.model.name == "工件_A"


def test_template_service_segments_roi_without_changing_domain_algorithm(tmp_path: Path) -> None:
    image, roi, _mask = _scene()
    output = TemplateService(tmp_path).segment_roi(image, roi, "border_color")
    assert output.mask.shape == (roi[3], roi[2])
    assert output.method == "border_color"
    assert output.quality.coverage > 0.02
