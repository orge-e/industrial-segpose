from industrial_segpose.ui_tk.app import detection_workflow_status, template_workflow_status


def test_template_workflow_reports_next_missing_step():
    assert template_workflow_status(
        has_image=False, has_roi=False, has_mask=False, has_name=False
    )[0].startswith("第 1 步")
    assert "分割轮廓" in template_workflow_status(
        has_image=True, has_roi=True, has_mask=False, has_name=False
    )[0]
    assert "填写工件类型" in template_workflow_status(
        has_image=True, has_roi=True, has_mask=True, has_name=False
    )[0]
    assert "可以保存" in template_workflow_status(
        has_image=True, has_roi=True, has_mask=True, has_name=True
    )[0]


def test_detection_workflow_prioritizes_templates_then_image_then_run():
    assert "没有可用模板" in detection_workflow_status(
        valid_template_count=0, has_image=True, running=False, has_result=False
    )[0]
    assert "选择检测图像" in detection_workflow_status(
        valid_template_count=2, has_image=False, running=False, has_result=False
    )[0]
    assert "开始检测" in detection_workflow_status(
        valid_template_count=2, has_image=True, running=False, has_result=False
    )[0]
    assert "正在检测" in detection_workflow_status(
        valid_template_count=2, has_image=True, running=True, has_result=False
    )[0]
    assert "检测完成" in detection_workflow_status(
        valid_template_count=2, has_image=True, running=False, has_result=True
    )[0]
