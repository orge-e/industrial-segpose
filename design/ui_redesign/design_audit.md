# Phase 1 — Existing UI Design Audit

This audit records responsibilities and state from the current Tkinter UI. It
does not preserve the old visual arrangement.

## 建立模板

### Workflow

1. Select a local or industrial-camera reference image.
2. Define a single-object ROI.
3. Create, inspect and optionally repair the segmentation mask.
4. Set the template name and independent matching parameters.
5. Save to the persistent library or continue to detection.

### Required state

- Source image and source path.
- ROI coordinates and dimensions.
- Mask, coverage and mask-quality state.
- Template name.
- Threshold, rotation range, scale range and feature mode.
- Save readiness.

### Prototype response

The workflow remains visible as a five-step rail, while the viewer becomes the
dominant surface. Import and save are page actions. Detailed properties and
save actions are placed in the inspector. Original/Mask/Overlay belong to the
viewer instead of a separate control card.

## 实时视觉

### Production data

- Current-frame object count and classification breakdown.
- Template/type, center X/Y, rotation angle, pick point and confidence.
- Frame rate, latency and frame-quality status.
- Camera, video-source and detector state.
- Optional auxiliary cumulative count.

### Configuration responsibilities

- Source/camera index or video path.
- Network industrial-camera source.
- Resolution, FPS, exposure, FOURCC and Windows capture backend.
- Detection maximum dimension and algorithm.
- Quality gate, detection ROI and NMS.
- Counting line, position and direction.

### Prototype response

Production data stays on the main page. Device, detection and counting
configuration moves to a right-side drawer opened from **设备设置**. The clean
state gives the camera view most of the workspace; the drawer is temporary and
does not define the normal production layout.

## Boundaries retained

- No camera lifecycle is invoked.
- No OpenCV detector is invoked.
- No ROI or mask data is read or written.
- No template library is loaded.
- No external device connection or package build occurs.
- No existing application entry point is changed.
