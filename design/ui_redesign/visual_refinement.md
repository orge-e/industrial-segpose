# FlexPose Vision Prototype — Visual Refinement Baseline

Status: **Design Freeze Candidate**

Review size: **1440 × 900**

Scope: isolated PySide6 mockup only

## Final refinement changes

1. Unified product-facing section names in Chinese while retaining ROI, Mask,
   FPS and OpenCV as established technical terms.
2. Reduced the header from 64 px to 54 px and removed the persistent slogan.
3. Reduced the workflow rail from 188 px to 158 px and replaced filled step
   rows with a vertical state sequence: completed, active and pending.
4. Rebuilt the template inspector as five context pages. Workflow selection now
   switches the inspector between source, ROI, Mask, parameters and deployment.
5. Preserved current-object and cumulative-count values as primary metrics;
   FPS and processing latency now use a smaller secondary metric style.
6. Reduced blue selection surfaces. Navigation keeps a thin indicator and a
   very weak background; table selection is now neutral graphite-blue.
7. Replaced fluorescent mock parts with gray-white flexible material on a dark
   industrial surface. Detection boxes remain restrained blue and center marks
   use cyan.
8. Tightened table rows, viewer toolbar rhythm, inspector spacing, button/input
   heights and status-bar language.

## Frozen structure

- Global navigation and compact header.
- Template workflow + dominant viewer + contextual inspector.
- Live camera viewer + production panel + current-frame results table.
- Temporary settings drawer for device, detector and counting configuration.
- Bottom device state bar.

## Explicit non-goals

No OpenCV, camera, template library, calibration or existing Tkinter code
is connected or modified by this prototype.
