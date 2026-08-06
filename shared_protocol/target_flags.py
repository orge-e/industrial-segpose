"""Stable target-quality bit definitions shared by PC, K230 and STM32."""


FLAG_TOUCHES_BORDER = 1 << 0
FLAG_SUSPECTED_OVERLAP = 1 << 1
FLAG_PICK_AREA_INSUFFICIENT = 1 << 2
FLAG_LOW_CONFIDENCE = 1 << 3
FLAG_UNREACHABLE = 1 << 4
FLAG_DUPLICATE = 1 << 5
FLAG_SIZE_ABNORMAL = 1 << 6
FLAG_IMAGE_QUALITY = 1 << 7
FLAG_TEMPLATE_CONFLICT = 1 << 8

FLAG_NAMES = {
    FLAG_TOUCHES_BORDER: "touches_border",
    FLAG_SUSPECTED_OVERLAP: "suspected_overlap",
    FLAG_PICK_AREA_INSUFFICIENT: "pick_area_insufficient",
    FLAG_LOW_CONFIDENCE: "low_confidence",
    FLAG_UNREACHABLE: "unreachable",
    FLAG_DUPLICATE: "duplicate",
    FLAG_SIZE_ABNORMAL: "size_abnormal",
    FLAG_IMAGE_QUALITY: "image_quality",
    FLAG_TEMPLATE_CONFLICT: "template_conflict",
}

AUTO_PICK_BLOCKING_FLAGS = (
    FLAG_TOUCHES_BORDER
    | FLAG_SUSPECTED_OVERLAP
    | FLAG_PICK_AREA_INSUFFICIENT
    | FLAG_LOW_CONFIDENCE
    | FLAG_UNREACHABLE
    | FLAG_DUPLICATE
    | FLAG_SIZE_ABNORMAL
    | FLAG_IMAGE_QUALITY
    | FLAG_TEMPLATE_CONFLICT
)


def flag_names(value):
    """Return deterministic human-readable names for an integer bit mask."""
    integer = int(value)
    return [name for bit, name in sorted(FLAG_NAMES.items()) if integer & bit]


def auto_pick_allowed(value):
    return (int(value) & AUTO_PICK_BLOCKING_FLAGS) == 0
