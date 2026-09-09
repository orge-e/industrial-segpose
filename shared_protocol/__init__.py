"""Wire protocol shared by the workstation, K230 and controller simulators."""

from .messages import (
    PROTOCOL_VERSION,
    decode_message,
    encode_message,
    make_heartbeat,
    make_pick_ack,
    make_pick_target,
    validate_message,
)
from .target_flags import (
    AUTO_PICK_BLOCKING_FLAGS,
    FLAG_DUPLICATE,
    FLAG_IMAGE_QUALITY,
    FLAG_LOW_CONFIDENCE,
    FLAG_PICK_AREA_INSUFFICIENT,
    FLAG_SIZE_ABNORMAL,
    FLAG_SUSPECTED_OVERLAP,
    FLAG_TEMPLATE_CONFLICT,
    FLAG_TOUCHES_BORDER,
    FLAG_UNREACHABLE,
    auto_pick_allowed,
    flag_names,
)

__all__ = [
    "PROTOCOL_VERSION",
    "decode_message",
    "encode_message",
    "make_heartbeat",
    "make_pick_ack",
    "make_pick_target",
    "validate_message",
    "AUTO_PICK_BLOCKING_FLAGS",
    "FLAG_DUPLICATE",
    "FLAG_IMAGE_QUALITY",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_PICK_AREA_INSUFFICIENT",
    "FLAG_SIZE_ABNORMAL",
    "FLAG_SUSPECTED_OVERLAP",
    "FLAG_TEMPLATE_CONFLICT",
    "FLAG_TOUCHES_BORDER",
    "FLAG_UNREACHABLE",
    "auto_pick_allowed",
    "flag_names",
]
