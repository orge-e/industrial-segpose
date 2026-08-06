"""Small JSON-lines protocol with no CPython-only dependencies.

The module intentionally avoids dataclasses and third-party packages so the
same file can be copied to a CanMV MicroPython filesystem.
"""

try:
    import ujson as json
except ImportError:  # CPython
    import json


PROTOCOL_VERSION = 1
_STATUSES = ("ready", "ambiguous", "rejected")
_ACK_STATUSES = ("accepted", "picked", "failed", "expired")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % name)
    return float(value)


def make_pick_target(
    track_id,
    template_id,
    template_name,
    confidence,
    image_center,
    pick_point,
    angle_deg,
    timestamp_ms,
    world_point=None,
    arrival_time_ms=None,
    status="ready",
):
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "pick_target",
        "track_id": int(track_id),
        "template_id": str(template_id),
        "template_name": str(template_name),
        "confidence": float(confidence),
        "image_center": [float(image_center[0]), float(image_center[1])],
        "pick_point": [float(pick_point[0]), float(pick_point[1])],
        "angle_deg": float(angle_deg),
        "timestamp_ms": int(timestamp_ms),
        "status": status,
    }
    if world_point is not None:
        message["world_point_mm"] = [float(world_point[0]), float(world_point[1])]
    if arrival_time_ms is not None:
        message["arrival_time_ms"] = int(arrival_time_ms)
    validate_message(message)
    return message


def make_pick_ack(track_id, status, timestamp_ms, vacuum_ok=None, reason=None):
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "pick_ack",
        "track_id": int(track_id),
        "status": str(status),
        "timestamp_ms": int(timestamp_ms),
    }
    if vacuum_ok is not None:
        message["vacuum_ok"] = bool(vacuum_ok)
    if reason:
        message["reason"] = str(reason)
    validate_message(message)
    return message


def make_heartbeat(device_id, timestamp_ms, state="running", counters=None):
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "heartbeat",
        "device_id": str(device_id),
        "timestamp_ms": int(timestamp_ms),
        "state": str(state),
        "counters": dict(counters or {}),
    }
    validate_message(message)
    return message


def validate_message(message):
    if not isinstance(message, dict):
        raise ValueError("message must be an object")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    message_type = message.get("message_type")
    if message_type == "pick_target":
        if int(message.get("track_id", -1)) < 0:
            raise ValueError("track_id must be non-negative")
        if not message.get("template_id") or not message.get("template_name"):
            raise ValueError("template identity is required")
        confidence = _number(message.get("confidence"), "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        for key in ("image_center", "pick_point"):
            point = message.get(key)
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("%s must contain two coordinates" % key)
            _number(point[0], key)
            _number(point[1], key)
        _number(message.get("angle_deg"), "angle_deg")
        if message.get("status") not in _STATUSES:
            raise ValueError("invalid target status")
    elif message_type == "pick_ack":
        if int(message.get("track_id", -1)) < 0:
            raise ValueError("track_id must be non-negative")
        if message.get("status") not in _ACK_STATUSES:
            raise ValueError("invalid acknowledgement status")
    elif message_type == "heartbeat":
        if not message.get("device_id"):
            raise ValueError("device_id is required")
    else:
        raise ValueError("unsupported message_type")
    if int(message.get("timestamp_ms", -1)) < 0:
        raise ValueError("timestamp_ms must be non-negative")
    return message


def encode_message(message):
    validate_message(message)
    return json.dumps(message) + "\n"


def decode_message(line):
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    message = json.loads(line.strip())
    return validate_message(message)
