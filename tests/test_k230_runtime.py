from k230_runtime import K230Runtime
from k230_runtime.simulation import MemoryTransport, SequenceCamera, SequenceDetector, StepClock
from shared_protocol import decode_message


def _detection(confidence=0.9, status="ready"):
    return {
        "track_id": 12,
        "template_id": "textile-a",
        "template_name": "textile A",
        "confidence": confidence,
        "image_center": [320.0, 180.0],
        "pick_point": [326.0, 186.0],
        "angle_deg": 42.0,
        "world_point_mm": [150.0, 60.0],
        "arrival_time_ms": 700,
        "status": status,
    }


def test_runtime_emits_target_and_heartbeat():
    camera = SequenceCamera(["frame-1", "frame-2"])
    detector = SequenceDetector([[ _detection() ], []])
    transport = MemoryTransport()
    runtime = K230Runtime(
        {
            "device_id": "test-k230",
            "minimum_confidence": 0.6,
            "emit_ambiguous": False,
            "heartbeat_interval_ms": 40,
        },
        camera,
        detector,
        transport,
        StepClock(start=100, step=40),
    )

    assert runtime.run() == 2
    messages = [decode_message(line) for line in transport.messages]

    assert [item["message_type"] for item in messages] == ["pick_target", "heartbeat", "heartbeat"]
    assert messages[0]["world_point_mm"] == [150.0, 60.0]
    assert messages[0]["pick_point"] == [326.0, 186.0]
    assert runtime.targets == 1


def test_runtime_filters_low_confidence_and_ambiguous_targets():
    camera = SequenceCamera([1, 2])
    detector = SequenceDetector([[ _detection(0.4) ], [ _detection(0.9, "ambiguous") ]])
    transport = MemoryTransport()
    runtime = K230Runtime(
        {"minimum_confidence": 0.6, "emit_ambiguous": False, "heartbeat_interval_ms": 1000},
        camera,
        detector,
        transport,
        StepClock(),
    )

    runtime.run()
    messages = [decode_message(line) for line in transport.messages]

    assert all(item["message_type"] != "pick_target" for item in messages)
    assert runtime.rejected == 2
