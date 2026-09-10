import pytest

from shared_protocol import (
    decode_message,
    encode_message,
    make_heartbeat,
    make_pick_ack,
    make_pick_target,
)


def test_pick_target_json_lines_round_trip():
    message = make_pick_target(
        7,
        "template-a",
        "粉色纺织件",
        0.91,
        (120.5, 88.25),
        (124.0, 91.0),
        36.0,
        123456,
        world_point=(184.2, 72.1),
        arrival_time_ms=820,
        quality_flags=4,
        candidate_pick_points=[[130.0, 95.0]],
        safe_radius_px=11.5,
        auto_pick_allowed=False,
        scan_id=3,
        view_id=2,
    )

    encoded = encode_message(message)

    assert encoded.endswith("\n")
    assert decode_message(encoded) == message
    assert message["quality_flags"] == 4
    assert message["scan_id"] == 3


def test_protocol_rejects_unsafe_values():
    with pytest.raises(ValueError, match="confidence"):
        make_pick_target(1, "a", "A", 1.2, (1, 2), (1, 2), 0, 1)
    with pytest.raises(ValueError, match="acknowledgement"):
        make_pick_ack(1, "unknown", 100)


def test_heartbeat_contains_counters():
    message = make_heartbeat("vision-station-01", 1000, counters={"targets": 4})
    assert decode_message(encode_message(message))["counters"]["targets"] == 4
