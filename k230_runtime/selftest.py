"""Offline smoke test for the board-independent K230 production loop."""

from k230_runtime.app import K230Runtime
from k230_runtime.config import load_config
from k230_runtime.simulation import MemoryTransport, SequenceCamera, SequenceDetector, StepClock
from shared_protocol import decode_message


def run_selftest():
    config = load_config()
    config["heartbeat_interval_ms"] = 1
    camera = SequenceCamera([{"frame": 1}, {"frame": 2}])
    detector = SequenceDetector(
        [
            [
                {
                    "track_id": 7,
                    "template_id": "demo-template",
                    "template_name": "offline-demo",
                    "confidence": 0.93,
                    "image_center": [320.0, 240.0],
                    "pick_point": [322.0, 238.0],
                    "angle_deg": 18.5,
                    "status": "ready",
                }
            ],
            [],
        ]
    )
    transport = MemoryTransport()
    runtime = K230Runtime(config, camera, detector, transport, StepClock(step=20))
    processed = runtime.run(maximum_frames=2)
    messages = [decode_message(payload) for payload in transport.messages]
    pick_targets = [message for message in messages if message["message_type"] == "pick_target"]
    heartbeats = [message for message in messages if message["message_type"] == "heartbeat"]
    if processed != 2 or len(pick_targets) != 1 or not heartbeats:
        raise RuntimeError("K230 offline self-test failed")
    return {
        "processed_frames": processed,
        "pick_targets": len(pick_targets),
        "heartbeats": len(heartbeats),
        "target": pick_targets[0],
    }


def main():
    result = run_selftest()
    print("K230 offline self-test passed")
    print(result)


if __name__ == "__main__":
    main()
