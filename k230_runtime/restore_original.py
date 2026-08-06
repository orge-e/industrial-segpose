"""Restore the root startup program backed up by ``install_autostart.py``."""

ROOT_MAIN = "/sdcard/main.py"
RECOVERY_MAIN = "/sdcard/industrial_vision/recovery/original_main.py"


def restore():
    with open(RECOVERY_MAIN, "rb") as input_stream:
        with open(ROOT_MAIN, "wb") as output_stream:
            while True:
                chunk = input_stream.read(4096)
                if not chunk:
                    break
                output_stream.write(chunk)
    print("original /sdcard/main.py restored")


if __name__ == "__main__":
    restore()
