"""One-time CanMV script that installs the safe auto-start launcher."""

APP_ROOT = "/sdcard/industrial_vision"
ROOT_MAIN = "/sdcard/main.py"
LAUNCHER = APP_ROOT + "/autostart_main.py"
RECOVERY_DIR = APP_ROOT + "/recovery"
RECOVERY_MAIN = RECOVERY_DIR + "/original_main.py"


def _copy(source, destination):
    with open(source, "rb") as input_stream:
        with open(destination, "wb") as output_stream:
            while True:
                chunk = input_stream.read(4096)
                if not chunk:
                    break
                output_stream.write(chunk)


def install():
    import os

    try:
        os.stat(RECOVERY_DIR)
    except OSError:
        os.mkdir(RECOVERY_DIR)
    try:
        os.stat(RECOVERY_MAIN)
    except OSError:
        _copy(ROOT_MAIN, RECOVERY_MAIN)
        print("original main.py backed up", RECOVERY_MAIN)
    _copy(LAUNCHER, ROOT_MAIN)
    print("industrial vision auto-start installed")
    print("hold the board key during boot to run the recovery program")


if __name__ == "__main__":
    install()
