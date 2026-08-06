"""Apply a staged K230 runtime update and preserve a one-version backup."""

try:
    import uos as os
except ImportError:
    import os


APP_ROOT = "/sdcard/industrial_vision"
UPDATE_ROOT = APP_ROOT + "/k230_runtime_update"
RUNTIME_ROOT = APP_ROOT + "/k230_runtime"
FILES = ("detector.py", "device_app.py", "error_reporter.py", "main.py", "overlay.py")


def _copy(source, destination):
    with open(source, "rb") as input_stream:
        with open(destination, "wb") as output_stream:
            while True:
                block = input_stream.read(4096)
                if not block:
                    break
                output_stream.write(block)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def apply():
    completed = []
    for name in FILES:
        source = UPDATE_ROOT + "/" + name
        target = RUNTIME_ROOT + "/" + name
        temporary = target + ".new"
        backup = target + ".bak"
        _remove(temporary)
        _copy(source, temporary)
        _remove(backup)
        try:
            os.rename(target, backup)
        except OSError:
            pass
        try:
            os.rename(temporary, target)
        except Exception:
            try:
                os.rename(backup, target)
            except OSError:
                pass
            raise
        completed.append(name)
        print("updated", name)
    with open(APP_ROOT + "/logs/runtime_update.log", "w") as stream:
        stream.write("status: success\n")
        stream.write("files: %s\n" % ",".join(completed))
        stream.write("restart_required: yes\n")
    print("runtime update complete; restart K230")
    return completed


if __name__ == "__main__":
    apply()
