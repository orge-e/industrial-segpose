"""Safe root ``/sdcard/main.py`` launcher installed after hardware validation."""

import sys

APP_ROOT = "/sdcard/industrial_vision"
RECOVERY_MAIN = APP_ROOT + "/recovery/original_main.py"
DISABLE_FLAG = APP_ROOT + "/disable_autostart"
ERROR_LOG = APP_ROOT + "/logs/startup_error.log"


def _exists(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _safe_mode_requested():
    if _exists(DISABLE_FLAG):
        return True
    try:
        from ybUtils.YbKey import YbKey
        return YbKey().is_pressed() == 1
    except Exception:
        return False


def _run_file(path):
    with open(path, "r") as stream:
        source = stream.read()
    namespace = {"__name__": "__main__", "__file__": path}
    exec(source, namespace, namespace)


def _record_error(error):
    try:
        with open(ERROR_LOG, "a") as stream:
            stream.write("startup error: %s\n" % error)
    except Exception:
        pass


def main():
    if _safe_mode_requested():
        print("industrial vision safe mode")
        if _exists(RECOVERY_MAIN):
            _run_file(RECOVERY_MAIN)
        return
    if APP_ROOT not in sys.path:
        sys.path.insert(0, APP_ROOT)
    try:
        from k230_runtime.main import main as run_application
        run_application()
    except BaseException as error:
        _record_error(error)
        print("industrial vision startup failed", error)
        if _exists(RECOVERY_MAIN):
            _run_file(RECOVERY_MAIN)
        else:
            raise


if __name__ == "__main__":
    main()
