"""PySide6 正式界面：创建应用对象并启动主窗口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import build_stylesheet


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlexPose Vision PySide6 staged desktop UI")
    parser.add_argument("--check", action="store_true", help="construct the UI and exit without entering the event loop")
    parser.add_argument("--render", type=Path, help="save a 1440x900 review screenshot and exit")
    parser.add_argument("--page", choices=("template", "detection", "live", "validation"), default="template")
    parser.add_argument("--review-sample", action="store_true", help="load checked-in real reference assets for visual QA")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args, qt_args = parser.parse_known_args(argv)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("FlexPose Vision")
    app.setOrganizationName("FlexPose")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(build_stylesheet())
    window = MainWindow(project_root())
    window.show_page(args.page)
    if args.review_sample:
        window.template_page.load_review_sample()
    if args.check:
        app.processEvents()
        return 0
    if args.render:
        window.resize(args.width, args.height)
        window.show()
        app.processEvents()
        args.render.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.render), "PNG"):
            raise OSError(f"无法保存截图：{args.render}")
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
"""PySide6 正式界面：创建应用对象并启动主窗口。"""
