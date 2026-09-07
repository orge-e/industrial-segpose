"""Qt 主题：根据设计令牌生成全局 QSS 样式表。"""

from .tokens import COLORS


def build_stylesheet() -> str:
    c = COLORS
    return f"""
    * {{ font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 12px; color: {c['text']}; outline: none; }}
    QMainWindow, QWidget#AppRoot {{ background: {c['app']}; }}
    QFrame#Header {{ background: {c['navigation']}; border-bottom: 1px solid {c['border']}; }}
    QFrame#Sidebar {{ background: {c['navigation']}; border-right: 1px solid {c['border']}; }}
    QFrame#Footer {{ background: {c['navigation']}; border-top: 1px solid {c['divider']}; }}
    QFrame#Panel {{ background: {c['panel']}; border-radius: 6px; }}
    QFrame#Elevated {{ background: {c['elevated']}; border-radius: 6px; }}
    QFrame#Viewer {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 7px; }}
    QFrame#Divider {{ background: {c['divider']}; min-height: 1px; max-height: 1px; }}
    QLabel#AppName {{ font-size: 21px; font-weight: 600; }}
    QLabel#PageTitle {{ font-size: 24px; font-weight: 600; }}
    QLabel#PageHint, QLabel#Secondary {{ color: {c['secondary']}; }}
    QLabel#Muted {{ color: {c['muted']}; font-size: 10px; }}
    QLabel#SectionTitle {{ color: {c['secondary']}; font-size: 11px; font-weight: 600; }}
    QLabel#FieldLabel {{ color: {c['muted']}; font-size: 10px; }}
    QLabel#Success {{ color: {c['success']}; font-size: 11px; font-weight: 600; }}
    QLabel#Warning {{ color: {c['warning']}; font-size: 11px; font-weight: 600; }}
    QLabel#Error {{ color: {c['error']}; font-size: 11px; font-weight: 600; }}
    QLabel#Info {{ color: {c['info']}; font-size: 11px; font-weight: 600; }}
    QPushButton {{ min-height: 34px; padding: 0 14px; border-radius: 4px; background: {c['elevated']}; border: 1px solid {c['border']}; }}
    QPushButton:hover {{ background: #202D3E; border-color: #34445B; }}
    QPushButton:pressed {{ background: #111A26; }}
    QPushButton:disabled {{ color: {c['disabled']}; background: {c['panel']}; border-color: {c['divider']}; }}
    QPushButton#PrimaryButton {{ background: {c['primary']}; border-color: {c['primary']}; font-weight: 600; }}
    QPushButton#PrimaryButton:hover {{ background: {c['primary_hover']}; }}
    QPushButton#GhostButton {{ background: transparent; border-color: transparent; color: {c['secondary']}; }}
    QPushButton#GhostButton:hover {{ background: {c['elevated']}; color: {c['text']}; }}
    QToolButton {{ min-width: 30px; min-height: 30px; border: 0; border-radius: 4px; background: transparent; color: {c['secondary']}; padding: 2px; }}
    QToolButton:hover {{ background: {c['elevated']}; color: {c['text']}; }}
    QToolButton:checked {{ background: #141F2E; color: {c['text']}; }}
    QToolButton#NavButton {{ min-height: 42px; padding: 0 16px; text-align: left; border-radius: 0; border-left: 3px solid transparent; color: {c['secondary']}; }}
    QToolButton#NavButton:hover {{ background: #121D2B; color: {c['text']}; }}
    QToolButton#NavButton:checked {{ background: #111A27; border-left-color: {c['primary']}; color: {c['text']}; }}
    QPushButton#StepButton {{ text-align: left; min-height: 32px; padding: 0 4px; background: transparent; border: 0; color: {c['muted']}; }}
    QPushButton#StepButton:hover, QPushButton#StepButton:checked {{ color: {c['text']}; background: transparent; }}
    QPushButton#SegmentButton {{ min-height: 28px; padding: 0 9px; border: 0; border-radius: 0; background: transparent; color: {c['muted']}; }}
    QPushButton#SegmentButton:hover {{ color: {c['text']}; }}
    QPushButton#SegmentButton:checked {{ color: {c['text']}; border-bottom: 2px solid {c['primary']}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 32px; padding: 0 9px; background: #0F1722; border: 1px solid {c['border']}; border-radius: 4px; selection-background-color: #294E80; }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c['primary']}; }}
    QComboBox::drop-down {{ border: 0; width: 24px; }}
    QGraphicsView {{ background: {c['canvas']}; border: 0; }}
    QScrollBar:vertical {{ background: transparent; width: 7px; }}
    QScrollBar::handle:vertical {{ background: #334155; min-height: 30px; border-radius: 3px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QSplitter::handle {{ background: {c['divider']}; width: 1px; }}
    QProgressBar {{ min-height: 3px; max-height: 3px; background: {c['divider']}; border: 0; }}
    QProgressBar::chunk {{ background: {c['primary']}; }}
    """
"""Qt 主题：根据设计令牌生成全局 QSS 样式表。"""
