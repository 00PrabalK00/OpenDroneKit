"""Application theme and icon helpers for a modern desktop look."""

from __future__ import annotations

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QStyle, QWidget


def standard_icon(widget: QWidget, *candidates: str) -> QIcon:
    """Return first available standard icon from candidate enum names."""
    app = QApplication.instance()
    style = app.style() if app is not None else widget.style()
    for name in candidates:
        pix = getattr(QStyle.StandardPixmap, name, None)
        if pix is not None:
            return style.standardIcon(pix)
    return QIcon()


def app_stylesheet() -> str:
    """Global dark-blue professional theme."""
    return """
QWidget {
    color: #e6edf7;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow, QWidget#central {
    background-color: #0f1724;
}
QGroupBox {
    background-color: #182335;
    border: 1px solid #2b3a52;
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #b9c8dd;
}
QTabWidget::pane {
    border: 1px solid #2b3a52;
    border-radius: 10px;
    background: #111b2a;
    top: -1px;
}
QTabBar::tab {
    background: #1a2740;
    color: #c8d6ea;
    border: 1px solid #30405a;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 12px;
    min-width: 96px;
}
QTabBar::tab:selected {
    background: #2b4f80;
    color: #f5f8ff;
}
QTabBar::tab:hover:!selected {
    background: #233654;
}
QPushButton {
    background-color: #2b4f80;
    border: 1px solid #3d628f;
    border-radius: 8px;
    padding: 6px 10px;
    color: #f4f8ff;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3867a5;
}
QPushButton:pressed {
    background-color: #25466f;
}
QPushButton:disabled {
    background-color: #263449;
    border-color: #2d3d52;
    color: #8ea0b7;
}
QLineEdit, QPlainTextEdit, QTextEdit, QListWidget, QTreeWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #111b2a;
    border: 1px solid #2f425f;
    border-radius: 8px;
    padding: 5px 7px;
    color: #e4ecf8;
    selection-background-color: #3d7ddd;
    selection-color: #f8fbff;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QListWidget:focus, QTreeWidget:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #4f8fe8;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #4a607f;
    background: #111b2a;
    border-radius: 4px;
}
QCheckBox::indicator:checked {
    border: 1px solid #4a607f;
    background: #3d7ddd;
    border-radius: 4px;
}
QProgressBar {
    border: 1px solid #365076;
    border-radius: 6px;
    background: #111b2a;
    text-align: center;
    color: #dbe6f5;
    min-height: 18px;
}
QProgressBar::chunk {
    border-radius: 5px;
    background-color: #3d7ddd;
}
QStatusBar {
    background: #121d2b;
    border-top: 1px solid #2b3a52;
    color: #b9c8dd;
}
QLabel {
    color: #d9e4f3;
}
QSplitter::handle {
    background: #253448;
}
"""
