"""Reusable UI components for OpenDroneKit."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _lbl(text: str, obj_name: str = "", word_wrap: bool = False) -> QLabel:
    l = QLabel(text)
    if obj_name:
        l.setObjectName(obj_name)
    if word_wrap:
        l.setWordWrap(True)
    return l


# ─── StatusChip ───────────────────────────────────────────────────────────────

class StatusChip(QLabel):
    """Colored pill label: 'online'|'ready'|'complete'|'warning'|'error'|'missing'|'idle'|'running'."""

    _CHIP_NAMES = {
        "online": "chip_online",
        "ready": "chip_ready",
        "complete": "chip_complete",
        "warning": "chip_warning",
        "error": "chip_error",
        "missing": "chip_missing",
        "danger": "chip_danger",
        "offline": "chip_offline",
        "idle": "chip_idle",
        "none": "chip_none",
        "info": "chip_info",
        "running": "chip_running",
    }

    def __init__(self, text: str, level: str = "idle", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.set_level(level)

    def set_level(self, level: str) -> None:
        obj = self._CHIP_NAMES.get(level.lower(), "chip_idle")
        self.setObjectName(obj)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_text_level(self, text: str, level: str) -> None:
        self.setText(text)
        self.set_level(level)


# ─── Banner ───────────────────────────────────────────────────────────────────

class Banner(QFrame):
    """Inline notification strip: 'warning'|'error'|'info'|'success'."""

    def __init__(self, text: str, kind: str = "info", parent: QWidget | None = None):
        super().__init__(parent)
        names = {
            "warning": "bannerWarning",
            "error": "bannerError",
            "info": "bannerInfo",
            "success": "bannerSuccess",
        }
        self.setObjectName(names.get(kind, "bannerInfo"))

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 7, 12, 7)
        row.setSpacing(8)

        icon_map = {"warning": "⚠", "error": "✖", "info": "ℹ", "success": "✔"}
        icon_lbl = QLabel(icon_map.get(kind, "ℹ"))
        icon_lbl.setObjectName("muted")
        row.addWidget(icon_lbl)

        self._msg = QLabel(text)
        self._msg.setWordWrap(True)
        self._msg.setObjectName("secondary")
        row.addWidget(self._msg, stretch=1)

    def set_text(self, text: str) -> None:
        self._msg.setText(text)


# ─── EmptyState ───────────────────────────────────────────────────────────────

class EmptyState(QWidget):
    """Centered empty-state widget with icon, title, description, and action button."""

    actionClicked = pyqtSignal()

    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        action_label: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(40, 40, 40, 40)

        icon_lbl = QLabel(icon)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 36px;")
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setObjectName("sectionTitle")
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setWordWrap(True)
        desc_lbl.setObjectName("muted")
        desc_lbl.setMaximumWidth(420)
        layout.addWidget(desc_lbl)

        if action_label:
            layout.addSpacing(8)
            self._btn = QPushButton(action_label)
            self._btn.setObjectName("primary")
            self._btn.setFixedWidth(200)
            self._btn.clicked.connect(self.actionClicked)
            layout.addWidget(self._btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_action_label(self, label: str) -> None:
        if hasattr(self, "_btn"):
            self._btn.setText(label)


# ─── PathInput ────────────────────────────────────────────────────────────────

class PathInput(QWidget):
    """Path input with browse button, tooltip on long paths, optional validation."""

    pathChanged = pyqtSignal(str)

    def __init__(
        self,
        placeholder: str = "Select folder or file...",
        directory_only: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._directory_only = directory_only

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        from PyQt6.QtWidgets import QLineEdit
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.setMinimumHeight(32)
        self._edit.textChanged.connect(self._on_text_changed)
        row.addWidget(self._edit, stretch=1)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setObjectName("ghost")
        self._browse_btn.setFixedWidth(70)
        self._browse_btn.setMinimumHeight(32)
        self._browse_btn.clicked.connect(self._browse)
        row.addWidget(self._browse_btn)

    def _browse(self) -> None:
        if self._directory_only:
            path = QFileDialog.getExistingDirectory(self, "Select Folder", self._edit.text() or "")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select File", self._edit.text() or "",
                "All Files (*)",
            )
        if path:
            self._edit.setText(path)

    def _on_text_changed(self, text: str) -> None:
        self._edit.setToolTip(text)
        self.pathChanged.emit(text)

    def text(self) -> str:
        return self._edit.text().strip()

    def setText(self, text: str) -> None:
        self._edit.setText(text)

    def set_file_filter(self, flt: str) -> None:
        self._file_filter = flt


# ─── SectionCard ──────────────────────────────────────────────────────────────

class SectionCard(QFrame):
    """Styled card frame with optional title bar."""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        if title:
            title_lbl = QLabel(title.upper())
            title_lbl.setObjectName("sectionTitle")
            root.addWidget(title_lbl)

        self._body = QVBoxLayout()
        self._body.setSpacing(6)
        root.addLayout(self._body)

    @property
    def body(self) -> QVBoxLayout:
        return self._body

    def add_widget(self, widget: QWidget) -> None:
        self._body.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._body.addLayout(layout)


# ─── MetricCard ───────────────────────────────────────────────────────────────

class MetricCard(QFrame):
    """Small card displaying a metric value + label."""

    def __init__(self, label: str, value: str = "—", unit: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("subCard")
        self.setMinimumWidth(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet("font-size: 14pt; font-weight: 700; color: #edf2ff;")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._value_lbl)

        if unit:
            unit_lbl = QLabel(unit)
            unit_lbl.setObjectName("muted")
            unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(unit_lbl)

        lbl = QLabel(label)
        lbl.setObjectName("muted")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

    def set_value(self, value: str) -> None:
        self._value_lbl.setText(value)


# ─── ReadinessCard ────────────────────────────────────────────────────────────

class ReadinessCard(QFrame):
    """Card showing a readiness state with icon and label."""

    def __init__(self, label: str, ready: bool = False, note: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("subCard")
        self.setMinimumHeight(60)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self._icon = QLabel("●")
        self._icon.setFixedWidth(14)
        layout.addWidget(self._icon)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        self._label = QLabel(label)
        self._label.setObjectName("fieldLabel")
        text_col.addWidget(self._label)
        if note:
            self._note = QLabel(note)
            self._note.setObjectName("muted")
            text_col.addWidget(self._note)
        else:
            self._note = None
        layout.addLayout(text_col, stretch=1)

        self._chip = StatusChip("", "idle")
        layout.addWidget(self._chip)

        self.set_ready(ready, note)

    def set_ready(self, ready: bool, note: str = "") -> None:
        if ready:
            self._icon.setStyleSheet("color: #22c55e; font-size: 14px;")
            self._chip.set_text_level("Ready", "ready")
        else:
            self._icon.setStyleSheet("color: #5a7898; font-size: 14px;")
            self._chip.set_text_level("Not ready", "idle")
        if note and self._note:
            self._note.setText(note)

    def set_state(self, text: str, level: str, note: str = "") -> None:
        icons = {
            "ready": ("●", "#22c55e"),
            "complete": ("●", "#22c55e"),
            "online": ("●", "#22c55e"),
            "warning": ("●", "#f59e0b"),
            "error": ("●", "#ef4444"),
            "missing": ("●", "#ef4444"),
            "running": ("●", "#3b82f6"),
            "idle": ("●", "#5a7898"),
        }
        icon_char, color = icons.get(level, ("●", "#5a7898"))
        self._icon.setText(icon_char)
        self._icon.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._chip.set_text_level(text, level)
        if note and self._note:
            self._note.setText(note)


# ─── TelemetryLabel ───────────────────────────────────────────────────────────

class TelemetryLabel(QFrame):
    """Telemetry value display with large number + label."""

    def __init__(self, label: str, value: str = "—", unit: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("subCard")
        self.setMinimumHeight(64)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top = QHBoxLayout()
        self._value = QLabel(value)
        self._value.setStyleSheet("font-size: 16pt; font-weight: 700; color: #edf2ff;")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self._value)
        if unit:
            self._unit = QLabel(unit)
            self._unit.setObjectName("muted")
            self._unit.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
            top.addWidget(self._unit)
        layout.addLayout(top)

        lbl = QLabel(label)
        lbl.setObjectName("muted")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

    def set_value(self, value: str) -> None:
        self._value.setText(value)


# ─── PipelineStageCard ────────────────────────────────────────────────────────

_STAGE_COLORS = {
    "ready":    ("#22c55e", "#041f10", "#14532d"),
    "running":  ("#3b82f6", "#0b1e3d", "#1d4ed8"),
    "complete": ("#22c55e", "#041f10", "#14532d"),
    "failed":   ("#ef4444", "#2c0808", "#7f1d1d"),
    "skipped":  ("#5a7898", "#0a1422", "#1a2d40"),
    "idle":     ("#5a7898", "#0f1c2e", "#213349"),
}


class PipelineStageCard(QFrame):
    runClicked = pyqtSignal()

    def __init__(
        self,
        number: int,
        title: str,
        description: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("subCard")
        self.setMinimumHeight(60)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        self._num = QLabel(str(number))
        self._num.setFixedSize(28, 28)
        self._num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._num.setStyleSheet(
            "background:#1b2c3f; border-radius:14px; color:#9ab4cc; font-weight:700; font-size:10pt;"
        )
        layout.addWidget(self._num)

        text = QVBoxLayout()
        text.setSpacing(2)
        self._title = QLabel(title)
        self._title.setObjectName("fieldLabel")
        text.addWidget(self._title)
        if description:
            self._desc = QLabel(description)
            self._desc.setObjectName("muted")
            text.addWidget(self._desc)
        layout.addLayout(text, stretch=1)

        self._chip = StatusChip("Idle", "idle")
        layout.addWidget(self._chip)

        self._run_btn = QPushButton("Run")
        self._run_btn.setObjectName("ghost")
        self._run_btn.setFixedWidth(64)
        self._run_btn.setMinimumHeight(28)
        self._run_btn.clicked.connect(self.runClicked)
        layout.addWidget(self._run_btn)

    def set_state(self, state: str) -> None:
        color, bg, border = _STAGE_COLORS.get(state, _STAGE_COLORS["idle"])
        self._num.setStyleSheet(
            f"background:{bg}; border: 1px solid {border}; border-radius:14px;"
            f"color:{color}; font-weight:700; font-size:10pt;"
        )
        self._chip.set_text_level(state.capitalize(), state)
        self._run_btn.setEnabled(state not in ("running",))

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(enabled)

    def set_run_label(self, label: str) -> None:
        self._run_btn.setText(label)


# ─── Separator ────────────────────────────────────────────────────────────────

def h_separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("QFrame { color: #192535; }")
    sep.setFixedHeight(1)
    return sep


def make_label(text: str, obj_name: str = "fieldLabel") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName(obj_name)
    return lbl
