"""Modern operator-console UI for the Drone Inspection Toolkit.

This module keeps the existing PyQt entrypoint while presenting the product as a
compact industrial inspection platform: top bar, grouped sidebar navigation,
map-first mission planning, live inspection controls, media review, defect
analysis, reports, fleet management, and settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .components import StatusChip
from .workspace import AppSession


BLUE = "#2f7df6"
GREEN = "#6bd66f"
YELLOW = "#f6bd3a"
RED = "#ef5555"
PANEL = "#111a25"
PANEL_2 = "#151f2b"
BORDER = "#253244"
TEXT = "#eef4ff"
MUTED = "#8fa2b7"


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _project_root(session: AppSession) -> Path:
    project = session.active_project if isinstance(session.active_project, dict) else None
    root = Path(str(project.get("root_dir", ""))) if project else Path("final_toolkit_outputs") / "projects" / "field_project_01"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _project_id(session: AppSession) -> str:
    pid = session.active_project_id()
    return str(pid if pid is not None and pid >= 0 else "default")


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json", "{}") if isinstance(row, dict) else "{}"
    try:
        data = json.loads(str(raw or "{}"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _default_polygon_lonlat() -> list[list[float]]:
    return [
        [-122.0000, 37.0000],
        [-121.9987, 37.0001],
        [-121.9985, 37.0012],
        [-121.9994, 37.0017],
        [-122.0004, 37.0010],
    ]


def _mode_from_label(label: str) -> str:
    mapping = {
        "Manual Assisted": "waypoints",
        "Grid Scan": "grid",
        "Orbit Scan": "orbit",
        "Facade Scan": "facade",
        "Corridor Scan": "linear_inspection",
        "Custom Waypoints": "waypoints",
    }
    return mapping.get(str(label), "grid")


def _camera_from_label(label: str) -> str:
    text = str(label).lower()
    if "thermal" in text:
        return "thermal_640"
    if "zoom" in text:
        return "zoom_inspection"
    if "45" in text:
        return "rgb_45mp"
    return "mavic2pro"


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())


def _pixmap_from_path(path: str | Path, max_size: QSize) -> QPixmap:
    pix = QPixmap(str(path))
    if pix.isNull():
        return QPixmap()
    return pix.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def _soft_shadow(widget: QWidget, blur: float = 24.0, y: float = 8.0, alpha: int = 70) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(float(blur))
    effect.setOffset(0, float(y))
    effect.setColor(QColor(0, 0, 0, int(alpha)))
    widget.setGraphicsEffect(effect)


def _style_icon(widget: QWidget, name: str) -> Any:
    pix = getattr(QStyle.StandardPixmap, name, None)
    if pix is None:
        pix = QStyle.StandardPixmap.SP_FileIcon
    return widget.style().standardIcon(pix)


def _page_title(text: str, subtitle: str = "") -> QWidget:
    box = QWidget()
    row = QVBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(3)
    title = QLabel(text)
    title.setObjectName("modernPageTitle")
    row.addWidget(title)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("modernMuted")
        row.addWidget(sub)
    return box


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("modernLabel")
    return lbl


def _muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("modernMuted")
    return lbl


def _section(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("modernSection")
    return lbl


def _button(text: str, kind: str = "secondary") -> QPushButton:
    btn = QPushButton(text)
    names = {
        "primary": "modernPrimary",
        "secondary": "modernSecondary",
        "ghost": "modernGhost",
        "danger": "modernDanger",
    }
    btn.setObjectName(names.get(kind, "modernSecondary"))
    btn.setMinimumHeight(34)
    return btn


def _icon_button(parent: QWidget, icon_name: str, tooltip: str) -> QToolButton:
    btn = QToolButton(parent)
    btn.setObjectName("modernIconButton")
    btn.setIcon(_style_icon(parent, icon_name))
    btn.setIconSize(QSize(18, 18))
    btn.setToolTip(tooltip)
    btn.setFixedSize(36, 36)
    return btn


def _combo(items: list[str], current: int = 0) -> QComboBox:
    combo = QComboBox()
    combo.setObjectName("modernInput")
    combo.setMinimumHeight(34)
    combo.setEditable(False)
    combo.addItems(items)
    combo.setCurrentIndex(min(max(current, 0), max(0, combo.count() - 1)))
    return combo


def _spin(value: float, suffix: str = "", low: float = 0, high: float = 999) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setObjectName("modernInput")
    spin.setRange(low, high)
    spin.setValue(value)
    spin.setDecimals(1 if suffix not in {"%", "m"} else 0)
    spin.setSuffix(f" {suffix}" if suffix else "")
    spin.setMinimumHeight(34)
    return spin


class DroneSelectionField(QFrame):
    changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("droneSelect")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.options = [
            ("Matrice 350 RTK", "H20T payload", "Available"),
            ("Mavic 3 Enterprise", "RGB 45 MP", "Standby"),
            ("Phantom 4 RTK", "Survey RGB", "Standby"),
            ("Skydio X10", "Thermal/RGB", "Offline"),
        ]
        self.selected = self.options[0]
        _soft_shadow(self, blur=12, y=3, alpha=34)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(9)

        self.icon = QLabel("DRN")
        self.icon.setObjectName("droneIcon")
        self.icon.setFixedSize(34, 30)
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.icon)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        self.model = QLabel()
        self.model.setObjectName("droneModel")
        self.status = QLabel()
        self.status.setObjectName("droneStatus")
        text_col.addWidget(self.model)
        text_col.addWidget(self.status)
        row.addLayout(text_col, stretch=1)

        self.dot = QLabel()
        self.dot.setObjectName("availabilityDot")
        self.dot.setFixedSize(8, 8)
        row.addWidget(self.dot)

        self.chevron = QLabel("v")
        self.chevron.setObjectName("fieldChevron")
        row.addWidget(self.chevron)
        self._sync()

    def currentText(self) -> str:  # noqa: N802
        return f"{self.selected[0]} - {self.selected[2]}"

    def mousePressEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        for option in self.options:
            action = QAction(f"{option[0]}   {option[2]}", menu)
            action.setData(option)
            menu.addAction(action)
        action = menu.exec(self.mapToGlobal(event.pos()))
        if action is not None and isinstance(action.data(), tuple):
            self.selected = action.data()
            self._sync()
            self.changed.emit(self.currentText())
        super().mousePressEvent(event)

    def _sync(self) -> None:
        model, payload, state = self.selected
        self.model.setText(model)
        self.status.setText(f"{payload}  |  {state}")
        self.dot.setProperty("available", state == "Available")
        self.dot.style().unpolish(self.dot)
        self.dot.style().polish(self.dot)


class Panel(QFrame):
    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("modernPanel")
        _soft_shadow(self, blur=18, y=5, alpha=48)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)
        if title:
            self.layout.addWidget(_section(title))


class ScrollPanel(QScrollArea):
    def __init__(self, content: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, note: str, level: str = "idle"):
        super().__init__()
        self.setObjectName("modernMetric")
        _soft_shadow(self, blur=18, y=5, alpha=42)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(_muted(title), stretch=1)
        chip = StatusChip(note, level)
        top.addWidget(chip)
        lay.addLayout(top)
        val = QLabel(value)
        val.setObjectName("modernMetricValue")
        lay.addWidget(val)


class MiniMapPreview(QWidget):
    def __init__(self, mode: str = "dashboard", parent: QWidget | None = None):
        super().__init__(parent)
        self.mode = mode
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        self._paint_aerial_background(p, rect)

        p.setPen(QPen(QColor(188, 180, 151, 110), 5))
        p.drawPath(self._road_path(rect))
        p.setPen(QPen(QColor(40, 45, 40, 90), 2))
        p.drawPath(self._road_path(rect))

        poly = QPolygonF([
            QPointF(rect.width() * 0.20, rect.height() * 0.26),
            QPointF(rect.width() * 0.70, rect.height() * 0.18),
            QPointF(rect.width() * 0.84, rect.height() * 0.55),
            QPointF(rect.width() * 0.55, rect.height() * 0.82),
            QPointF(rect.width() * 0.18, rect.height() * 0.68),
        ])
        p.setPen(QPen(QColor(78, 152, 255, 185), 2))
        p.setBrush(QColor(47, 125, 246, 30))
        p.drawPolygon(poly)

        p.setPen(QPen(QColor("#eaf4ff"), 1.2, Qt.PenStyle.SolidLine))
        for row in range(8):
            y = rect.height() * (0.34 + row * 0.045)
            p.drawLine(QPointF(rect.width() * 0.31, y), QPointF(rect.width() * 0.74, y + 20))

        p.setPen(QPen(QColor(RED), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(239, 85, 85, 28))
        p.drawEllipse(QPointF(rect.width() * 0.57, rect.height() * 0.22), 48, 48)
        for offset in range(-42, 44, 10):
            p.drawLine(
                QPointF(rect.width() * 0.57 - 42 + offset, rect.height() * 0.22 + 42),
                QPointF(rect.width() * 0.57 + 42 + offset, rect.height() * 0.22 - 42),
            )

        p.setPen(QPen(QColor(YELLOW), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(246, 189, 58, 24))
        p.drawRect(QRectF(rect.width() * 0.58, rect.height() * 0.68, 120, 60))

    def _paint_aerial_background(self, p: QPainter, rect) -> None:
        grad = QLinearGradient(0, 0, rect.width(), rect.height())
        grad.setColorAt(0, QColor("#24352b"))
        grad.setColorAt(0.32, QColor("#31432d"))
        grad.setColorAt(0.68, QColor("#1d2b26"))
        grad.setColorAt(1, QColor("#161f22"))
        p.fillRect(rect, grad)

        patch_colors = ["#415034", "#2c3d2e", "#5b553b", "#25372e", "#4a402d"]
        for i in range(34):
            cx = (i * 97) % max(rect.width(), 1)
            cy = (i * 61 + 37) % max(rect.height(), 1)
            w = 110 + (i * 23) % 190
            h = 44 + (i * 17) % 105
            color = QColor(patch_colors[i % len(patch_colors)])
            color.setAlpha(38 + (i * 11) % 42)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(cx, cy), w / 2, h / 2)

        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        for i in range(-rect.height(), rect.width(), 58):
            p.drawLine(i, rect.height(), i + rect.height(), 0)
        p.setPen(QPen(QColor(0, 0, 0, 28), 1))
        for i in range(0, rect.width(), 96):
            p.drawLine(i, 0, int(i + rect.height() * 0.4), rect.height())

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(220):
            x = (i * 47 + 31) % max(rect.width(), 1)
            y = (i * 83 + 19) % max(rect.height(), 1)
            radius = 1.2 + (i % 4) * 0.45
            alpha = 32 + (i % 5) * 12
            p.setBrush(QColor(26, 50, 31, alpha))
            p.drawEllipse(QPointF(x, y), radius, radius)

    def _road_path(self, rect):
        path = QPainterPath()
        path.moveTo(rect.width() * 0.03, rect.height() * 0.75)
        path.cubicTo(
            rect.width() * 0.25,
            rect.height() * 0.55,
            rect.width() * 0.28,
            rect.height() * 0.10,
            rect.width() * 0.54,
            rect.height() * 0.23,
        )
        path.cubicTo(
            rect.width() * 0.86,
            rect.height() * 0.38,
            rect.width() * 0.76,
            rect.height() * 0.88,
            rect.width() * 0.98,
            rect.height() * 0.78,
        )
        return path


class MissionMapCanvas(MiniMapPreview):
    activeToolChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.active_tool = "Select"
        self.last_plan_summary: dict[str, Any] = {}
        self.setMinimumSize(640, 520)
        self.setObjectName("missionMap")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        top = QHBoxLayout()
        self.layer = _combo(["Satellite", "Street", "Terrain", "Blueprint", "Imported Map"])
        self.layer.setFixedWidth(140)
        top.addWidget(self.layer)
        top.addStretch(1)
        for icon, tip in [
            ("SP_ArrowUp", "Zoom in"),
            ("SP_ArrowDown", "Zoom out"),
            ("SP_BrowserReload", "Fit to mission"),
            ("SP_TitleBarMaxButton", "Full screen"),
        ]:
            top.addWidget(_icon_button(self, icon, tip))
        root.addLayout(top)

        root.addStretch(1)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        tools = Panel()
        tools.setObjectName("mapTools")
        tools.setFixedWidth(52)
        tools.setFixedHeight(310)
        tools.layout.setContentsMargins(7, 7, 7, 7)
        tools.layout.setSpacing(6)
        self.tool_buttons: dict[str, QToolButton] = {}
        for name, icon in [
            ("Select", "SP_ArrowCursor"),
            ("Draw Polygon", "SP_FileDialogNewFolder"),
            ("Waypoint", "SP_DialogYesButton"),
            ("No-Fly Zone", "SP_MessageBoxCritical"),
            ("Inspect Point", "SP_FileDialogContentsView"),
            ("Measure", "SP_ComputerIcon"),
            ("Erase", "SP_TrashIcon"),
        ]:
            b = QToolButton(self)
            b.setObjectName("mapToolButton")
            b.setIcon(_style_icon(self, icon))
            b.setIconSize(QSize(17, 17))
            b.setToolTip(name)
            b.setCheckable(True)
            b.setFixedSize(36, 34)
            b.clicked.connect(lambda _=False, n=name: self._tool_feedback(n))
            self.tool_buttons[name] = b
            tools.layout.addWidget(b, alignment=Qt.AlignmentFlag.AlignCenter)
        self.tool_buttons["Select"].setChecked(True)
        bottom.addWidget(tools, stretch=0, alignment=Qt.AlignmentFlag.AlignBottom)
        bottom.addStretch(1)

        timeline = Panel()
        timeline.setObjectName("mapOverlay")
        timeline.setFixedSize(360, 104)
        timeline.layout.setContentsMargins(12, 10, 12, 10)
        row = QHBoxLayout()
        row.setSpacing(10)
        for idx, (stage, tm) in enumerate([
            ("Takeoff", "00:01"),
            ("Transit", "00:03"),
            ("Inspect", "00:18"),
            ("Return", "00:04"),
            ("Landing", "00:01"),
        ], start=1):
            col = QVBoxLayout()
            col.setSpacing(2)
            dot = QLabel(str(idx))
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setFixedSize(26, 26)
            dot.setStyleSheet(
                f"background:{GREEN if idx in (1, 5) else BLUE}; border-radius:13px; color:white; font-weight:700;"
            )
            col.addWidget(dot, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(_label(stage), alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(_muted(tm), alignment=Qt.AlignmentFlag.AlignCenter)
            row.addLayout(col)
        timeline.layout.addLayout(row)
        bottom.addWidget(timeline, stretch=0, alignment=Qt.AlignmentFlag.AlignBottom)

        stats = Panel()
        stats.setObjectName("mapOverlay")
        stats.setFixedSize(300, 104)
        stats.layout.setContentsMargins(12, 10, 12, 10)
        statrow = QHBoxLayout()
        statrow.setSpacing(2)
        self.stat_labels: dict[str, QLabel] = {}
        for title, value in [("Time", "--"), ("Dist", "--"), ("Batt", "--"), ("Cov", "--")]:
            col = QVBoxLayout()
            col.setSpacing(2)
            val = QLabel(value)
            val.setObjectName("modernMapStat")
            self.stat_labels[title] = val
            col.addWidget(val, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(_muted(title), alignment=Qt.AlignmentFlag.AlignCenter)
            statrow.addLayout(col)
        stats.layout.addLayout(statrow)
        bottom.addWidget(stats, stretch=0, alignment=Qt.AlignmentFlag.AlignBottom)
        root.addLayout(bottom)

    def _tool_feedback(self, name: str) -> None:
        self.active_tool = name
        for tool_name, button in self.tool_buttons.items():
            button.setChecked(tool_name == name)
        self.activeToolChanged.emit(name)

    def set_plan_summary(self, summary: dict[str, Any]) -> None:
        self.last_plan_summary = dict(summary or {})
        distance_m = float(summary.get("distance_m", 0.0) or 0.0)
        time_min = float(summary.get("estimated_time_min", 0.0) or 0.0)
        coverage = float(summary.get("coverage_pct", 0.0) or 0.0)
        battery = min(100.0, max(10.0, time_min / 22.0 * 100.0))
        self.stat_labels["Time"].setText(f"{time_min:.0f}m" if time_min else "--")
        self.stat_labels["Dist"].setText(f"{distance_m / 1000.0:.1f}k" if distance_m else "--")
        self.stat_labels["Batt"].setText(f"{battery:.0f}%" if time_min else "--")
        self.stat_labels["Cov"].setText(f"{coverage:.0f}%" if coverage else "--")
        self.update()

    def _road_path(self, rect):
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(rect.width() * 0.03, rect.height() * 0.75)
        path.cubicTo(
            rect.width() * 0.25,
            rect.height() * 0.55,
            rect.width() * 0.28,
            rect.height() * 0.10,
            rect.width() * 0.54,
            rect.height() * 0.23,
        )
        path.cubicTo(
            rect.width() * 0.86,
            rect.height() * 0.38,
            rect.width() * 0.76,
            rect.height() * 0.88,
            rect.width() * 0.98,
            rect.height() * 0.78,
        )
        return path

    def paintEvent(self, event) -> None:  # noqa: N802
        MiniMapPreview.paintEvent(self, event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(20, 60, -20, -170)
        cx = r.center().x()
        cy = r.center().y()

        poly_points = [
            QPointF(r.left() + r.width() * 0.18, r.top() + r.height() * 0.28),
            QPointF(r.left() + r.width() * 0.70, r.top() + r.height() * 0.18),
            QPointF(r.left() + r.width() * 0.84, r.top() + r.height() * 0.54),
            QPointF(r.left() + r.width() * 0.58, r.top() + r.height() * 0.84),
            QPointF(r.left() + r.width() * 0.25, r.top() + r.height() * 0.78),
            QPointF(r.left() + r.width() * 0.14, r.top() + r.height() * 0.48),
        ]
        p.setPen(QPen(QColor(35, 80, 140, 170), 7))
        for a, b in zip(poly_points, poly_points[1:] + poly_points[:1]):
            p.drawLine(a, b)

        p.setPen(QPen(QColor(72, 154, 255, 230), 2.5))
        p.setBrush(QColor(47, 125, 246, 52))
        p.drawPolygon(QPolygonF(poly_points))

        p.setPen(QPen(QColor("#eaf4ff"), 1.5))
        for row in range(9):
            t = row / 8.0
            left = QPointF(
                poly_points[0].x() * (1 - t) + poly_points[5].x() * t + 24,
                poly_points[0].y() * (1 - t) + poly_points[5].y() * t + 12,
            )
            right = QPointF(
                poly_points[1].x() * (1 - t) + poly_points[3].x() * t - 28,
                poly_points[1].y() * (1 - t) + poly_points[3].y() * t + 18,
            )
            p.drawLine(left, right)
            arrow_x = left.x() * 0.42 + right.x() * 0.58
            arrow_y = left.y() * 0.42 + right.y() * 0.58
            p.drawLine(QPointF(arrow_x - 6, arrow_y - 3), QPointF(arrow_x, arrow_y))
            p.drawLine(QPointF(arrow_x - 6, arrow_y + 3), QPointF(arrow_x, arrow_y))

        for idx, pt in enumerate(poly_points, start=1):
            p.setPen(QPen(QColor("#dceeff"), 2))
            p.setBrush(QColor(47, 125, 246, 235))
            p.drawEllipse(pt, 11, 11)
            p.setPen(QPen(QColor(0, 0, 0, 80), 1))
            p.drawEllipse(pt, 14, 14)
            p.setPen(QColor("white"))
            p.drawText(QRectF(pt.x() - 10, pt.y() - 10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(idx))

        nofly_center = QPointF(r.left() + r.width() * 0.60, r.top() + r.height() * 0.19)
        nofly_path = QPainterPath()
        nofly_path.addEllipse(nofly_center, 56, 46)
        p.setPen(QPen(QColor(239, 85, 85, 220), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(239, 85, 85, 36))
        p.drawPath(nofly_path)
        p.save()
        p.setClipPath(nofly_path)
        p.setPen(QPen(QColor(239, 85, 85, 130), 1))
        for offset in range(-120, 130, 10):
            p.drawLine(QPointF(nofly_center.x() - 80 + offset, nofly_center.y() + 62), QPointF(nofly_center.x() + 80 + offset, nofly_center.y() - 62))
        p.restore()

        ezone = QRectF(r.left() + r.width() * 0.58, r.top() + r.height() * 0.76, 118, 58)
        p.setPen(QPen(QColor(246, 189, 58, 230), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(246, 189, 58, 26))
        p.drawRoundedRect(ezone, 8, 8)
        p.setPen(QColor(YELLOW))
        p.drawText(ezone, Qt.AlignmentFlag.AlignCenter, "E1")

        p.setPen(QPen(QColor("#f3f7ff"), 5))
        p.drawLine(QPointF(cx - 36, cy + 44), QPointF(cx + 38, cy - 54))
        p.drawLine(QPointF(cx - 5, cy - 8), QPointF(cx - 88, cy - 36))
        p.drawLine(QPointF(cx - 4, cy - 9), QPointF(cx + 96, cy + 8))
        p.setBrush(QColor("#f3f7ff"))
        p.drawEllipse(QPointF(cx, cy), 8, 8)


class TopBar(QWidget):
    settingsRequested = pyqtSignal()

    def __init__(self, session: AppSession, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("modernTopBar")
        self.setFixedHeight(60)
        row = QHBoxLayout(self)
        row.setContentsMargins(18, 7, 18, 7)
        row.setSpacing(12)

        logo = QLabel("ODK")
        logo.setObjectName("modernLogo")
        logo.setFixedSize(36, 36)
        row.addWidget(logo)
        product = QLabel("Drone Inspection Toolkit")
        product.setObjectName("modernProduct")
        row.addWidget(product)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedHeight(30)
        sep.setStyleSheet(f"color:{BORDER};")
        row.addWidget(sep)

        self.workspace = QToolButton()
        self.workspace.setObjectName("workspaceSelector")
        self.workspace.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.workspace.setText("Workspace:  Northwind Energy")
        self.workspace.setMinimumHeight(34)
        menu = QMenu(self.workspace)
        for name in ["Northwind Energy", "Bridge Assets", "Solar Operations"]:
            menu.addAction(name)
        self.workspace.setMenu(menu)
        row.addWidget(self.workspace)

        self.status = StatusChip("Planning", "running")
        row.addWidget(self.status)
        row.addStretch(1)

        search_wrap = QFrame()
        search_wrap.setObjectName("searchWrap")
        srow = QHBoxLayout(search_wrap)
        srow.setContentsMargins(12, 0, 10, 0)
        srow.setSpacing(8)
        srow.addWidget(QLabel("Search"))
        self.search = QLineEdit()
        self.search.setObjectName("topSearch")
        self.search.setPlaceholderText("Search missions, drones, defects, assets")
        srow.addWidget(self.search, stretch=1)
        hint = QLabel("Ctrl K")
        hint.setObjectName("shortcutHint")
        srow.addWidget(hint)
        search_wrap.setFixedWidth(420)
        row.addWidget(search_wrap)
        row.addStretch(1)

        bell = _icon_button(self, "SP_MessageBoxInformation", "Notifications")
        row.addWidget(bell)
        self.connection = QLabel("Disconnected\nLink Quality: --")
        self.connection.setObjectName("connectionPill")
        self.connection.setFixedHeight(44)
        row.addWidget(self.connection)
        self.battery = QLabel("--%\nEst. -- min")
        self.battery.setObjectName("batteryPill")
        self.battery.setFixedHeight(44)
        row.addWidget(self.battery)
        settings = _icon_button(self, "SP_FileDialogDetailedView", "Settings")
        settings.clicked.connect(self.settingsRequested)
        row.addWidget(settings)

        profile = QToolButton()
        profile.setObjectName("profileButton")
        profile.setText("James Patel")
        profile.setMinimumHeight(36)
        profile.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        profile_menu = QMenu(profile)
        profile_menu.addAction("Profile")
        profile_menu.addAction("Account preferences")
        profile_menu.addSeparator()
        profile_menu.addAction("Sign Out")
        profile.setMenu(profile_menu)
        row.addWidget(profile)

        self.session.projectChanged.connect(self._on_project_changed)
        self.session.droneConnected.connect(self._on_drone_connected)
        self.session.telemetryUpdated.connect(self._on_telemetry)

    def _on_project_changed(self, project: dict[str, Any]) -> None:
        name = project.get("name", "Northwind Energy") if isinstance(project, dict) else "Northwind Energy"
        self.workspace.setText(f"Workspace:  {name}")

    def _on_drone_connected(self, connected: bool) -> None:
        self.connection.setText("Connected\nLink Quality: --" if connected else "Disconnected\nLink Quality: --")
        self.status.set_text_level("Idle" if connected else "Offline", "ready" if connected else "idle")

    def _on_telemetry(self, telem: Any) -> None:
        try:
            link = float(getattr(telem, "link_quality_pct", 0.0))
            battery = float(getattr(telem, "battery_pct", 0.0))
            connected = bool(getattr(telem, "connected", False))
            self.connection.setText(("Connected" if connected else "Disconnected") + f"\nLink Quality: {link:.0f}%")
            est_min = max(0, int(round(battery / 100.0 * 32.0)))
            self.battery.setText(f"{battery:.0f}%\nEst. {est_min} min")
        except Exception:
            pass


class NavButton(QPushButton):
    def __init__(self, label: str, index: int, icon_name: str, parent: QWidget | None = None):
        super().__init__(label)
        self.index = index
        self.full_label = label
        self.setObjectName("modernNav")
        self.setCheckable(True)
        self.setIcon(_style_icon(parent or self, icon_name))
        self.setIconSize(QSize(18, 18))
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed:
            self.setText("")
            self.setToolTip(self.full_label)
        else:
            self.setText(self.full_label)
            self.setToolTip("")


class Sidebar(QWidget):
    navigate = pyqtSignal(int)

    GROUPS = [
        ("Operations", [("Dashboard", 0, "SP_ComputerIcon"), ("Mission Planner", 1, "SP_FileDialogDetailedView"), ("Live Inspection", 2, "SP_MediaPlay")]),
        ("Analysis", [("Media Review", 3, "SP_FileDialogContentsView"), ("Defect Analysis", 4, "SP_MessageBoxWarning")]),
        ("Library", [("Asset Library", 5, "SP_DirIcon"), ("Reports", 6, "SP_FileIcon"), ("Drone Fleet", 7, "SP_DriveNetIcon")]),
        ("System", [("Settings", 8, "SP_FileDialogInfoView")]),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("modernSidebar")
        self.setFixedWidth(184)
        self.buttons: list[NavButton] = []
        self.headers: list[QLabel] = []
        self.collapsed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 12, 0, 12)
        root.setSpacing(4)

        for group, items in self.GROUPS:
            header = QLabel(group)
            header.setObjectName("navGroup")
            root.addWidget(header)
            self.headers.append(header)
            for label, idx, icon in items:
                btn = NavButton(label, idx, icon, self)
                btn.clicked.connect(lambda _=False, i=idx: self.set_active(i))
                root.addWidget(btn)
                self.buttons.append(btn)
            root.addSpacing(8)
        root.addStretch(1)
        self.collapse_btn = _button("Collapse", "ghost")
        self.collapse_btn.setIcon(_style_icon(self, "SP_TitleBarShadeButton"))
        self.collapse_btn.setIconSize(QSize(16, 16))
        self.collapse_btn.clicked.connect(self.toggle_collapsed)
        root.addWidget(self.collapse_btn)
        self.set_active(1)

    def set_active(self, index: int) -> None:
        for btn in self.buttons:
            btn.setChecked(btn.index == index)
        self.navigate.emit(index)

    def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed
        self.setFixedWidth(72 if self.collapsed else 184)
        for header in self.headers:
            header.setVisible(not self.collapsed)
        for btn in self.buttons:
            btn.set_collapsed(self.collapsed)
        self.collapse_btn.setText("Expand" if self.collapsed else "Collapse")


class DashboardPage(QWidget):
    def __init__(self, session: AppSession, navigate: Callable[[int], None], feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.navigate = navigate
        self.feedback = feedback
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(_page_title("Dashboard", "Operational overview for inspections, defects, drones, and reports."))
        top.addStretch(1)
        root.addLayout(top)

        cards = QGridLayout()
        cards.setHorizontalSpacing(16)
        cards.setVerticalSpacing(16)
        for col, args in enumerate([
            ("Active Drone", "Matrice 350 RTK", "Safe", "ready"),
            ("Next Mission", "Turbine T-17", "14:30", "running"),
            ("Open Defects", "23", "5 critical", "warning"),
            ("Reports Pending", "7", "Draft", "idle"),
        ]):
            cards.addWidget(MetricCard(*args), 0, col)
        root.addLayout(cards)

        middle = QHBoxLayout()
        map_panel = Panel("Mission Map Preview")
        map_panel.layout.addWidget(MiniMapPreview(), stretch=1)
        middle.addWidget(map_panel, stretch=3)
        recent = Panel("Recent Inspections")
        for text, chip in [
            ("WTG-17 detailed inspection", "Reviewing"),
            ("Bridge B12 corrosion pass", "Report Ready"),
            ("Solar block C thermal scan", "Flying"),
            ("Warehouse roof survey", "Complete"),
        ]:
            row = QHBoxLayout()
            row.addWidget(_label(text), stretch=1)
            row.addWidget(StatusChip(chip, "running" if chip == "Flying" else "ready"))
            recent.layout.addLayout(row)
        middle.addWidget(recent, stretch=1)
        health = Panel("Drone Health")
        for label, value in [("Fleet availability", "8 / 10"), ("Average battery health", "91%"), ("Sensors ready", "14 / 16"), ("Maintenance due", "2 drones")]:
            row = QHBoxLayout()
            row.addWidget(_muted(label), stretch=1)
            row.addWidget(_label(value))
            health.layout.addLayout(row)
        middle.addWidget(health, stretch=1)
        root.addLayout(middle, stretch=1)

        bottom = QHBoxLayout()
        for title, items in [
            ("Recent Media Uploads", ["IMG_2041 thermal set", "Video pass 03", "Turbine nacelle closeups"]),
            ("Critical Alerts", ["No-fly conflict near turbine T-17", "Drone D-04 firmware out of date", "High wind warning for 16:00"]),
        ]:
            panel = Panel(title)
            for item in items:
                panel.layout.addWidget(_label(item))
            bottom.addWidget(panel, stretch=1)
        quick = Panel("Quick Actions")
        qrow = QHBoxLayout()
        actions = [
            ("Create Mission", lambda: self.navigate(1), "primary"),
            ("Import Map", self._import_map, "secondary"),
            ("Upload Inspection Media", lambda: self.navigate(3), "secondary"),
            ("Generate Report", lambda: self.navigate(6), "secondary"),
        ]
        for text, cb, kind in actions:
            b = _button(text, kind)
            b.clicked.connect(cb)
            qrow.addWidget(b)
        quick.layout.addLayout(qrow)
        bottom.addWidget(quick, stretch=2)
        root.addLayout(bottom)

    def _import_map(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import map or asset boundary",
            "",
            "Map files (*.geojson *.json *.kml *.kmz *.mbtiles *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        try:
            src = Path(path)
            out_dir = _project_root(self.session) / "maps"
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / src.name
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())
            pid = self.session.active_project_id()
            if pid is not None and pid >= 0:
                self.session.store.append_audit_event(pid, "map_imported", {"source": str(src), "path": str(dst)})
            self.feedback(f"Map imported: {dst.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Import Map", str(exc))


class MissionConfigPanel(Panel):
    def __init__(self):
        super().__init__("Mission Configuration")
        self.mission_name = QLineEdit("Turbine_T-17_Main_Inspection")
        self.mission_name.setObjectName("modernInput")
        self.mission_name.setMaxLength(100)
        self.layout.addWidget(_label("Mission Name"))
        self.layout.addWidget(self.mission_name)
        self.layout.addWidget(_muted("27 / 100"))
        self.asset_type = _combo(["Building", "Bridge", "Solar Farm", "Wind Turbine", "Pipeline", "Warehouse", "Custom"], 3)
        self.inspection_type = _combo(["Visual Survey", "Thermal Inspection", "Crack Detection", "Corrosion Inspection", "Roof Inspection", "Structural Scan"], 5)
        self.drone_selection = DroneSelectionField()
        self.camera_profile = _combo(["H20T Triple Sensor", "RGB 45 MP", "Thermal 640", "Zoom Inspection"], 0)
        self.flight_mode = _combo(["Manual Assisted", "Grid Scan", "Orbit Scan", "Facade Scan", "Corridor Scan", "Custom Waypoints"], 1)
        self.altitude = _spin(80, "m", 5, 150)
        self.overlap = _spin(75, "%", 20, 95)
        self.speed = _spin(8.0, "m/s", 1, 20)
        fields: list[tuple[str, QWidget, str]] = [
            ("Asset Type", self.asset_type, "Target asset family."),
            ("Inspection Type", self.inspection_type, "Determines capture density and AI workflow."),
            ("Drone Selection", self.drone_selection, "Only available aircraft are selectable."),
            ("Camera Profile", self.camera_profile, "Camera and lens preset."),
            ("Flight Mode", self.flight_mode, "Route compiler mode."),
            ("Altitude (AGL)", self.altitude, "Allowed range 20-150 m."),
            ("Overlap", self.overlap, "Front overlap. Side overlap uses mission defaults."),
            ("Speed", self.speed, "Recommended 3-15 m/s."),
        ]
        for label, widget, help_text in fields:
            self.layout.addWidget(_label(label))
            self.layout.addWidget(widget)
            self.layout.addWidget(_muted(help_text))
        self.return_to_home = QCheckBox("Return to Home")
        self.return_to_home.setChecked(True)
        self.obstacle_avoidance = QCheckBox("Obstacle Avoidance")
        self.obstacle_avoidance.setChecked(True)
        self.layout.addWidget(self.return_to_home)
        self.layout.addWidget(self.obstacle_avoidance)
        self.layout.addWidget(_label("Emergency Landing Zone"))
        self.emergency_zone = _combo(["E1 - North Field", "E2 - Service Road", "E3 - Gravel Pad"], 0)
        self.layout.addWidget(self.emergency_zone)
        self.layout.addStretch(1)
        weather = QLabel("Wind: 12 km/h NE   Temp: 18 C   Clouds: 20%")
        weather.setObjectName("weatherChip")
        self.layout.addWidget(weather)

    def values(self) -> dict[str, Any]:
        return {
            "mission_name": self.mission_name.text().strip() or "Mission",
            "asset_type": self.asset_type.currentText(),
            "inspection_type": self.inspection_type.currentText(),
            "drone": self.drone_selection.currentText(),
            "camera": self.camera_profile.currentText(),
            "mode": self.flight_mode.currentText(),
            "altitude_m": float(self.altitude.value()),
            "overlap_pct": float(self.overlap.value()),
            "speed_m_s": float(self.speed.value()),
            "return_to_home": self.return_to_home.isChecked(),
            "obstacle_avoidance": self.obstacle_avoidance.isChecked(),
            "emergency_landing_zone": self.emergency_zone.currentText(),
        }


class ValidationPanel(Panel):
    saveRequested = pyqtSignal()
    simulateRequested = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(self):
        super().__init__("Mission Validation")
        self.check_body = QVBoxLayout()
        self.check_body.setSpacing(8)
        self.layout.addLayout(self.check_body)
        self.layout.addSpacing(12)
        self.layout.addWidget(_section("Mission Output Summary"))
        self.output_labels: dict[str, QLabel] = {}
        for label, value in [("Expected Images", "--"), ("Video Duration", "--"), ("Thermal Captures", "--"), ("AI Defect Mode", "Enabled")]:
            row = QHBoxLayout()
            row.addWidget(_muted(label), stretch=1)
            val = _label(value)
            self.output_labels[label] = val
            row.addWidget(val)
            self.layout.addLayout(row)
        self.layout.addSpacing(8)
        self.layout.addWidget(_section("Mission Risk Summary"))
        self.risk_labels: dict[str, QLabel] = {}
        for label, value in [
            ("Risk", "Low"),
            ("Weather", "12 km/h wind"),
            ("Storage", "--"),
            ("AI Model", "Structural v1"),
        ]:
            row = QHBoxLayout()
            row.addWidget(_muted(label), stretch=1)
            val = _label(value)
            self.risk_labels[label] = val
            row.addWidget(val)
            self.layout.addLayout(row)
        self.layout.addStretch(1)
        save = _button("Save Mission", "primary")
        sim = _button("Simulate Mission", "secondary")
        sim.setObjectName("modernOutline")
        clear = _button("Clear Plan", "danger")
        save.clicked.connect(self.saveRequested)
        sim.clicked.connect(self.simulateRequested)
        clear.clicked.connect(self.clearRequested)
        self.layout.addWidget(save)
        self.layout.addWidget(sim)
        divider = QLabel("OR")
        divider.setObjectName("modernMuted")
        divider.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(divider)
        self.layout.addWidget(clear)
        self.set_checks([])

    def set_checks(self, checks: list[Any]) -> None:
        _clear_layout(self.check_body)
        if not checks:
            checks = [
                {"label": "Flight path complete", "ok": False, "level": "idle", "message": "Generate or save a mission."},
                {"label": "Battery sufficient", "ok": False, "level": "idle", "message": "Waiting for estimate."},
                {"label": "Emergency landing zone assigned", "ok": False, "level": "idle", "message": "Select a landing zone."},
            ]
        for check in checks:
            data = check.to_dict() if hasattr(check, "to_dict") else dict(check)
            ok = bool(data.get("ok"))
            level = str(data.get("level") or ("ready" if ok else "warning"))
            chip_level = "ready" if ok and level == "ok" else ("error" if level == "error" else ("warning" if level == "warning" else "idle"))
            row = QHBoxLayout()
            row.addWidget(StatusChip("OK" if ok else "!", chip_level))
            row.addWidget(_label(str(data.get("label", "Check"))), stretch=1)
            row.addWidget(StatusChip(str(data.get("message", "OK" if ok else "Review"))[:28], chip_level))
            self.check_body.addLayout(row)

    def set_output_summary(self, summary: dict[str, Any]) -> None:
        waypoints = int(summary.get("waypoints", 0) or 0)
        time_min = float(summary.get("estimated_time_min", 0.0) or 0.0)
        camera = str(summary.get("camera", "") or "Enabled")
        self.output_labels["Expected Images"].setText(f"{max(0, waypoints):,}")
        self.output_labels["Video Duration"].setText(f"{time_min:.0f} min" if time_min else "--")
        self.output_labels["Thermal Captures"].setText(f"{max(0, waypoints // 4):,}")
        self.output_labels["AI Defect Mode"].setText("Enabled" if camera else "Ready")
        storage_gb = max(0.2, waypoints * 9.5 / 1024.0)
        battery = min(100.0, max(10.0, time_min / 22.0 * 100.0)) if time_min else 0.0
        self.risk_labels["Risk"].setText("Low" if battery < 75 else "Review")
        self.risk_labels["Weather"].setText("Wind 12 km/h NE")
        self.risk_labels["Storage"].setText(f"{storage_gb:.1f} GB est.")
        self.risk_labels["AI Model"].setText("Structural + thermal")


class MissionPlannerPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.current_save: Any | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 12, 24, 24)
        root.setSpacing(12)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.config = MissionConfigPanel()
        left_layout.addWidget(self.config)
        left_scroll = ScrollPanel(left_content)
        left_scroll.setFixedWidth(310)
        root.addWidget(left_scroll)

        self.map_canvas = MissionMapCanvas()
        self.map_canvas.activeToolChanged.connect(lambda name: self.feedback(f"Map tool active: {name}"))
        root.addWidget(self.map_canvas, stretch=1)

        self.validation = ValidationPanel()
        self.validation.setFixedWidth(340)
        self.validation.saveRequested.connect(self._save_mission)
        self.validation.simulateRequested.connect(self._simulate_mission)
        self.validation.clearRequested.connect(self._clear_plan)
        root.addWidget(self.validation)
        self._generate_plan(show_errors=False)

    def _manager(self):
        from core.mission_manager import MissionManager

        return MissionManager(project_root=_project_root(self.session), project_id=_project_id(self.session), drone=self.session.drone_client)

    def _request(self):
        from core.mission_manager import MissionPlanRequest

        vals = self.config.values()
        mode = _mode_from_label(str(vals["mode"]))
        polygon = _default_polygon_lonlat()
        extra: dict[str, Any] = {}
        if mode in {"orbit"}:
            extra["orbit_center_lonlat"] = polygon[0]
            extra["orbit_radius_m"] = 45.0
        elif mode in {"linear_inspection", "waypoints"}:
            extra["linear_path_lonlat"] = polygon[:3]
            extra["waypoint_path_lonlat"] = polygon[:4]
        return MissionPlanRequest(
            mission_name=str(vals["mission_name"]),
            polygon_lonlat=polygon,
            altitude_m=float(vals["altitude_m"]),
            front_overlap_pct=float(vals["overlap_pct"]),
            side_overlap_pct=max(20.0, float(vals["overlap_pct"]) - 10.0),
            speed_m_s=float(vals["speed_m_s"]),
            mode=mode,
            camera=_camera_from_label(str(vals["camera"])),
            drone_id=str(vals["drone"]).split(" - ")[0],
            asset_type=str(vals["asset_type"]),
            inspection_type=str(vals["inspection_type"]),
            emergency_landing_zone=str(vals["emergency_landing_zone"]),
            weather_safe=True,
            extra=extra,
        )

    def _generate_plan(self, show_errors: bool = True):
        try:
            manager = self._manager()
            req = self._request()
            plan = manager.generate_plan(req)
            checks = manager.validate_plan(
                plan,
                emergency_landing_zone=req.emergency_landing_zone,
                weather_safe=req.weather_safe,
            )
            summary = manager.summarize_plan(plan)
            summary["camera"] = req.camera
            self.session.publish_plan(plan)
            self.validation.set_checks(checks)
            self.validation.set_output_summary(summary)
            self.map_canvas.set_plan_summary(summary)
            return manager, plan, checks, summary
        except Exception as exc:
            self.validation.set_checks([{"label": "Mission generation", "ok": False, "level": "error", "message": str(exc)}])
            if show_errors:
                QMessageBox.critical(self, "Mission Planning Error", str(exc))
            return None

    def _save_mission(self) -> None:
        generated = self._generate_plan(show_errors=True)
        if not generated:
            return
        manager, plan, _checks, _summary = generated
        mission_name = self.config.values()["mission_name"]
        try:
            save = manager.save_mission(plan, mission_name=mission_name)
            self.current_save = save
            try:
                self.session.save_mission_version(mission_name=str(mission_name), note=f"Saved to {save.output_dir}")
            except Exception:
                pass
            self.feedback(f"Mission saved: {Path(save.summary_path).name}")
            QMessageBox.information(self, "Mission Saved", f"Mission files written to:\n{save.output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Mission Save Error", str(exc))

    def _simulate_mission(self) -> None:
        generated = self._generate_plan(show_errors=True)
        if not generated:
            return
        _manager, plan, checks, summary = generated
        blockers = [c for c in checks if not getattr(c, "ok", False) and getattr(c, "level", "") == "error"]
        if blockers:
            self.feedback("Simulation blocked by validation errors.")
            return
        self.feedback(
            f"Simulation ready: {len(plan.waypoints)} waypoints, "
            f"{float(summary.get('estimated_time_min', 0.0)):.1f} min estimate."
        )

    def _clear_plan(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear Plan",
            "Clear the current mission plan and drawing layers?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.session.current_plan = None
            self.session.missionPlanChanged.emit(None)
            self.validation.set_checks([])
            self.validation.set_output_summary({})
            self.map_canvas.set_plan_summary({})
            self.feedback("Mission plan cleared.")


class TelemetryPanel(Panel):
    def __init__(self):
        super().__init__("Drone Telemetry")
        self.values: dict[str, QLabel] = {}
        for label, value in [
            ("Battery", "--"),
            ("GPS Lock", "--"),
            ("Altitude", "--"),
            ("Speed", "--"),
            ("Signal Strength", "--"),
            ("Storage Remaining", "128 GB"),
            ("Camera Status", "Ready"),
            ("Temperature", "--"),
            ("Wind Estimate", "--"),
        ]:
            row = QHBoxLayout()
            row.addWidget(_muted(label), stretch=1)
            val = _label(value)
            self.values[label] = val
            row.addWidget(val)
            self.layout.addLayout(row)
        self.layout.addStretch(1)

    def set_telemetry(self, telem: Any) -> None:
        try:
            self.values["Battery"].setText(f"{float(getattr(telem, 'battery_pct', 0.0)):.0f}%")
            self.values["GPS Lock"].setText("RTK Fixed" if int(getattr(telem, "gps_fix", 0)) >= 4 else "Searching")
            self.values["Altitude"].setText(f"{float(getattr(telem, 'altitude_rel_m', 0.0)):.1f} m")
            self.values["Speed"].setText(f"{float(getattr(telem, 'speed_mps', 0.0)):.1f} m/s")
            self.values["Signal Strength"].setText(f"{float(getattr(telem, 'link_quality_pct', 0.0)):.0f}%")
            self.values["Temperature"].setText(f"{float(getattr(telem, 'temperature_c', 0.0)):.0f} C" if hasattr(telem, "temperature_c") else "--")
        except Exception:
            pass


class VideoFeedPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("videoFeed")
        self.setMinimumHeight(320)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.addStretch(1)
        title = QLabel("LIVE RGB FEED - H20T")
        title.setObjectName("videoTitle")
        root.addWidget(title, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        root.addStretch(1)
        controls = QHBoxLayout()
        for text in ["Snapshot", "Start Recording", "Pause Stream"]:
            controls.addWidget(_button(text, "ghost"))
        for text in ["Thermal View", "AI Overlay", "Defect Boxes"]:
            chk = QCheckBox(text)
            chk.setChecked(text != "Thermal View")
            controls.addWidget(chk)
        controls.addStretch(1)
        root.addLayout(controls)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0, QColor("#1e2b32"))
        grad.setColorAt(1, QColor("#080d13"))
        p.fillRect(self.rect(), grad)
        p.setPen(QPen(QColor("#e9f1ff"), 2))
        cx = self.width() / 2
        cy = self.height() / 2
        p.drawLine(QPointF(cx - 80, cy + 44), QPointF(cx + 80, cy - 36))
        p.drawEllipse(QPointF(cx, cy), 36, 36)
        p.setPen(QPen(QColor(RED), 2))
        p.drawRect(QRectF(cx + 90, cy - 50, 130, 90))


class LiveInspectionPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.flight_session: Any | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        self.telemetry = TelemetryPanel()
        root.addWidget(self.telemetry, stretch=1)
        center = QVBoxLayout()
        center.addWidget(_page_title("Live Inspection", "Operator control room for active missions."))
        center.addWidget(VideoFeedPanel(), stretch=2)
        map_panel = Panel("Live Map")
        map_panel.layout.addWidget(MiniMapPreview(), stretch=1)
        center.addWidget(map_panel, stretch=1)
        root.addLayout(center, stretch=3)

        right = Panel("Mission Controls")
        self.state = StatusChip("Ready", "ready")
        right.layout.addWidget(self.state)
        self.control_buttons: dict[str, QPushButton] = {}
        for text in ["Start Mission", "Pause Mission", "Resume Mission", "Return Home"]:
            b = _button(text, "primary" if text == "Start Mission" else "secondary")
            b.clicked.connect(lambda _=False, t=text: self._control(t))
            self.control_buttons[text] = b
            right.layout.addWidget(b)
        emergency = _button("Emergency Land", "danger")
        emergency.clicked.connect(self._emergency)
        right.layout.addSpacing(12)
        right.layout.addWidget(emergency)
        right.layout.addSpacing(14)
        right.layout.addWidget(_section("Alerts"))
        for level, text in [("info", "Route progress nominal."), ("warning", "Wind gusts increasing near return leg."), ("danger", "No-fly buffer warning at waypoint 2.")]:
            chip = StatusChip(level.title(), "error" if level == "danger" else level)
            row = QHBoxLayout()
            row.addWidget(chip)
            row.addWidget(_label(text), stretch=1)
            right.layout.addLayout(row)
        right.layout.addStretch(1)
        root.addWidget(right, stretch=1)
        self.session.telemetryUpdated.connect(self.telemetry.set_telemetry)
        self.session.droneConnected.connect(self._on_drone_connected)
        self.session.missionPlanChanged.connect(lambda _plan: self._refresh_buttons())
        self._refresh_buttons()

    def _control(self, text: str) -> None:
        try:
            if text == "Start Mission":
                self._start_mission()
                return
            if text == "Pause Mission":
                from core.flight import get_flight_manager

                result = get_flight_manager().pause_flight()
            elif text == "Resume Mission":
                from core.flight import get_flight_manager

                result = get_flight_manager().resume_flight()
            elif text == "Return Home":
                from core.flight import get_flight_manager

                result = get_flight_manager().trigger_rth()
            else:
                return
            level = "running" if result.success else "warning"
            self.state.set_text_level(text.replace(" Mission", ""), level)
            self.feedback(f"{result.command}: {result.message}")
        except Exception as exc:
            QMessageBox.warning(self, "Flight Command", str(exc))

    def _start_mission(self) -> None:
        if self.session.current_plan is None:
            QMessageBox.warning(self, "Live Inspection", "Save or simulate a mission in Mission Planner first.")
            return
        if not self.session.drone_client.is_connected():
            if not self.session.connect_drone("mock://", "mock"):
                QMessageBox.warning(self, "Live Inspection", "Unable to connect mock drone.")
                return
        reply = QMessageBox.question(
            self,
            "Start Mission",
            "Upload the current mission, run preflight checks, and start the drone mission?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from core.mission_manager import MissionManager

            manager = MissionManager(
                project_root=_project_root(self.session),
                project_id=_project_id(self.session),
                drone=self.session.drone_client,
            )
            manager.current_plan = self.session.current_plan
            upload = manager.upload_mission(self.session.current_plan)
            if not upload.success:
                raise RuntimeError(upload.message)
            report = manager.run_preflight(
                self.session.current_plan,
                mission_uploaded=True,
                emergency_landing_zone="E1 - North Field",
                weather_acknowledged=True,
            )
            self.flight_session = manager.start_flight(self.session.current_plan, require_preflight=False)
            self.state.set_text_level("Flying", "running")
            self.feedback(f"Mission started. Preflight checks: {len(report.checks)}")
            self._refresh_buttons()
        except Exception as exc:
            QMessageBox.critical(self, "Start Mission Failed", str(exc))

    def _on_drone_connected(self, connected: bool) -> None:
        self.state.set_text_level("Ready" if connected else "No Drone", "ready" if connected else "idle")
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        has_plan = self.session.current_plan is not None
        connected = bool(self.session.drone_client.is_connected())
        self.control_buttons.get("Start Mission", QPushButton()).setEnabled(has_plan)
        for key in ["Pause Mission", "Resume Mission", "Return Home"]:
            self.control_buttons.get(key, QPushButton()).setEnabled(connected)

    def _emergency(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Emergency Land Confirmation")
        lay = QVBoxLayout(dlg)
        lay.addWidget(_page_title("Emergency Land", "This immediately diverts Matrice 350 RTK to the selected landing zone."))
        lay.addWidget(_label("Drone: Matrice 350 RTK"))
        lay.addWidget(_label("Current altitude: 82 m"))
        lay.addWidget(_label("Battery: 78%"))
        confirm = QLineEdit()
        confirm.setPlaceholderText("Type CONFIRM")
        lay.addWidget(confirm)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        land = buttons.addButton("Emergency Land", QDialogButtonBox.ButtonRole.AcceptRole)
        land.setObjectName("modernDanger")
        land.setEnabled(False)
        confirm.textChanged.connect(lambda t: land.setEnabled(t.strip().upper() == "CONFIRM"))
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                from core.flight import get_flight_manager

                result = get_flight_manager().abort_flight(reason="operator_emergency_land")
                self.state.set_text_level("Emergency", "error")
                self.feedback(f"Emergency command: {result.message}")
            except Exception as exc:
                QMessageBox.critical(self, "Emergency Land", str(exc))


@dataclass
class MediaItem:
    name: str
    media_type: str
    timestamp: str
    location: str
    defect: str
    confidence: str
    status: str
    file_path: str = ""
    thumbnail_path: str = ""
    image_id: str = ""
    dataset_id: str = ""
    metadata: dict[str, Any] | None = None


class MediaCard(QFrame):
    clicked = pyqtSignal(object)

    def __init__(self, item: MediaItem):
        super().__init__()
        self.item = item
        self.setObjectName("mediaCard")
        self.setMinimumHeight(138)
        lay = QVBoxLayout(self)
        thumb = QLabel(item.media_type)
        thumb.setObjectName("mediaThumb")
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setMinimumHeight(64)
        if item.thumbnail_path and Path(item.thumbnail_path).exists():
            pix = _pixmap_from_path(item.thumbnail_path, QSize(220, 86))
            if not pix.isNull():
                thumb.setPixmap(pix)
        lay.addWidget(thumb)
        lay.addWidget(_label(item.timestamp))
        row = QHBoxLayout()
        row.addWidget(StatusChip(item.defect, "warning" if item.defect != "None" else "idle"))
        row.addWidget(_muted(item.confidence))
        lay.addLayout(row)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self.item)
        super().mousePressEvent(event)


class FilterPanel(Panel):
    changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__("Filters")
        self.filters: dict[str, QComboBox] = {}
        options = {
            "Mission": ["All Missions", "Turbine T-17", "Bridge B12", "Solar Block C"],
            "Date Range": ["Last 7 days", "Last 30 days", "This quarter"],
            "Media Type": ["All Media", "Photos", "Video", "Thermal"],
            "Defect Type": ["Any Defect", "Crack", "Corrosion", "Hotspot", "Delamination"],
            "Severity": ["Any Severity", "Critical", "High", "Medium", "Low"],
            "Drone": ["Any Drone", "Matrice 350 RTK", "Mavic 3 Enterprise"],
        }
        for label, vals in options.items():
            self.layout.addWidget(_label(label))
            combo = _combo(vals)
            combo.currentTextChanged.connect(self._emit)
            self.filters[label] = combo
            self.layout.addWidget(combo)
        self.layout.addWidget(_label("AI Confidence"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(72)
        slider.valueChanged.connect(lambda _v: self._emit())
        self.conf_slider = slider
        self.layout.addWidget(slider)
        self.layout.addStretch(1)

    def _emit(self) -> None:
        values = {k: v.currentText() for k, v in self.filters.items()}
        values["Confidence"] = f">= {self.conf_slider.value()}%"
        self.changed.emit(values)


class MediaReviewPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.items: list[MediaItem] = []
        self.current_item: MediaItem | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        self.filters = FilterPanel()
        self.filters.setFixedWidth(260)
        self.filters.changed.connect(self._update_chips)
        root.addWidget(self.filters)
        mid = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.addWidget(_page_title("Media Review", "Review photos, video, thermal captures, and AI detections."), stretch=1)
        import_btn = _button("Upload Inspection Media", "primary")
        import_btn.clicked.connect(self._import_media)
        refresh_btn = _button("Refresh", "secondary")
        refresh_btn.clicked.connect(self._refresh_media)
        title_row.addWidget(refresh_btn)
        title_row.addWidget(import_btn)
        mid.addLayout(title_row)
        self.chips = QHBoxLayout()
        mid.addLayout(self.chips)
        grid_wrap = QWidget()
        self.grid = QGridLayout(grid_wrap)
        self.grid.setSpacing(16)
        mid.addWidget(ScrollPanel(grid_wrap), stretch=1)
        root.addLayout(mid, stretch=2)
        self.detail = Panel("Selected Media")
        self.detail.setFixedWidth(340)
        root.addWidget(self.detail)
        self.session.datasetImported.connect(lambda _row: self._refresh_media())
        self.session.activeImageChanged.connect(lambda _path: self._refresh_media(select_path=_path))
        self._refresh_media()

    def _project_root_id(self) -> tuple[Path, str]:
        return _project_root(self.session), _project_id(self.session)

    def _load_media_items(self) -> list[MediaItem]:
        root, _pid = self._project_root_id()
        out: list[MediaItem] = []
        for row in self.session.list_datasets():
            meta = _row_metadata(row)
            dataset_id = str(meta.get("backend_dataset_id", "") or "")
            if dataset_id:
                try:
                    from core.data_library import get_image_assets

                    for asset in get_image_assets(root, dataset_id, page=0, page_size=200):
                        path = Path(asset.file_path)
                        out.append(
                            MediaItem(
                                name=path.name,
                                media_type="THERMAL" if "therm" in path.stem.lower() else "PHOTO",
                                timestamp=str(asset.captured_at or row.get("captured_at", "")),
                                location=str(row.get("name", "") or "Imported Dataset"),
                                defect="None",
                                confidence="0%",
                                status="Unreviewed",
                                file_path=str(path),
                                thumbnail_path=str(asset.thumbnail_path or ""),
                                image_id=asset.id,
                                dataset_id=dataset_id,
                                metadata=asset.to_dict(),
                            )
                        )
                except Exception as exc:
                    self.feedback(f"Media backend read failed: {exc}")
            else:
                folder = Path(str(row.get("path", "")))
                if folder.exists():
                    for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS)[:200]:
                        out.append(
                            MediaItem(
                                name=path.name,
                                media_type="PHOTO",
                                timestamp=str(row.get("captured_at", "")),
                                location=str(row.get("name", "") or folder.name),
                                defect="None",
                                confidence="0%",
                                status="Unreviewed",
                                file_path=str(path),
                                dataset_id="",
                            )
                        )
        return out

    def _refresh_media(self, select_path: str = "") -> None:
        self.items = self._load_media_items()
        _clear_layout(self.grid)
        if not self.items:
            empty = QLabel("Upload inspection media or run a mission to start review")
            empty.setObjectName("selectedPreview")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setMinimumHeight(180)
            self.grid.addWidget(empty, 0, 0)
            self._render_empty_detail()
            return
        selected = self.items[0]
        for idx, item in enumerate(self.items):
            card = MediaCard(item)
            card.clicked.connect(self._select_media)
            self.grid.addWidget(card, idx // 3, idx % 3)
            if select_path and Path(item.file_path) == Path(select_path):
                selected = item
        self._select_media(selected)

    def _import_media(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select inspection media folder")
        if not path:
            return
        folder = Path(path)
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        if not files:
            QMessageBox.warning(self, "Media Import", "No supported image files found in that folder.")
            return
        try:
            row = self.session.import_dataset(folder=folder, name=folder.name, metadata={"dataset_type": "rgb", "image_count": len(files)})
            meta = _row_metadata(row)
            err = meta.get("backend_import_error")
            self.feedback(f"Imported {meta.get('image_count', len(files))} media files." + (f" Backend warning: {err}" if err else ""))
            self._refresh_media()
        except Exception as exc:
            QMessageBox.critical(self, "Media Import Error", str(exc))

    def _update_chips(self, values: dict) -> None:
        while self.chips.count():
            item = self.chips.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for key, value in values.items():
            if value.startswith("All") or value.startswith("Any"):
                continue
            self.chips.addWidget(StatusChip(f"{key}: {value}", "info"))
        self.chips.addStretch(1)

    def _select_media(self, item: MediaItem) -> None:
        self.current_item = item
        if item.file_path and Path(self.session.active_image_path or "") != Path(item.file_path):
            self.session.set_active_image(item.file_path)
        while self.detail.layout.count() > 1:
            child = self.detail.layout.takeAt(1)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    sub = child.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        preview = QLabel(item.media_type)
        preview.setObjectName("selectedPreview")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumHeight(180)
        source = item.thumbnail_path if item.thumbnail_path and Path(item.thumbnail_path).exists() else item.file_path
        if source and Path(source).exists():
            pix = _pixmap_from_path(source, QSize(300, 180))
            if not pix.isNull():
                preview.setPixmap(pix)
        self.detail.layout.addWidget(preview)
        meta = item.metadata or {}
        for label, value in [
            ("Timestamp", item.timestamp),
            ("Location", item.location),
            ("GPS", f"{meta.get('gps_lat')}, {meta.get('gps_lon')}" if meta.get("gps_lat") is not None else "Not embedded"),
            ("Altitude", f"{meta.get('altitude_m')} m" if meta.get("altitude_m") is not None else "Unknown"),
            ("Camera", str(meta.get("camera_model") or "Unknown")),
            ("AI Detections", f"{item.defect} ({item.confidence})"),
        ]:
            row = QHBoxLayout()
            row.addWidget(_muted(label), stretch=1)
            row.addWidget(_label(value))
            self.detail.layout.addLayout(row)
        self.detail.layout.addWidget(_label("Manual Notes"))
        self.detail.layout.addWidget(QPlainTextEdit("Inspect blade root seam during repair window."))
        self.detail.layout.addWidget(_section("Annotation Tools"))
        for text in ["Draw Box", "Draw Polygon", "Add Marker", "Add Comment", "Mark False Positive", "Confirm Defect"]:
            b = _button(text, "secondary" if text != "Confirm Defect" else "primary")
            b.clicked.connect(lambda _=False, t=text, i=item: self._annotate_media(i, t))
            self.detail.layout.addWidget(b)

    def _render_empty_detail(self) -> None:
        while self.detail.layout.count() > 1:
            child = self.detail.layout.takeAt(1)
            if child.widget():
                child.widget().deleteLater()
        self.detail.layout.addWidget(_muted("No media selected."))

    def _annotate_media(self, item: MediaItem, action: str) -> None:
        if not item.file_path:
            return
        try:
            from core.annotations import create_annotation

            annotation_type = {
                "Draw Box": "rectangle",
                "Draw Polygon": "polygon",
                "Add Marker": "point",
                "Add Comment": "free_text",
                "Mark False Positive": "false_positive",
                "Confirm Defect": "defect_confirm",
            }.get(action, "free_text")
            ann = create_annotation(
                project_root=_project_root(self.session),
                project_id=_project_id(self.session),
                source_type="image",
                source_id=item.file_path,
                annotation_type=annotation_type,
                geometry={"ui_action": action, "bbox": [0.25, 0.25, 0.5, 0.5]},
                label=action,
                severity="medium" if action == "Confirm Defect" else None,
                note=f"{action} from Media Review",
            )
            self.feedback(f"Annotation saved: {ann.label}")
        except Exception as exc:
            QMessageBox.critical(self, "Annotation Error", str(exc))


class DefectAnalysisPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.defect_rows: list[dict[str, Any]] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(_page_title("Defect Analysis", "Structured defect triage with AI confidence, maps, trends, and repair notes."), stretch=1)
        for opts in [
            ["All Missions", "Turbine T-17"],
            ["All Assets", "WTG-17"],
            ["All Types", "Crack", "Corrosion"],
            ["All Severity", "Critical", "High"],
            ["All Status", "Open", "Assigned"],
        ]:
            top.addWidget(_combo(opts))
        run_btn = _button("Run Analysis", "primary")
        run_btn.clicked.connect(self._run_analysis)
        top.addWidget(run_btn)
        root.addLayout(top)

        summary = QHBoxLayout()
        for args in [
            ("Critical", "5", "Open", "error"),
            ("High", "12", "Assigned", "warning"),
            ("Avg Confidence", "86%", "AI", "running"),
            ("Repair Orders", "4", "Draft", "idle"),
        ]:
            summary.addWidget(MetricCard(*args))
        root.addLayout(summary)

        body = QHBoxLayout()
        left = QVBoxLayout()
        map_panel = Panel("Defect Map")
        map_panel.layout.addWidget(MiniMapPreview(), stretch=1)
        left.addWidget(map_panel, stretch=1)
        charts = QHBoxLayout()
        for title in ["Trend Chart", "AI Confidence Distribution"]:
            p = Panel(title)
            bar = QProgressBar()
            bar.setValue(72 if "Confidence" in title else 48)
            p.layout.addWidget(bar)
            p.layout.addWidget(_muted("Updated from the selected mission filters."))
            charts.addWidget(p)
        left.addLayout(charts)

        table_panel = Panel("Defect Table")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Defect ID", "Type", "Severity", "Asset Area", "Confidence", "Status", "Assigned To", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(lambda r, _c: self._show_defect(self.defect_rows[r]) if 0 <= r < len(self.defect_rows) else None)
        table_panel.layout.addWidget(self.table)
        left.addWidget(table_panel, stretch=1)
        body.addLayout(left, stretch=3)

        self.drawer = Panel("Defect Detail")
        self.drawer.setFixedWidth(360)
        body.addWidget(self.drawer, stretch=0)
        root.addLayout(body, stretch=1)
        self.session.datasetImported.connect(lambda _row: self._refresh_defects())
        self._refresh_defects()

    def _latest_defect_summary(self) -> dict[str, Any] | None:
        root = _project_root(self.session)
        summaries = sorted((root / "analysis" / "defects").glob("*/defects.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in summaries:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["_summary_path"] = str(path)
                    return data
            except Exception:
                continue
        return None

    def _refresh_defects(self) -> None:
        data = self._latest_defect_summary()
        defects = data.get("defects", []) if isinstance(data, dict) else []
        self.defect_rows = [d for d in defects if isinstance(d, dict)]
        self.table.setRowCount(len(self.defect_rows))
        for r, d in enumerate(self.defect_rows):
            vals = [
                str(d.get("id", ""))[:8],
                str(d.get("defect_type", "")),
                str(d.get("severity", "")).title(),
                Path(str(d.get("image_path", ""))).name,
                f"{float(d.get('confidence', 0.0)) * 100.0:.0f}%",
                "Open",
                "Unassigned",
                "View",
            ]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(val))
        if self.defect_rows:
            self._show_defect(self.defect_rows[0])
        else:
            self._show_empty_defect()

    def _active_backend_dataset_id(self) -> str:
        for row in self.session.list_datasets():
            meta = _row_metadata(row)
            dataset_id = str(meta.get("backend_dataset_id", "") or "")
            if dataset_id:
                return dataset_id
        return ""

    def _run_analysis(self) -> None:
        dataset_id = self._active_backend_dataset_id()
        if not dataset_id:
            QMessageBox.warning(self, "Defect Analysis", "Import inspection media before running defect analysis.")
            return
        try:
            from core.defect_engine import DefectDetectionConfig, export_defect_table, run_defect_detection

            result = run_defect_detection(
                project_root=_project_root(self.session),
                dataset_id=dataset_id,
                config=DefectDetectionConfig(mode="classical", threshold=0.25, min_area_px=30),
            )
            export_defect_table(result, "csv")
            self.session.publish_run_artifacts({"source": "defect_analysis", "run_dir": result.output_dir, "summary_path": str(Path(result.output_dir) / "defects.json")})
            self.feedback(f"Defect analysis complete: {len(result.defects)} defects across {result.image_count} images.")
            self._refresh_defects()
        except Exception as exc:
            QMessageBox.critical(self, "Defect Analysis Error", str(exc))

    def _show_empty_defect(self) -> None:
        while self.drawer.layout.count() > 1:
            child = self.drawer.layout.takeAt(1)
            if child.widget():
                child.widget().deleteLater()
        self.drawer.layout.addWidget(_muted("No defects detected for the selected filters."))

    def _show_defect(self, row: dict[str, Any]) -> None:
        while self.drawer.layout.count() > 1:
            child = self.drawer.layout.takeAt(1)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    sub = child.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        preview = QLabel("AI DETECTION OVERLAY")
        preview.setObjectName("selectedPreview")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumHeight(180)
        overlay = str(row.get("overlay_path", "") or "")
        image_path = str(row.get("image_path", "") or "")
        source = overlay if overlay and Path(overlay).exists() else image_path
        if source and Path(source).exists():
            pix = _pixmap_from_path(source, QSize(320, 180))
            if not pix.isNull():
                preview.setPixmap(pix)
        self.drawer.layout.addWidget(preview)
        values = [
            str(row.get("id", "")),
            str(row.get("defect_type", "")),
            str(row.get("severity", "")).title(),
            Path(image_path).name,
            f"{float(row.get('confidence', 0.0)) * 100.0:.0f}%",
            "Open",
            "Unassigned",
        ]
        for label, value in zip(["Defect ID", "Type", "Severity", "Area", "Confidence", "Status", "Assigned"], values):
            line = QHBoxLayout()
            line.addWidget(_muted(label), stretch=1)
            line.addWidget(_label(value))
            self.drawer.layout.addLayout(line)
        self.drawer.layout.addWidget(_label("Manual Inspection Notes"))
        self.drawer.layout.addWidget(QPlainTextEdit("Recommend close-up verification and repair planning."))
        self.drawer.layout.addWidget(_label("Severity"))
        severity = str(row.get("severity", "medium")).lower()
        self.drawer.layout.addWidget(_combo(["Critical", "High", "Medium", "Low"], {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(severity, 2)))
        self.drawer.layout.addWidget(_label("Status"))
        self.drawer.layout.addWidget(_combo(["Open", "Assigned", "In Repair", "Resolved"], 0))
        export = _button("Export Defect", "primary")
        export.clicked.connect(lambda _=False, d=row: self._export_defect(d))
        self.drawer.layout.addWidget(export)

    def _export_defect(self, row: dict[str, Any]) -> None:
        try:
            out_dir = _project_root(self.session) / "analysis" / "defects" / "exports"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{str(row.get('id', 'defect'))}.json"
            out.write_text(json.dumps(row, indent=2), encoding="utf-8")
            self.feedback(f"Defect exported: {out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Defect", str(exc))


class ReportsPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.latest_report_path = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(_page_title("Reports", "Professional inspection report generation and export."))
        top.addStretch(1)
        top.addWidget(_combo(["Executive Structural", "Thermal Summary", "Repair Package"]))
        top.addWidget(_combo(["Turbine T-17", "Bridge B12", "Solar Block C"]))
        top.addWidget(_combo(["Last 7 days", "Last 30 days"]))
        create = _button("Create Report", "primary")
        create.clicked.connect(self._generate_report)
        top.addWidget(create)
        root.addLayout(top)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Report Name", "Mission", "Asset", "Generated", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_report_row(r))
        root.addWidget(self.table, stretch=1)

        builder = QHBoxLayout()
        sections = Panel("Sections")
        for s in ["Executive Summary", "Mission Details", "Drone Details", "Inspection Coverage", "Detected Defects", "Media Evidence", "Risk Assessment", "Recommendations", "Appendix"]:
            sections.layout.addWidget(_label(s))
        builder.addWidget(sections, stretch=1)
        preview = Panel("Report Preview")
        actions = QHBoxLayout()
        for text in ["Generate PDF", "Export DOCX", "Share Report", "Save Draft"]:
            b = _button(text, "primary" if text == "Generate PDF" else "secondary")
            b.clicked.connect(lambda _=False, t=text: self._report_action(t))
            actions.addWidget(b)
        preview.layout.addLayout(actions)
        doc = QTextEdit()
        doc.setPlainText("WTG-17 inspection report preview\n\nExecutive Summary\nMission Details\nDetected Defects\nRecommendations")
        preview.layout.addWidget(doc)
        builder.addWidget(preview, stretch=3)
        settings = Panel("Report Settings")
        for label, opts in [("Template", ["Executive Structural", "Insurance Package"]), ("Evidence Density", ["Compact", "Detailed"]), ("Risk Model", ["Industrial Default", "Conservative"])]:
            settings.layout.addWidget(_label(label))
            settings.layout.addWidget(_combo(opts))
        builder.addWidget(settings, stretch=1)
        root.addLayout(builder, stretch=2)
        self.session.reportSaved.connect(lambda _row: self._refresh_reports())
        self._refresh_reports()

    def _report_context(self) -> dict[str, Any]:
        project = self.session.active_project if isinstance(self.session.active_project, dict) else {}
        datasets = self.session.list_datasets()
        reports = self.session.list_reports()
        missions = self.session.list_mission_versions()
        defects = []
        root = _project_root(self.session)
        latest = sorted((root / "analysis" / "defects").glob("*/defects.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if latest:
            try:
                defects = json.loads(latest[0].read_text(encoding="utf-8")).get("defects", [])
            except Exception:
                defects = []
        plan = self.session.current_plan
        return {
            "project": project,
            "datasets": datasets,
            "reports": reports,
            "missions": missions,
            "defects": defects,
            "plan": plan,
        }

    def _generate_report(self) -> None:
        try:
            context = self._report_context()
            project = context["project"]
            title = f"{project.get('name', 'Inspection')} Report"
            out_dir = _project_root(self.session) / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = out_dir / f"inspection_report_{stamp}.html"
            plan = context["plan"]
            plan_rows = ""
            if plan is not None:
                plan_rows = (
                    f"<li>Waypoints: {len(plan.waypoints)}</li>"
                    f"<li>Distance: {float(plan.path_distance_m):.1f} m</li>"
                    f"<li>Estimated time: {float(plan.estimated_time_min):.1f} min</li>"
                )
            defect_rows = "\n".join(
                f"<tr><td>{Path(str(d.get('image_path', ''))).name}</td><td>{d.get('defect_type')}</td><td>{d.get('severity')}</td><td>{float(d.get('confidence', 0))*100:.0f}%</td></tr>"
                for d in context["defects"][:200]
                if isinstance(d, dict)
            )
            html_path.write_text(
                f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:36px;color:#172033}}h1{{color:#0b57d0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5e0;padding:8px;text-align:left}}.muted{{color:#65758b}}</style>
</head><body>
<h1>{title}</h1>
<p class='muted'>Generated by Drone Inspection Toolkit.</p>
<h2>Mission Summary</h2><ul>{plan_rows or '<li>No mission saved yet.</li>'}</ul>
<h2>Datasets</h2><ul>{''.join(f"<li>{r.get('name')} - {r.get('path')}</li>" for r in context['datasets']) or '<li>No datasets imported.</li>'}</ul>
<h2>Detected Defects</h2><table><tr><th>Image</th><th>Type</th><th>Severity</th><th>Confidence</th></tr>{defect_rows or '<tr><td colspan=4>No defect run available.</td></tr>'}</table>
<h2>Recommendations</h2><p>Review high severity defects, verify annotations, and export the final package after operator approval.</p>
</body></html>""",
                encoding="utf-8",
            )
            row = self.session.save_report(title=title, report_type="inspection_html", content_path=html_path, metadata={"source": "modern_report_builder"})
            self.latest_report_path = str(html_path)
            self.feedback(f"Report generated: {Path(row.get('content_path', html_path)).name}")
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path.resolve())))
        except Exception as exc:
            QMessageBox.critical(self, "Report Generation Error", str(exc))

    def _refresh_reports(self) -> None:
        rows = self.session.list_reports()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [
                str(row.get("title", "")),
                "Current Mission",
                str((self.session.active_project or {}).get("name", "Asset")),
                str(row.get("created_at", ""))[:10],
                "Ready" if Path(str(row.get("content_path", ""))).exists() else "Missing",
            ]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, row)
                self.table.setItem(r, c, item)

    def _open_report_row(self, row: int) -> None:
        item = self.table.item(row, 0)
        data = item.data(Qt.ItemDataRole.UserRole) if item else None
        path = Path(str(data.get("content_path", ""))) if isinstance(data, dict) else Path()
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _report_action(self, action: str) -> None:
        if action in {"Save Draft", "Generate PDF"}:
            self._generate_report()
            if action == "Generate PDF":
                self._try_export_pdf()
            return
        if action == "Export DOCX":
            self._export_docx()
            return
        if action == "Share Report":
            reports_dir = _project_root(self.session) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(reports_dir.resolve())))

    def _try_export_pdf(self) -> None:
        if not self.latest_report_path:
            return
        try:
            from weasyprint import HTML

            html_path = Path(self.latest_report_path)
            pdf_path = html_path.with_suffix(".pdf")
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            self.session.save_report(
                title=html_path.stem + " PDF",
                report_type="inspection_pdf",
                content_path=pdf_path,
                metadata={"source_html": str(html_path)},
            )
            self.feedback(f"PDF generated: {pdf_path.name}")
        except Exception as exc:
            self.feedback(f"PDF export skipped: {exc}")

    def _export_docx(self) -> None:
        if not self.latest_report_path:
            self._generate_report()
        if not self.latest_report_path:
            return
        try:
            import re
            import zipfile

            html_path = Path(self.latest_report_path)
            text = html_path.read_text(encoding="utf-8", errors="ignore")
            plain = re.sub(r"<[^>]+>", "\n", text)
            plain = "\n".join(line.strip() for line in plain.splitlines() if line.strip())
            docx_path = html_path.with_suffix(".docx")
            body = "".join(
                f"<w:p><w:r><w:t>{line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</w:t></w:r></w:p>"
                for line in plain.splitlines()
            )
            with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("[Content_Types].xml", "<?xml version='1.0' encoding='UTF-8'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/><Default Extension='xml' ContentType='application/xml'/><Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>")
                z.writestr("_rels/.rels", "<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>")
                z.writestr("word/document.xml", f"<?xml version='1.0' encoding='UTF-8'?><w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body>{body}</w:body></w:document>")
            self.session.save_report(
                title=html_path.stem + " DOCX",
                report_type="inspection_docx",
                content_path=docx_path,
                metadata={"source_html": str(html_path)},
            )
            self.feedback(f"DOCX exported: {docx_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "DOCX Export", str(exc))


class DroneFleetPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        root.addWidget(_page_title("Drone Fleet", "Manage drones, sensors, camera profiles, maintenance, and assignments."))
        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        drones = [
            ("Matrice 350 RTK", "DJI M350", "Connected", "94%", "v10.02", "H20T", "WTG-17", "OK"),
            ("Mavic 3 Enterprise", "M3E", "Standby", "88%", "v07.01", "RGB 45 MP", "Bridge B12", "Due"),
            ("Skydio X10", "X10", "Offline", "82%", "v2.4", "Thermal", "Solar C", "OK"),
            ("Phantom 4 RTK", "P4 RTK", "Charging", "76%", "v01.08", "RGB", "Warehouse", "Due"),
        ]
        for idx, drone in enumerate(drones):
            card = Panel(drone[0])
            for label, val in zip(["Model", "Connection", "Battery Health", "Firmware", "Camera", "Last Mission", "Maintenance"], drone[1:]):
                row = QHBoxLayout()
                row.addWidget(_muted(label), stretch=1)
                row.addWidget(_label(val))
                card.layout.addLayout(row)
            actions = QHBoxLayout()
            for text in ["View Details", "Calibrate", "Assign Mission"]:
                b = _button(text, "secondary")
                b.clicked.connect(lambda _=False, t=text, d=drone[0]: self._drone_action(t, d))
                actions.addWidget(b)
            card.layout.addLayout(actions)
            grid.addWidget(card, idx // 2, idx % 2)
        root.addWidget(ScrollPanel(grid_wrap), stretch=1)

    def _drone_action(self, action: str, drone_name: str) -> None:
        try:
            if action == "View Details":
                if not self.session.drone_client.is_connected():
                    self.session.connect_drone("mock://", "mock")
                telem = self.session.drone_client.get_telemetry()
                QMessageBox.information(
                    self,
                    drone_name,
                    f"Connected: {telem.connected}\nBattery: {telem.battery_pct:.0f}%\nGPS fix: {telem.gps_fix}\nMode: {telem.flight_mode}",
                )
                return
            if action == "Calibrate":
                pid = self.session.active_project_id()
                if pid is not None and pid >= 0:
                    self.session.store.append_audit_event(pid, "drone_calibration_requested", {"drone": drone_name})
                self.feedback(f"Calibration workflow recorded for {drone_name}.")
                return
            if action == "Assign Mission":
                if self.session.current_plan is None:
                    QMessageBox.warning(self, "Assign Mission", "Generate a mission before assigning a drone.")
                    return
                self.session.connect_drone("mock://", "mock")
                self.feedback(f"{drone_name} assigned to current mission.")
        except Exception as exc:
            QMessageBox.critical(self, "Drone Fleet", str(exc))


class AssetLibraryPage(QWidget):
    def __init__(self, navigate: Callable[[int], None]):
        super().__init__()
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        assets = Panel("Asset Library")
        for name in ["WTG-17 Wind Turbine", "Bridge B12", "Solar Block C", "North Warehouse Roof", "Pipeline Segment 08"]:
            row = QHBoxLayout()
            row.addWidget(_label(name), stretch=1)
            b = _button("Plan Mission", "secondary")
            b.clicked.connect(lambda _=False: navigate(1))
            row.addWidget(b)
            assets.layout.addLayout(row)
        root.addWidget(assets, stretch=1)
        map_panel = Panel("Asset Boundaries")
        map_panel.layout.addWidget(MiniMapPreview(), stretch=1)
        root.addWidget(map_panel, stretch=2)


class SettingsPage(QWidget):
    def __init__(self, session: AppSession, feedback: Callable[[str], None]):
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.setting_widgets: dict[str, list[tuple[str, QWidget]]] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        root.addWidget(_page_title("Settings", "Workspace, maps, AI models, safety, integrations, users, and appearance."))
        tabs = QTabWidget()
        tab_defs = {
            "General": ["Workspace name", "Units", "Time zone", "Default mission folder"],
            "Map": ["Default map layer", "Imported map management", "No fly zone library", "Asset boundary library"],
            "AI Models": ["Defect detection model", "Thermal anomaly model", "Confidence threshold", "Auto classify defects"],
            "Camera Profiles": ["RGB camera profile", "Thermal camera profile", "Zoom camera profile", "Resolution", "Frame rate", "Capture interval"],
            "Safety": ["Minimum battery return threshold", "Maximum altitude", "Maximum speed", "Geofence enforcement", "Emergency landing behavior", "Obstacle avoidance sensitivity"],
            "Integrations": ["Cloud storage", "GIS system", "Report export destination", "Webhook endpoint"],
            "Users": ["Operators", "Reviewers", "Admins", "Invitations"],
            "Appearance": ["Theme selector", "Compact mode", "Sidebar collapsed by default"],
        }
        for title, fields in tab_defs.items():
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(16, 16, 16, 16)
            self.setting_widgets[title] = []
            for field in fields:
                lay.addWidget(_label(field))
                if "toggle" in field.lower() or field in {"Auto classify defects", "Compact mode", "Sidebar collapsed by default", "Geofence enforcement"}:
                    chk = QCheckBox(field)
                    chk.setChecked(True)
                    lay.addWidget(chk)
                    self.setting_widgets[title].append((field, chk))
                else:
                    widget = _combo(["Default", "Metric", "Imperial"] if field == "Units" else ["Default", "Option A", "Option B"]) if "selector" in field.lower() or "layer" in field.lower() or "Units" in field else QLineEdit()
                    if isinstance(widget, QLineEdit) and field == "Workspace name" and isinstance(self.session.active_project, dict):
                        widget.setText(str(self.session.active_project.get("name", "")))
                    lay.addWidget(widget)
                    self.setting_widgets[title].append((field, widget))
            save = _button("Save Settings", "primary")
            save.clicked.connect(lambda _=False, t=title: self._save_settings(t))
            lay.addWidget(save, alignment=Qt.AlignmentFlag.AlignRight)
            lay.addStretch(1)
            tabs.addTab(page, title)
        root.addWidget(tabs, stretch=1)

    def _widget_value(self, widget: QWidget) -> Any:
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return ""

    def _save_settings(self, section: str) -> None:
        values = {field: self._widget_value(widget) for field, widget in self.setting_widgets.get(section, [])}
        try:
            self.session.store.set_setting(f"modern_settings:{section}", values)
            if section == "General":
                from core.settings import AppSettings, load_app_settings, save_app_settings

                current = load_app_settings()
                units = str(values.get("Units", current.default_units)).lower()
                units = "imperial" if "imperial" in units else "metric"
                save_app_settings(
                    AppSettings(
                        **{
                            **current.to_dict(),
                            "default_units": units,
                            "workspace_root": str(_project_root(self.session).parent),
                        }
                    )
                )
            if section == "Safety":
                from core.settings import DroneProfile, save_drone_profiles

                save_drone_profiles([DroneProfile(name="Default Safety Profile")])
            self.feedback(f"{section} settings saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Settings", str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Inspection Toolkit")
        self.resize(1720, 960)
        self.setMinimumSize(1260, 760)
        self.session = AppSession()

        central = QWidget()
        central.setObjectName("modernCentral")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.topbar = TopBar(self.session)
        self.topbar.settingsRequested.connect(lambda: self.sidebar.set_active(8))
        root.addWidget(self.topbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.sidebar = Sidebar()
        self.sidebar.navigate.connect(self._navigate)
        body.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, stretch=1)
        root.addLayout(body, stretch=1)

        self._build_pages()
        self.sidebar.set_active(1)
        self.session.statusChanged.connect(self._toast)
        self.session.projectChanged.connect(self._project_changed)

    def _build_pages(self) -> None:
        from ui.workspace import WorkflowTemplatesTab
        from ui.measurements_tab import MeasurementsTab
        pages = [
            DashboardPage(self.session, self._navigate, self._toast),
            MissionPlannerPage(self.session, self._toast),
            LiveInspectionPage(self.session, self._toast),
            MediaReviewPage(self.session, self._toast),
            DefectAnalysisPage(self.session, self._toast),
            AssetLibraryPage(self._navigate),
            ReportsPage(self.session, self._toast),
            DroneFleetPage(self.session, self._toast),
            SettingsPage(self.session, self._toast),
            WorkflowTemplatesTab(self.session),
            MeasurementsTab(self.session),
        ]
        for page in pages:
            self.stack.addWidget(page)

    def _navigate(self, index: int) -> None:
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)

    def _toast(self, message: str) -> None:
        self.statusBar().showMessage(message, 4500)

    def _project_changed(self, project: dict[str, Any]) -> None:
        name = project.get("name", "") if isinstance(project, dict) else ""
        self.setWindowTitle(f"Drone Inspection Toolkit - {name}" if name else "Drone Inspection Toolkit")

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.session.store.close()
        except Exception:
            pass
        super().closeEvent(event)
