"""Design system for OpenDroneKit — professional dark inspection theme."""

from __future__ import annotations

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QStyle, QWidget

# ─── Palette constants ─────────────────────────────────────────────────────────
C_BG_DEEP      = "#060e1b"
C_BG_BASE      = "#0a1422"
C_SURFACE_1    = "#0f1c2e"
C_SURFACE_2    = "#152233"
C_SURFACE_3    = "#1b2c3f"
C_SURFACE_4    = "#223347"

C_BORDER_FAINT = "#192535"
C_BORDER_MED   = "#213349"
C_BORDER_BOLD  = "#2c445e"

C_TEXT_1       = "#edf2ff"   # primary
C_TEXT_2       = "#9ab4cc"   # secondary
C_TEXT_3       = "#5a7898"   # muted / helper
C_TEXT_DIS     = "#2e4260"   # disabled

C_BLUE         = "#3b82f6"
C_BLUE_H       = "#2563eb"
C_BLUE_P       = "#1d4ed8"
C_BLUE_BG      = "#0b1e3d"

C_GREEN        = "#22c55e"
C_GREEN_BG     = "#041f10"
C_GREEN_BD     = "#14532d"
C_GREEN_TEXT   = "#86efac"

C_AMBER        = "#f59e0b"
C_AMBER_BG     = "#2c1800"
C_AMBER_BD     = "#6b3a00"
C_AMBER_TEXT   = "#fcd34d"

C_RED          = "#ef4444"
C_RED_BG       = "#2c0808"
C_RED_BD       = "#7f1d1d"
C_RED_TEXT     = "#fca5a5"

C_PURPLE       = "#a78bfa"
C_PURPLE_BG    = "#1e1140"


def standard_icon(widget: QWidget, *candidates: str) -> QIcon:
    app = QApplication.instance()
    style = app.style() if app is not None else widget.style()
    for name in candidates:
        pix = getattr(QStyle.StandardPixmap, name, None)
        if pix is not None:
            return style.standardIcon(pix)
    return QIcon()


def app_stylesheet() -> str:
    return f"""
/* ── Base ─────────────────────────────────────────────────────────────────── */
QWidget {{
    color: {C_TEXT_1};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
    background-color: transparent;
}}
QMainWindow, QWidget#central {{
    background-color: {C_BG_BASE};
}}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
QWidget#sidebar {{
    background-color: {C_BG_DEEP};
    border-right: 1px solid {C_BORDER_MED};
}}
QPushButton#navBtn {{
    background: transparent;
    border: none;
    border-radius: 8px;
    color: {C_TEXT_2};
    font-size: 9.5pt;
    font-weight: 500;
    padding: 9px 12px;
    text-align: left;
}}
QPushButton#navBtn:hover {{
    background: {C_SURFACE_2};
    color: {C_TEXT_1};
}}
QPushButton#navBtn[active="true"] {{
    background: {C_BLUE_BG};
    color: {C_BLUE};
    font-weight: 700;
    border-left: 3px solid {C_BLUE};
    padding-left: 9px;
}}

/* ── App bar ──────────────────────────────────────────────────────────────── */
QWidget#appBar {{
    background-color: {C_BG_DEEP};
    border-bottom: 1px solid {C_BORDER_MED};
}}

/* ── Status bar ───────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {C_BG_DEEP};
    border-top: 1px solid {C_BORDER_MED};
    color: {C_TEXT_2};
    font-size: 8.5pt;
}}
QStatusBar::item {{
    border: none;
}}

/* ── Scroll areas ─────────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {C_SURFACE_1};
    width: 7px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER_BOLD};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C_TEXT_3};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {C_SURFACE_1};
    height: 7px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {C_BORDER_BOLD};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Cards / sections ─────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {C_SURFACE_1};
    border: 1px solid {C_BORDER_MED};
    border-radius: 10px;
    margin-top: 14px;
    padding: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {C_TEXT_2};
    font-size: 9pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QFrame#card {{
    background-color: {C_SURFACE_1};
    border: 1px solid {C_BORDER_MED};
    border-radius: 10px;
}}
QFrame#subCard {{
    background-color: {C_SURFACE_2};
    border: 1px solid {C_BORDER_FAINT};
    border-radius: 7px;
}}

/* ── Inputs ───────────────────────────────────────────────────────────────── */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {C_SURFACE_3};
    border: 1px solid {C_BORDER_MED};
    border-radius: 6px;
    padding: 5px 8px;
    color: {C_TEXT_1};
    selection-background-color: {C_BLUE};
    selection-color: white;
    min-height: 30px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {C_BLUE};
    background-color: {C_SURFACE_4};
}}
QLineEdit:read-only {{
    background-color: {C_SURFACE_2};
    color: {C_TEXT_2};
    border-color: {C_BORDER_FAINT};
}}
QLineEdit::placeholder, QPlainTextEdit::placeholder {{
    color: {C_TEXT_3};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {C_SURFACE_3};
    border: 1px solid {C_BORDER_BOLD};
    selection-background-color: {C_BLUE};
    selection-color: white;
    color: {C_TEXT_1};
    padding: 2px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {C_SURFACE_4};
    border: none;
    width: 16px;
    border-radius: 3px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {C_BORDER_BOLD};
}}
QListWidget, QTreeWidget {{
    background-color: {C_SURFACE_1};
    border: 1px solid {C_BORDER_MED};
    border-radius: 6px;
    padding: 3px;
    color: {C_TEXT_1};
    outline: none;
}}
QListWidget::item, QTreeWidget::item {{
    padding: 5px 6px;
    border-radius: 5px;
    min-height: 26px;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {C_BLUE_BG};
    color: {C_BLUE};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {C_SURFACE_2};
}}
QTreeWidget::branch {{
    background: transparent;
}}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {C_SURFACE_3};
    border: 1px solid {C_BORDER_BOLD};
    border-radius: 7px;
    padding: 6px 14px;
    color: {C_TEXT_1};
    font-weight: 600;
    min-height: 30px;
}}
QPushButton:hover {{
    background-color: {C_SURFACE_4};
    border-color: {C_TEXT_3};
}}
QPushButton:pressed {{
    background-color: {C_SURFACE_2};
}}
QPushButton:disabled {{
    background-color: {C_SURFACE_1};
    border-color: {C_BORDER_FAINT};
    color: {C_TEXT_DIS};
}}
QPushButton#primary {{
    background-color: {C_BLUE};
    border-color: {C_BLUE_H};
    color: white;
    font-weight: 700;
}}
QPushButton#primary:hover {{
    background-color: {C_BLUE_H};
}}
QPushButton#primary:pressed {{
    background-color: {C_BLUE_P};
}}
QPushButton#primary:disabled {{
    background-color: {C_BLUE_BG};
    border-color: {C_BORDER_MED};
    color: {C_TEXT_DIS};
}}
QPushButton#danger {{
    background-color: {C_RED_BG};
    border: 2px solid {C_RED};
    color: {C_RED_TEXT};
    font-weight: 700;
    font-size: 10.5pt;
    min-height: 36px;
    padding: 8px 18px;
}}
QPushButton#danger:hover {{
    background-color: #4a1010;
    color: white;
}}
QPushButton#danger:pressed {{
    background-color: {C_RED};
    color: white;
}}
QPushButton#danger:disabled {{
    background-color: {C_SURFACE_1};
    border-color: {C_BORDER_MED};
    color: {C_TEXT_DIS};
}}
QPushButton#warning {{
    background-color: {C_AMBER_BG};
    border: 1px solid {C_AMBER};
    color: {C_AMBER_TEXT};
    font-weight: 700;
}}
QPushButton#warning:hover {{
    background-color: #3d2200;
    color: white;
}}
QPushButton#warning:disabled {{
    background-color: {C_SURFACE_1};
    border-color: {C_BORDER_FAINT};
    color: {C_TEXT_DIS};
}}
QPushButton#ghost {{
    background: transparent;
    border: 1px solid {C_BORDER_MED};
    color: {C_TEXT_2};
    font-weight: 500;
}}
QPushButton#ghost:hover {{
    background: {C_SURFACE_2};
    color: {C_TEXT_1};
}}
QPushButton#success {{
    background-color: {C_GREEN_BG};
    border: 1px solid {C_GREEN};
    color: {C_GREEN_TEXT};
    font-weight: 700;
}}

/* ── Checkboxes ───────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {C_TEXT_1};
    min-height: 24px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
}}
QCheckBox::indicator:unchecked {{
    background: {C_SURFACE_3};
    border: 1px solid {C_BORDER_BOLD};
}}
QCheckBox::indicator:checked {{
    background: {C_BLUE};
    border: 1px solid {C_BLUE_H};
    image: url(none);
}}
QCheckBox::indicator:disabled {{
    background: {C_SURFACE_2};
    border-color: {C_BORDER_FAINT};
}}
QCheckBox:disabled {{
    color: {C_TEXT_DIS};
}}

/* ── Progress bars ────────────────────────────────────────────────────────── */
QProgressBar {{
    border: 1px solid {C_BORDER_MED};
    border-radius: 6px;
    background: {C_SURFACE_1};
    text-align: center;
    color: {C_TEXT_1};
    min-height: 20px;
    font-size: 8.5pt;
}}
QProgressBar::chunk {{
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C_BLUE_H}, stop:1 {C_BLUE});
}}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {C_BORDER_MED};
    border-radius: 8px;
    background: {C_SURFACE_1};
    top: -1px;
}}
QTabBar::tab {{
    background: {C_SURFACE_2};
    color: {C_TEXT_2};
    border: 1px solid {C_BORDER_MED};
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    padding: 7px 14px;
    min-width: 90px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {C_BLUE_BG};
    color: {C_BLUE};
    font-weight: 700;
    border-color: {C_BLUE};
}}
QTabBar::tab:hover:!selected {{
    background: {C_SURFACE_3};
    color: {C_TEXT_1};
}}

/* ── Splitter ─────────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {C_BORDER_FAINT};
}}
QSplitter::handle:hover {{
    background: {C_BORDER_BOLD};
}}
QSplitter::handle:horizontal {{
    width: 4px;
}}
QSplitter::handle:vertical {{
    height: 4px;
}}

/* ── Labels ───────────────────────────────────────────────────────────────── */
QLabel {{
    color: {C_TEXT_1};
    background: transparent;
}}
QLabel#muted {{
    color: {C_TEXT_3};
    font-size: 9pt;
}}
QLabel#secondary {{
    color: {C_TEXT_2};
}}
QLabel#pageTitle {{
    font-size: 15pt;
    font-weight: 700;
    color: {C_TEXT_1};
}}
QLabel#sectionTitle {{
    font-size: 10pt;
    font-weight: 600;
    color: {C_TEXT_2};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QLabel#fieldLabel {{
    color: {C_TEXT_2};
    font-size: 9.5pt;
    font-weight: 500;
}}

/* ── Chip labels ──────────────────────────────────────────────────────────── */
QLabel#chip_online, QLabel#chip_ready, QLabel#chip_complete {{
    background: {C_GREEN_BG};
    color: {C_GREEN_TEXT};
    border: 1px solid {C_GREEN_BD};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 8.5pt;
    font-weight: 700;
}}
QLabel#chip_warning {{
    background: {C_AMBER_BG};
    color: {C_AMBER_TEXT};
    border: 1px solid {C_AMBER_BD};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 8.5pt;
    font-weight: 700;
}}
QLabel#chip_error, QLabel#chip_missing, QLabel#chip_danger {{
    background: {C_RED_BG};
    color: {C_RED_TEXT};
    border: 1px solid {C_RED_BD};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 8.5pt;
    font-weight: 700;
}}
QLabel#chip_offline, QLabel#chip_idle, QLabel#chip_none {{
    background: {C_BG_DEEP};
    color: {C_TEXT_3};
    border: 1px solid {C_BORDER_MED};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 8.5pt;
    font-weight: 600;
}}
QLabel#chip_info, QLabel#chip_running {{
    background: {C_BLUE_BG};
    color: {C_BLUE};
    border: 1px solid {C_BLUE_P};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 8.5pt;
    font-weight: 700;
}}

/* ── Banner frames ────────────────────────────────────────────────────────── */
QFrame#bannerWarning {{
    background: {C_AMBER_BG};
    border: 1px solid {C_AMBER_BD};
    border-radius: 7px;
}}
QFrame#bannerError {{
    background: {C_RED_BG};
    border: 1px solid {C_RED_BD};
    border-radius: 7px;
}}
QFrame#bannerInfo {{
    background: {C_BLUE_BG};
    border: 1px solid {C_BORDER_BOLD};
    border-radius: 7px;
}}
QFrame#bannerSuccess {{
    background: {C_GREEN_BG};
    border: 1px solid {C_GREEN_BD};
    border-radius: 7px;
}}

/* ── Form layout label alignment ──────────────────────────────────────────── */
/* Modern operator console */
QWidget#modernCentral {{
    background: {C_BG_BASE};
}}
QWidget#modernTopBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b121b, stop:1 #060b11);
    border-bottom: 1px solid #1e2d3f;
}}
QLabel#modernLogo {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0d2d59, stop:1 #071a33);
    color: #8fc1ff;
    border: 1px solid #2c75d8;
    border-radius: 10px;
    font-weight: 800;
    font-size: 10pt;
    qproperty-alignment: AlignCenter;
}}
QLabel#modernProduct {{
    color: {C_TEXT_1};
    font-weight: 800;
    font-size: 13pt;
}}
QToolButton#workspaceSelector, QToolButton#profileButton {{
    background: rgba(10, 17, 27, 120);
    border: 1px solid transparent;
    border-radius: 9px;
    color: {C_TEXT_1};
    padding: 6px 10px;
    font-weight: 600;
}}
QToolButton#workspaceSelector:hover, QToolButton#profileButton:hover {{
    background: rgba(21, 34, 51, 190);
    border-color: {C_BORDER_MED};
}}
QFrame#searchWrap {{
    background: rgba(8, 15, 24, 220);
    border: 1px solid #26384c;
    border-radius: 12px;
}}
QLineEdit#topSearch {{
    background: transparent;
    border: none;
    min-height: 34px;
    padding: 0;
}}
QLabel#shortcutHint {{
    background: {C_SURFACE_2};
    border: 1px solid {C_BORDER_MED};
    border-radius: 6px;
    color: {C_TEXT_2};
    padding: 2px 6px;
    font-size: 8pt;
}}
QToolButton#modernIconButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
}}
QToolButton#modernIconButton:hover {{
    background: {C_SURFACE_2};
    border-color: {C_BORDER_MED};
}}
QLabel#connectionPill, QLabel#batteryPill {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(17, 28, 42, 230), stop:1 rgba(8, 15, 24, 230));
    border: 1px solid #2b3d52;
    border-radius: 10px;
    padding: 5px 12px;
    color: {C_TEXT_1};
    font-size: 8.5pt;
}}
QWidget#modernSidebar {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #060b11, stop:1 #09111a);
    border-right: 1px solid #1f2d3d;
}}
QLabel#navGroup {{
    color: {C_TEXT_3};
    font-size: 8pt;
    font-weight: 700;
    padding: 12px 16px 4px 16px;
    text-transform: uppercase;
}}
QPushButton#modernNav {{
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 9px;
    color: {C_TEXT_2};
    text-align: left;
    padding: 8px 14px 8px 12px;
    font-weight: 600;
    margin: 1px 8px;
}}
QPushButton#modernNav:hover {{
    background: rgba(21, 34, 51, 160);
    color: {C_TEXT_1};
}}
QPushButton#modernNav:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(20, 83, 168, 170), stop:1 rgba(11, 30, 61, 90));
    color: #7db7ff;
    border-left-color: {C_BLUE};
}}
QFrame#modernPanel, QFrame#modernMetric {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(20, 31, 43, 232), stop:1 rgba(10, 17, 27, 232));
    border: 1px solid rgba(65, 86, 112, 160);
    border-radius: 16px;
}}
QFrame#modernMetric {{
    min-height: 92px;
}}
QFrame#mapTools, QFrame#mapOverlay {{
    background: rgba(7, 13, 20, 205);
    border: 1px solid rgba(100, 126, 158, 145);
    border-radius: 14px;
}}
QToolButton#mapToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 5px;
}}
QToolButton#mapToolButton:hover {{
    background: rgba(35, 52, 75, 180);
    border-color: rgba(100, 126, 158, 130);
}}
QToolButton#mapToolButton:checked {{
    background: rgba(47, 125, 246, 170);
    border-color: #7bb4ff;
}}
QWidget#missionMap {{
    background: #111b20;
    border: 1px solid rgba(72, 95, 121, 170);
    border-radius: 16px;
}}
QFrame#droneSelect {{
    background: rgba(8, 15, 24, 220);
    border: 1px solid #31465e;
    border-radius: 10px;
    min-height: 42px;
}}
QFrame#droneSelect:hover {{
    border-color: {C_BLUE};
    background: rgba(13, 23, 36, 235);
}}
QLabel#droneIcon {{
    background: rgba(47, 125, 246, 40);
    border: 1px solid rgba(80, 145, 255, 130);
    border-radius: 8px;
    color: #9ec8ff;
    font-size: 8pt;
    font-weight: 900;
}}
QLabel#droneModel {{
    color: {C_TEXT_1};
    font-size: 9.5pt;
    font-weight: 800;
}}
QLabel#droneStatus {{
    color: {C_TEXT_3};
    font-size: 8pt;
    font-weight: 600;
}}
QLabel#availabilityDot {{
    background: #596979;
    border-radius: 4px;
}}
QLabel#availabilityDot[available="true"] {{
    background: {C_GREEN};
}}
QLabel#fieldChevron {{
    color: {C_TEXT_2};
    font-size: 9pt;
    font-weight: 900;
}}
QFrame#mediaCard {{
    background: {C_SURFACE_1};
    border: 1px solid {C_BORDER_MED};
    border-radius: 16px;
}}
QFrame#mediaCard:hover {{
    border-color: {C_BLUE};
    background: {C_SURFACE_2};
}}
QLabel#mediaThumb, QLabel#selectedPreview {{
    background: #0b131f;
    border: 1px solid {C_BORDER_MED};
    border-radius: 12px;
    color: {C_TEXT_2};
    font-weight: 800;
}}
QWidget#videoFeed {{
    border: 1px solid {C_BORDER_MED};
    border-radius: 16px;
}}
QLabel#videoTitle {{
    color: {C_TEXT_1};
    background: rgba(7, 13, 20, 190);
    border: 1px solid {C_BORDER_BOLD};
    border-radius: 8px;
    padding: 5px 10px;
    font-weight: 800;
}}
QLabel#modernPageTitle {{
    color: {C_TEXT_1};
    font-size: 18pt;
    font-weight: 800;
}}
QLabel#modernMetricValue {{
    color: {C_TEXT_1};
    font-size: 22pt;
    font-weight: 800;
}}
QLabel#modernMapStat {{
    color: {C_TEXT_1};
    font-size: 13pt;
    font-weight: 800;
}}
QLabel#modernLabel {{
    color: {C_TEXT_1};
    font-weight: 600;
}}
QLabel#modernMuted {{
    color: {C_TEXT_3};
    font-size: 9pt;
}}
QLabel#modernSection {{
    color: {C_TEXT_2};
    font-size: 9pt;
    font-weight: 800;
    letter-spacing: 0.5px;
}}
QLabel#weatherChip {{
    background: rgba(27, 44, 63, 170);
    border: 1px solid rgba(72, 95, 121, 150);
    border-radius: 10px;
    color: {C_TEXT_2};
    padding: 8px;
}}
QPushButton#modernPrimary {{
    background: {C_BLUE};
    border: 1px solid {C_BLUE_H};
    border-radius: 10px;
    color: white;
    font-weight: 800;
    padding: 7px 14px;
}}
QPushButton#modernPrimary:hover {{
    background: {C_BLUE_H};
}}
QPushButton#modernSecondary {{
    background: rgba(8, 15, 24, 90);
    border: 1px solid rgba(72, 95, 121, 150);
    border-radius: 10px;
    color: {C_TEXT_1};
    font-weight: 700;
    padding: 7px 14px;
}}
QPushButton#modernSecondary:hover {{
    border-color: {C_BLUE};
    background: rgba(21, 34, 51, 160);
}}
QPushButton#modernOutline {{
    background: rgba(8, 15, 24, 70);
    border: 1px solid #2d7be8;
    border-radius: 10px;
    color: #b8d6ff;
    font-weight: 800;
    padding: 7px 14px;
}}
QPushButton#modernOutline:hover {{
    border-color: #69a8ff;
    background: rgba(17, 47, 92, 95);
}}
QPushButton#modernGhost {{
    background: rgba(8, 15, 24, 75);
    border: 1px solid rgba(72, 95, 121, 110);
    border-radius: 10px;
    color: {C_TEXT_2};
    font-weight: 600;
    padding: 6px 10px;
}}
QPushButton#modernGhost:hover {{
    background: {C_SURFACE_2};
    color: {C_TEXT_1};
}}
QPushButton#modernDanger {{
    background: transparent;
    border: 1px solid {C_RED};
    border-radius: 10px;
    color: {C_RED_TEXT};
    font-weight: 800;
    padding: 7px 14px;
}}
QPushButton#modernDanger:hover {{
    background: {C_RED_BG};
}}
QLineEdit#modernInput, QComboBox#modernInput, QDoubleSpinBox#modernInput {{
    background: rgba(8, 15, 24, 220);
    border: 1px solid #31465e;
    border-radius: 10px;
    min-height: 32px;
    padding: 4px 10px;
    color: {C_TEXT_1};
}}
QLineEdit#modernInput:hover, QComboBox#modernInput:hover, QDoubleSpinBox#modernInput:hover {{
    border-color: #46617f;
    background: rgba(13, 23, 36, 235);
}}
QLineEdit#modernInput:focus, QComboBox#modernInput:focus, QDoubleSpinBox#modernInput:focus {{
    border-color: {C_BLUE};
}}
QComboBox#modernInput {{
    padding-right: 28px;
}}
QComboBox#modernInput::drop-down {{
    width: 26px;
    border: none;
    border-left: 1px solid rgba(72, 95, 121, 120);
}}
QComboBox#modernInput::down-arrow {{
    image: none;
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {C_TEXT_2};
    margin-right: 8px;
}}
QComboBox#modernInput QAbstractItemView {{
    background: #101b29;
    border: 1px solid #36506b;
    selection-background-color: {C_BLUE_BG};
    selection-color: #dcecff;
    padding: 4px;
}}
QFormLayout > QLabel {{
    color: {C_TEXT_2};
    font-size: 9.5pt;
    padding-top: 4px;
}}
"""
