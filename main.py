"""GUI entrypoint for the finalized structural inspection toolkit."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StartupSplash(QWidget):
    """Simple startup splash with logo, module load list, and progress."""

    def __init__(self, logo_path: Path):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setWindowTitle("Launching OpenDroneKit")
        self.setFixedSize(760, 460)
        self.setStyleSheet(
            """
            QWidget#startup {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #111a29, stop:1 #1c2c44);
                border: 1px solid #365076;
                border-radius: 12px;
            }
            QLabel#title {
                color: #f2f7ff;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#status {
                color: #c4d6ee;
                font-size: 13px;
            }
            QListWidget {
                background: #0f1a2a;
                color: #dbe8f7;
                border: 1px solid #365076;
                border-radius: 8px;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #365076;
                border-radius: 6px;
                background: #0f1a2a;
                color: #e6edf5;
                text-align: center;
                height: 22px;
            }
            QProgressBar::chunk {
                background: #3d7ddd;
                border-radius: 5px;
            }
            """
        )
        self.setObjectName("startup")

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        self.logo = QLabel()
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                self.logo.setPixmap(
                    pix.scaled(
                        280,
                        160,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                self.logo.setText("OpenDroneKit")
        else:
            self.logo.setText("OpenDroneKit")
        root.addWidget(self.logo)

        self.title = QLabel("OpenDroneKit")
        self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.title)

        self.status = QLabel("Preparing startup...")
        self.status.setObjectName("status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.status)

        self.log = QListWidget()
        root.addWidget(self.log, stretch=1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

    def center_on_screen(self, app: QApplication) -> None:
        screen = app.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        self.move(
            geom.x() + (geom.width() - self.width()) // 2,
            geom.y() + (geom.height() - self.height()) // 2,
        )

    def set_step(self, message: str, idx: int, total: int) -> None:
        safe_total = max(total, 1)
        pct = int(round((max(idx, 0) / safe_total) * 100.0))
        self.status.setText(message)
        self.progress.setValue(int(max(0, min(100, pct))))
        self.log.addItem(f"[{idx}/{safe_total}] {message}")
        self.log.scrollToBottom()


def _bootstrap_modules(
    splash: StartupSplash,
    app: QApplication,
) -> type:
    steps: list[tuple[str, str | None]] = [
        ("Loading core model registry", "core.models"),
        ("Loading defect detection module", "core.detection"),
        ("Loading crack propagation module", "core.propagation"),
        ("Loading reconstruction pipeline", "core.reconstruction"),
        ("Loading end-to-end pipeline", "core.pipeline"),
        ("Loading mission planner", "mission.planner"),
        ("Loading workspace UI", "ui.workspace"),
        ("Loading analysis tabs", "ui.tabs"),
        ("Loading main window", "ui.main_window"),
    ]

    main_window_cls = None
    total_steps = len(steps) + 2
    for idx, (label, module_name) in enumerate(steps, start=1):
        splash.set_step(label, idx, total_steps)
        app.processEvents()
        if module_name:
            module = importlib.import_module(module_name)
            if module_name == "ui.main_window":
                main_window_cls = getattr(module, "MainWindow", None)
        time.sleep(0.07)

    splash.set_step("Initializing application workspace", len(steps) + 1, total_steps)
    app.processEvents()
    if main_window_cls is None:
        raise RuntimeError("MainWindow class could not be loaded.")
    time.sleep(0.08)

    splash.set_step("Startup complete", len(steps) + 2, total_steps)
    app.processEvents()
    time.sleep(0.08)
    return main_window_cls


def main() -> int:
    root = Path(__file__).resolve().parent
    logo_path = root / "logo.png"

    # QtWebEngine requires this attribute before QApplication is created.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    try:
        # Import WebEngine early to satisfy Qt startup constraints.
        from PyQt6 import QtWebEngineWidgets  # noqa: F401
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        from ui.theme import app_stylesheet

        app.setStyleSheet(app_stylesheet())
    except Exception:
        pass
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    splash = StartupSplash(logo_path=logo_path)
    splash.center_on_screen(app)
    splash.show()
    app.processEvents()

    try:
        MainWindow = _bootstrap_modules(splash=splash, app=app)
        window = MainWindow()
    except Exception as exc:
        splash.close()
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Application failed to launch.\n\n{exc}",
        )
        return 1

    if logo_path.exists():
        window.setWindowIcon(QIcon(str(logo_path)))
    window.show()
    splash.close()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
