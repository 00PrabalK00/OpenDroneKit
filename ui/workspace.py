"""Workspace-style UI with project, mission, fly, data, report, and settings areas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mission import MissionPlan

from .mission_planner import MissionPlannerTab
from .project_store import ProjectStore
from .theme import standard_icon
from .components import (
    Banner,
    EmptyState,
    MetricCard,
    ReadinessCard,
    SectionCard,
    StatusChip,
    TelemetryLabel,
    h_separator,
    make_label,
    PathInput,
)


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pixmap_from_bgr(image_bgr) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _elide_path(path: str, max_len: int = 60) -> str:
    if len(path) <= max_len:
        return path
    half = (max_len - 3) // 2
    return path[:half] + "..." + path[-half:]


def _scrollable(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    return sa


def _primary_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("primary")
    btn.setMinimumHeight(34)
    return btn


def _danger_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("danger")
    return btn


def _warning_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("warning")
    btn.setMinimumHeight(34)
    return btn


def _ghost_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("ghost")
    btn.setMinimumHeight(32)
    return btn


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("sectionTitle")
    return lbl


@dataclass
class MissionEstimate:
    distance_m: float = 0.0
    time_min: float = 0.0
    batteries: int = 0
    coverage_pct: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# AppSession (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class AppSession(QObject):
    """Shared app state and offline-first project storage facade."""

    projectChanged = pyqtSignal(object)
    missionPlanChanged = pyqtSignal(object)
    activeDatasetChanged = pyqtSignal(str)
    activeImageChanged = pyqtSignal(str)
    runArtifactsChanged = pyqtSignal(object)
    missionVersionSaved = pyqtSignal(object)
    datasetImported = pyqtSignal(object)
    reportSaved = pyqtSignal(object)
    statusChanged = pyqtSignal(str)

    droneConnected = pyqtSignal(bool)
    telemetryUpdated = pyqtSignal(object)

    def __init__(self, store: ProjectStore | None = None):
        super().__init__()
        self.store = store or ProjectStore()
        self.active_project: dict[str, Any] | None = None
        self.current_plan: MissionPlan | None = None
        self.active_dataset_dir: str = ""
        self.active_image_path: str = ""
        self.latest_run_artifacts: dict[str, Any] = {}
        # Drone client — starts as mock, can be swapped for MAVSDK
        from core.drone import MockDroneClient
        self.drone_client: MockDroneClient = MockDroneClient()
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(1000)
        self._telemetry_timer.timeout.connect(self._poll_telemetry)
        self.ensure_active_project()

    def connect_drone(self, uri: str = "mock://", driver: str = "mock") -> bool:
        from core.drone import create_drone_client
        try:
            self.drone_client = create_drone_client(driver)
            self.drone_client.connect(uri)
            self._telemetry_timer.start()
            self.droneConnected.emit(True)
            self.statusChanged.emit(f"Drone connected: {uri}")
            return True
        except Exception as exc:
            self.statusChanged.emit(f"Drone connection failed: {exc}")
            self.droneConnected.emit(False)
            return False

    def disconnect_drone(self) -> None:
        self._telemetry_timer.stop()
        try:
            self.drone_client.disconnect()
        except Exception:
            pass
        self.droneConnected.emit(False)
        self.statusChanged.emit("Drone disconnected.")

    def _poll_telemetry(self) -> None:
        try:
            telem = self.drone_client.get_telemetry()
            self.telemetryUpdated.emit(telem)
        except Exception:
            pass

    def ensure_active_project(self) -> dict[str, Any]:
        project = self.store.get_active_project()
        if project is None:
            projects = self.store.list_projects()
            if projects:
                project = projects[0]
                self.store.set_active_project(int(project["id"]))
            else:
                project = self.store.create_project(
                    name="Field Project 01",
                    root_dir=Path("final_toolkit_outputs") / "projects" / "field_project_01",
                    description="Default offline project",
                )
        self.active_project = project
        self.active_dataset_dir = ""
        self.active_image_path = ""
        self.projectChanged.emit(dict(project))
        return project

    def create_project(self, name: str, root_dir: str = "", description: str = "") -> dict[str, Any]:
        project = self.store.create_project(
            name=name,
            root_dir=(Path(root_dir) if str(root_dir).strip() else None),
            description=description,
        )
        self.active_project = project
        self.projectChanged.emit(dict(project))
        self.statusChanged.emit(f"Active project: {project['name']}")
        return project

    def set_active_project(self, project_id: int) -> dict[str, Any]:
        project = self.store.get_project(int(project_id))
        if project is None:
            raise ValueError(f"Project not found: {project_id}")
        self.store.set_active_project(int(project_id))
        self.active_project = project
        self.active_dataset_dir = ""
        self.active_image_path = ""
        self.projectChanged.emit(dict(project))
        self.statusChanged.emit(f"Active project: {project['name']}")
        return project

    def active_project_id(self) -> int | None:
        if not isinstance(self.active_project, dict):
            return None
        return _safe_int(self.active_project.get("id"), default=-1)

    def _refresh_active_project(self, *, emit: bool = True) -> None:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            return
        project = self.store.get_project(pid)
        if project is None:
            return
        self.active_project = project
        if emit:
            self.projectChanged.emit(dict(project))

    def publish_plan(self, plan: MissionPlan) -> None:
        self.current_plan = plan
        self.missionPlanChanged.emit(plan)

    def save_mission_version(self, mission_name: str, note: str = "") -> dict[str, Any]:
        if self.current_plan is None:
            raise ValueError("No current mission plan to save.")
        pid = self.active_project_id()
        if pid is None or pid < 0:
            raise ValueError("No active project selected.")

        recipe = self.current_plan.flight_recipe if isinstance(self.current_plan.flight_recipe, dict) else {}
        coverage = self.current_plan.expected_coverage or {}
        summary = {
            "source": self.current_plan.source,
            "template": self.current_plan.template,
            "distance_m": float(self.current_plan.path_distance_m),
            "time_min": float(self.current_plan.estimated_time_min),
            "waypoints": int(len(self.current_plan.waypoints)),
            "autopilot_commands": int(len(self.current_plan.autopilot_commands)),
            "coverage_pct": float(coverage.get("achieved_coverage_pct", 0.0)),
            "repeat_enabled": bool(self.current_plan.repeat_enabled),
            "saved_at_utc": _utc_now(),
        }
        row = self.store.save_mission_version(
            project_id=pid,
            mission_name=str(mission_name).strip() or "mission",
            template=str(self.current_plan.template or "grid"),
            flight_recipe=recipe,
            plan_summary=summary,
            note=str(note or ""),
        )
        self._refresh_active_project()
        self.missionVersionSaved.emit(dict(row))
        self.statusChanged.emit(
            f"Mission version saved: {mission_name} v{_safe_int(row.get('version_num', 0))}"
        )
        return row

    def list_mission_versions(self, mission_name: str | None = None) -> list[dict[str, Any]]:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            return []
        return self.store.list_mission_versions(project_id=pid, mission_name=mission_name)

    def import_dataset(self, folder: str | Path, name: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            raise ValueError("No active project selected.")
        project = self.active_project if isinstance(self.active_project, dict) else self.store.get_project(pid)
        project_root = Path(str(project.get("root_dir", ""))) if isinstance(project, dict) else Path()
        folder_path = Path(folder)
        if not folder_path.exists():
            raise ValueError(f"Dataset path does not exist: {folder_path}")
        combined_metadata = dict(metadata or {})
        if project_root:
            try:
                from core.data_library import import_image_dataset

                backend_dataset = import_image_dataset(
                    project_root=project_root,
                    project_id=str(pid),
                    folder_path=folder_path,
                    dataset_name=str(name).strip() or folder_path.name,
                    dataset_type=str(combined_metadata.get("dataset_type", "rgb") or "rgb"),
                    generate_thumbnails=True,
                )
                combined_metadata.update(
                    {
                        "backend_dataset_id": backend_dataset.id,
                        "backend_dataset_root": backend_dataset.root_dir,
                        "image_count": backend_dataset.image_count,
                        "qa_status": backend_dataset.qa_status,
                        "has_gps_metadata": backend_dataset.has_gps_metadata,
                        "has_camera_metadata": backend_dataset.has_camera_metadata,
                    }
                )
            except Exception as exc:
                combined_metadata.setdefault("backend_import_error", str(exc))
        row = self.store.save_dataset_entry(
            project_id=pid,
            name=str(name).strip() or folder_path.name,
            path=str(folder_path),
            metadata=combined_metadata,
        )
        self._refresh_active_project()
        self.datasetImported.emit(dict(row))
        self.statusChanged.emit(f"Dataset imported: {row.get('name', '')}")
        self.set_active_dataset(str(folder_path))
        return row

    def list_datasets(self) -> list[dict[str, Any]]:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            return []
        return self.store.list_datasets(project_id=pid)

    def save_report(
        self,
        title: str,
        report_type: str,
        content_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            raise ValueError("No active project selected.")
        row = self.store.save_report(
            project_id=pid,
            title=title,
            report_type=report_type,
            content_path=content_path,
            metadata=metadata or {},
        )
        self._refresh_active_project()
        self.reportSaved.emit(dict(row))
        self.statusChanged.emit(f"Report saved: {row.get('title', '')}")
        return row

    def list_reports(self) -> list[dict[str, Any]]:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            return []
        return self.store.list_reports(project_id=pid)

    def list_projects(self) -> list[dict[str, Any]]:
        return self.store.list_projects()

    def list_audit_events(self, limit: int = 200) -> list[dict[str, Any]]:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            return []
        return self.store.list_audit_events(pid, limit=limit)

    def set_sync_status(self, status: str) -> None:
        pid = self.active_project_id()
        if pid is None or pid < 0:
            return
        self.store.mark_sync_status(pid, status)
        project = self.store.get_project(pid)
        if project is not None:
            self.active_project = project
            self.projectChanged.emit(dict(project))
            self.statusChanged.emit(f"Project sync status: {status}")

    def set_active_dataset(self, dataset_dir: str) -> None:
        path = str(dataset_dir or "").strip()
        if not path:
            return
        self.active_dataset_dir = path
        self.activeDatasetChanged.emit(path)
        self.statusChanged.emit(f"Active dataset: {Path(path).name}")

    def set_active_image(self, image_path: str) -> None:
        path = str(image_path or "").strip()
        if not path:
            return
        self.active_image_path = path
        self.activeImageChanged.emit(path)

    def publish_run_artifacts(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.latest_run_artifacts = dict(payload)
        self.runArtifactsChanged.emit(dict(payload))

        run_dir = str(payload.get("run_dir", "") or "").strip()
        report_path = str(payload.get("report_path", "") or "").strip()
        summary_path = str(payload.get("summary_path", "") or "").strip()
        point_cloud_path = str(payload.get("point_cloud_path", "") or "").strip()
        if run_dir:
            self.statusChanged.emit(f"Latest run: {Path(run_dir).name}")

        project = self.active_project if isinstance(self.active_project, dict) else None
        if project is None:
            return

        if report_path and Path(report_path).exists():
            existing_paths = {str(row.get("content_path", "")) for row in self.list_reports()}
            if report_path not in existing_paths:
                title = f"Pipeline Report {Path(run_dir).name}" if run_dir else "Pipeline Report"
                try:
                    self.save_report(
                        title=title,
                        report_type="pipeline_auto",
                        content_path=report_path,
                        metadata={
                            "run_dir": run_dir,
                            "summary_path": summary_path,
                            "point_cloud_path": point_cloud_path,
                            "source": str(payload.get("source", "pipeline")),
                        },
                    )
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Projects tab
# ─────────────────────────────────────────────────────────────────────────────

class ProjectsTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._build_ui()
        self._refresh()
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.datasetImported.connect(lambda _row: self._refresh())
        self.session.missionVersionSaved.connect(lambda _row: self._refresh())
        self.session.reportSaved.connect(lambda _row: self._refresh())

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = self._make_header()
        root.addWidget(header)

        # Main content area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(16, 12, 16, 12)

        # Left: project list
        left = self._build_project_list()
        splitter.addWidget(left)

        # Right: project dashboard
        self._right_stack_outer = QWidget()
        right_root = QVBoxLayout(self._right_stack_outer)
        right_root.setContentsMargins(0, 0, 0, 0)

        self._empty_state = EmptyState(
            "📁",
            "No Projects Yet",
            "Create your first inspection project to begin. Projects hold your missions, datasets, and analysis results.",
            "Create Project",
        )
        self._empty_state.actionClicked.connect(self._focus_create_form)
        right_root.addWidget(self._empty_state, stretch=1)

        self._dashboard = self._build_dashboard()
        self._dashboard.hide()
        right_root.addWidget(self._dashboard, stretch=1)

        splitter.addWidget(self._right_stack_outer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _make_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("appBar")
        header.setFixedHeight(54)
        row = QHBoxLayout(header)
        row.setContentsMargins(20, 0, 20, 0)
        row.setSpacing(12)

        title = QLabel("Projects")
        title.setObjectName("pageTitle")
        row.addWidget(title)
        row.addStretch(1)

        self.btn_create = _primary_btn("+ New Project")
        self.btn_create.setFixedWidth(140)
        self.btn_create.clicked.connect(self._show_create_dialog)
        row.addWidget(self.btn_create)

        return header

    def _build_project_list(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(240)
        w.setMaximumWidth(300)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        lbl = _section_label("All Projects")
        layout.addWidget(lbl)

        self.projects_list = QListWidget()
        self.projects_list.setSpacing(2)
        self.projects_list.currentItemChanged.connect(self._on_project_selected)
        layout.addWidget(self.projects_list, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_set_active = _ghost_btn("Set Active")
        self.btn_set_active.clicked.connect(self._set_active_from_selection)
        btn_row.addWidget(self.btn_set_active)

        btn_refresh = _ghost_btn("Refresh")
        btn_refresh.clicked.connect(self._refresh)
        btn_row.addWidget(btn_refresh)
        layout.addLayout(btn_row)

        return w

    def _build_dashboard(self) -> QWidget:
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 16)
        layout.setSpacing(12)

        # Project header
        top_row = QHBoxLayout()
        self.lbl_project_name = QLabel("Project")
        self.lbl_project_name.setStyleSheet("font-size: 14pt; font-weight: 700; color: #edf2ff;")
        top_row.addWidget(self.lbl_project_name, stretch=1)
        self.chip_sync = StatusChip("offline", "idle")
        top_row.addWidget(self.chip_sync)
        layout.addLayout(top_row)

        self.lbl_project_desc = QLabel("")
        self.lbl_project_desc.setObjectName("secondary")
        self.lbl_project_desc.setWordWrap(True)
        layout.addWidget(self.lbl_project_desc)

        self.lbl_project_root = QLabel("")
        self.lbl_project_root.setObjectName("muted")
        self.lbl_project_root.setWordWrap(False)
        layout.addWidget(self.lbl_project_root)

        layout.addWidget(h_separator())

        # Readiness grid
        layout.addWidget(_section_label("Project Readiness"))
        grid_row1 = QHBoxLayout()
        grid_row2 = QHBoxLayout()
        self.card_mission    = ReadinessCard("Mission",          False, "No missions planned")
        self.card_dataset    = ReadinessCard("Dataset",          False, "No datasets imported")
        self.card_offline    = ReadinessCard("Offline Map",      False, "Map not cached")
        self.card_analysis   = ReadinessCard("Analysis Results", False, "No analysis run yet")
        self.card_report     = ReadinessCard("Reports",          False, "No reports generated")
        for c in (self.card_mission, self.card_dataset, self.card_offline):
            grid_row1.addWidget(c)
        for c in (self.card_analysis, self.card_report):
            grid_row2.addWidget(c)
        grid_row2.addStretch(1)
        layout.addLayout(grid_row1)
        layout.addLayout(grid_row2)

        layout.addWidget(h_separator())

        # Actions
        layout.addWidget(_section_label("Actions"))
        action_row = QHBoxLayout()
        self.btn_download_offline = _ghost_btn("Download Offline Map")
        self.btn_download_offline.clicked.connect(self._download_offline_region)
        self.btn_open_folder = _ghost_btn("Open Project Folder")
        self.btn_open_folder.clicked.connect(self._open_project_folder)
        self.btn_mark_synced = _ghost_btn("Mark Synced")
        self.btn_mark_synced.clicked.connect(lambda: self.session.set_sync_status("synced"))
        action_row.addWidget(self.btn_download_offline)
        action_row.addWidget(self.btn_open_folder)
        action_row.addWidget(self.btn_mark_synced)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        layout.addWidget(h_separator())

        # Audit trail (compact)
        layout.addWidget(_section_label("Recent Activity"))
        self.audit_log = QPlainTextEdit()
        self.audit_log.setReadOnly(True)
        self.audit_log.setMinimumHeight(150)
        self.audit_log.setObjectName("muted")
        self.audit_log.setStyleSheet(
            "QPlainTextEdit { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px;"
            " font-size: 8.5pt; color: #5a7898; }"
        )
        layout.addWidget(self.audit_log, stretch=1)

        sa = _scrollable(scroll_content)
        wrapper = QWidget()
        wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(sa, stretch=1)
        return wrapper

    # ── Data ──────────────────────────────────────────────────────────────────

    def _selected_project_id(self) -> int | None:
        item = self.projects_list.currentItem()
        if item is None:
            return None
        pid = item.data(Qt.ItemDataRole.UserRole)
        return _safe_int(pid, default=-1) if pid is not None else None

    def _refresh(self):
        projects = self.session.list_projects()
        self.projects_list.blockSignals(True)
        self.projects_list.clear()
        active_pid = self.session.active_project_id()
        active_row = -1
        for idx, row in enumerate(projects):
            name = str(row.get("name", "[unnamed]"))
            mv   = _safe_int(row.get("mission_versions_count", 0))
            ds   = _safe_int(row.get("datasets_count", 0))
            sync = str(row.get("sync_status", "offline"))
            text = f"{name}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, _safe_int(row.get("id"), default=-1))
            item.setToolTip(f"Missions: {mv}  Datasets: {ds}  Sync: {sync}")
            self.projects_list.addItem(item)
            if active_pid is not None and _safe_int(row.get("id"), default=-1) == active_pid:
                active_row = idx
        self.projects_list.blockSignals(False)
        if active_row >= 0:
            self.projects_list.setCurrentRow(active_row)
        elif self.projects_list.count() > 0:
            self.projects_list.setCurrentRow(0)
        self._update_dashboard()

    def _update_dashboard(self):
        project = self.session.active_project
        has_projects = self.projects_list.count() > 0

        if not isinstance(project, dict) or not has_projects:
            self._empty_state.show()
            self._dashboard.hide()
            return

        self._empty_state.hide()
        self._dashboard.show()

        name  = str(project.get("name", "Project"))
        desc  = str(project.get("description", ""))
        root  = str(project.get("root_dir", ""))
        sync  = str(project.get("sync_status", "offline"))

        self.lbl_project_name.setText(name)
        self.lbl_project_desc.setText(desc)
        self.lbl_project_root.setText(_elide_path(root, 70) if root else "No root folder set")
        self.lbl_project_root.setToolTip(root)

        level = "online" if sync == "synced" else "warning" if sync == "offline_ready" else "idle"
        self.chip_sync.set_text_level(sync.replace("_", " ").title(), level)

        mv = _safe_int(project.get("mission_versions_count", 0))
        ds = _safe_int(project.get("datasets_count", 0))
        rp = _safe_int(project.get("reports_count", 0))

        self.card_mission.set_ready(mv > 0, f"{mv} version(s)" if mv else "No missions planned")
        self.card_dataset.set_ready(ds > 0, f"{ds} dataset(s)" if ds else "No datasets imported")
        self.card_offline.set_ready(sync in ("offline_ready", "synced"),
                                    "Cached" if sync in ("offline_ready", "synced") else "Not cached")
        self.card_analysis.set_ready(
            bool(self.session.latest_run_artifacts),
            "Results available" if self.session.latest_run_artifacts else "No analysis run",
        )
        self.card_report.set_ready(rp > 0, f"{rp} report(s)" if rp else "No reports generated")

        events = self.session.list_audit_events(limit=20)
        if events:
            lines = []
            for ev in events[:15]:
                ts = str(ev.get("created_at", ""))[:19].replace("T", " ")
                et = str(ev.get("event_type", ""))
                lines.append(f"{ts}  {et}")
            self.audit_log.setPlainText("\n".join(lines))
        else:
            self.audit_log.setPlainText("No recent activity.")

    def _on_project_selected(self, current: QListWidgetItem | None, _prev):
        del current, _prev
        self._update_dashboard()

    def _on_project_changed(self, _project: dict):
        self._refresh()

    def _set_active_from_selection(self):
        pid = self._selected_project_id()
        if pid is None or pid < 0:
            QMessageBox.warning(self, "Project", "Select a project first.")
            return
        try:
            self.session.set_active_project(pid)
        except Exception as exc:
            QMessageBox.critical(self, "Project", str(exc))

    def _focus_create_form(self):
        self._show_create_dialog()

    def _show_create_dialog(self):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Create New Project")
        dlg.setMinimumWidth(440)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        layout.addWidget(_section_label("New Inspection Project"))

        form = QFormLayout()
        form.setSpacing(10)
        form.setContentsMargins(0, 0, 0, 0)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., Bridge Corridor — Feb 2026")
        name_edit.setMinimumHeight(32)
        form.addRow("Name *", name_edit)

        root_row = QHBoxLayout()
        root_edit = QLineEdit()
        root_edit.setPlaceholderText("Optional custom root folder")
        root_edit.setMinimumHeight(32)
        browse_btn = _ghost_btn("Browse")
        browse_btn.setFixedWidth(80)
        def _browse():
            p = QFileDialog.getExistingDirectory(dlg, "Select project root folder")
            if p:
                root_edit.setText(p)
        browse_btn.clicked.connect(_browse)
        root_row.addWidget(root_edit)
        root_row.addWidget(browse_btn)
        form.addRow("Root Folder", root_row)

        desc_edit = QLineEdit()
        desc_edit.setPlaceholderText("Short description")
        desc_edit.setMinimumHeight(32)
        form.addRow("Description", desc_edit)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Project", "Project name is required.")
            return
        try:
            self.session.create_project(
                name=name,
                root_dir=root_edit.text().strip(),
                description=desc_edit.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Project", str(exc))
        self._refresh()

    def _download_offline_region(self):
        project = self.session.active_project or {}
        if not project:
            QMessageBox.warning(self, "Offline", "No active project.")
            return
        root_dir = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
        cache_dir = root_dir / "offline_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "project_id": _safe_int(project.get("id"), default=-1),
            "project_name": str(project.get("name", "")),
            "cached_at_utc": _utc_now(),
            "basemap_tiles": "requested",
            "elevation_data": "requested",
            "status": "offline_ready",
        }
        (cache_dir / "offline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.session.set_sync_status("offline_ready")
        QMessageBox.information(self, "Offline Cache", f"Offline cache written to:\n{cache_dir}")

    def _open_project_folder(self):
        project = self.session.active_project
        if not isinstance(project, dict):
            return
        root = Path(str(project.get("root_dir", "")))
        if root and root.exists():
            QDesktopServices.openUrl(root.resolve().as_uri())
        else:
            QMessageBox.warning(self, "Project", "Project root folder not found.")


# ─────────────────────────────────────────────────────────────────────────────
# Missions tab
# ─────────────────────────────────────────────────────────────────────────────

class MissionsWorkspaceTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self.planner_tab = MissionPlannerTab()
        self._version_rows: list[dict[str, Any]] = []
        self._build_ui()
        self._wire_signals()
        self._refresh_versions()
        self._on_project_changed(self.session.active_project or {})

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top toolbar
        toolbar = QFrame()
        toolbar.setObjectName("appBar")
        toolbar.setFixedHeight(54)
        tb_row = QHBoxLayout(toolbar)
        tb_row.setContentsMargins(16, 0, 16, 0)
        tb_row.setSpacing(10)

        self.lbl_mission_project = QLabel("Project: —")
        self.lbl_mission_project.setObjectName("secondary")
        tb_row.addWidget(self.lbl_mission_project)

        tb_row.addWidget(self._v_sep())

        lbl = QLabel("Mission:")
        lbl.setObjectName("muted")
        tb_row.addWidget(lbl)
        self.mission_name = QLineEdit("Mission A")
        self.mission_name.setFixedWidth(160)
        self.mission_name.setMinimumHeight(30)
        tb_row.addWidget(self.mission_name)

        lbl2 = QLabel("Note:")
        lbl2.setObjectName("muted")
        tb_row.addWidget(lbl2)
        self.mission_note = QLineEdit()
        self.mission_note.setPlaceholderText("What changed in this version")
        self.mission_note.setFixedWidth(200)
        self.mission_note.setMinimumHeight(30)
        tb_row.addWidget(self.mission_note)

        tb_row.addStretch(1)

        self.btn_save_version = _primary_btn("Save Version")
        self.btn_save_version.setFixedWidth(120)
        self.btn_save_version.clicked.connect(self._save_current_version)
        tb_row.addWidget(self.btn_save_version)

        self.btn_load_version = _ghost_btn("Load Version")
        self.btn_load_version.setFixedWidth(110)
        self.btn_load_version.clicked.connect(self._load_selected_version)
        tb_row.addWidget(self.btn_load_version)

        self.btn_undo = _ghost_btn("← Older")
        self.btn_undo.setFixedWidth(80)
        self.btn_undo.clicked.connect(self._undo_version)
        tb_row.addWidget(self.btn_undo)

        self.btn_redo = _ghost_btn("Newer →")
        self.btn_redo.setFixedWidth(80)
        self.btn_redo.clicked.connect(self._redo_version)
        tb_row.addWidget(self.btn_redo)

        root.addWidget(toolbar)

        # Version list (compact, collapsible-style)
        version_bar = QFrame()
        version_bar.setObjectName("subCard")
        version_bar.setFixedHeight(70)
        vb_layout = QHBoxLayout(version_bar)
        vb_layout.setContentsMargins(14, 6, 14, 6)
        vb_layout.setSpacing(8)

        vb_layout.addWidget(make_label("Version History:", "muted"))
        self.version_list = QListWidget()
        self.version_list.setMaximumHeight(52)
        self.version_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        vb_layout.addWidget(self.version_list, stretch=1)
        root.addWidget(version_bar)

        # Metric strip
        metric_strip = QFrame()
        metric_strip.setObjectName("subCard")
        metric_strip.setFixedHeight(52)
        ms_row = QHBoxLayout(metric_strip)
        ms_row.setContentsMargins(16, 6, 16, 6)
        ms_row.setSpacing(20)

        self.ribbon_time     = QLabel("ETA: —")
        self.ribbon_distance = QLabel("Distance: —")
        self.ribbon_batt     = QLabel("Batteries: —")
        self.ribbon_cov      = QLabel("Coverage: —")
        for lbl in (self.ribbon_time, self.ribbon_distance, self.ribbon_batt, self.ribbon_cov):
            lbl.setObjectName("secondary")
            ms_row.addWidget(lbl)
        ms_row.addStretch(1)

        self.chip_mission_status = StatusChip("No mission", "idle")
        ms_row.addWidget(self.chip_mission_status)
        root.addWidget(metric_strip)

        # Map (main area)
        root.addWidget(self.planner_tab, stretch=1)

    def _v_sep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #192535;")
        sep.setFixedHeight(28)
        return sep

    def _wire_signals(self):
        self.planner_tab.planGenerated.connect(self._on_plan_generated)
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.missionVersionSaved.connect(lambda _row: self._refresh_versions())

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "—") if isinstance(project, dict) else "—"
        self.lbl_mission_project.setText(f"Project: {name}")
        self._refresh_versions()

    def _on_plan_generated(self, plan: MissionPlan):
        self.session.publish_plan(plan)
        coverage = plan.expected_coverage or {}
        est = MissionEstimate(
            distance_m=float(plan.path_distance_m),
            time_min=float(plan.estimated_time_min),
            batteries=max(1, int(math.ceil(max(0.1, float(plan.estimated_time_min)) / 22.0))),
            coverage_pct=float(coverage.get("achieved_coverage_pct", 0.0)),
        )
        self.ribbon_time.setText(f"ETA: {est.time_min:.2f} min")
        self.ribbon_distance.setText(f"Distance: {est.distance_m:.1f} m")
        self.ribbon_batt.setText(f"Batteries: {est.batteries}")
        self.ribbon_cov.setText(f"Coverage: {est.coverage_pct:.1f}%")
        self.chip_mission_status.set_text_level(
            f"{len(plan.waypoints)} waypoints", "ready"
        )

    def _save_current_version(self):
        if self.session.current_plan is None:
            QMessageBox.warning(self, "Mission", "Generate a mission first.")
            return
        try:
            self.session.save_mission_version(
                mission_name=self.mission_name.text().strip() or "Mission",
                note=self.mission_note.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Mission", str(exc))
            return
        self.mission_note.clear()
        self._refresh_versions()

    def _refresh_versions(self):
        mission_name = self.mission_name.text().strip() or None
        rows = self.session.list_mission_versions(mission_name=mission_name)
        self._version_rows = rows
        self.version_list.clear()
        for row in rows:
            v        = _safe_int(row.get("version_num"), default=0)
            template = str(row.get("template", "grid"))
            created  = str(row.get("created_at", ""))[:16].replace("T", " ")
            note     = str(row.get("note", ""))
            label    = f"v{v}  {template}  {created}" + (f"  — {note}" if note else "")
            item     = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.version_list.addItem(item)
        if self.version_list.count() > 0:
            self.version_list.setCurrentRow(0)

    def _selected_version_row(self) -> dict[str, Any] | None:
        item = self.version_list.currentItem()
        if item is None:
            return None
        payload = item.data(Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _load_version_payload(self, row: dict[str, Any]) -> bool:
        try:
            recipe_payload = json.loads(str(row.get("flight_recipe_json", "{}")))
        except Exception as exc:
            QMessageBox.critical(self, "Mission", f"Invalid stored flight recipe:\n{exc}")
            return False
        try:
            recipe_obj = self.planner_tab.planner._coerce_recipe(recipe_payload)  # noqa: SLF001
        except Exception as exc:
            QMessageBox.critical(self, "Mission", f"Unable to load mission version:\n{exc}")
            return False
        self.planner_tab.loaded_repeat_recipe = recipe_obj
        self.planner_tab.repeat_recipe_path   = f"db://mission_versions/{row.get('id')}"
        self.planner_tab.recipe_label.setText(
            f"Loaded: {recipe_obj.recipe_id} v{recipe_obj.version} (from history)"
        )
        self.planner_tab.repeat_mode.setChecked(True)
        try:
            self.planner_tab.generate_plan()
        except Exception as exc:
            QMessageBox.critical(self, "Mission", f"Failed to regenerate mission:\n{exc}")
            return False
        return True

    def _load_selected_version(self):
        row = self._selected_version_row()
        if row is None:
            QMessageBox.warning(self, "Mission", "Select a version first.")
            return
        self._load_version_payload(row)

    def _undo_version(self):
        cur = self.version_list.currentRow()
        if cur < 0:
            return
        nxt = cur + 1
        if nxt >= self.version_list.count():
            QMessageBox.information(self, "Mission", "No older version available.")
            return
        self.version_list.setCurrentRow(nxt)
        row = self._selected_version_row()
        if row is not None:
            self._load_version_payload(row)

    def _redo_version(self):
        cur = self.version_list.currentRow()
        if cur <= 0:
            QMessageBox.information(self, "Mission", "No newer version available.")
            return
        self.version_list.setCurrentRow(cur - 1)
        row = self._selected_version_row()
        if row is not None:
            self._load_version_payload(row)


# ─────────────────────────────────────────────────────────────────────────────
# Fly tab
# ─────────────────────────────────────────────────────────────────────────────

class FlyTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._eta_seconds   = 0
        self._waypoint_idx  = 0
        self._waypoint_count = 0
        self._running = False
        self._paused  = False
        self._timer   = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._update_button_states()
        self.session.missionPlanChanged.connect(self._on_plan_changed)
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.telemetryUpdated.connect(self._on_telemetry)
        self.session.droneConnected.connect(self._on_drone_connected)
        self._on_project_changed(self.session.active_project or {})

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Status bar (top) ──
        status_bar = QFrame()
        status_bar.setObjectName("appBar")
        status_bar.setFixedHeight(50)
        sb_row = QHBoxLayout(status_bar)
        sb_row.setContentsMargins(16, 0, 16, 0)
        sb_row.setSpacing(16)

        self.lbl_fly_project = QLabel("Project: —")
        self.lbl_fly_project.setObjectName("secondary")
        sb_row.addWidget(self.lbl_fly_project)

        sb_row.addWidget(self._v_sep())

        self.chip_flight_mode = StatusChip("Idle", "idle")
        sb_row.addWidget(QLabel("Mode:"))
        sb_row.addWidget(self.chip_flight_mode)

        sb_row.addWidget(self._v_sep())

        self.chip_drone = StatusChip("Disconnected", "idle")
        sb_row.addWidget(QLabel("Drone:"))
        sb_row.addWidget(self.chip_drone)

        sb_row.addWidget(self._v_sep())

        self.chip_gps = StatusChip("No Lock", "idle")
        sb_row.addWidget(QLabel("GPS:"))
        sb_row.addWidget(self.chip_gps)

        sb_row.addWidget(self._v_sep())

        self.chip_battery_top = StatusChip("—", "idle")
        sb_row.addWidget(QLabel("Battery:"))
        sb_row.addWidget(self.chip_battery_top)

        sb_row.addStretch(1)

        self.btn_connect_drone = QPushButton("Connect Drone")
        self.btn_connect_drone.setObjectName("ghost")
        self.btn_connect_drone.setFixedHeight(32)
        self.btn_connect_drone.setFixedWidth(130)
        self.btn_connect_drone.clicked.connect(self._toggle_drone_connection)
        sb_row.addWidget(self.btn_connect_drone)

        root.addWidget(status_bar)

        # ── RTH/Max altitude warning ──
        self._rth_banner = Banner(
            "RTH altitude exceeds max altitude. Check Settings → Flight Safety.",
            kind="warning",
        )
        self._rth_banner.hide()
        root.addWidget(self._rth_banner)

        # ── Main content (splitter) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        # Left: preflight checklist
        left = self._build_preflight_panel()
        splitter.addWidget(left)

        # Right: HUD + controls
        right = self._build_flight_panel()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        # ── Command log ──
        log_frame = QFrame()
        log_frame.setObjectName("subCard")
        log_frame.setFixedHeight(110)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(12, 8, 12, 8)
        log_layout.setSpacing(4)
        log_layout.addWidget(make_label("Command Log", "muted"))
        self.fly_log = QPlainTextEdit()
        self.fly_log.setReadOnly(True)
        self.fly_log.setStyleSheet(
            "QPlainTextEdit { background: transparent; border: none; font-size: 8.5pt;"
            " color: #5a7898; font-family: 'Consolas', monospace; }"
        )
        log_layout.addWidget(self.fly_log)
        root.addWidget(log_frame)

    def _build_preflight_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(280)
        w.setMaximumWidth(320)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        layout.addWidget(_section_label("Preflight Checklist"))

        # Mission readiness
        mission_card = QGroupBox("Mission")
        mc_layout = QVBoxLayout(mission_card)
        self.fly_summary = QLabel("No mission loaded — go to Mission Planner first.")
        self.fly_summary.setWordWrap(True)
        self.fly_summary.setObjectName("secondary")
        mc_layout.addWidget(self.fly_summary)
        self.chip_mission_ready = StatusChip("Not Loaded", "idle")
        mc_layout.addWidget(self.chip_mission_ready)
        layout.addWidget(mission_card)

        # System checks
        checks_card = QGroupBox("System Checks")
        cc_layout = QVBoxLayout(checks_card)
        cc_layout.setSpacing(4)

        self.chk_connection = self._make_check("Drone connected", auto=True)
        self.chk_gps        = self._make_check("GPS / RTK lock", auto=True)
        self.chk_battery    = self._make_check("Battery health verified", auto=True)

        for item_layout in (self.chk_connection, self.chk_gps, self.chk_battery):
            cc_layout.addLayout(item_layout)
        layout.addWidget(checks_card)

        # Safety checks
        safety_card = QGroupBox("Safety Checks")
        sc_layout = QVBoxLayout(safety_card)
        sc_layout.setSpacing(4)

        self.chk_rth      = self._make_check("RTH altitude reviewed", auto=False)
        self.chk_oa       = self._make_check("Obstacle avoidance confirmed", auto=False)
        self.chk_geofence = self._make_check("Geofence & no-fly validated", auto=False)

        for item_layout in (self.chk_rth, self.chk_oa, self.chk_geofence):
            sc_layout.addLayout(item_layout)
        layout.addWidget(safety_card)

        layout.addStretch(1)
        return w

    def _make_check(self, label: str, auto: bool) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        chk = QCheckBox(label)
        chk.stateChanged.connect(self._update_button_states)
        row.addWidget(chk, stretch=1)
        if auto:
            badge = QLabel("AUTO")
            badge.setObjectName("chip_info")
            badge.setFixedWidth(42)
            row.addWidget(badge)
        row.setProperty("_chk", chk)
        # store chk on row for access — use a trick: attach to the layout via a dict
        # Actually, for simplicity, we'll return (row, chk) tuple
        # Let's use a different approach: store checkbox on the row as attribute
        row._checkbox = chk  # type: ignore[attr-defined]
        return row

    def _build_flight_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Telemetry HUD
        hud_card = QGroupBox("Live Telemetry")
        hud_layout = QVBoxLayout(hud_card)
        hud_layout.setSpacing(8)

        tele_row1 = QHBoxLayout()
        self.tele_wp      = TelemetryLabel("Waypoint", "—/—")
        self.tele_eta     = TelemetryLabel("ETA", "—", "s")
        self.tele_battery = TelemetryLabel("Battery", "—", "%")
        self.tele_link    = TelemetryLabel("Link", "—")
        for t in (self.tele_wp, self.tele_eta, self.tele_battery, self.tele_link):
            tele_row1.addWidget(t)
        hud_layout.addLayout(tele_row1)

        self.lbl_hud_mode = QLabel("State: idle")
        self.lbl_hud_mode.setObjectName("secondary")
        self.lbl_hud_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hud_layout.addWidget(self.lbl_hud_mode)
        layout.addWidget(hud_card)

        # Flight controls
        ctrl_card = QGroupBox("Flight Controls")
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setSpacing(12)

        # Primary: Run Mission
        self.btn_run = _primary_btn("▶  Run Mission")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._run_clicked)
        ctrl_layout.addWidget(self.btn_run)

        # Secondary row: Pause + Resume
        secondary_row = QHBoxLayout()
        self.btn_pause  = _ghost_btn("⏸  Pause")
        self.btn_resume = _ghost_btn("▶  Resume")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_resume.clicked.connect(self._resume)
        secondary_row.addWidget(self.btn_pause)
        secondary_row.addWidget(self.btn_resume)
        ctrl_layout.addLayout(secondary_row)

        # Safety row: RTH + Abort
        safety_row = QHBoxLayout()
        self.btn_rth = _warning_btn("🏠  Return To Home")
        self.btn_rth.clicked.connect(self._rth)
        safety_row.addWidget(self.btn_rth, stretch=2)

        self.btn_abort = _danger_btn("✖  ABORT")
        self.btn_abort.setMinimumWidth(120)
        self.btn_abort.clicked.connect(self._abort)
        safety_row.addWidget(self.btn_abort, stretch=1)
        ctrl_layout.addLayout(safety_row)

        layout.addWidget(ctrl_card)

        # Battery swap
        swap_card = QGroupBox("Battery Swap / Resume")
        swap_layout = QFormLayout(swap_card)
        swap_layout.setSpacing(8)

        self.resume_segment = QSpinBox()
        self.resume_segment.setRange(0, 99999)
        self.resume_segment.setValue(0)
        self.resume_segment.setMinimumHeight(32)
        swap_layout.addRow("Last completed segment:", self.resume_segment)

        self.btn_resume_segment = _ghost_btn("Resume From Segment")
        self.btn_resume_segment.clicked.connect(self._resume_from_segment)
        swap_layout.addRow("", self.btn_resume_segment)
        layout.addWidget(swap_card)

        layout.addStretch(1)
        return w

    # ── Checkbox helpers ───────────────────────────────────────────────────────

    def _get_checks(self) -> dict[str, QCheckBox]:
        return {
            "Drone connected":         self.chk_connection._checkbox,  # type: ignore
            "GPS/RTK lock":            self.chk_gps._checkbox,         # type: ignore
            "Battery health":          self.chk_battery._checkbox,     # type: ignore
            "RTH altitude reviewed":   self.chk_rth._checkbox,         # type: ignore
            "Obstacle avoidance":      self.chk_oa._checkbox,          # type: ignore
            "Geofence validated":      self.chk_geofence._checkbox,    # type: ignore
        }

    def _check_preflight(self) -> tuple[bool, list[str]]:
        checks = self._get_checks()
        missing = [name for name, chk in checks.items() if not chk.isChecked()]
        return len(missing) == 0, missing

    def _v_sep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #192535;")
        sep.setFixedHeight(24)
        return sep

    # ── State management ───────────────────────────────────────────────────────

    def _update_button_states(self):
        has_mission  = self.session.current_plan is not None
        preflight_ok = all(chk.isChecked() for chk in self._get_checks().values())
        in_flight    = self._running and not self._paused
        in_pause     = self._running and self._paused
        can_fly      = has_mission and preflight_ok and not self._running

        self.btn_run.setEnabled(can_fly)
        self.btn_pause.setEnabled(in_flight)
        self.btn_resume.setEnabled(in_pause)
        self.btn_rth.setEnabled(self._running)
        self.btn_abort.setEnabled(self._running)
        self.btn_resume_segment.setEnabled(not self._running)

        if not has_mission:
            self.btn_run.setToolTip("No mission loaded — go to Mission Planner first")
            self.chip_mission_ready.set_text_level("Not Loaded", "idle")
        elif not preflight_ok:
            self.btn_run.setToolTip("Complete all preflight checks first")
            self.chip_mission_ready.set_text_level("Preflight Incomplete", "warning")
        else:
            self.btn_run.setToolTip("")
            self.chip_mission_ready.set_text_level("Ready to Fly", "ready")

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "—") if isinstance(project, dict) else "—"
        self.lbl_fly_project.setText(f"Project: {name}")

    def _toggle_drone_connection(self):
        if self.session.drone_client.is_connected():
            self.session.disconnect_drone()
        else:
            self.session.connect_drone("mock://", "mock")

    def _on_drone_connected(self, connected: bool):
        if connected:
            self.chip_drone.set_text_level("Connected", "ready")
            self.btn_connect_drone.setText("Disconnect")
            self._append_log(f"[{_utc_now()}]  Drone connected")
        else:
            self.chip_drone.set_text_level("Disconnected", "idle")
            self.btn_connect_drone.setText("Connect Drone")
            self._append_log(f"[{_utc_now()}]  Drone disconnected")
        self._update_button_states()

    def _on_telemetry(self, telem) -> None:
        try:
            gps_text = f"{telem.satellites} sats" if telem.gps_ok else "No Fix"
            gps_level = "ready" if telem.gps_ok else "error"
            self.chip_gps.set_text_level(gps_text, gps_level)
            bat_pct = int(telem.battery_pct)
            bat_level = "ready" if bat_pct > 30 else "warning" if bat_pct > 15 else "error"
            self.chip_battery_top.set_text_level(f"{bat_pct}%", bat_level)
            drone_level = "ready" if telem.connected else "idle"
            self.chip_drone.set_text_level(
                "Connected" if telem.connected else "Disconnected", drone_level
            )
            mode_level = "running" if self._running else "ready" if telem.connected else "idle"
            if not self._running:
                self.chip_flight_mode.set_text_level(telem.flight_mode or "—", mode_level)
            # Update telemetry HUD
            if self._running:
                self.tele_wp.set_value(f"{telem.waypoint_index}/{max(telem.waypoint_total, self._waypoint_count)}")
                self.tele_battery.set_value(str(bat_pct))
                self.tele_link.set_value(f"{int(telem.link_quality_pct)}%")
        except Exception:
            pass

    def _on_plan_changed(self, plan: MissionPlan):
        self._waypoint_count = max(0, len(plan.waypoints))
        self._waypoint_idx   = 0
        self._eta_seconds    = int(max(0.0, float(plan.estimated_time_min)) * 60.0)
        self.fly_summary.setText(
            f"{plan.template} template  ·  {len(plan.waypoints)} waypoints  ·  "
            f"{plan.path_distance_m:.0f} m  ·  {plan.estimated_time_min:.1f} min"
        )
        self.tele_wp.set_value(f"0/{self._waypoint_count}")
        self.tele_eta.set_value(str(self._eta_seconds))
        self.tele_battery.set_value("100")
        self.tele_link.set_value("—")
        self._update_button_states()

    def _append_log(self, text: str):
        current = self.fly_log.toPlainText().strip()
        self.fly_log.setPlainText((current + "\n" if current else "") + text)
        self.fly_log.verticalScrollBar().setValue(self.fly_log.verticalScrollBar().maximum())

    # ── Actions ────────────────────────────────────────────────────────────────

    def _run_clicked(self):
        if self.session.current_plan is None:
            QMessageBox.warning(self, "Fly", "Generate a mission in Mission Planner first.")
            return
        ok, missing = self._check_preflight()
        if not ok:
            QMessageBox.warning(self, "Preflight Blocked",
                                "The following checks are not complete:\n• " + "\n• ".join(missing))
            return
        # Upload + start via drone client
        try:
            wp_items = [
                {"lat": float(wp.lat), "lon": float(wp.lon), "alt_m": float(wp.alt_m)}
                for wp in self.session.current_plan.waypoints
            ]
            upload_result = self.session.drone_client.upload_mission(wp_items)
            if not upload_result.success:
                self._append_log(f"[{_utc_now()}]  Mission upload failed: {upload_result.message}")
            start_result = self.session.drone_client.start_mission()
            if not start_result.success:
                self._append_log(f"[{_utc_now()}]  Start failed: {start_result.message}")
        except Exception as exc:
            self._append_log(f"[{_utc_now()}]  Drone command error: {exc}")
        self._running = True
        self._paused  = False
        self.chip_flight_mode.set_text_level("Running", "running")
        self._append_log(f"[{_utc_now()}]  Mission started")
        self._timer.start()
        self._update_button_states()

    def _pause(self):
        if not self._running:
            return
        self._timer.stop()
        self._paused = True
        try:
            self.session.drone_client.pause_mission()
        except Exception:
            pass
        self.chip_flight_mode.set_text_level("Paused", "warning")
        self._append_log(f"[{_utc_now()}]  Mission paused")
        self._update_button_states()

    def _resume(self):
        if not self._running:
            return
        try:
            self.session.drone_client.resume_mission()
        except Exception:
            pass
        self._timer.start()
        self._paused = False
        self.chip_flight_mode.set_text_level("Running", "running")
        self._append_log(f"[{_utc_now()}]  Mission resumed")
        self._update_button_states()

    def _abort(self):
        if not self._running:
            return
        reply = QMessageBox.question(
            self, "Confirm Abort",
            "Abort the active mission? The drone will hover in place.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self.session.drone_client.abort_mission()
        except Exception:
            pass
        self._timer.stop()
        self._running = False
        self._paused  = False
        self.chip_flight_mode.set_text_level("Aborted", "error")
        self._append_log(f"[{_utc_now()}]  MISSION ABORTED")
        self._update_button_states()

    def _rth(self):
        if not self._running:
            return
        try:
            self.session.drone_client.return_to_home()
        except Exception:
            pass
        self._timer.stop()
        self._running = False
        self._paused  = False
        self.chip_flight_mode.set_text_level("RTH", "warning")
        self._append_log(f"[{_utc_now()}]  Return-to-home initiated")
        self._update_button_states()

    def _resume_from_segment(self):
        seg = int(self.resume_segment.value())
        self._append_log(f"[{_utc_now()}]  Resume requested from segment {seg}")
        self.chip_flight_mode.set_text_level("Resume Pending", "warning")

    def _tick(self):
        if not self._running:
            return
        if self._eta_seconds > 0:
            self._eta_seconds -= 1
        if self._waypoint_count > 0 and self._waypoint_idx < self._waypoint_count:
            step = max(1, int(max(1, self._eta_seconds + 1) / max(1, self._waypoint_count - self._waypoint_idx)))
            if self._eta_seconds % step == 0:
                self._waypoint_idx = min(self._waypoint_count, self._waypoint_idx + 1)
        bat = max(5, int(100.0 * max(0.0, float(self._eta_seconds)) / max(1.0, float(max(self._eta_seconds, 1) + 120))))
        self.tele_wp.set_value(f"{self._waypoint_idx}/{self._waypoint_count}")
        self.tele_eta.set_value(str(self._eta_seconds))
        self.tele_battery.set_value(str(bat))
        self.chip_battery_top.set_text_level(f"{bat}%", "ready" if bat > 30 else "warning" if bat > 15 else "error")

        if self._eta_seconds <= 0 or (self._waypoint_count > 0 and self._waypoint_idx >= self._waypoint_count):
            self._timer.stop()
            self._running = False
            self._paused  = False
            self.chip_flight_mode.set_text_level("Complete", "complete")
            self._append_log(f"[{_utc_now()}]  Mission complete")
            self._update_button_states()


# ─────────────────────────────────────────────────────────────────────────────
# Data tab
# ─────────────────────────────────────────────────────────────────────────────

class DataTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._current_image_paths: list[str] = []
        self._build_ui()
        self._wire_signals()
        self._refresh_tree()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QFrame()
        header.setObjectName("appBar")
        header.setFixedHeight(54)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(16, 0, 16, 0)
        hb.setSpacing(12)

        self.lbl_data_project = QLabel("Project: —")
        self.lbl_data_project.setObjectName("secondary")
        hb.addWidget(self.lbl_data_project)

        hb.addStretch(1)

        # Filters (compact)
        hb.addWidget(make_label("Mission:", "muted"))
        self.filter_mission = QComboBox()
        self.filter_mission.addItem("All", "")
        self.filter_mission.setFixedWidth(130)
        self.filter_mission.setMinimumHeight(30)
        hb.addWidget(self.filter_mission)

        hb.addWidget(make_label("Tag:", "muted"))
        self.filter_tag = QComboBox()
        self.filter_tag.addItem("Any tag", "")
        self.filter_tag.setFixedWidth(100)
        self.filter_tag.setMinimumHeight(30)
        hb.addWidget(self.filter_tag)

        self.btn_import_dataset = _primary_btn("Import Dataset")
        self.btn_import_dataset.setFixedWidth(140)
        self.btn_import_dataset.clicked.connect(self._import_dataset_folder)
        hb.addWidget(self.btn_import_dataset)

        root.addWidget(header)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        # Left: project tree
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(260)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(6)
        ll.addWidget(_section_label("Project Assets"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name"])
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        ll.addWidget(self.tree, stretch=1)
        splitter.addWidget(left)

        # Center: image list
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 8, 0)
        cl.setSpacing(6)
        cl.addWidget(_section_label("Images"))

        self._empty_data = EmptyState(
            "🗂",
            "No Dataset Loaded",
            "Import an image folder to begin. Expected format: RGB drone images in a flat folder, "
            "with optional masks in a matching subfolder.",
            "Import Dataset",
        )
        self._empty_data.actionClicked.connect(self._import_dataset_folder)
        cl.addWidget(self._empty_data)

        self.photo_list = QListWidget()
        self.photo_list.currentTextChanged.connect(self._on_photo_selected)
        self.photo_list.hide()
        cl.addWidget(self.photo_list, stretch=1)
        splitter.addWidget(center)

        # Right: image inspector
        right = QWidget()
        right.setMinimumWidth(240)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        rl.addWidget(_section_label("Image Inspector"))

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setStyleSheet("QLabel { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px; }")
        self.preview.setText("Select an image")
        self.preview.setObjectName("muted")
        rl.addWidget(self.preview, stretch=1)

        self.qa_box = QPlainTextEdit()
        self.qa_box.setReadOnly(True)
        self.qa_box.setMaximumHeight(120)
        self.qa_box.setPlaceholderText("Image metadata and QA status will appear here.")
        self.qa_box.setStyleSheet(
            "QPlainTextEdit { background: #0f1c2e; border: 1px solid #192535;"
            " font-size: 8.5pt; color: #5a7898; border-radius: 6px; }"
        )
        rl.addWidget(self.qa_box)

        # Annotation strip
        ann_card = QGroupBox("Annotate")
        ann_f = QFormLayout(ann_card)
        ann_f.setSpacing(6)
        self.ann_label = QLineEdit()
        self.ann_label.setPlaceholderText("Label…")
        self.ann_label.setMinimumHeight(28)
        ann_f.addRow("Label:", self.ann_label)
        self.ann_severity = QComboBox()
        self.ann_severity.addItems(["— none —", "critical", "high", "medium", "low", "info"])
        self.ann_severity.setMinimumHeight(28)
        ann_f.addRow("Severity:", self.ann_severity)
        self.ann_note = QLineEdit()
        self.ann_note.setPlaceholderText("Optional note…")
        self.ann_note.setMinimumHeight(28)
        ann_f.addRow("Note:", self.ann_note)
        btn_ann = _ghost_btn("Save Annotation")
        btn_ann.clicked.connect(self._save_annotation)
        ann_f.addRow("", btn_ann)
        rl.addWidget(ann_card)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        root.addWidget(splitter, stretch=1)

    # ── Signals & data ────────────────────────────────────────────────────────

    def _wire_signals(self):
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.datasetImported.connect(lambda _row: self._refresh_tree())
        self.session.missionVersionSaved.connect(lambda _row: self._refresh_tree())
        self._on_project_changed(self.session.active_project or {})

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "—") if isinstance(project, dict) else "—"
        self.lbl_data_project.setText(f"Project: {name}")
        self._refresh_tree()

    def _refresh_tree(self):
        self.tree.clear()
        datasets = self.session.list_datasets()
        versions = self.session.list_mission_versions()

        ds_root = QTreeWidgetItem(["Datasets"])
        for ds in datasets:
            item = QTreeWidgetItem([f"{ds.get('name', '')}"])
            cnt = _safe_int(ds.get("metadata", {}).get("image_count", 0)) if isinstance(ds.get("metadata"), dict) else 0
            item.setToolTip(0, f"{ds.get('path', '')}  ({cnt} images)")
            item.setData(0, Qt.ItemDataRole.UserRole, {"type": "dataset", "row": ds})
            ds_root.addChild(item)
        self.tree.addTopLevelItem(ds_root)

        mv_root = QTreeWidgetItem(["Mission Versions"])
        names = sorted({str(v.get("mission_name", "mission")) for v in versions})
        for nm in names:
            grp = QTreeWidgetItem([nm])
            for v in [r for r in versions if str(r.get("mission_name", "")) == nm][:20]:
                child = QTreeWidgetItem([
                    f"v{v.get('version_num', 0)}  {v.get('template', '')}  "
                    f"{str(v.get('created_at', ''))[:10]}"
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, {"type": "mission_version", "row": v})
                grp.addChild(child)
            mv_root.addChild(grp)
        self.tree.addTopLevelItem(mv_root)
        self.tree.expandAll()

        self.filter_mission.blockSignals(True)
        self.filter_mission.clear()
        self.filter_mission.addItem("All", "")
        for nm in names:
            self.filter_mission.addItem(nm, nm)
        self.filter_mission.blockSignals(False)

        has_data = len(datasets) > 0
        self._empty_data.setVisible(not has_data)
        self.photo_list.setVisible(has_data)

    def _import_dataset_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image dataset folder")
        if not path:
            return
        root = Path(path)
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        files.sort()
        if not files:
            QMessageBox.warning(self, "Data", "No supported images found in that folder.")
            return
        try:
            self.session.import_dataset(
                folder=path,
                name=root.name,
                metadata={"image_count": len(files)},
            )
        except Exception as exc:
            QMessageBox.critical(self, "Data", str(exc))
            return
        self.session.set_active_dataset(path)
        self._load_image_list([str(p) for p in files])
        self._refresh_tree()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _col: int):
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        kind = str(payload.get("type", ""))
        row  = payload.get("row", {})
        if kind == "dataset" and isinstance(row, dict):
            path = Path(str(row.get("path", "")))
            if path.exists() and path.is_dir():
                files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS])
                self.session.set_active_dataset(str(path))
                self._load_image_list([str(p) for p in files])
        elif kind == "mission_version" and isinstance(row, dict):
            self.qa_box.setPlainText(
                f"Mission: {row.get('mission_name', '')}\n"
                f"Version: v{row.get('version_num', 0)}\n"
                f"Template: {row.get('template', '')}\n"
                f"Created: {row.get('created_at', '')}\n"
                f"Note: {row.get('note', '')}"
            )

    def _load_image_list(self, paths: list[str]):
        self._current_image_paths = paths
        self.photo_list.clear()
        self.photo_list.addItems([Path(p).name for p in paths])
        self.photo_list.show()
        self._empty_data.hide()
        if paths:
            self.photo_list.setCurrentRow(0)

    def _on_photo_selected(self, name: str):
        if not name:
            return
        match = next((p for p in self._current_image_paths if Path(p).name == name), None)
        if match is None:
            return
        self.session.set_active_image(match)
        img = cv2.imread(match, cv2.IMREAD_COLOR)
        if img is None:
            self.preview.setText("Unable to load image")
            return
        self.preview.setPixmap(
            _pixmap_from_bgr(img).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        qa = self._quick_qa(img)
        self.qa_box.setPlainText(
            f"File: {name}\n"
            f"Sharpness: {qa['sharpness']:.0f}  ({qa['blur_status']})\n"
            f"Exposure: {qa['exposure_status']}\n"
            f"Dark clip: {qa['dark_clip_pct']:.1f}%\n"
            f"Bright clip: {qa['bright_clip_pct']:.1f}%"
        )

    def _save_annotation(self):
        img_path = self.session.active_image_path
        if not img_path:
            QMessageBox.warning(self, "Annotate", "Select an image first.")
            return
        label = self.ann_label.text().strip()
        if not label:
            QMessageBox.warning(self, "Annotate", "Enter a label.")
            return
        project = self.session.active_project or {}
        project_id = str(project.get("id", ""))
        root_dir = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
        sev = self.ann_severity.currentText()
        severity = sev if sev != "— none —" else None
        try:
            from core.annotations import create_annotation
            create_annotation(
                project_root=root_dir,
                project_id=project_id,
                source_type="image",
                source_id=img_path,
                annotation_type="severity_tag" if severity else "free_text",
                geometry={"type": "Point", "pixel_coords": [0, 0]},
                label=label,
                severity=severity,
                note=self.ann_note.text().strip() or None,
            )
            self.ann_label.clear()
            self.ann_note.clear()
            self.ann_severity.setCurrentIndex(0)
            self.session.statusChanged.emit(f"Annotation saved: {label}")
        except Exception as exc:
            QMessageBox.critical(self, "Annotate", str(exc))

    def _quick_qa(self, image_bgr) -> dict[str, Any]:
        gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sharp   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hist    = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total   = float(max(1.0, hist.sum()))
        dark    = float(hist[:5].sum() / total * 100.0)
        bright  = float(hist[-5:].sum() / total * 100.0)
        exp_st  = "ok" if dark <= 5.0 and bright <= 5.0 else ("under-exposed" if dark > 5.0 else "over-exposed")
        blur_st = "ok" if sharp >= 80.0 else "potential blur"
        return {"sharpness": sharp, "dark_clip_pct": dark, "bright_clip_pct": bright,
                "exposure_status": exp_st, "blur_status": blur_st}


# ─────────────────────────────────────────────────────────────────────────────
# Reports tab
# ─────────────────────────────────────────────────────────────────────────────

class ReportsTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._build_ui()
        self._wire_signals()
        self._refresh_reports()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("appBar")
        header.setFixedHeight(54)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(16, 0, 16, 0)
        hb.setSpacing(12)
        self.lbl_report_project = QLabel("Project: —")
        self.lbl_report_project.setObjectName("secondary")
        hb.addWidget(self.lbl_report_project)
        hb.addStretch(1)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        # Left: config + readiness
        left = QWidget()
        left.setMinimumWidth(260)
        left.setMaximumWidth(340)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(10)

        ll.addWidget(_section_label("Report Configuration"))

        config_card = QGroupBox("Report Setup")
        cf = QFormLayout(config_card)
        cf.setSpacing(10)
        self.report_title = QLineEdit("Inspection Report")
        self.report_title.setMinimumHeight(32)
        cf.addRow("Title:", self.report_title)
        self.report_type = QComboBox()
        self.report_type.addItem("Standard Inspection", "standard")
        self.report_type.addItem("Detailed Engineering", "advanced")
        self.report_type.addItem("Executive Summary", "executive")
        self.report_type.setMinimumHeight(32)
        cf.addRow("Type:", self.report_type)
        ll.addWidget(config_card)

        sections_card = QGroupBox("Sections")
        sc = QVBoxLayout(sections_card)
        sc.setSpacing(4)
        self.sec_overview     = QCheckBox("Overview & Project Summary")
        self.sec_map          = QCheckBox("Mission Map")
        self.sec_photos       = QCheckBox("Key Photos")
        self.sec_annotations  = QCheckBox("Annotations")
        self.sec_measurements = QCheckBox("Measurements")
        for chk in (self.sec_overview, self.sec_map, self.sec_photos, self.sec_annotations, self.sec_measurements):
            chk.setChecked(True)
            sc.addWidget(chk)
        ll.addWidget(sections_card)

        # Readiness checklist
        ll.addWidget(_section_label("Report Readiness"))
        self.ready_project  = ReadinessCard("Project active",    False)
        self.ready_mission  = ReadinessCard("Mission loaded",    False)
        self.ready_dataset  = ReadinessCard("Dataset available", False)
        self.ready_analysis = ReadinessCard("Analysis results",  False)
        for card in (self.ready_project, self.ready_mission, self.ready_dataset, self.ready_analysis):
            ll.addWidget(card)

        ll.addStretch(1)

        self.btn_generate = _primary_btn("Generate Report")
        self.btn_generate.clicked.connect(self._generate_report)
        ll.addWidget(self.btn_generate)

        self.btn_open_folder = _ghost_btn("Open Reports Folder")
        self.btn_open_folder.clicked.connect(self._open_report_folder)
        ll.addWidget(self.btn_open_folder)

        splitter.addWidget(left)

        # Right: reports list + preview
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        rl.addWidget(_section_label("Generated Reports"))

        self.report_list = QListWidget()
        self.report_list.setMaximumHeight(160)
        self.report_list.currentItemChanged.connect(self._on_report_selected)
        rl.addWidget(self.report_list)

        self._empty_reports = EmptyState(
            "📄",
            "No Reports Yet",
            "Configure the report options on the left and click Generate Report.\n"
            "Requires at least one active project.",
        )
        rl.addWidget(self._empty_reports)

        rl.addWidget(_section_label("Preview"))
        self.report_preview = QPlainTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setPlaceholderText("Generated report content will appear here.")
        rl.addWidget(self.report_preview, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _wire_signals(self):
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.reportSaved.connect(lambda _row: self._refresh_reports())
        self.session.missionPlanChanged.connect(lambda _plan: self._update_readiness())
        self.session.datasetImported.connect(lambda _row: self._update_readiness())
        self._on_project_changed(self.session.active_project or {})

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "—") if isinstance(project, dict) else "—"
        self.lbl_report_project.setText(f"Project: {name}")
        self._update_readiness()
        self._refresh_reports()

    def _update_readiness(self):
        project = self.session.active_project
        has_project  = isinstance(project, dict)
        has_mission  = self.session.current_plan is not None
        has_dataset  = bool(self.session.active_dataset_dir)
        has_analysis = bool(self.session.latest_run_artifacts)

        self.ready_project.set_ready(has_project)
        self.ready_mission.set_ready(has_mission)
        self.ready_dataset.set_ready(has_dataset)
        self.ready_analysis.set_ready(has_analysis, note="Optional but recommended")

        self.btn_generate.setEnabled(has_project)
        if not has_project:
            self.btn_generate.setToolTip("No active project")
        else:
            self.btn_generate.setToolTip("")

    def _selected_sections(self) -> list[str]:
        out = []
        if self.sec_overview.isChecked():     out.append("overview")
        if self.sec_map.isChecked():          out.append("map")
        if self.sec_photos.isChecked():       out.append("photos")
        if self.sec_annotations.isChecked():  out.append("annotations")
        if self.sec_measurements.isChecked(): out.append("measurements")
        return out

    def _project_root(self) -> Path | None:
        project = self.session.active_project
        if not isinstance(project, dict):
            return None
        root = Path(str(project.get("root_dir", "")))
        if not root:
            return None
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _generate_report(self):
        project = self.session.active_project
        if not isinstance(project, dict):
            QMessageBox.warning(self, "Reports", "No active project selected.")
            return
        root = self._project_root()
        if root is None:
            QMessageBox.warning(self, "Reports", "Project root is not available.")
            return
        sections = self._selected_sections()
        if not sections:
            QMessageBox.warning(self, "Reports", "Select at least one section.")
            return

        report_type = str(self.report_type.currentData() or "standard")
        title       = self.report_title.text().strip() or "Inspection Report"
        versions    = self.session.list_mission_versions()[:10]
        datasets    = self.session.list_datasets()[:10]
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"report_{report_type}_{stamp}.md"

        lines: list[str] = [
            f"# {title}", "",
            f"- Project: {project.get('name', '')}",
            f"- Type: {report_type}",
            f"- Generated: {_utc_now()}", "",
        ]
        if "overview" in sections:
            lines += [
                "## Overview",
                f"- Mission versions: {len(versions)}",
                f"- Datasets: {len(datasets)}",
                f"- Mission loaded: {'yes' if self.session.current_plan is not None else 'no'}", "",
            ]
        if "map" in sections:
            lines.append("## Mission Map Summary")
            if self.session.current_plan is not None:
                plan = self.session.current_plan
                lines += [
                    f"- Template: {plan.template}",
                    f"- Distance: {plan.path_distance_m:.1f} m",
                    f"- Estimated time: {plan.estimated_time_min:.2f} min",
                    f"- Waypoints: {len(plan.waypoints)}", "",
                ]
            else:
                lines += ["- No active mission plan.", ""]
        if "photos" in sections:
            lines.append("## Key Photos")
            if datasets:
                newest = datasets[0]
                path   = Path(str(newest.get("path", "")))
                images = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]) if path.exists() else []
                for p in images[:8]:
                    lines.append(f"- {p.name}")
            else:
                lines.append("- No datasets available.")
            lines.append("")
        if "annotations" in sections:
            lines += ["## Annotations", "- Add defect annotations and engineer notes here.", ""]
        if "measurements" in sections:
            lines += ["## Measurements", "- Add measured spans, widths, and hotspot deltas here.", ""]

        report_path.write_text("\n".join(lines), encoding="utf-8")
        try:
            self.session.save_report(
                title=title,
                report_type=report_type,
                content_path=report_path,
                metadata={"sections": sections},
            )
        except Exception as exc:
            QMessageBox.critical(self, "Reports", str(exc))
            return
        self._refresh_reports()
        self.report_preview.setPlainText(report_path.read_text(encoding="utf-8"))

    def _refresh_reports(self):
        rows = self.session.list_reports()
        self.report_list.clear()
        for row in rows:
            title   = str(row.get("title", "Report"))
            rtype   = str(row.get("report_type", "standard"))
            created = str(row.get("created_at", ""))[:16].replace("T", " ")
            item    = QListWidgetItem(f"{title}  ·  {rtype}  ·  {created}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.report_list.addItem(item)
        has_reports = self.report_list.count() > 0
        self._empty_reports.setVisible(not has_reports)
        if has_reports:
            self.report_list.setCurrentRow(0)

    def _on_report_selected(self, current: QListWidgetItem | None, _prev):
        if current is None:
            return
        row = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(row, dict):
            return
        path = Path(str(row.get("content_path", "")))
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f"[Binary file or non-UTF-8 content]\nPath: {path}"
        else:
            text = f"File not found:\n{path}"
        self.report_preview.setPlainText(text)

    def _open_report_folder(self):
        root = self._project_root()
        if root is None:
            QMessageBox.warning(self, "Reports", "No active project.")
            return
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(reports_dir.resolve().as_uri())


# ─────────────────────────────────────────────────────────────────────────────
# Settings tab
# ─────────────────────────────────────────────────────────────────────────────

class SettingsTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._build_ui()
        self._load_defaults()
        self._wire_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("appBar")
        header.setFixedHeight(54)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        hb.addWidget(title)
        hb.addStretch(1)
        self.btn_save = _primary_btn("Save Changes")
        self.btn_save.setFixedWidth(130)
        self.btn_save.clicked.connect(self._save_defaults)
        hb.addWidget(self.btn_save)
        root.addWidget(header)

        # RTH safety banner (shown when RTH > max alt)
        self._rth_banner = Banner(
            "RTH altitude is higher than max altitude. This may cause unexpected behavior. "
            "RTH altitude should typically be ≤ max altitude unless you have a deliberate reason.",
            kind="warning",
        )
        self._rth_banner.hide()
        root.addWidget(self._rth_banner)

        # Scrollable content
        scroll_content = QWidget()
        sc = QVBoxLayout(scroll_content)
        sc.setContentsMargins(16, 12, 16, 16)
        sc.setSpacing(14)

        # ── Drone Profile ──
        drone_card = QGroupBox("Drone Profile")
        df = QFormLayout(drone_card)
        df.setSpacing(10)
        self.default_drone = QComboBox()
        self.default_drone.addItem("Mavic 2 Pro",       "mavic2pro")
        self.default_drone.addItem("Phantom 4 RTK",     "phantom4rtk")
        self.default_drone.addItem("Custom",            "custom")
        self.default_drone.setMinimumHeight(32)
        df.addRow("Drone model:", self.default_drone)
        self.default_payload = QComboBox()
        self.default_payload.addItem("RGB",                 "rgb")
        self.default_payload.addItem("RGB + Thermal",       "rgb_thermal")
        self.default_payload.addItem("RGB + LiDAR",         "rgb_lidar")
        self.default_payload.addItem("RGB + Multispectral", "rgb_multispectral")
        self.default_payload.setMinimumHeight(32)
        df.addRow("Payload:", self.default_payload)
        self.unit_system = QComboBox()
        self.unit_system.addItem("Metric",   "metric")
        self.unit_system.addItem("Imperial", "imperial")
        self.unit_system.setMinimumHeight(32)
        df.addRow("Units:", self.unit_system)
        sc.addWidget(drone_card)

        # ── Flight Safety ──
        safety_card = QGroupBox("Flight Safety")
        sf = QFormLayout(safety_card)
        sf.setSpacing(10)

        self.default_min_alt = QDoubleSpinBox()
        self.default_min_alt.setRange(5.0, 300.0)
        self.default_min_alt.setValue(30.0)
        self.default_min_alt.setSuffix(" m")
        self.default_min_alt.setMinimumHeight(32)
        sf.addRow("Minimum altitude:", self.default_min_alt)

        self.default_max_alt = QDoubleSpinBox()
        self.default_max_alt.setRange(10.0, 600.0)
        self.default_max_alt.setValue(120.0)
        self.default_max_alt.setSuffix(" m")
        self.default_max_alt.setMinimumHeight(32)
        self.default_max_alt.valueChanged.connect(self._validate_altitude)
        sf.addRow("Maximum altitude:", self.default_max_alt)

        self.default_rth = QDoubleSpinBox()
        self.default_rth.setRange(10.0, 600.0)
        self.default_rth.setValue(120.0)
        self.default_rth.setSuffix(" m")
        self.default_rth.setMinimumHeight(32)
        self.default_rth.valueChanged.connect(self._validate_altitude)
        sf.addRow("RTH altitude:", self.default_rth)

        rth_note = QLabel("RTH altitude should equal or be less than max altitude unless explicitly overriding safety limits.")
        rth_note.setObjectName("muted")
        rth_note.setWordWrap(True)
        sf.addRow("", rth_note)
        sc.addWidget(safety_card)

        # ── Team & Workflow ──
        team_card = QGroupBox("Team & Workflow")
        tf = QFormLayout(team_card)
        tf.setSpacing(10)
        self.role = QComboBox()
        self.role.addItem("Pilot",    "pilot")
        self.role.addItem("Reviewer", "reviewer")
        self.role.addItem("Admin",    "admin")
        self.role.setMinimumHeight(32)
        tf.addRow("Role:", self.role)
        self.team_workflow = QCheckBox("Enable team workflow gates  (plan → fly → review → report)")
        self.team_workflow.setChecked(True)
        tf.addRow("", self.team_workflow)
        self.allow_sharing = QCheckBox("Allow mission file sharing (import/export)")
        self.allow_sharing.setChecked(True)
        tf.addRow("", self.allow_sharing)
        sc.addWidget(team_card)

        # ── Offline Planning ──
        offline_card = QGroupBox("Offline Planning")
        of = QFormLayout(offline_card)
        of.setSpacing(10)
        self.offline_region = QLineEdit()
        self.offline_region.setPlaceholderText("Region name for basemap / elevation cache")
        self.offline_region.setMinimumHeight(32)
        of.addRow("Region:", self.offline_region)

        offline_buttons = QHBoxLayout()
        self.btn_download_cache = _ghost_btn("Download Map Cache")
        self.btn_download_cache.clicked.connect(self._download_cache)
        self.btn_set_offline = _ghost_btn("Mark Offline Ready")
        self.btn_set_offline.clicked.connect(lambda: self.session.set_sync_status("offline_ready"))
        offline_buttons.addWidget(self.btn_download_cache)
        offline_buttons.addWidget(self.btn_set_offline)
        offline_buttons.addStretch(1)
        of.addRow("", offline_buttons)
        sc.addWidget(offline_card)

        # ── File Portability ──
        io_card = QGroupBox("Mission File Portability")
        io_layout = QHBoxLayout(io_card)
        self.btn_import_mission = _ghost_btn("Import Mission File")
        self.btn_import_mission.clicked.connect(self._import_mission_file)
        self.btn_export_mission = _ghost_btn("Export Latest Mission")
        self.btn_export_mission.clicked.connect(self._export_latest_mission_file)
        io_layout.addWidget(self.btn_import_mission)
        io_layout.addWidget(self.btn_export_mission)
        io_layout.addStretch(1)
        sc.addWidget(io_card)

        # Activity log
        sc.addWidget(_section_label("Activity Log"))
        self.settings_log = QPlainTextEdit()
        self.settings_log.setReadOnly(True)
        self.settings_log.setMaximumHeight(120)
        self.settings_log.setStyleSheet(
            "QPlainTextEdit { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px;"
            " font-size: 8.5pt; color: #5a7898; }"
        )
        sc.addWidget(self.settings_log)

        sc.addStretch(1)

        sa = _scrollable(scroll_content)
        root.addWidget(sa, stretch=1)

    def _validate_altitude(self):
        rth = self.default_rth.value()
        max_alt = self.default_max_alt.value()
        if rth > max_alt:
            self._rth_banner.set_text(
                f"RTH altitude ({rth:.0f} m) is higher than max altitude ({max_alt:.0f} m). "
                "This may cause unexpected behavior. Adjust one of these values."
            )
            self._rth_banner.show()
        else:
            self._rth_banner.hide()

    def _wire_signals(self):
        self.session.projectChanged.connect(lambda _p: self._append_log("Active project updated."))

    def _append_log(self, text: str):
        now     = _utc_now()
        current = self.settings_log.toPlainText().strip()
        line    = f"[{now}] {text}"
        self.settings_log.setPlainText(line if not current else current + "\n" + line)

    def _save_defaults(self):
        defaults = {
            "default_drone":           str(self.default_drone.currentData()),
            "default_payload":         str(self.default_payload.currentData()),
            "unit_system":             str(self.unit_system.currentData()),
            "default_min_altitude_m":  float(self.default_min_alt.value()),
            "default_max_altitude_m":  float(self.default_max_alt.value()),
            "default_rth_altitude_m":  float(self.default_rth.value()),
            "org_role":                str(self.role.currentData()),
            "team_workflow_enabled":   bool(self.team_workflow.isChecked()),
            "mission_sharing_enabled": bool(self.allow_sharing.isChecked()),
            "offline_region":          self.offline_region.text().strip(),
        }
        self.session.store.set_setting("ui_defaults", defaults)
        self._append_log("Defaults saved.")

    def _load_defaults(self):
        defaults = self.session.store.get_setting("ui_defaults", {})
        if not isinstance(defaults, dict):
            return
        for combo, key in (
            (self.default_drone,   "default_drone"),
            (self.default_payload, "default_payload"),
            (self.unit_system,     "unit_system"),
            (self.role,            "org_role"),
        ):
            idx = combo.findData(defaults.get(key))
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self.default_min_alt.setValue(_safe_float(defaults.get("default_min_altitude_m"), 30.0))
        self.default_max_alt.setValue(_safe_float(defaults.get("default_max_altitude_m"), 120.0))
        self.default_rth.setValue(_safe_float(defaults.get("default_rth_altitude_m"), 120.0))
        self.team_workflow.setChecked(bool(defaults.get("team_workflow_enabled", True)))
        self.allow_sharing.setChecked(bool(defaults.get("mission_sharing_enabled", True)))
        self.offline_region.setText(str(defaults.get("offline_region", "")))
        self._validate_altitude()

    def _download_cache(self):
        project = self.session.active_project or {}
        if not project:
            QMessageBox.warning(self, "Settings", "No active project selected.")
            return
        root    = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
        cache   = root / "offline_cache"
        cache.mkdir(parents=True, exist_ok=True)
        region  = self.offline_region.text().strip() or "unspecified_region"
        payload = {
            "region": region,
            "basemap_tiles": "cached_placeholder",
            "elevation": "cached_placeholder",
            "cached_at_utc": _utc_now(),
        }
        (cache / "map_cache_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_log(f"Offline cache manifest saved for region '{region}'.")

    def _import_mission_file(self):
        if not self.allow_sharing.isChecked():
            QMessageBox.warning(self, "Settings", "Mission sharing is disabled.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import Mission File", "", "JSON (*.json);;All Files (*)")
        if not path:
            return
        project = self.session.active_project or {}
        if not project:
            QMessageBox.warning(self, "Settings", "No active project selected.")
            return
        root    = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
        dst_dir = root / "mission_imports"
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        self._append_log(f"Mission file imported: {dst}")

    def _export_latest_mission_file(self):
        if not self.allow_sharing.isChecked():
            QMessageBox.warning(self, "Settings", "Mission sharing is disabled.")
            return
        rows = self.session.list_mission_versions()
        if not rows:
            QMessageBox.warning(self, "Settings", "No mission versions available to export.")
            return
        latest = rows[0]
        try:
            payload = json.loads(str(latest.get("flight_recipe_json", "{}")))
        except Exception as exc:
            QMessageBox.critical(self, "Settings", f"Invalid mission payload:\n{exc}")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Mission File",
            f"{latest.get('mission_name', 'mission')}_v{latest.get('version_num', 0)}.json",
            "JSON (*.json);;All Files (*)",
        )
        if not out_path:
            return
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_log(f"Mission exported: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Templates tab
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowTemplatesTab(QWidget):
    """Browse and apply the 11 built-in workflow templates."""

    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._templates: list[Any] = []
        self._build_ui()
        self._load_templates()
        self.session.projectChanged.connect(lambda _p: self._load_templates())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("appBar")
        header.setFixedHeight(54)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(16, 0, 16, 0)
        hb.setSpacing(12)
        lbl = QLabel("Workflow Templates")
        lbl.setObjectName("secondary")
        hb.addWidget(lbl)
        hb.addStretch(1)
        self.btn_assign = _primary_btn("Assign to Project")
        self.btn_assign.setFixedWidth(160)
        self.btn_assign.setEnabled(False)
        self.btn_assign.clicked.connect(self._assign_template)
        hb.addWidget(self.btn_assign)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        # Left: template list
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(6)
        ll.addWidget(_section_label("Templates"))
        self.tmpl_list = QListWidget()
        self.tmpl_list.currentRowChanged.connect(self._on_template_selected)
        ll.addWidget(self.tmpl_list, stretch=1)
        splitter.addWidget(left)

        # Right: detail
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        rl.addWidget(_section_label("Template Details"))

        detail_card = QGroupBox("Overview")
        dc = QVBoxLayout(detail_card)
        self.lbl_tmpl_name = QLabel("—")
        self.lbl_tmpl_name.setStyleSheet("font-size: 13pt; font-weight: 700;")
        self.lbl_tmpl_desc = QLabel("Select a template.")
        self.lbl_tmpl_desc.setWordWrap(True)
        self.lbl_tmpl_desc.setObjectName("secondary")
        dc.addWidget(self.lbl_tmpl_name)
        dc.addWidget(self.lbl_tmpl_desc)
        rl.addWidget(detail_card)

        stages_card = QGroupBox("Processing Stages")
        sc = QVBoxLayout(stages_card)
        self.lbl_stages = QLabel("—")
        self.lbl_stages.setWordWrap(True)
        self.lbl_stages.setObjectName("secondary")
        sc.addWidget(self.lbl_stages)
        rl.addWidget(stages_card)

        inputs_card = QGroupBox("Required Inputs")
        ic = QVBoxLayout(inputs_card)
        self.lbl_inputs = QLabel("—")
        self.lbl_inputs.setWordWrap(True)
        self.lbl_inputs.setObjectName("secondary")
        ic.addWidget(self.lbl_inputs)
        rl.addWidget(inputs_card)

        params_card = QGroupBox("Default Parameters")
        pc = QVBoxLayout(params_card)
        self.lbl_params = QPlainTextEdit()
        self.lbl_params.setReadOnly(True)
        self.lbl_params.setMaximumHeight(120)
        self.lbl_params.setStyleSheet(
            "QPlainTextEdit { background: #0f1c2e; border: 1px solid #192535;"
            " font-size: 8.5pt; color: #5a7898; border-radius: 6px; }"
        )
        pc.addWidget(self.lbl_params)
        rl.addWidget(params_card)

        rl.addStretch(1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

    def _load_templates(self):
        try:
            from core.workflows import list_workflow_templates
            project = self.session.active_project or {}
            workspace = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
            self._templates = list_workflow_templates(workspace)
        except Exception:
            self._templates = []

        self.tmpl_list.blockSignals(True)
        self.tmpl_list.clear()
        for t in self._templates:
            self.tmpl_list.addItem(f"{t.name}")
        self.tmpl_list.blockSignals(False)

        if self._templates:
            self.tmpl_list.setCurrentRow(0)
            self._on_template_selected(0)

    def _on_template_selected(self, row: int):
        if row < 0 or row >= len(self._templates):
            self.btn_assign.setEnabled(False)
            return
        t = self._templates[row]
        self.lbl_tmpl_name.setText(t.name)
        self.lbl_tmpl_desc.setText(t.description)
        self.lbl_stages.setText(", ".join(t.processing_stages) if t.processing_stages else "—")
        self.lbl_inputs.setText(", ".join(t.required_inputs) if t.required_inputs else "—")
        params_text = json.dumps(t.default_parameters, indent=2) if t.default_parameters else "{}"
        self.lbl_params.setPlainText(params_text)
        self.btn_assign.setEnabled(True)

    def _assign_template(self):
        row = self.tmpl_list.currentRow()
        if row < 0 or row >= len(self._templates):
            return
        t = self._templates[row]
        project = self.session.active_project or {}
        if not project:
            QMessageBox.warning(self, "Workflows", "No active project selected.")
            return
        try:
            self.session.store.set_setting(
                f"project_{project.get('id')}_workflow", {"template_id": t.id, "template_name": t.name}
            )
            self.session.statusChanged.emit(f"Workflow '{t.name}' assigned to project.")
            QMessageBox.information(
                self, "Workflows", f"Template '{t.name}' assigned to project '{project.get('name', '')}'."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Workflows", str(exc))
