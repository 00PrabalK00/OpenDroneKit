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
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from final_toolkit.mission import MissionPlan

from .mission_planner import MissionPlannerTab
from .project_store import ProjectStore
from .theme import standard_icon


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


@dataclass
class MissionEstimate:
    distance_m: float = 0.0
    time_min: float = 0.0
    batteries: int = 0
    coverage_pct: float = 0.0


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

    def __init__(self, store: ProjectStore | None = None):
        super().__init__()
        self.store = store or ProjectStore()
        self.active_project: dict[str, Any] | None = None
        self.current_plan: MissionPlan | None = None
        self.active_dataset_dir: str = ""
        self.active_image_path: str = ""
        self.latest_run_artifacts: dict[str, Any] = {}
        self.ensure_active_project()

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
        folder_path = Path(folder)
        if not folder_path.exists():
            raise ValueError(f"Dataset path does not exist: {folder_path}")
        row = self.store.save_dataset_entry(
            project_id=pid,
            name=str(name).strip() or folder_path.name,
            path=str(folder_path),
            metadata=metadata or {},
        )
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

    def _build_ui(self):
        root = QVBoxLayout(self)

        status_row = QHBoxLayout()
        self.active_project_label = QLabel("Active project: [none]")
        self.offline_badge = QLabel("Offline Mode: Enabled")
        self.offline_badge.setStyleSheet("QLabel { color: #0b8f5d; font-weight: 600; }")
        self.sync_status_label = QLabel("Sync: offline")
        status_row.addWidget(self.active_project_label, stretch=1)
        status_row.addWidget(self.offline_badge)
        status_row.addWidget(self.sync_status_label)
        root.addLayout(status_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        project_group = QGroupBox("Projects")
        project_layout = QVBoxLayout(project_group)
        self.projects_list = QListWidget()
        self.projects_list.currentItemChanged.connect(self._on_project_selected)
        project_layout.addWidget(self.projects_list)
        project_buttons = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setIcon(standard_icon(self, "SP_BrowserReload", "SP_DialogResetButton"))
        self.btn_refresh.clicked.connect(self._refresh)
        self.btn_set_active = QPushButton("Set Active")
        self.btn_set_active.setIcon(standard_icon(self, "SP_DialogApplyButton", "SP_DialogYesButton"))
        self.btn_set_active.clicked.connect(self._set_active_from_selection)
        project_buttons.addWidget(self.btn_refresh)
        project_buttons.addWidget(self.btn_set_active)
        project_layout.addLayout(project_buttons)
        left_layout.addWidget(project_group)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        create_group = QGroupBox("Create / Configure Project")
        create_form = QFormLayout(create_group)
        self.new_project_name = QLineEdit()
        self.new_project_name.setPlaceholderText("e.g., Bridge Corridor - Feb 2026")
        create_form.addRow("Name:", self.new_project_name)
        root_row = QHBoxLayout()
        self.new_project_root = QLineEdit()
        self.new_project_root.setPlaceholderText("Optional custom project root folder")
        btn_root = QPushButton("Browse")
        btn_root.clicked.connect(self._browse_project_root)
        root_row.addWidget(self.new_project_root)
        root_row.addWidget(btn_root)
        create_form.addRow("Root:", root_row)
        self.new_project_desc = QLineEdit()
        self.new_project_desc.setPlaceholderText("Short description")
        create_form.addRow("Description:", self.new_project_desc)
        actions = QHBoxLayout()
        self.btn_create = QPushButton("Create Project")
        self.btn_create.setIcon(standard_icon(self, "SP_FileDialogNewFolder", "SP_DirIcon"))
        self.btn_create.clicked.connect(self._create_project)
        self.btn_download_offline = QPushButton("Download Map For Offline")
        self.btn_download_offline.setIcon(standard_icon(self, "SP_DialogOpenButton", "SP_DriveHDIcon"))
        self.btn_download_offline.clicked.connect(self._download_offline_region)
        self.btn_mark_synced = QPushButton("Mark Sync Complete")
        self.btn_mark_synced.setIcon(standard_icon(self, "SP_DialogApplyButton", "SP_DialogYesButton"))
        self.btn_mark_synced.clicked.connect(lambda: self.session.set_sync_status("synced"))
        actions.addWidget(self.btn_create)
        actions.addWidget(self.btn_download_offline)
        actions.addWidget(self.btn_mark_synced)
        create_form.addRow("", actions)
        right_layout.addWidget(create_group)

        details_group = QGroupBox("Project Details & Audit Trail")
        details_layout = QVBoxLayout(details_group)
        self.project_details = QPlainTextEdit()
        self.project_details.setReadOnly(True)
        details_layout.addWidget(self.project_details)
        right_layout.addWidget(details_group)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _selected_project_id(self) -> int | None:
        item = self.projects_list.currentItem()
        if item is None:
            return None
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid is None:
            return None
        return _safe_int(pid, default=-1)

    def _refresh(self):
        projects = self.session.list_projects()
        self.projects_list.blockSignals(True)
        self.projects_list.clear()
        active_pid = self.session.active_project_id()
        active_row = -1
        for idx, row in enumerate(projects):
            name = str(row.get("name", "[unnamed]"))
            mv = _safe_int(row.get("mission_versions_count", 0))
            ds = _safe_int(row.get("datasets_count", 0))
            rp = _safe_int(row.get("reports_count", 0))
            text = f"{name} | missions:{mv} datasets:{ds} reports:{rp}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, _safe_int(row.get("id"), default=-1))
            self.projects_list.addItem(item)
            if active_pid is not None and _safe_int(row.get("id"), default=-1) == active_pid:
                active_row = idx
        self.projects_list.blockSignals(False)
        if active_row >= 0:
            self.projects_list.setCurrentRow(active_row)
        elif self.projects_list.count() > 0:
            self.projects_list.setCurrentRow(0)
        self._update_details()

    def _on_project_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None):
        del current, _previous
        self._update_details()

    def _set_active_from_selection(self):
        pid = self._selected_project_id()
        if pid is None or pid < 0:
            QMessageBox.warning(self, "Project", "Select a project first.")
            return
        try:
            self.session.set_active_project(pid)
        except Exception as exc:
            QMessageBox.critical(self, "Project", str(exc))

    def _on_project_changed(self, _project: dict):
        self._refresh()

    def _browse_project_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select project root folder")
        if path:
            self.new_project_root.setText(path)

    def _create_project(self):
        name = self.new_project_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Project", "Project name is required.")
            return
        try:
            self.session.create_project(
                name=name,
                root_dir=self.new_project_root.text().strip(),
                description=self.new_project_desc.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Project", str(exc))
            return
        self.new_project_name.clear()
        self.new_project_root.clear()
        self.new_project_desc.clear()
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
        QMessageBox.information(self, "Offline Cache", f"Offline cache manifest written:\n{cache_dir}")

    def _update_details(self):
        project = self.session.active_project
        if not isinstance(project, dict):
            self.active_project_label.setText("Active project: [none]")
            self.sync_status_label.setText("Sync: offline")
            self.project_details.setPlainText("No project selected.")
            return

        self.active_project_label.setText(f"Active project: {project.get('name', '[unnamed]')}")
        self.sync_status_label.setText(f"Sync: {project.get('sync_status', 'offline')}")

        events = self.session.list_audit_events(limit=40)
        lines = [
            f"Project ID: {_safe_int(project.get('id'), default=-1)}",
            f"Root: {project.get('root_dir', '')}",
            f"Description: {project.get('description', '')}",
            f"Created: {project.get('created_at', '')}",
            f"Updated: {project.get('updated_at', '')}",
            "",
            "Recent audit events:",
        ]
        if not events:
            lines.append("- [none]")
        for ev in events[:20]:
            payload = {}
            try:
                payload = json.loads(str(ev.get("payload_json", "{}")))
            except Exception:
                payload = {}
            lines.append(
                f"- {ev.get('created_at', '')} | {ev.get('event_type', '')} | {json.dumps(payload, ensure_ascii=True)}"
            )
        self.project_details.setPlainText("\n".join(lines))


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
        root.setSpacing(8)

        context = QGroupBox("Mission Context")
        context_form = QFormLayout(context)
        self.project_label = QLabel("Project: [none]")
        context_form.addRow("", self.project_label)
        self.mission_name = QLineEdit("Mission A")
        context_form.addRow("Mission Name:", self.mission_name)
        self.mission_note = QLineEdit()
        self.mission_note.setPlaceholderText("Version note (what changed)")
        context_form.addRow("Note:", self.mission_note)

        ctx_buttons = QHBoxLayout()
        self.btn_save_version = QPushButton("Save Mission Version")
        self.btn_save_version.clicked.connect(self._save_current_version)
        self.btn_refresh_versions = QPushButton("Refresh Versions")
        self.btn_refresh_versions.clicked.connect(self._refresh_versions)
        self.btn_load_version = QPushButton("Load Selected Version (Repeat)")
        self.btn_load_version.clicked.connect(self._load_selected_version)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self._undo_version)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self._redo_version)
        ctx_buttons.addWidget(self.btn_save_version)
        ctx_buttons.addWidget(self.btn_refresh_versions)
        ctx_buttons.addWidget(self.btn_load_version)
        ctx_buttons.addWidget(self.btn_undo)
        ctx_buttons.addWidget(self.btn_redo)
        context_form.addRow("", ctx_buttons)

        self.version_list = QListWidget()
        self.version_list.setMaximumHeight(120)
        context_form.addRow("History:", self.version_list)
        root.addWidget(context)

        estimate = QFrame()
        estimate.setFrameShape(QFrame.Shape.StyledPanel)
        estimate_layout = QHBoxLayout(estimate)
        estimate_layout.setContentsMargins(8, 6, 8, 6)
        self.ribbon_time = QLabel("ETA: -")
        self.ribbon_distance = QLabel("Distance: -")
        self.ribbon_batt = QLabel("Batteries: -")
        self.ribbon_cov = QLabel("Coverage: -")
        estimate_layout.addWidget(self.ribbon_time)
        estimate_layout.addWidget(self.ribbon_distance)
        estimate_layout.addWidget(self.ribbon_batt)
        estimate_layout.addWidget(self.ribbon_cov)
        estimate_layout.addStretch(1)
        root.addWidget(estimate)

        root.addWidget(self.planner_tab, stretch=1)

    def _wire_signals(self):
        self.planner_tab.planGenerated.connect(self._on_plan_generated)
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.missionVersionSaved.connect(lambda _row: self._refresh_versions())

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "[none]") if isinstance(project, dict) else "[none]"
        self.project_label.setText(f"Project: {name}")
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
            v = _safe_int(row.get("version_num"), default=0)
            template = str(row.get("template", "grid"))
            created = str(row.get("created_at", ""))
            note = str(row.get("note", ""))
            item = QListWidgetItem(f"v{v} | {template} | {created} | {note}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.version_list.addItem(item)
        if self.version_list.count() > 0:
            self.version_list.setCurrentRow(0)

    def _selected_version_row(self) -> dict[str, Any] | None:
        item = self.version_list.currentItem()
        if item is None:
            return None
        payload = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            return payload
        return None

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
        self.planner_tab.repeat_recipe_path = f"db://mission_versions/{row.get('id')}"
        self.planner_tab.recipe_label.setText(
            f"Loaded: {recipe_obj.recipe_id} v{recipe_obj.version} (from history)"
        )
        self.planner_tab.repeat_mode.setChecked(True)
        try:
            self.planner_tab.generate_plan()
        except Exception as exc:
            QMessageBox.critical(self, "Mission", f"Failed to regenerate mission from version:\n{exc}")
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
        nxt = cur - 1
        self.version_list.setCurrentRow(nxt)
        row = self._selected_version_row()
        if row is not None:
            self._load_version_payload(row)


class FlyTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._eta_seconds = 0
        self._waypoint_idx = 0
        self._waypoint_count = 0
        self._running = False
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self.session.missionPlanChanged.connect(self._on_plan_changed)
        self.session.projectChanged.connect(self._on_project_changed)
        self._on_project_changed(self.session.active_project or {})

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        top = QGroupBox("Preflight Funnel")
        form = QFormLayout(top)
        self.fly_project_label = QLabel("Project: [none]")
        form.addRow("", self.fly_project_label)
        self.chk_connection = QCheckBox("Drone connected")
        self.chk_gps = QCheckBox("GPS/RTK lock")
        self.chk_rth = QCheckBox("RTH altitude reviewed")
        self.chk_oa = QCheckBox("Obstacle avoidance profile confirmed")
        self.chk_geofence = QCheckBox("Geofence + no-fly validated")
        self.chk_battery = QCheckBox("Battery health verified")
        for chk in (
            self.chk_connection,
            self.chk_gps,
            self.chk_rth,
            self.chk_oa,
            self.chk_geofence,
            self.chk_battery,
        ):
            form.addRow("", chk)

        self.fly_summary = QLabel("No mission loaded.")
        self.fly_summary.setWordWrap(True)
        form.addRow("Mission:", self.fly_summary)

        start_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Mission")
        self.btn_run.clicked.connect(self._run_clicked)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_resume = QPushButton("Resume")
        self.btn_resume.clicked.connect(self._resume)
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.clicked.connect(self._abort)
        self.btn_rth = QPushButton("RTH")
        self.btn_rth.clicked.connect(self._rth)
        start_row.addWidget(self.btn_run)
        start_row.addWidget(self.btn_pause)
        start_row.addWidget(self.btn_resume)
        start_row.addWidget(self.btn_abort)
        start_row.addWidget(self.btn_rth)
        form.addRow("", start_row)

        root.addWidget(top)

        hud = QGroupBox("Live HUD")
        hud_form = QFormLayout(hud)
        self.hud_wp = QLabel("Waypoint: -")
        self.hud_eta = QLabel("ETA: -")
        self.hud_battery = QLabel("Battery: -")
        self.hud_link = QLabel("Link: -")
        self.hud_mode = QLabel("Mode: idle")
        hud_form.addRow("Current:", self.hud_wp)
        hud_form.addRow("ETA:", self.hud_eta)
        hud_form.addRow("Battery:", self.hud_battery)
        hud_form.addRow("Link:", self.hud_link)
        hud_form.addRow("State:", self.hud_mode)
        root.addWidget(hud)

        resume = QGroupBox("Battery Swap / Resume")
        resume_form = QFormLayout(resume)
        self.resume_segment = QSpinBox()
        self.resume_segment.setRange(0, 99999)
        self.resume_segment.setValue(0)
        resume_form.addRow("Last Completed Segment:", self.resume_segment)
        self.btn_resume_segment = QPushButton("Resume From Segment")
        self.btn_resume_segment.clicked.connect(self._resume_from_segment)
        resume_form.addRow("", self.btn_resume_segment)
        root.addWidget(resume)

        self.fly_log = QPlainTextEdit()
        self.fly_log.setReadOnly(True)
        root.addWidget(self.fly_log, stretch=1)

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "[none]") if isinstance(project, dict) else "[none]"
        self.fly_project_label.setText(f"Project: {name}")

    def _on_plan_changed(self, plan: MissionPlan):
        self._waypoint_count = max(0, len(plan.waypoints))
        self._waypoint_idx = 0
        self._eta_seconds = int(max(0.0, float(plan.estimated_time_min)) * 60.0)
        self.fly_summary.setText(
            f"Template {plan.template} | {len(plan.waypoints)} waypoints | "
            f"{plan.path_distance_m:.1f} m | ETA {plan.estimated_time_min:.2f} min"
        )
        self.hud_wp.setText(f"Waypoint: 0/{self._waypoint_count}")
        self.hud_eta.setText(f"ETA: {self._eta_seconds}s")
        self.hud_battery.setText("Battery: 100%")
        self.hud_link.setText("Link: strong")

    def _check_preflight(self) -> tuple[bool, list[str]]:
        checks = {
            "Drone connected": self.chk_connection.isChecked(),
            "GPS/RTK lock": self.chk_gps.isChecked(),
            "RTH altitude reviewed": self.chk_rth.isChecked(),
            "Obstacle avoidance profile confirmed": self.chk_oa.isChecked(),
            "Geofence + no-fly validated": self.chk_geofence.isChecked(),
            "Battery health verified": self.chk_battery.isChecked(),
        }
        missing = [name for name, ok in checks.items() if not ok]
        return len(missing) == 0, missing

    def _append_log(self, text: str):
        current = self.fly_log.toPlainText().strip()
        if current:
            self.fly_log.setPlainText(current + "\n" + text)
        else:
            self.fly_log.setPlainText(text)
        self.fly_log.verticalScrollBar().setValue(self.fly_log.verticalScrollBar().maximum())

    def _run_clicked(self):
        if self.session.current_plan is None:
            QMessageBox.warning(self, "Fly", "Generate a mission in Missions first.")
            return
        ok, missing = self._check_preflight()
        if not ok:
            QMessageBox.warning(self, "Preflight Blocked", "Missing checks:\n- " + "\n- ".join(missing))
            return
        self._running = True
        self.hud_mode.setText("Mode: running")
        self._append_log(f"[{_utc_now()}] Mission started")
        self._timer.start()

    def _pause(self):
        if not self._running:
            return
        self._timer.stop()
        self.hud_mode.setText("Mode: paused")
        self._append_log(f"[{_utc_now()}] Mission paused")

    def _resume(self):
        if not self._running:
            return
        self._timer.start()
        self.hud_mode.setText("Mode: running")
        self._append_log(f"[{_utc_now()}] Mission resumed")

    def _abort(self):
        if not self._running:
            return
        self._timer.stop()
        self._running = False
        self.hud_mode.setText("Mode: aborted")
        self._append_log(f"[{_utc_now()}] Mission aborted")

    def _rth(self):
        if not self._running:
            return
        self._timer.stop()
        self._running = False
        self.hud_mode.setText("Mode: RTH")
        self._append_log(f"[{_utc_now()}] Return-to-home initiated")

    def _resume_from_segment(self):
        seg = int(self.resume_segment.value())
        self._append_log(f"[{_utc_now()}] Resume requested from segment {seg}")
        self.hud_mode.setText("Mode: resume-pending")

    def _tick(self):
        if not self._running:
            return
        if self._eta_seconds > 0:
            self._eta_seconds -= 1
        if self._waypoint_count > 0 and self._waypoint_idx < self._waypoint_count:
            step_interval = max(1, int(max(1, self._eta_seconds + 1) / max(1, self._waypoint_count - self._waypoint_idx)))
            if self._eta_seconds % step_interval == 0:
                self._waypoint_idx = min(self._waypoint_count, self._waypoint_idx + 1)
        battery_pct = max(5, int(100.0 * max(0.0, float(self._eta_seconds)) / max(1.0, float(max(self._eta_seconds, 1) + 120))))
        self.hud_wp.setText(f"Waypoint: {self._waypoint_idx}/{self._waypoint_count}")
        self.hud_eta.setText(f"ETA: {self._eta_seconds}s")
        self.hud_battery.setText(f"Battery: {battery_pct}%")
        self.hud_link.setText("Link: strong")
        if self._eta_seconds <= 0 or (self._waypoint_count > 0 and self._waypoint_idx >= self._waypoint_count):
            self._timer.stop()
            self._running = False
            self.hud_mode.setText("Mode: completed")
            self._append_log(f"[{_utc_now()}] Mission complete")


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
        root.setSpacing(8)

        top = QGroupBox("Data Project")
        form = QFormLayout(top)
        self.data_project_label = QLabel("Project: [none]")
        form.addRow("", self.data_project_label)

        filter_row = QHBoxLayout()
        self.filter_mission = QComboBox()
        self.filter_mission.addItem("All missions", "")
        self.filter_date = QComboBox()
        self.filter_date.addItem("Any date", "")
        self.filter_tag = QComboBox()
        self.filter_tag.addItem("Any QA tag", "")
        self.btn_apply_filter = QPushButton("Apply Filters")
        self.btn_apply_filter.clicked.connect(self._apply_filters)
        filter_row.addWidget(QLabel("Mission:"))
        filter_row.addWidget(self.filter_mission)
        filter_row.addWidget(QLabel("Date:"))
        filter_row.addWidget(self.filter_date)
        filter_row.addWidget(QLabel("Tag:"))
        filter_row.addWidget(self.filter_tag)
        filter_row.addWidget(self.btn_apply_filter)
        form.addRow("Filters:", filter_row)

        import_row = QHBoxLayout()
        self.btn_import_dataset = QPushButton("Import Image Dataset Folder")
        self.btn_import_dataset.clicked.connect(self._import_dataset_folder)
        self.btn_refresh_tree = QPushButton("Refresh")
        self.btn_refresh_tree.clicked.connect(self._refresh_tree)
        import_row.addWidget(self.btn_import_dataset)
        import_row.addWidget(self.btn_refresh_tree)
        form.addRow("", import_row)
        root.addWidget(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QGroupBox("Project Tree")
        left_layout = QVBoxLayout(left)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Project Assets"])
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self.tree)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_list = QListWidget()
        self.photo_list.currentTextChanged.connect(self._on_photo_selected)
        right_layout.addWidget(self.photo_list)

        preview_row = QHBoxLayout()
        self.preview = QLabel("Photo preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        preview_row.addWidget(self.preview, stretch=2)
        self.qa_box = QPlainTextEdit()
        self.qa_box.setReadOnly(True)
        preview_row.addWidget(self.qa_box, stretch=1)
        right_layout.addLayout(preview_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _wire_signals(self):
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.datasetImported.connect(lambda _row: self._refresh_tree())
        self.session.missionVersionSaved.connect(lambda _row: self._refresh_tree())
        self._on_project_changed(self.session.active_project or {})

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "[none]") if isinstance(project, dict) else "[none]"
        self.data_project_label.setText(f"Project: {name}")
        self._refresh_tree()

    def _refresh_tree(self):
        self.tree.clear()
        datasets = self.session.list_datasets()
        versions = self.session.list_mission_versions()

        root_datasets = QTreeWidgetItem(["Datasets"])
        for ds in datasets:
            item = QTreeWidgetItem([f"{ds.get('name', '')} | {ds.get('captured_at', '')}"])
            item.setData(0, Qt.ItemDataRole.UserRole, {"type": "dataset", "row": ds})
            root_datasets.addChild(item)
        self.tree.addTopLevelItem(root_datasets)

        root_missions = QTreeWidgetItem(["Mission Versions"])
        mission_names = sorted({str(v.get("mission_name", "mission")) for v in versions})
        for name in mission_names:
            group = QTreeWidgetItem([name])
            for v in [r for r in versions if str(r.get("mission_name", "")) == name][:30]:
                child = QTreeWidgetItem([f"v{v.get('version_num', 0)} | {v.get('template', '')} | {v.get('created_at', '')}"])
                child.setData(0, Qt.ItemDataRole.UserRole, {"type": "mission_version", "row": v})
                group.addChild(child)
            root_missions.addChild(group)
        self.tree.addTopLevelItem(root_missions)
        self.tree.expandAll()

        self.filter_mission.blockSignals(True)
        self.filter_mission.clear()
        self.filter_mission.addItem("All missions", "")
        for name in mission_names:
            self.filter_mission.addItem(name, name)
        self.filter_mission.blockSignals(False)

    def _import_dataset_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if not path:
            return
        root = Path(path)
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        files.sort()
        if not files:
            QMessageBox.warning(self, "Data", "No supported images found in selected folder.")
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

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int):
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        kind = str(payload.get("type", ""))
        row = payload.get("row", {})
        if kind == "dataset" and isinstance(row, dict):
            path = Path(str(row.get("path", "")))
            if path.exists() and path.is_dir():
                files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
                files.sort()
                self.session.set_active_dataset(str(path))
                self._load_image_list([str(p) for p in files])
        elif kind == "mission_version" and isinstance(row, dict):
            note = str(row.get("note", ""))
            self.qa_box.setPlainText(
                "\n".join(
                    [
                        "Mission version selected",
                        f"Mission: {row.get('mission_name', '')}",
                        f"Version: {row.get('version_num', 0)}",
                        f"Template: {row.get('template', '')}",
                        f"Created: {row.get('created_at', '')}",
                        f"Note: {note}",
                    ]
                )
            )

    def _load_image_list(self, paths: list[str]):
        self._current_image_paths = paths
        self.photo_list.clear()
        self.photo_list.addItems([Path(p).name for p in paths])
        if paths:
            self.photo_list.setCurrentRow(0)

    def _on_photo_selected(self, name: str):
        if not name:
            return
        match = None
        for p in self._current_image_paths:
            if Path(p).name == name:
                match = p
                break
        if match is None:
            return
        self.session.set_active_image(match)
        img = cv2.imread(match, cv2.IMREAD_COLOR)
        if img is None:
            self.preview.setText("Unable to load image")
            self.preview.setPixmap(QPixmap())
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
            "\n".join(
                [
                    f"Image: {name}",
                    f"Sharpness (Laplacian variance): {qa['sharpness']:.2f}",
                    f"Dark clip: {qa['dark_clip_pct']:.2f}%",
                    f"Bright clip: {qa['bright_clip_pct']:.2f}%",
                    f"Exposure status: {qa['exposure_status']}",
                    f"Blur status: {qa['blur_status']}",
                    "Map-linked review: select mission version in tree for context.",
                ]
            )
        )

    def _quick_qa(self, image_bgr) -> dict[str, Any]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = float(max(1.0, hist.sum()))
        dark_clip = float(hist[:5].sum() / total * 100.0)
        bright_clip = float(hist[-5:].sum() / total * 100.0)
        exposure_status = "ok"
        if dark_clip > 5.0:
            exposure_status = "under-exposed"
        elif bright_clip > 5.0:
            exposure_status = "over-exposed"
        blur_status = "ok" if sharpness >= 80.0 else "potential blur"
        return {
            "sharpness": sharpness,
            "dark_clip_pct": dark_clip,
            "bright_clip_pct": bright_clip,
            "exposure_status": exposure_status,
            "blur_status": blur_status,
        }

    def _apply_filters(self):
        selected_mission = str(self.filter_mission.currentData() or "")
        selected_tag = str(self.filter_tag.currentData() or "")
        selected_date = str(self.filter_date.currentData() or "")
        text = (
            f"Filters applied:\n"
            f"- mission={selected_mission or 'all'}\n"
            f"- date={selected_date or 'any'}\n"
            f"- tag={selected_tag or 'any'}"
        )
        current = self.qa_box.toPlainText().strip()
        if current:
            self.qa_box.setPlainText(current + "\n\n" + text)
        else:
            self.qa_box.setPlainText(text)


class ReportsTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._build_ui()
        self._wire_signals()
        self._refresh_reports()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        top = QGroupBox("Report Builder")
        form = QFormLayout(top)
        self.report_project_label = QLabel("Project: [none]")
        form.addRow("", self.report_project_label)
        self.report_title = QLineEdit("Inspection Report")
        form.addRow("Title:", self.report_title)
        self.report_type = QComboBox()
        self.report_type.addItem("Standard", "standard")
        self.report_type.addItem("Advanced", "advanced")
        form.addRow("Type:", self.report_type)

        sections = QHBoxLayout()
        self.sec_overview = QCheckBox("Overview")
        self.sec_overview.setChecked(True)
        self.sec_map = QCheckBox("Map")
        self.sec_map.setChecked(True)
        self.sec_photos = QCheckBox("Key Photos")
        self.sec_photos.setChecked(True)
        self.sec_annotations = QCheckBox("Annotations")
        self.sec_annotations.setChecked(True)
        self.sec_measurements = QCheckBox("Measurements")
        self.sec_measurements.setChecked(True)
        for chk in (
            self.sec_overview,
            self.sec_map,
            self.sec_photos,
            self.sec_annotations,
            self.sec_measurements,
        ):
            sections.addWidget(chk)
        form.addRow("Sections:", sections)

        action_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Report")
        self.btn_generate.setIcon(standard_icon(self, "SP_DialogSaveButton", "SP_FileIcon"))
        self.btn_generate.clicked.connect(self._generate_report)
        self.btn_open_folder = QPushButton("Open Report Folder")
        self.btn_open_folder.setIcon(standard_icon(self, "SP_DirOpenIcon", "SP_DialogOpenButton"))
        self.btn_open_folder.clicked.connect(self._open_report_folder)
        action_row.addWidget(self.btn_generate)
        action_row.addWidget(self.btn_open_folder)
        form.addRow("", action_row)
        root.addWidget(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QGroupBox("Generated Reports")
        left_layout = QVBoxLayout(left)
        self.report_list = QListWidget()
        self.report_list.currentItemChanged.connect(self._on_report_selected)
        left_layout.addWidget(self.report_list)
        splitter.addWidget(left)

        right = QGroupBox("Preview")
        right_layout = QVBoxLayout(right)
        self.report_preview = QPlainTextEdit()
        self.report_preview.setReadOnly(True)
        right_layout.addWidget(self.report_preview)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _wire_signals(self):
        self.session.projectChanged.connect(self._on_project_changed)
        self.session.reportSaved.connect(lambda _row: self._refresh_reports())
        self._on_project_changed(self.session.active_project or {})

    def _on_project_changed(self, project: dict[str, Any]):
        name = project.get("name", "[none]") if isinstance(project, dict) else "[none]"
        self.report_project_label.setText(f"Project: {name}")
        self._refresh_reports()

    def _selected_sections(self) -> list[str]:
        out = []
        if self.sec_overview.isChecked():
            out.append("overview")
        if self.sec_map.isChecked():
            out.append("map")
        if self.sec_photos.isChecked():
            out.append("photos")
        if self.sec_annotations.isChecked():
            out.append("annotations")
        if self.sec_measurements.isChecked():
            out.append("measurements")
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
        title = self.report_title.text().strip() or "Inspection Report"
        versions = self.session.list_mission_versions()[:10]
        datasets = self.session.list_datasets()[:10]
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"report_{report_type}_{stamp}.md"

        lines: list[str] = [f"# {title}", "", f"- Project: {project.get('name', '')}", f"- Type: {report_type}", f"- Generated: {_utc_now()}", ""]
        if "overview" in sections:
            lines.extend(
                [
                    "## Overview",
                    f"- Mission versions: {len(versions)}",
                    f"- Datasets: {len(datasets)}",
                    f"- Current mission loaded: {'yes' if self.session.current_plan is not None else 'no'}",
                    "",
                ]
            )
        if "map" in sections:
            lines.append("## Mission Map Summary")
            if self.session.current_plan is not None:
                plan = self.session.current_plan
                lines.extend(
                    [
                        f"- Template: {plan.template}",
                        f"- Distance: {plan.path_distance_m:.1f} m",
                        f"- Estimated time: {plan.estimated_time_min:.2f} min",
                        f"- Waypoints: {len(plan.waypoints)}",
                    ]
                )
            else:
                lines.append("- No active mission plan.")
            lines.append("")
        if "photos" in sections:
            lines.append("## Key Photos")
            if datasets:
                newest = datasets[0]
                path = Path(str(newest.get("path", "")))
                images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS] if path.exists() else []
                images.sort()
                for p in images[:8]:
                    lines.append(f"- {p.name}")
            else:
                lines.append("- No datasets available.")
            lines.append("")
        if "annotations" in sections:
            lines.extend(["## Annotations", "- Add defect annotations and engineer notes here.", ""])
        if "measurements" in sections:
            lines.extend(["## Measurements", "- Add measured spans, widths, and hotspot deltas here.", ""])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        metadata = {
            "sections": sections,
            "share_link": f"local://project/{project.get('id')}/reports/{report_path.name}",
        }
        try:
            self.session.save_report(
                title=title,
                report_type=report_type,
                content_path=report_path,
                metadata=metadata,
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
            title = str(row.get("title", "Report"))
            rtype = str(row.get("report_type", "standard"))
            created = str(row.get("created_at", ""))
            item = QListWidgetItem(f"{title} | {rtype} | {created}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.report_list.addItem(item)
        if self.report_list.count() > 0:
            self.report_list.setCurrentRow(0)
        else:
            self.report_preview.setPlainText("No reports generated yet.")

    def _on_report_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None):
        if current is None:
            return
        row = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(row, dict):
            return
        path = Path(str(row.get("content_path", "")))
        if path.exists():
            self.report_preview.setPlainText(path.read_text(encoding="utf-8"))
        else:
            self.report_preview.setPlainText(f"Report file not found:\n{path}")

    def _open_report_folder(self):
        root = self._project_root()
        if root is None:
            QMessageBox.warning(self, "Reports", "No active project.")
            return
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(reports_dir.resolve().as_uri())


class SettingsTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._build_ui()
        self._load_defaults()
        self._wire_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        pilot = QGroupBox("Pilot Defaults")
        pilot_form = QFormLayout(pilot)
        self.default_drone = QComboBox()
        self.default_drone.addItem("Mavic 2 Pro", "mavic2pro")
        self.default_drone.addItem("Phantom 4 RTK", "phantom4rtk")
        self.default_drone.addItem("Custom", "custom")
        pilot_form.addRow("Drone:", self.default_drone)
        self.default_payload = QComboBox()
        self.default_payload.addItem("RGB", "rgb")
        self.default_payload.addItem("RGB + Thermal", "rgb_thermal")
        self.default_payload.addItem("RGB + LiDAR", "rgb_lidar")
        self.default_payload.addItem("RGB + Multispectral", "rgb_multispectral")
        pilot_form.addRow("Payload:", self.default_payload)
        self.unit_system = QComboBox()
        self.unit_system.addItem("Metric", "metric")
        self.unit_system.addItem("Imperial", "imperial")
        pilot_form.addRow("Units:", self.unit_system)

        self.default_min_alt = QDoubleSpinBox()
        self.default_min_alt.setRange(5.0, 300.0)
        self.default_min_alt.setValue(30.0)
        self.default_min_alt.setSuffix(" m")
        pilot_form.addRow("Min Altitude:", self.default_min_alt)
        self.default_max_alt = QDoubleSpinBox()
        self.default_max_alt.setRange(10.0, 600.0)
        self.default_max_alt.setValue(120.0)
        self.default_max_alt.setSuffix(" m")
        pilot_form.addRow("Max Altitude:", self.default_max_alt)
        self.default_rth = QDoubleSpinBox()
        self.default_rth.setRange(10.0, 600.0)
        self.default_rth.setValue(140.0)
        self.default_rth.setSuffix(" m")
        pilot_form.addRow("RTH Altitude:", self.default_rth)
        root.addWidget(pilot)

        org = QGroupBox("Org Defaults & Team")
        org_form = QFormLayout(org)
        self.role = QComboBox()
        self.role.addItem("Pilot", "pilot")
        self.role.addItem("Reviewer", "reviewer")
        self.role.addItem("Admin", "admin")
        org_form.addRow("Role:", self.role)
        self.team_workflow = QCheckBox("Enable team workflow gates (plan -> fly -> review -> report)")
        self.team_workflow.setChecked(True)
        org_form.addRow("", self.team_workflow)
        self.allow_sharing = QCheckBox("Allow mission file sharing (import/export)")
        self.allow_sharing.setChecked(True)
        org_form.addRow("", self.allow_sharing)
        root.addWidget(org)

        offline = QGroupBox("Offline Planning")
        offline_form = QFormLayout(offline)
        self.offline_badge = QLabel("Offline mode: active")
        self.offline_badge.setStyleSheet("QLabel { color: #0b8f5d; font-weight: 600; }")
        offline_form.addRow("", self.offline_badge)
        self.offline_region = QLineEdit()
        self.offline_region.setPlaceholderText("Region name for basemap/elevation cache")
        offline_form.addRow("Region:", self.offline_region)
        offline_buttons = QHBoxLayout()
        self.btn_download_cache = QPushButton("Download Offline Map Cache")
        self.btn_download_cache.setIcon(standard_icon(self, "SP_DialogOpenButton", "SP_DriveHDIcon"))
        self.btn_download_cache.clicked.connect(self._download_cache)
        self.btn_set_offline = QPushButton("Set Project Offline Ready")
        self.btn_set_offline.setIcon(standard_icon(self, "SP_DialogApplyButton", "SP_DialogYesButton"))
        self.btn_set_offline.clicked.connect(lambda: self.session.set_sync_status("offline_ready"))
        offline_buttons.addWidget(self.btn_download_cache)
        offline_buttons.addWidget(self.btn_set_offline)
        offline_form.addRow("", offline_buttons)
        root.addWidget(offline)

        io_group = QGroupBox("Mission File Portability")
        io_form = QFormLayout(io_group)
        io_btn_row = QHBoxLayout()
        self.btn_import_mission = QPushButton("Import Mission File")
        self.btn_import_mission.setIcon(standard_icon(self, "SP_DialogOpenButton", "SP_DirOpenIcon"))
        self.btn_import_mission.clicked.connect(self._import_mission_file)
        self.btn_export_mission = QPushButton("Export Latest Mission File")
        self.btn_export_mission.setIcon(standard_icon(self, "SP_DialogSaveButton", "SP_DriveFDIcon"))
        self.btn_export_mission.clicked.connect(self._export_latest_mission_file)
        io_btn_row.addWidget(self.btn_import_mission)
        io_btn_row.addWidget(self.btn_export_mission)
        io_form.addRow("", io_btn_row)
        self.settings_log = QPlainTextEdit()
        self.settings_log.setReadOnly(True)
        io_form.addRow("", self.settings_log)
        root.addWidget(io_group, stretch=1)

        footer = QHBoxLayout()
        self.btn_save_defaults = QPushButton("Save Defaults")
        self.btn_save_defaults.setIcon(standard_icon(self, "SP_DialogSaveButton", "SP_DialogApplyButton"))
        self.btn_save_defaults.clicked.connect(self._save_defaults)
        footer.addStretch(1)
        footer.addWidget(self.btn_save_defaults)
        root.addLayout(footer)

    def _wire_signals(self):
        self.session.projectChanged.connect(lambda _project: self._append_log("Active project updated."))

    def _append_log(self, text: str):
        now = _utc_now()
        current = self.settings_log.toPlainText().strip()
        line = f"[{now}] {text}"
        self.settings_log.setPlainText(line if not current else current + "\n" + line)

    def _save_defaults(self):
        defaults = {
            "default_drone": str(self.default_drone.currentData()),
            "default_payload": str(self.default_payload.currentData()),
            "unit_system": str(self.unit_system.currentData()),
            "default_min_altitude_m": float(self.default_min_alt.value()),
            "default_max_altitude_m": float(self.default_max_alt.value()),
            "default_rth_altitude_m": float(self.default_rth.value()),
            "org_role": str(self.role.currentData()),
            "team_workflow_enabled": bool(self.team_workflow.isChecked()),
            "mission_sharing_enabled": bool(self.allow_sharing.isChecked()),
            "offline_region": self.offline_region.text().strip(),
        }
        self.session.store.set_setting("ui_defaults", defaults)
        self._append_log("Defaults saved.")

    def _load_defaults(self):
        defaults = self.session.store.get_setting("ui_defaults", {})
        if not isinstance(defaults, dict):
            return
        idx = self.default_drone.findData(defaults.get("default_drone"))
        if idx >= 0:
            self.default_drone.setCurrentIndex(idx)
        idx = self.default_payload.findData(defaults.get("default_payload"))
        if idx >= 0:
            self.default_payload.setCurrentIndex(idx)
        idx = self.unit_system.findData(defaults.get("unit_system"))
        if idx >= 0:
            self.unit_system.setCurrentIndex(idx)
        self.default_min_alt.setValue(_safe_float(defaults.get("default_min_altitude_m"), 30.0))
        self.default_max_alt.setValue(_safe_float(defaults.get("default_max_altitude_m"), 120.0))
        self.default_rth.setValue(_safe_float(defaults.get("default_rth_altitude_m"), 140.0))
        idx = self.role.findData(defaults.get("org_role"))
        if idx >= 0:
            self.role.setCurrentIndex(idx)
        self.team_workflow.setChecked(bool(defaults.get("team_workflow_enabled", True)))
        self.allow_sharing.setChecked(bool(defaults.get("mission_sharing_enabled", True)))
        self.offline_region.setText(str(defaults.get("offline_region", "")))

    def _download_cache(self):
        project = self.session.active_project or {}
        if not project:
            QMessageBox.warning(self, "Settings", "No active project selected.")
            return
        root = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
        cache = root / "offline_cache"
        cache.mkdir(parents=True, exist_ok=True)
        region = self.offline_region.text().strip() or "unspecified_region"
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
            QMessageBox.warning(self, "Settings", "Mission sharing is disabled by org defaults.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mission File",
            "",
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        project = self.session.active_project or {}
        if not project:
            QMessageBox.warning(self, "Settings", "No active project selected.")
            return
        root = Path(str(project.get("root_dir", "final_toolkit_outputs/projects")))
        dst_dir = root / "mission_imports"
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        self._append_log(f"Mission file imported: {dst}")

    def _export_latest_mission_file(self):
        if not self.allow_sharing.isChecked():
            QMessageBox.warning(self, "Settings", "Mission sharing is disabled by org defaults.")
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
            self,
            "Export Latest Mission File",
            f"{latest.get('mission_name', 'mission')}_v{latest.get('version_num', 0)}.json",
            "JSON (*.json);;All Files (*)",
        )
        if not out_path:
            return
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_log(f"Mission file exported: {out_path}")
