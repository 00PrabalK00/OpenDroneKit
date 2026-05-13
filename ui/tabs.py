"""PyQt tabs for the finalized toolkit workflow."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QMessageBox,
)
MATPLOTLIB_AVAILABLE = False
MATPLOTLIB_ERROR = ""
FigureCanvas = None
Figure = None
for _backend in (
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
):
    try:
        if _backend.endswith("backend_qtagg"):
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        else:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        MATPLOTLIB_AVAILABLE = True
        MATPLOTLIB_ERROR = ""
        break
    except Exception as exc:  # pragma: no cover - optional UI dependency
        MATPLOTLIB_ERROR = str(exc)

from legacy_bridge import legacy_summary
from core.detection import (
    detect_cracks,
    detect_metal_defects,
    detect_solar_defects,
    detect_structural_defects,
    load_image,
)
from core.pipeline import PipelineConfig, run_pipeline
from core.propagation import CrackPropagationForecaster, PropagationPhysicsConfig
from core.reconstruction import CustomDroneReconstructor, available_reconstruction_profiles
from .theme import standard_icon
from .components import (
    Banner,
    EmptyState,
    PipelineStageCard,
    ReadinessCard,
    StatusChip,
    TelemetryLabel,
    h_separator,
    make_label,
)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ── Local button helpers ──────────────────────────────────────────────────────

def _primary_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("primary")
    btn.setMinimumHeight(34)
    return btn


def _ghost_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("ghost")
    btn.setMinimumHeight(32)
    return btn


def _section_lbl(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("sectionTitle")
    return lbl


def _pixmap_from_bgr(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _set_preview_from_path(label: QLabel, path: str, empty_text: str):
    p = Path(path)
    if not p.exists():
        label.setText(empty_text)
        label.setPixmap(QPixmap())
        return
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        label.setText(empty_text)
        label.setPixmap(QPixmap())
        return
    label.setPixmap(
        _pixmap_from_bgr(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )


def _first_image_in_dir(folder: str | Path) -> str:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return ""
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    files.sort()
    return str(files[0]) if files else ""


class PipelineRunWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, str)

    def __init__(self, image_dir: str, output_dir: str, config: PipelineConfig):
        super().__init__()
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.config = config

    @pyqtSlot()
    def run(self):
        try:
            def _progress(percent: int, message: str) -> None:
                self.progress.emit(int(percent), str(message))

            result = run_pipeline(
                image_dir=self.image_dir,
                output_dir=self.output_dir,
                config=self.config,
                progress_callback=_progress,
            )
            self.finished.emit(result.to_dict(), "")
        except Exception:
            self.finished.emit({}, traceback.format_exc())


class ReconstructionRunWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)

    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        crack_mask_dir: str | None,
        profile: str,
        execution_mode: str,
        use_cache: bool,
    ):
        super().__init__()
        self.image_dir = str(image_dir or "")
        self.output_dir = str(output_dir or "")
        self.crack_mask_dir = str(crack_mask_dir or "").strip() or None
        self.profile = str(profile or "standard")
        self.execution_mode = str(execution_mode or "local")
        self.use_cache = bool(use_cache)

    @pyqtSlot()
    def run(self):
        try:
            def _progress(percent: int, message: str) -> None:
                self.progress.emit(int(percent), str(message))

            recon = CustomDroneReconstructor(
                profile=self.profile,
                execution_mode=self.execution_mode,
                use_cache=self.use_cache,
            ).reconstruct(
                image_dir=self.image_dir,
                crack_mask_dir=self.crack_mask_dir,
                output_dir=self.output_dir or "final_toolkit_outputs/ui_reconstruction",
                progress_callback=_progress,
            )
            self.finished.emit(recon.to_dict(), "")
        except Exception:
            self.finished.emit({}, traceback.format_exc())


class CrackAnalysisTab(QWidget):
    def __init__(self, session: Any = None):
        super().__init__()
        self.session = session
        self.forecaster = CrackPropagationForecaster()
        self._build_ui()
        self._bind_session()

    def _bind_session(self):
        if self.session is None:
            return
        if hasattr(self.session, "activeImageChanged"):
            self.session.activeImageChanged.connect(self._on_active_image_changed)
        if hasattr(self.session, "active_image_path"):
            path = str(getattr(self.session, "active_image_path", "") or "")
            if path:
                self.image_path.setText(path)
        if hasattr(self.session, "active_dataset_dir") and not self.image_path.text().strip():
            first = _first_image_in_dir(str(getattr(self.session, "active_dataset_dir", "") or ""))
            if first:
                self.image_path.setText(first)

    def _on_active_image_changed(self, path: str):
        p = str(path or "").strip()
        if p:
            self.image_path.setText(p)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        # Left: setup panel (scrollable)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(_section_lbl("Crack Analysis Setup"))

        # Input section
        input_card = QGroupBox("Input Image")
        input_form = QFormLayout(input_card)
        input_form.setSpacing(10)
        image_row = QHBoxLayout()
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select drone image...")
        self.image_path.setMinimumHeight(32)
        browse = QPushButton("Browse")
        browse.setObjectName("ghost")
        browse.setFixedWidth(72)
        browse.setMinimumHeight(32)
        browse.clicked.connect(self._browse_image)
        image_row.addWidget(self.image_path)
        image_row.addWidget(browse)
        input_form.addRow("Image:", image_row)
        left_layout.addWidget(input_card)

        # Propagation model
        model_card = QGroupBox("Propagation Model")
        model_form = QFormLayout(model_card)
        model_form.setSpacing(10)

        self.steps = QSpinBox()
        self.steps.setRange(1, 30)
        self.steps.setValue(6)
        self.steps.setMinimumHeight(32)
        model_form.addRow("Forecast steps:", self.steps)

        self.propagation_mode = QComboBox()
        self.propagation_mode.addItem("Geometric (fast)", "geometric")
        self.propagation_mode.addItem("Physics Informed (detailed)", "physics_informed")
        self.propagation_mode.setMinimumHeight(32)
        model_form.addRow("Propagation mode:", self.propagation_mode)
        left_layout.addWidget(model_card)

        # Stress profile
        stress_card = QGroupBox("Stress Profile")
        stress_form = QFormLayout(stress_card)
        stress_form.setSpacing(10)

        self.prop_sigma = QDoubleSpinBox()
        self.prop_sigma.setRange(0.1, 2000.0)
        self.prop_sigma.setDecimals(2)
        self.prop_sigma.setValue(35.0)
        self.prop_sigma.setSuffix(" MPa")
        self.prop_sigma.setMinimumHeight(32)
        stress_form.addRow("Nominal stress σ:", self.prop_sigma)

        self.prop_delta_sigma = QDoubleSpinBox()
        self.prop_delta_sigma.setRange(0.01, 2000.0)
        self.prop_delta_sigma.setDecimals(2)
        self.prop_delta_sigma.setValue(12.0)
        self.prop_delta_sigma.setSuffix(" MPa")
        self.prop_delta_sigma.setMinimumHeight(32)
        stress_form.addRow("Stress range Δσ:", self.prop_delta_sigma)

        self.prop_horizon_years = QDoubleSpinBox()
        self.prop_horizon_years.setRange(0.01, 20.0)
        self.prop_horizon_years.setDecimals(1)
        self.prop_horizon_years.setValue(1.0)
        self.prop_horizon_years.setSuffix(" years")
        self.prop_horizon_years.setMinimumHeight(32)
        stress_form.addRow("Forecast horizon:", self.prop_horizon_years)

        self.prop_cycles_year = QDoubleSpinBox()
        self.prop_cycles_year.setRange(1.0, 1.0e9)
        self.prop_cycles_year.setDecimals(0)
        self.prop_cycles_year.setValue(1_000_000.0)
        self.prop_cycles_year.setSingleStep(10000.0)
        self.prop_cycles_year.setMinimumHeight(32)
        stress_form.addRow("Cycles per year:", self.prop_cycles_year)
        left_layout.addWidget(stress_card)

        # Advanced (collapsed visual grouping)
        adv_card = QGroupBox("Advanced Parameters (0 = use defaults)")
        adv_form = QFormLayout(adv_card)
        adv_form.setSpacing(8)

        self.prop_pixel_size_mm = QDoubleSpinBox()
        self.prop_pixel_size_mm.setRange(0.01, 20.0)
        self.prop_pixel_size_mm.setDecimals(3)
        self.prop_pixel_size_mm.setValue(0.5)
        self.prop_pixel_size_mm.setSingleStep(0.05)
        self.prop_pixel_size_mm.setSuffix(" mm/px")
        self.prop_pixel_size_mm.setMinimumHeight(32)
        adv_form.addRow("Pixel size:", self.prop_pixel_size_mm)

        self.prop_cycles_step = QDoubleSpinBox()
        self.prop_cycles_step.setRange(0.0, 1.0e12)
        self.prop_cycles_step.setDecimals(0)
        self.prop_cycles_step.setValue(0.0)
        self.prop_cycles_step.setSingleStep(10000.0)
        self.prop_cycles_step.setMinimumHeight(32)
        adv_form.addRow("Cycles per step (0=auto):", self.prop_cycles_step)

        self.prop_kic = QDoubleSpinBox()
        self.prop_kic.setRange(0.0, 500.0)
        self.prop_kic.setDecimals(2)
        self.prop_kic.setValue(0.0)
        self.prop_kic.setMinimumHeight(32)
        adv_form.addRow("Fracture toughness KIC:", self.prop_kic)

        self.prop_paris_c = QDoubleSpinBox()
        self.prop_paris_c.setRange(0.0, 1.0)
        self.prop_paris_c.setDecimals(16)
        self.prop_paris_c.setValue(0.0)
        self.prop_paris_c.setSingleStep(1e-11)
        self.prop_paris_c.setMinimumHeight(32)
        adv_form.addRow("Paris law C:", self.prop_paris_c)

        self.prop_paris_m = QDoubleSpinBox()
        self.prop_paris_m.setRange(0.0, 10.0)
        self.prop_paris_m.setDecimals(3)
        self.prop_paris_m.setValue(0.0)
        self.prop_paris_m.setMinimumHeight(32)
        adv_form.addRow("Paris law m:", self.prop_paris_m)
        left_layout.addWidget(adv_card)

        left_layout.addStretch(1)

        run_btn = _primary_btn("Run Crack Analysis")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self._run)
        left_layout.addWidget(run_btn)

        from PyQt6.QtWidgets import QScrollArea, QFrame
        left_sa = QScrollArea()
        left_sa.setWidget(left_content)
        left_sa.setWidgetResizable(True)
        left_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_sa.setFrameShape(QFrame.Shape.NoFrame)
        left_sa.setMinimumWidth(280)
        left_sa.setMaximumWidth(360)
        splitter.addWidget(left_sa)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_layout.addWidget(_section_lbl("Results"))

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet(
            "QLabel { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px; color: #5a7898; }"
        )
        self.preview.setText("Run analysis to see crack overlay and propagation forecast")
        self.preview.setWordWrap(True)
        right_layout.addWidget(self.preview, stretch=2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("Analysis metrics will appear here after running.")
        right_layout.addWidget(self.log, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self.image_path.setText(path)

    def _run(self):
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing Input", "Select an image first.")
            return

        try:
            image = load_image(path)
            crack = detect_cracks(image)
            mode = str(self.propagation_mode.currentData() or "geometric")
            physics_cfg = None
            if mode == "physics_informed":
                physics_cfg = PropagationPhysicsConfig.from_context(
                    structure_type="generic",
                    material_type="concrete",
                    pixel_size_mm=float(self.prop_pixel_size_mm.value()),
                    sigma_nominal_mpa=float(self.prop_sigma.value()),
                    delta_sigma_mpa=float(self.prop_delta_sigma.value()),
                    cycles_per_year=float(self.prop_cycles_year.value()),
                    horizon_years=float(self.prop_horizon_years.value()),
                    cycles_per_step=float(self.prop_cycles_step.value()),
                    fracture_toughness_mpa_sqrt_m=float(self.prop_kic.value()) if self.prop_kic.value() > 0.0 else None,
                    paris_c=float(self.prop_paris_c.value()) if self.prop_paris_c.value() > 0.0 else None,
                    paris_m=float(self.prop_paris_m.value()) if self.prop_paris_m.value() > 0.0 else None,
                )
            seq, metrics = self.forecaster.forecast(
                crack.mask,
                steps=self.steps.value(),
                mode=mode,
                physics=physics_cfg,
            )

            out_dir = Path("final_toolkit_outputs") / "ui_crack_analysis" / Path(path).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "crack_overlay.png"), crack.overlay)
            cv2.imwrite(str(out_dir / "crack_mask.png"), crack.mask)
            self.forecaster.save_sequence(seq, out_dir / "forecast", prefix="step")

            self.preview.setPixmap(
                _pixmap_from_bgr(crack.overlay).scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

            final_m = metrics[-1]
            self.log.setPlainText(
                "\n".join(
                    [
                        f"Image: {path}",
                        f"Crack pixels: {crack.crack_pixels}",
                        f"Crack ratio: {crack.crack_ratio:.5f}",
                        f"Estimated crack length (px): {crack.estimated_length_px:.1f}",
                        f"Estimated max crack width (px): {crack.estimated_max_width_px:.1f}",
                        f"Propagation mode: {mode}",
                        f"Forecast steps: {self.steps.value()}",
                        f"Forecast final crack ratio: {final_m.crack_ratio:.5f}",
                        f"Final crack length (m): {float(final_m.crack_length_m):.6f}",
                        f"Final K_I (MPa*sqrt(m)): {float(final_m.stress_intensity_mpa_sqrt_m):.4f}",
                        f"Final DeltaK (MPa*sqrt(m)): {float(final_m.delta_k_mpa_sqrt_m):.4f}",
                        f"Final da/dN (m/cycle): {float(final_m.fatigue_da_dn_m_per_cycle):.3e}",
                        f"Final failure probability (horizon): {100.0 * float(final_m.failure_probability_horizon):.2f}%",
                        f"Saved to: {out_dir}",
                    ]
                )
            )
            if self.session is not None and hasattr(self.session, "set_active_image"):
                self.session.set_active_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Crack Analysis Error", str(exc))


class MetalDefectTab(QWidget):
    def __init__(self, session: Any = None):
        super().__init__()
        self.session = session
        self._build_ui()
        self._bind_session()

    def _bind_session(self):
        if self.session is None:
            return
        if hasattr(self.session, "activeImageChanged"):
            self.session.activeImageChanged.connect(self._on_active_image_changed)
        if hasattr(self.session, "active_image_path"):
            path = str(getattr(self.session, "active_image_path", "") or "")
            if path:
                self.image_path.setText(path)
        if hasattr(self.session, "active_dataset_dir") and not self.image_path.text().strip():
            first = _first_image_in_dir(str(getattr(self.session, "active_dataset_dir", "") or ""))
            if first:
                self.image_path.setText(first)

    def _on_active_image_changed(self, path: str):
        p = str(path or "").strip()
        if p:
            self.image_path.setText(p)

    def _build_ui(self):
        from PyQt6.QtWidgets import QScrollArea, QFrame
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        # Left: config panel (scrollable)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(_section_lbl("Defect Detection Setup"))

        # Input card
        input_card = QGroupBox("Input Image")
        input_form = QFormLayout(input_card)
        input_form.setSpacing(10)
        image_row = QHBoxLayout()
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select drone image...")
        self.image_path.setMinimumHeight(32)
        browse = QPushButton("Browse")
        browse.setObjectName("ghost")
        browse.setFixedWidth(72)
        browse.setMinimumHeight(32)
        browse.clicked.connect(self._browse_image)
        image_row.addWidget(self.image_path)
        image_row.addWidget(browse)
        input_form.addRow("Image:", image_row)
        left_layout.addWidget(input_card)

        # Analysis type card
        type_card = QGroupBox("Analysis Settings")
        type_form = QFormLayout(type_card)
        type_form.setSpacing(10)

        self.analysis_type = QComboBox()
        self.analysis_type.addItem("Metal Defects (Classical)", "metal")
        self.analysis_type.addItem("Structural Defects (AI Model)", "structural")
        self.analysis_type.addItem("Solar Defects (AI Model)", "solar")
        self.analysis_type.setMinimumHeight(32)
        self.analysis_type.currentIndexChanged.connect(self._sync_analysis_controls)
        type_form.addRow("Analysis type:", self.analysis_type)

        self.use_model = QCheckBox("Use AI model when available")
        self.use_model.setChecked(True)
        type_form.addRow("", self.use_model)

        self.model_key = QLineEdit("structural_multiclass_detector")
        self.model_key.setMinimumHeight(32)
        type_form.addRow("Model key:", self.model_key)
        left_layout.addWidget(type_card)

        left_layout.addStretch(1)

        run_btn = _primary_btn("Run Defect Detection")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self._run)
        left_layout.addWidget(run_btn)

        left_sa = QScrollArea()
        left_sa.setWidget(left_content)
        left_sa.setWidgetResizable(True)
        left_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_sa.setFrameShape(QFrame.Shape.NoFrame)
        left_sa.setMinimumWidth(260)
        left_sa.setMaximumWidth(340)
        splitter.addWidget(left_sa)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_layout.addWidget(_section_lbl("Detection Results"))

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet(
            "QLabel { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px; color: #5a7898; }"
        )
        self.preview.setText("Run detection to see defect overlay")
        self.preview.setWordWrap(True)
        right_layout.addWidget(self.preview, stretch=2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("Detection metrics will appear here after running.")
        right_layout.addWidget(self.log, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)
        self._sync_analysis_controls()

    def _sync_analysis_controls(self):
        mode = str(self.analysis_type.currentData() or "metal")
        if mode == "metal":
            self.use_model.setChecked(False)
            self.use_model.setEnabled(False)
            self.model_key.setText("")
            self.model_key.setEnabled(False)
            return
        self.use_model.setEnabled(True)
        self.model_key.setEnabled(True)
        if mode == "structural" and not self.model_key.text().strip():
            self.model_key.setText("structural_multiclass_detector")
        if mode == "solar" and (
            not self.model_key.text().strip() or self.model_key.text().strip() == "structural_multiclass_detector"
        ):
            self.model_key.setText("solar_defect_detector")

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self.image_path.setText(path)

    def _run(self):
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing Input", "Select an image first.")
            return

        try:
            image = load_image(path)
            mode = str(self.analysis_type.currentData() or "metal")
            out_dir = Path("final_toolkit_outputs") / "ui_defect_analysis" / Path(path).stem
            out_dir.mkdir(parents=True, exist_ok=True)

            if mode == "metal":
                result = detect_metal_defects(image)
                overlay = result.overlay
                mask = result.mask
                cv2.imwrite(str(out_dir / "metal_overlay.png"), overlay)
                cv2.imwrite(str(out_dir / "metal_mask.png"), mask)
                lines = [
                    f"Image: {path}",
                    "Mode: metal",
                    f"Defect pixels: {result.defect_pixels}",
                    f"Defect ratio: {result.defect_ratio:.5f}",
                    f"Defect regions: {result.defect_regions}",
                    f"Saved to: {out_dir}",
                ]
            elif mode == "structural":
                key = self.model_key.text().strip() or "structural_multiclass_detector"
                result = detect_structural_defects(
                    image,
                    model_key=key,
                    use_model=self.use_model.isChecked(),
                )
                overlay = result.overlay
                mask = result.mask
                cv2.imwrite(str(out_dir / "structural_overlay.png"), overlay)
                cv2.imwrite(str(out_dir / "structural_mask.png"), mask)
                lines = [
                    f"Image: {path}",
                    "Mode: structural",
                    f"Model used: {result.model_used}",
                    f"Model available: {result.model_available}",
                    f"Defect pixels: {result.defect_pixels}",
                    f"Defect ratio: {result.defect_ratio:.5f}",
                    f"Detections: {len(result.detections)}",
                    f"Class ratios: {json.dumps(result.class_pixel_ratio, sort_keys=True)}",
                    f"Saved to: {out_dir}",
                ]
            else:
                key = self.model_key.text().strip() or "solar_defect_detector"
                result = detect_solar_defects(
                    image,
                    model_key=key,
                    use_model=self.use_model.isChecked(),
                )
                overlay = result.overlay
                mask = result.mask
                cv2.imwrite(str(out_dir / "solar_overlay.png"), overlay)
                cv2.imwrite(str(out_dir / "solar_mask.png"), mask)
                lines = [
                    f"Image: {path}",
                    "Mode: solar",
                    f"Model used: {result.model_used}",
                    f"Model available: {result.model_available}",
                    f"Defect pixels: {result.defect_pixels}",
                    f"Defect ratio: {result.defect_ratio:.5f}",
                    f"Detections: {len(result.detections)}",
                    f"Hotspot max temp C: {result.hotspot_max_temp_c}",
                    f"Class ratios: {json.dumps(result.class_pixel_ratio, sort_keys=True)}",
                    f"Saved to: {out_dir}",
                ]

            self.preview.setPixmap(
                _pixmap_from_bgr(overlay).scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.log.setPlainText("\n".join(lines))
            if self.session is not None and hasattr(self.session, "set_active_image"):
                self.session.set_active_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Defect Analysis Error", str(exc))


class ReconstructionTab(QWidget):
    def __init__(self, session: Any = None):
        super().__init__()
        self.session = session
        self.forecaster = CrackPropagationForecaster()
        self._point_links: list[dict[str, Any]] = []
        self._critical_points: list[dict[str, Any]] = []
        self._analysis_cache: dict[str, dict[str, Any]] = {}
        self._plot_point_indices = np.empty((0,), dtype=np.int32)
        self._scatter_artist = None
        self._selected_artist = None
        self._selected_point_index: int | None = None
        self._last_artifact_dir: str = ""
        self._fallback_canvas: QLabel | None = None
        self._fallback_plot_indices = np.empty((0,), dtype=np.int32)
        self._fallback_plot_xy = np.empty((0, 2), dtype=np.int32)
        self._fallback_plot_size = (0, 0)
        self._fallback_error_label: QLabel | None = None
        self._recon_thread: QThread | None = None
        self._recon_worker: ReconstructionRunWorker | None = None
        self._build_ui()
        self._bind_session()

    def _bind_session(self):
        if self.session is None:
            return
        if hasattr(self.session, "activeDatasetChanged"):
            self.session.activeDatasetChanged.connect(self._on_active_dataset_changed)
        if hasattr(self.session, "projectChanged"):
            self.session.projectChanged.connect(self._on_project_changed)
        if hasattr(self.session, "runArtifactsChanged"):
            self.session.runArtifactsChanged.connect(self._on_run_artifacts_changed)
        if hasattr(self.session, "active_dataset_dir"):
            dataset = str(getattr(self.session, "active_dataset_dir", "") or "").strip()
            if dataset:
                self.image_dir.setText(dataset)
        if hasattr(self.session, "active_project") and isinstance(self.session.active_project, dict):
            self._on_project_changed(self.session.active_project)

    def _on_active_dataset_changed(self, dataset_dir: str):
        path = str(dataset_dir or "").strip()
        if path:
            self.image_dir.setText(path)

    def _on_project_changed(self, project: dict):
        if not isinstance(project, dict):
            return
        root = Path(str(project.get("root_dir", "") or "")).expanduser()
        if not root:
            return
        recon_root = root / "analysis" / "reconstruction"
        recon_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.setText(str(recon_root))

    def _on_run_artifacts_changed(self, payload: dict):
        if not isinstance(payload, dict):
            return
        run_dir = str(payload.get("run_dir", "") or "").strip()
        if run_dir:
            self._last_artifact_dir = run_dir
            mask_dir = Path(run_dir) / "masks"
            if mask_dir.exists():
                self.mask_dir.setText(str(mask_dir))
        point_link = str(payload.get("point_link_path", "") or "").strip()
        critical = str(payload.get("critical_points_path", "") or "").strip()
        if point_link:
            self._load_point_link_artifacts(point_link_path=point_link, critical_points_path=critical)

    def _build_ui(self):
        from PyQt6.QtWidgets import QScrollArea, QFrame
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setContentsMargins(12, 10, 12, 10)

        # Left: config + controls (scrollable)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(_section_lbl("3D Reconstruction Setup"))

        # Paths card
        paths_card = QGroupBox("Source & Output")
        paths_form = QFormLayout(paths_card)
        paths_form.setSpacing(10)

        img_row = QHBoxLayout()
        self.image_dir = QLineEdit()
        self.image_dir.setPlaceholderText("Folder with drone images...")
        self.image_dir.setMinimumHeight(32)
        img_browse = QPushButton("Browse")
        img_browse.setObjectName("ghost")
        img_browse.setFixedWidth(72)
        img_browse.setMinimumHeight(32)
        img_browse.clicked.connect(self._browse_images)
        img_row.addWidget(self.image_dir)
        img_row.addWidget(img_browse)
        paths_form.addRow("Images:", img_row)

        mask_row = QHBoxLayout()
        self.mask_dir = QLineEdit()
        self.mask_dir.setPlaceholderText("Optional crack mask folder...")
        self.mask_dir.setMinimumHeight(32)
        mask_browse = QPushButton("Browse")
        mask_browse.setObjectName("ghost")
        mask_browse.setFixedWidth(72)
        mask_browse.setMinimumHeight(32)
        mask_browse.clicked.connect(self._browse_masks)
        mask_row.addWidget(self.mask_dir)
        mask_row.addWidget(mask_browse)
        paths_form.addRow("Masks (opt.):", mask_row)

        out_row = QHBoxLayout()
        self.output_dir = QLineEdit("final_toolkit_outputs/ui_reconstruction")
        self.output_dir.setMinimumHeight(32)
        out_browse = QPushButton("Browse")
        out_browse.setObjectName("ghost")
        out_browse.setFixedWidth(72)
        out_browse.setMinimumHeight(32)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_dir)
        out_row.addWidget(out_browse)
        paths_form.addRow("Output:", out_row)
        left_layout.addWidget(paths_card)

        # Reconstruction settings card
        recon_card = QGroupBox("Reconstruction Settings")
        recon_form = QFormLayout(recon_card)
        recon_form.setSpacing(10)

        self.profile = QComboBox()
        self.profile.setMinimumHeight(32)
        for name in available_reconstruction_profiles():
            label = {
                "fast_preview": "Fast Preview",
                "standard": "Standard",
                "inspection_high_accuracy": "Inspection — High Accuracy",
            }.get(name, name)
            self.profile.addItem(label, name)
        self.profile.setCurrentIndex(max(0, self.profile.findData("standard")))
        recon_form.addRow("Profile:", self.profile)

        self.execution_mode = QComboBox()
        self.execution_mode.setMinimumHeight(32)
        self.execution_mode.addItem("Local", "local")
        self.execution_mode.addItem("Cloud (fallback local)", "cloud")
        recon_form.addRow("Execution:", self.execution_mode)

        self.use_cache = QCheckBox("Reuse cached features and matches")
        self.use_cache.setChecked(True)
        recon_form.addRow("", self.use_cache)
        left_layout.addWidget(recon_card)

        # Propagation card
        prop_card = QGroupBox("Crack Propagation")
        prop_form = QFormLayout(prop_card)
        prop_form.setSpacing(10)

        self.propagation_steps = QSpinBox()
        self.propagation_steps.setRange(1, 30)
        self.propagation_steps.setValue(6)
        self.propagation_steps.setMinimumHeight(32)
        prop_form.addRow("Forecast steps:", self.propagation_steps)

        self.propagation_mode = QComboBox()
        self.propagation_mode.setMinimumHeight(32)
        self.propagation_mode.addItem("Geometric (fast)", "geometric")
        self.propagation_mode.addItem("Physics Informed", "physics_informed")
        prop_form.addRow("Mode:", self.propagation_mode)
        left_layout.addWidget(prop_card)

        # Progress card
        prog_card = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog_card)
        self.recon_progress = QProgressBar()
        self.recon_progress.setRange(0, 100)
        self.recon_progress.setValue(0)
        self.recon_progress.setFormat("%p%")
        self.recon_progress.setMinimumHeight(20)
        self.recon_progress_label = QLabel("Idle")
        self.recon_progress_label.setObjectName("muted")
        prog_layout.addWidget(self.recon_progress)
        prog_layout.addWidget(self.recon_progress_label)
        left_layout.addWidget(prog_card)

        left_layout.addStretch(1)

        # Action buttons
        self.run_btn = _primary_btn("Run Reconstruction")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run)
        left_layout.addWidget(self.run_btn)

        import_row = QHBoxLayout()
        load = _ghost_btn("Import Folder")
        load.clicked.connect(self._load_reconstruction_folder)
        load_file = _ghost_btn("Import File")
        load_file.clicked.connect(self._load_reconstruction_artifact)
        import_row.addWidget(load)
        import_row.addWidget(load_file)
        left_layout.addLayout(import_row)

        left_sa = QScrollArea()
        left_sa.setWidget(left_content)
        left_sa.setWidgetResizable(True)
        left_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_sa.setFrameShape(QFrame.Shape.NoFrame)
        left_sa.setMinimumWidth(280)
        left_sa.setMaximumWidth(360)
        main_split.addWidget(left_sa)

        # Right side: 3D viewer + point panel (keep existing splitter)
        right_host = QWidget()
        right_host_layout = QVBoxLayout(right_host)
        right_host_layout.setContentsMargins(0, 0, 0, 0)
        right_host_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        viewer_group = QGroupBox("3D Defect Risk Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        legend = QLabel("Legend: <b style='color:#34be54'>GREEN</b> no defects, <b style='color:#f5c421'>YELLOW</b> minor defects, <b style='color:#ff2424'>RED</b> severe defects")
        legend.setWordWrap(True)
        viewer_layout.addWidget(legend)
        if MATPLOTLIB_AVAILABLE and FigureCanvas is not None and Figure is not None:
            self.figure = Figure(figsize=(5.0, 4.0), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.canvas.mpl_connect("pick_event", self._on_pick_point)
            viewer_layout.addWidget(self.canvas, stretch=1)
        else:
            self.figure = None
            self.canvas = None
            self.ax = None
            self._fallback_error_label = QLabel(
                "Matplotlib 3D backend unavailable. Using built-in point-cloud fallback (click points to inspect)."
            )
            self._fallback_error_label.setWordWrap(True)
            viewer_layout.addWidget(self._fallback_error_label)
            self._fallback_canvas = QLabel("No point cloud loaded")
            self._fallback_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._fallback_canvas.setMinimumHeight(360)
            self._fallback_canvas.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #333; }")
            self._fallback_canvas.mousePressEvent = self._on_fallback_click
            viewer_layout.addWidget(self._fallback_canvas, stretch=1)
        left_layout.addWidget(viewer_group, stretch=2)

        critical_group = QGroupBox("Predicted Critical Points")
        critical_layout = QVBoxLayout(critical_group)
        self.critical_list = QListWidget()
        self.critical_list.itemSelectionChanged.connect(self._on_critical_selected)
        critical_layout.addWidget(self.critical_list)
        left_layout.addWidget(critical_group, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        selected_group = QGroupBox("Selected Point Analysis")
        selected_layout = QVBoxLayout(selected_group)
        self.selected_point_info = QPlainTextEdit()
        self.selected_point_info.setReadOnly(True)
        self.selected_point_info.setPlaceholderText("Click a point in the 3D viewer to inspect linked image evidence.")
        selected_layout.addWidget(self.selected_point_info)

        self.selected_image_preview = QLabel("Image analysis preview")
        self.selected_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_image_preview.setMinimumHeight(280)
        selected_layout.addWidget(self.selected_image_preview)

        action_row = QHBoxLayout()
        self.run_propagation_btn = QPushButton("Run Crack Propagation On Selection")
        self.run_propagation_btn.clicked.connect(self._run_selected_propagation)
        action_row.addWidget(self.run_propagation_btn)
        selected_layout.addLayout(action_row)

        self.propagation_log = QPlainTextEdit()
        self.propagation_log.setReadOnly(True)
        self.propagation_log.setPlaceholderText("Propagation forecast metrics will appear here.")
        selected_layout.addWidget(self.propagation_log)
        right_layout.addWidget(selected_group, stretch=1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        right_host_layout.addWidget(splitter, stretch=2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        self.log.setMaximumHeight(150)
        self.log.setPlaceholderText("Reconstruction log will appear here.")
        right_host_layout.addWidget(self.log)

        main_split.addWidget(right_host)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 3)
        root.addWidget(main_split, stretch=1)

    def _browse_images(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self.image_dir.setText(path)

    def _browse_masks(self):
        path = QFileDialog.getExistingDirectory(self, "Select mask folder")
        if path:
            self.mask_dir.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_dir.setText(path)

    @staticmethod
    def _resolve_artifact_path(base_json: Path, value: str) -> str:
        if not value:
            return ""
        candidate = Path(str(value))
        if candidate.is_absolute():
            return str(candidate)
        return str((base_json.parent / candidate).resolve())

    @staticmethod
    def _risk_from_rgb(rgb: tuple[int, int, int]) -> tuple[float, str]:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        palette = {
            "green": (52, 190, 84),
            "yellow": (245, 196, 33),
            "red": (255, 36, 36),
        }
        best_level = "green"
        best_dist = float("inf")
        for level, ref in palette.items():
            dr = float(r - ref[0])
            dg = float(g - ref[1])
            db = float(b - ref[2])
            dist = dr * dr + dg * dg + db * db
            if dist < best_dist:
                best_dist = dist
                best_level = level
        score = {"green": 0.12, "yellow": 0.42, "red": 0.82}.get(best_level, 0.12)
        return float(score), str(best_level)

    @staticmethod
    def _parse_ascii_ply_vertices(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
        xyz: list[list[float]] = []
        rgb: list[list[int]] = []
        with ply_path.open("rb") as handle:
            first = handle.readline().decode("utf-8", errors="ignore").strip().lower()
            if first != "ply":
                return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

            fmt = ""
            vertex_count = 0
            in_vertex = False
            props: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
                text = line.decode("utf-8", errors="ignore").strip()
                lower = text.lower()
                if lower.startswith("format "):
                    fmt = lower
                elif lower.startswith("element vertex "):
                    parts = lower.split()
                    try:
                        vertex_count = int(parts[-1])
                    except Exception:
                        vertex_count = 0
                    in_vertex = True
                    props = []
                elif lower.startswith("element ") and not lower.startswith("element vertex "):
                    in_vertex = False
                elif lower.startswith("property ") and in_vertex:
                    parts = lower.split()
                    if len(parts) >= 3:
                        props.append(parts[-1])
                elif lower == "end_header":
                    break

            if "ascii" not in fmt or vertex_count <= 0:
                return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

            if props and {"x", "y", "z"}.issubset(set(props)):
                ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
            else:
                ix, iy, iz = 0, 1, 2
            has_rgb = props and {"red", "green", "blue"}.issubset(set(props))
            ir = ig = ib = -1
            if has_rgb:
                ir, ig, ib = props.index("red"), props.index("green"), props.index("blue")

            for _ in range(vertex_count):
                line = handle.readline()
                if not line:
                    break
                parts = line.decode("utf-8", errors="ignore").strip().split()
                if len(parts) <= max(ix, iy, iz):
                    continue
                try:
                    x = float(parts[ix])
                    y = float(parts[iy])
                    z = float(parts[iz])
                except Exception:
                    continue
                xyz.append([x, y, z])
                if has_rgb and len(parts) > max(ir, ig, ib):
                    try:
                        rr = int(float(parts[ir]))
                        gg = int(float(parts[ig]))
                        bb = int(float(parts[ib]))
                    except Exception:
                        rr, gg, bb = 52, 190, 84
                    rgb.append([int(np.clip(rr, 0, 255)), int(np.clip(gg, 0, 255)), int(np.clip(bb, 0, 255))])

        xyz_arr = np.asarray(xyz, dtype=np.float32) if xyz else np.empty((0, 3), dtype=np.float32)
        rgb_arr = np.asarray(rgb, dtype=np.uint8) if rgb else np.empty((0, 3), dtype=np.uint8)
        return xyz_arr, rgb_arr

    def _load_ply_direct(self, ply_path: Path):
        coords, colors = self._parse_ascii_ply_vertices(ply_path)
        if coords.size == 0:
            QMessageBox.warning(
                self,
                "PLY Import",
                "Unable to parse vertices from PLY. Use an ASCII PLY or import point_link_map.json/digital_twin.json.",
            )
            return

        n = int(len(coords))
        if n > 350000:
            rng = np.random.default_rng(2026)
            sel = np.sort(rng.choice(n, size=350000, replace=False))
            coords = coords[sel]
            if len(colors) == n:
                colors = colors[sel]
            n = int(len(coords))
        point_links: list[dict[str, Any]] = []
        if len(colors) == n:
            for idx, (xyz, c) in enumerate(zip(coords, colors), start=0):
                score, level = self._risk_from_rgb((int(c[0]), int(c[1]), int(c[2])))
                point_links.append(
                    {
                        "point_index": int(idx),
                        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                        "risk_score": float(score),
                        "risk_level": str(level),
                        "criticality_score": float(score),
                        "source_image": "",
                        "source_image_name": "",
                        "paired_with": "",
                        "pixel_xy": [0, 0],
                        "defect_flags": {"crack": level == "red", "metal": False},
                        "image_summary": {"source": "ply_import", "notes": "No point-link metadata available."},
                    }
                )
        else:
            z = coords[:, 2].astype(np.float32)
            zmin, zmax = float(np.min(z)), float(np.max(z))
            span = max(1e-6, zmax - zmin)
            zn = np.clip((z - zmin) / span, 0.0, 1.0)
            for idx, (xyz, zv) in enumerate(zip(coords, zn.tolist()), start=0):
                score = float(np.clip(0.15 + 0.7 * zv, 0.0, 1.0))
                level = "green" if score < 0.25 else ("yellow" if score < 0.55 else "red")
                point_links.append(
                    {
                        "point_index": int(idx),
                        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                        "risk_score": float(score),
                        "risk_level": str(level),
                        "criticality_score": float(score),
                        "source_image": "",
                        "source_image_name": "",
                        "paired_with": "",
                        "pixel_xy": [0, 0],
                        "defect_flags": {"crack": level == "red", "metal": False},
                        "image_summary": {"source": "ply_import", "notes": "Risk estimated from geometry only."},
                    }
                )

        self._point_links = point_links
        self._last_artifact_dir = str(ply_path.parent)
        self._selected_point_index = None
        self._analysis_cache.clear()
        self._refresh_3d_plot()
        self._load_or_build_critical_points(critical_points_path="")
        self.selected_point_info.setPlainText(
            f"Loaded {len(self._point_links)} points from PLY only.\n"
            "Point-link metadata was not found, so image linkage/propagation requires a point_link_map.json run."
        )
        self.propagation_log.clear()

    @staticmethod
    def _risk_level_from_score(score: float) -> str:
        s = float(np.clip(score, 0.0, 1.0))
        if s < 0.25:
            return "green"
        if s < 0.55:
            return "yellow"
        return "red"

    def _resolve_legacy_image_root(self, frame_names: list[str], folder: Path) -> str:
        candidates: list[Path] = []
        image_dir_text = self.image_dir.text().strip()
        if image_dir_text:
            candidates.append(Path(image_dir_text))
        if self.session is not None and hasattr(self.session, "active_dataset_dir"):
            dataset = str(getattr(self.session, "active_dataset_dir", "") or "").strip()
            if dataset:
                candidates.append(Path(dataset))
        candidates.extend(
            [
                folder,
                folder.parent,
                folder.parent.parent,
                folder.parent.parent.parent,
            ]
        )

        best_path = ""
        best_score = -1
        unique: set[str] = set()
        sample = [name for name in frame_names if name][: min(20, len(frame_names))]
        if not sample:
            return ""

        for c in candidates:
            try:
                resolved = str(c.expanduser().resolve())
            except Exception:
                continue
            if resolved in unique:
                continue
            unique.add(resolved)
            path = Path(resolved)
            if not path.exists() or not path.is_dir():
                continue
            score = 0
            for name in sample:
                if (path / name).exists():
                    score += 1
            if score > best_score:
                best_score = score
                best_path = resolved
        return best_path if best_score > 0 else ""

    def _retrofit_point_links_from_legacy(
        self,
        folder: Path,
        point_cloud_path: Path,
        camera_pose_path: Path,
    ) -> tuple[str, str, str]:
        if not point_cloud_path.exists():
            return "", "", f"Point cloud missing: {point_cloud_path}"
        if not camera_pose_path.exists():
            return "", "", f"Camera poses missing: {camera_pose_path}"

        points, colors = self._parse_ascii_ply_vertices(point_cloud_path)
        if points.size == 0:
            return "", "", "Point cloud could not be parsed."

        try:
            pose_payload = json.loads(camera_pose_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return "", "", f"Unable to parse camera poses: {exc}"

        frames_raw = pose_payload.get("frames", []) if isinstance(pose_payload, dict) else []
        intrinsics = np.asarray(pose_payload.get("intrinsics", []), dtype=np.float64) if isinstance(pose_payload, dict) else np.empty((0,))
        if intrinsics.shape != (3, 3):
            return "", "", "Invalid intrinsics in camera_poses.json."
        if not isinstance(frames_raw, list) or not frames_raw:
            return "", "", "No camera frames in camera_poses.json."

        frame_names = [str(row.get("image", "") or "") for row in frames_raw if isinstance(row, dict)]
        image_root = self._resolve_legacy_image_root(frame_names, folder)
        if not image_root:
            return "", "", "Could not locate source image directory for camera frames."

        frame_meta: list[dict[str, Any]] = []
        for row in frames_raw:
            if not isinstance(row, dict):
                continue
            name = str(row.get("image", "") or "").strip()
            if not name:
                continue
            image_path = (Path(image_root) / name).resolve()
            if not image_path.exists():
                continue
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            try:
                R = np.asarray(row.get("R", []), dtype=np.float64).reshape(3, 3)
                t = np.asarray(row.get("t", []), dtype=np.float64).reshape(3, 1)
            except Exception:
                continue
            frame_meta.append(
                {
                    "name": name,
                    "path": str(image_path),
                    "w": int(w),
                    "h": int(h),
                    "R": R,
                    "t": t,
                }
            )
        if not frame_meta:
            return "", "", f"No matching images found under: {image_root}"

        pts = points.astype(np.float64)
        n = int(len(pts))
        best_frame = np.full(n, -1, dtype=np.int32)
        best_depth = np.full(n, np.inf, dtype=np.float64)
        best_u = np.zeros(n, dtype=np.int32)
        best_v = np.zeros(n, dtype=np.int32)

        for idx_frame, f in enumerate(frame_meta):
            cam = (f["R"] @ pts.T + f["t"]).T
            z = cam[:, 2]
            valid = z > 1e-6
            if not bool(np.any(valid)):
                continue
            proj = (intrinsics @ cam.T).T
            u = proj[:, 0] / np.clip(proj[:, 2], 1e-9, None)
            v = proj[:, 1] / np.clip(proj[:, 2], 1e-9, None)
            inside = valid & (u >= 0.0) & (u < float(f["w"])) & (v >= 0.0) & (v < float(f["h"]))
            better = inside & (z < best_depth)
            if not bool(np.any(better)):
                continue
            best_depth[better] = z[better]
            best_frame[better] = int(idx_frame)
            uu = np.round(u[better]).astype(np.int32)
            vv = np.round(v[better]).astype(np.int32)
            best_u[better] = np.clip(uu, 0, int(f["w"]) - 1)
            best_v[better] = np.clip(vv, 0, int(f["h"]) - 1)

        if len(colors) == n:
            risk_scores = np.zeros((n,), dtype=np.float32)
            risk_levels: list[str] = []
            for i, c in enumerate(colors.tolist()):
                score, level = self._risk_from_rgb((int(c[0]), int(c[1]), int(c[2])))
                risk_scores[i] = float(score)
                risk_levels.append(str(level))
        else:
            depth = best_depth.copy()
            missing = ~np.isfinite(depth)
            if np.all(missing):
                risk_scores = np.full((n,), 0.42, dtype=np.float32)
            else:
                finite = depth[~missing]
                dmin = float(np.min(finite))
                dmax = float(np.max(finite))
                span = max(1e-6, dmax - dmin)
                norm = np.zeros((n,), dtype=np.float32)
                norm[~missing] = np.clip((depth[~missing] - dmin) / span, 0.0, 1.0).astype(np.float32)
                risk_scores = np.clip(0.2 + 0.6 * (1.0 - norm), 0.0, 1.0)
                risk_scores[missing] = 0.42
            risk_levels = [self._risk_level_from_score(float(s)) for s in risk_scores.tolist()]

        criticality_scores = np.clip(0.85 * risk_scores + 0.15 * (best_frame >= 0).astype(np.float32), 0.0, 1.0)

        point_links: list[dict[str, Any]] = []
        image_summary: dict[str, dict[str, Any]] = {}
        for i in range(n):
            fi = int(best_frame[i])
            if fi >= 0:
                f = frame_meta[fi]
                source = str(f["path"])
                source_name = str(f["name"])
                pix = [int(best_u[i]), int(best_v[i])]
                image_summary.setdefault(source, {"source_image": source, "linked_points": 0})
                image_summary[source]["linked_points"] = int(image_summary[source]["linked_points"]) + 1
            else:
                source = ""
                source_name = ""
                pix = [0, 0]

            level = str(risk_levels[i])
            risk = float(risk_scores[i])
            crit = float(criticality_scores[i])
            point_links.append(
                {
                    "point_index": int(i),
                    "xyz": [float(points[i, 0]), float(points[i, 1]), float(points[i, 2])],
                    "risk_score": risk,
                    "risk_level": level,
                    "criticality_score": crit,
                    "source_image": source,
                    "source_image_name": source_name,
                    "paired_with": "",
                    "pixel_xy": pix,
                    "defect_flags": {"crack": level == "red", "metal": False},
                    "image_summary": {"source": "legacy_retrofit_projection"},
                }
            )

        point_link_path = folder / "point_link_map.json"
        critical_points_path = folder / "critical_points.json"
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "point_count": int(len(point_links)),
            "risk_thresholds": {"green_max": 0.25, "yellow_max": 0.55},
            "predictive_model": "legacy_projection_risk_v1",
            "image_summaries": image_summary,
            "points": point_links,
        }
        point_link_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        order = np.argsort(-criticality_scores)
        top_k = int(min(500, len(order)))
        critical_rows: list[dict[str, Any]] = []
        for rank, idx in enumerate(order[:top_k].tolist(), start=1):
            row = point_links[int(idx)]
            critical_rows.append(
                {
                    "rank": int(rank),
                    "point_index": int(idx),
                    "xyz": row.get("xyz", [0.0, 0.0, 0.0]),
                    "source_image": str(row.get("source_image", "")),
                    "pixel_xy": row.get("pixel_xy", [0, 0]),
                    "risk_level": str(row.get("risk_level", "green")),
                    "risk_score": float(row.get("risk_score", 0.0)),
                    "criticality_score": float(row.get("criticality_score", 0.0)),
                    "reason": ["legacy_projection_risk"],
                }
            )
        critical_points_path.write_text(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "predictive_model": "legacy_projection_risk_v1",
                    "critical_count": int(len(critical_rows)),
                    "points": critical_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        twin_path = folder / "digital_twin.json"
        if twin_path.exists():
            try:
                twin = json.loads(twin_path.read_text(encoding="utf-8"))
                if isinstance(twin, dict):
                    artifacts = twin.get("artifacts", {})
                    if not isinstance(artifacts, dict):
                        artifacts = {}
                    artifacts["point_links"] = str(point_link_path)
                    artifacts["critical_points"] = str(critical_points_path)
                    twin["artifacts"] = artifacts
                    twin_path.write_text(json.dumps(twin, indent=2), encoding="utf-8")
            except Exception:
                pass

        return str(point_link_path), str(critical_points_path), ""

    def _refresh_fallback_plot(self):
        if self._fallback_canvas is None:
            return
        w = max(480, int(self._fallback_canvas.width()))
        h = max(360, int(self._fallback_canvas.height()))
        canvas = np.full((h, w, 3), 18, dtype=np.uint8)

        if not self._point_links:
            cv2.putText(canvas, "No point cloud loaded", (24, max(48, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            self._fallback_plot_indices = np.empty((0,), dtype=np.int32)
            self._fallback_plot_xy = np.empty((0, 2), dtype=np.int32)
            self._fallback_plot_size = (w, h)
            self._fallback_canvas.setPixmap(_pixmap_from_bgr(canvas))
            return

        coords = np.asarray([p.get("xyz", [0.0, 0.0, 0.0]) for p in self._point_links], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            cv2.putText(canvas, "Invalid point geometry", (24, max(48, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 255), 2)
            self._fallback_canvas.setPixmap(_pixmap_from_bgr(canvas))
            return

        n = int(len(coords))
        if n > 25000:
            rng = np.random.default_rng(2026)
            sel = np.sort(rng.choice(n, size=25000, replace=False))
        else:
            sel = np.arange(n, dtype=np.int32)
        pts = coords[sel].astype(np.float32)

        pts -= np.mean(pts, axis=0, keepdims=True)
        std = np.std(pts, axis=0)
        std[std < 1e-6] = 1.0
        pts /= std[None, :]
        u = 0.74 * pts[:, 0] - 0.74 * pts[:, 1]
        v = 0.42 * pts[:, 0] + 0.42 * pts[:, 1] - 1.08 * pts[:, 2]
        uv = np.stack([u, v], axis=1)

        min_uv = np.min(uv, axis=0)
        max_uv = np.max(uv, axis=0)
        span = np.maximum(max_uv - min_uv, 1e-6)
        margin = 26.0
        uvn = (uv - min_uv[None, :]) / span[None, :]
        x = margin + uvn[:, 0] * float(max(1, w - int(2 * margin)))
        y = margin + uvn[:, 1] * float(max(1, h - int(2 * margin)))
        y = float(h) - y
        xy = np.stack([x, y], axis=1).astype(np.int32)

        bgr_map = {"green": (84, 190, 52), "yellow": (33, 196, 245), "red": (36, 36, 255)}
        for idx_local, idx_global in enumerate(sel.tolist()):
            level = str(self._point_links[int(idx_global)].get("risk_level", "green")).lower()
            color = bgr_map.get(level, (84, 190, 52))
            px, py = int(xy[idx_local, 0]), int(xy[idx_local, 1])
            if 1 <= px < w - 1 and 1 <= py < h - 1:
                cv2.circle(canvas, (px, py), 2, color, -1, lineType=cv2.LINE_AA)

        if self._selected_point_index is not None and int(self._selected_point_index) in set(sel.tolist()):
            hit = int(np.where(sel == int(self._selected_point_index))[0][0])
            sx, sy = int(xy[hit, 0]), int(xy[hit, 1])
            cv2.circle(canvas, (sx, sy), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        cv2.putText(canvas, "Fallback point viewer (no matplotlib)", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        if MATPLOTLIB_ERROR:
            msg = f"matplotlib error: {MATPLOTLIB_ERROR}"
            cv2.putText(canvas, msg[: min(len(msg), 95)], (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        self._fallback_plot_indices = sel.astype(np.int32)
        self._fallback_plot_xy = xy.astype(np.int32)
        self._fallback_plot_size = (w, h)
        self._fallback_canvas.setPixmap(_pixmap_from_bgr(canvas))

    def _on_fallback_click(self, event):
        if self._fallback_canvas is None or self._fallback_plot_indices.size == 0:
            return
        pix = self._fallback_canvas.pixmap()
        if pix is None:
            return
        ex = float(event.position().x()) if hasattr(event, "position") else float(event.x())
        ey = float(event.position().y()) if hasattr(event, "position") else float(event.y())
        pw, ph = int(pix.width()), int(pix.height())
        lw, lh = int(self._fallback_canvas.width()), int(self._fallback_canvas.height())
        ox = max(0, (lw - pw) // 2)
        oy = max(0, (lh - ph) // 2)
        x = int(round(ex - ox))
        y = int(round(ey - oy))
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return

        xy = self._fallback_plot_xy.astype(np.float32)
        d2 = np.sum((xy - np.array([[x, y]], dtype=np.float32)) ** 2, axis=1)
        if d2.size == 0:
            return
        hit = int(np.argmin(d2))
        if float(d2[hit]) > 13.0 * 13.0:
            return
        point_idx = int(self._fallback_plot_indices[hit])
        self._select_point(point_idx)

    def _load_reconstruction_folder(self):
        start_dir = self.output_dir.text().strip() or self._last_artifact_dir or "."
        folder = QFileDialog.getExistingDirectory(
            self,
            "Import Reconstruction Folder",
            start_dir,
        )
        if not folder:
            return
        self._import_reconstruction_folder(Path(folder))

    def _import_reconstruction_folder(self, folder_path: Path) -> bool:
        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Invalid Folder", f"Reconstruction folder not found:\n{folder}")
            return False

        self._last_artifact_dir = str(folder)

        point_link = folder / "point_link_map.json"
        critical = folder / "critical_points.json"
        if point_link.exists():
            self._load_point_link_artifacts(
                point_link_path=str(point_link),
                critical_points_path=str(critical if critical.exists() else ""),
            )
            return True

        twin = folder / "digital_twin.json"
        if twin.exists():
            try:
                payload = json.loads(twin.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

            if isinstance(payload, dict):
                artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {}
                p_link = self._resolve_artifact_path(twin, str(artifacts.get("point_links", "") or ""))
                c_path = self._resolve_artifact_path(twin, str(artifacts.get("critical_points", "") or ""))
                if p_link and Path(p_link).exists():
                    self._load_point_link_artifacts(
                        point_link_path=p_link,
                        critical_points_path=c_path if c_path and Path(c_path).exists() else "",
                    )
                    return True

                cloud_path = self._resolve_artifact_path(
                    twin,
                    str(artifacts.get("risk_point_cloud", "") or artifacts.get("point_cloud", "") or ""),
                )
                pose_path = self._resolve_artifact_path(
                    twin,
                    str(artifacts.get("camera_poses", "") or ""),
                )
                if cloud_path and pose_path and Path(cloud_path).exists() and Path(pose_path).exists():
                    retro_link, retro_critical, retro_err = self._retrofit_point_links_from_legacy(
                        folder=folder,
                        point_cloud_path=Path(cloud_path),
                        camera_pose_path=Path(pose_path),
                    )
                    if retro_link and Path(retro_link).exists():
                        self._load_point_link_artifacts(
                            point_link_path=retro_link,
                            critical_points_path=retro_critical if retro_critical and Path(retro_critical).exists() else "",
                        )
                        self.log.setPlainText(
                            "\n".join(
                                [
                                    "Legacy reconstruction upgraded.",
                                    f"Point-link map generated: {retro_link}",
                                    f"Critical points generated: {retro_critical or '[none]'}",
                                ]
                            )
                        )
                        return True
                    if retro_err:
                        needs_images = (
                            "image directory" in retro_err.lower()
                            or "matching images" in retro_err.lower()
                        )
                        if needs_images:
                            chosen = QFileDialog.getExistingDirectory(
                                self,
                                "Select Source Image Folder For Legacy Reconstruction",
                                self.image_dir.text().strip() or str(folder.parent),
                            )
                            if chosen:
                                self.image_dir.setText(chosen)
                                retro_link, retro_critical, retro_err2 = self._retrofit_point_links_from_legacy(
                                    folder=folder,
                                    point_cloud_path=Path(cloud_path),
                                    camera_pose_path=Path(pose_path),
                                )
                                if retro_link and Path(retro_link).exists():
                                    self._load_point_link_artifacts(
                                        point_link_path=retro_link,
                                        critical_points_path=retro_critical if retro_critical and Path(retro_critical).exists() else "",
                                    )
                                    self.log.setPlainText(
                                        "\n".join(
                                            [
                                                "Legacy reconstruction upgraded after selecting source image folder.",
                                                f"Point-link map generated: {retro_link}",
                                                f"Critical points generated: {retro_critical or '[none]'}",
                                            ]
                                        )
                                    )
                                    return True
                                if retro_err2:
                                    retro_err = retro_err2
                        self.log.setPlainText(
                            "\n".join(
                                [
                                    "Legacy reconstruction upgrade could not generate image links.",
                                    f"Reason: {retro_err}",
                                    "Falling back to PLY-only view.",
                                ]
                            )
                        )

                ply_candidates = [
                    self._resolve_artifact_path(twin, str(artifacts.get("risk_point_cloud", "") or "")),
                    self._resolve_artifact_path(twin, str(artifacts.get("point_cloud", "") or "")),
                ]
                for candidate in ply_candidates:
                    if candidate and Path(candidate).exists():
                        self._load_ply_direct(Path(candidate))
                        return True

        fallback_candidates: list[Path] = [
            folder / "custom_reconstruction_risk.ply",
            folder / "custom_reconstruction.ply",
        ]
        for p in sorted(folder.glob("*.ply")):
            if p not in fallback_candidates:
                fallback_candidates.append(p)
        for candidate in fallback_candidates:
            if candidate.exists() and candidate.is_file():
                self._load_ply_direct(candidate)
                return True

        QMessageBox.warning(
            self,
            "Import Failed",
            "No supported reconstruction artifacts found in this folder.\n"
            "Expected one of: point_link_map.json, digital_twin.json, *.ply",
        )
        return False

    def _load_reconstruction_artifact(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Reconstruction Artifact",
            self.output_dir.text().strip() or self._last_artifact_dir or "",
            "Reconstruction Files (*.json *.ply);;JSON Files (*.json);;PLY Files (*.ply);;All Files (*)",
        )
        if not path:
            return
        artifact_path = Path(path)
        if not artifact_path.exists():
            QMessageBox.warning(self, "Missing File", f"File not found: {path}")
            return

        point_link_path = ""
        critical_path = ""
        suffix = artifact_path.suffix.lower()
        if suffix == ".json":
            if artifact_path.name.lower() == "digital_twin.json":
                if self._import_reconstruction_folder(artifact_path.parent):
                    return
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception as exc:
                QMessageBox.warning(self, "Invalid JSON", str(exc))
                return
            if isinstance(payload, dict):
                artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {}
                if artifacts:
                    point_link_path = self._resolve_artifact_path(artifact_path, str(artifacts.get("point_links", "") or ""))
                    critical_path = self._resolve_artifact_path(artifact_path, str(artifacts.get("critical_points", "") or ""))
                    if not point_link_path:
                        cloud_candidate = str(artifacts.get("risk_point_cloud", "") or artifacts.get("point_cloud", "") or "")
                        cloud_resolved = self._resolve_artifact_path(artifact_path, cloud_candidate)
                        if cloud_resolved and Path(cloud_resolved).exists():
                            self._load_ply_direct(Path(cloud_resolved))
                            return
                elif isinstance(payload.get("points"), list) and payload.get("risk_thresholds") is not None:
                    point_link_path = str(artifact_path)
        elif suffix == ".ply":
            candidate_map = artifact_path.parent / "point_link_map.json"
            candidate_twin = artifact_path.parent / "digital_twin.json"
            if candidate_map.exists():
                point_link_path = str(candidate_map)
                critical_candidate = artifact_path.parent / "critical_points.json"
                if critical_candidate.exists():
                    critical_path = str(critical_candidate)
            elif candidate_twin.exists():
                try:
                    payload = json.loads(candidate_twin.read_text(encoding="utf-8"))
                    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {}
                    point_link_path = self._resolve_artifact_path(candidate_twin, str(artifacts.get("point_links", "") or ""))
                    critical_path = self._resolve_artifact_path(candidate_twin, str(artifacts.get("critical_points", "") or ""))
                except Exception:
                    point_link_path = ""

        if not point_link_path:
            if suffix == ".ply":
                self._load_ply_direct(artifact_path)
                return
            QMessageBox.warning(
                self,
                "Unsupported Artifact",
                "Could not resolve point-link metadata. Import digital_twin.json or point_link_map.json, or provide a PLY point cloud.",
            )
            return
        self._load_point_link_artifacts(point_link_path=point_link_path, critical_points_path=critical_path)

    def _load_point_link_artifacts(self, point_link_path: str, critical_points_path: str = ""):
        point_path = Path(str(point_link_path or "").strip())
        if not point_path.exists():
            QMessageBox.warning(self, "Missing Artifact", f"Point-link map not found:\n{point_path}")
            return
        try:
            payload = json.loads(point_path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "Artifact Error", f"Unable to read point-link map:\n{exc}")
            return
        points = payload.get("points", []) if isinstance(payload, dict) else []
        if not isinstance(points, list) or not points:
            QMessageBox.warning(self, "Artifact Error", "No point entries found in point-link map.")
            return

        self._point_links = [p for p in points if isinstance(p, dict)]
        self._last_artifact_dir = str(point_path.parent)
        self._selected_point_index = None
        self._analysis_cache.clear()
        self._refresh_3d_plot()
        self._load_or_build_critical_points(critical_points_path=critical_points_path)
        self.selected_point_info.setPlainText(f"Loaded {len(self._point_links)} linked 3D points.")
        self.propagation_log.clear()

    def _load_or_build_critical_points(self, critical_points_path: str = ""):
        loaded: list[dict[str, Any]] = []
        path = Path(str(critical_points_path or "").strip()) if critical_points_path else None
        if path is not None and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                points = payload.get("points", []) if isinstance(payload, dict) else []
                if isinstance(points, list):
                    loaded = [p for p in points if isinstance(p, dict)]
            except Exception:
                loaded = []
        if not loaded:
            scored = []
            for idx, p in enumerate(self._point_links):
                score = float(p.get("criticality_score", p.get("risk_score", 0.0)) or 0.0)
                scored.append((idx, score))
            scored.sort(key=lambda t: t[1], reverse=True)
            top = scored[: min(200, len(scored))]
            loaded = []
            for rank, (idx, score) in enumerate(top, start=1):
                p = self._point_links[idx]
                loaded.append(
                    {
                        "rank": int(rank),
                        "point_index": int(idx),
                        "criticality_score": float(score),
                        "risk_level": str(p.get("risk_level", "green")),
                        "risk_score": float(p.get("risk_score", 0.0)),
                        "source_image": str(p.get("source_image", "")),
                        "pixel_xy": p.get("pixel_xy", [0, 0]),
                        "reason": ["aggregated_risk"],
                    }
                )
        self._critical_points = loaded
        self.critical_list.clear()
        for row in self._critical_points:
            idx = int(row.get("point_index", -1))
            if idx < 0 or idx >= len(self._point_links):
                continue
            level = str(row.get("risk_level", "green")).upper()
            crit = float(row.get("criticality_score", 0.0))
            src = Path(str(row.get("source_image", "") or self._point_links[idx].get("source_image", ""))).name
            pix = row.get("pixel_xy", self._point_links[idx].get("pixel_xy", [0, 0]))
            if not isinstance(pix, list) or len(pix) < 2:
                pix = [0, 0]
            item = QListWidgetItem(f"#{int(row.get('rank', 0)):03d} [{level}] C={crit:.2f} {src} ({int(pix[0])},{int(pix[1])})")
            item.setData(Qt.ItemDataRole.UserRole, int(idx))
            self.critical_list.addItem(item)

    def _refresh_3d_plot(self):
        if self.ax is None or self.canvas is None:
            self._refresh_fallback_plot()
            return
        self.ax.clear()
        if not self._point_links:
            self.ax.set_title("No point-link data loaded")
            self.canvas.draw_idle()
            return

        coords = np.asarray([p.get("xyz", [0.0, 0.0, 0.0]) for p in self._point_links], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            self.ax.set_title("Invalid point geometry")
            self.canvas.draw_idle()
            return

        n = int(len(coords))
        if n > 25000:
            rng = np.random.default_rng(2026)
            sel = np.sort(rng.choice(n, size=25000, replace=False))
        else:
            sel = np.arange(n, dtype=np.int32)
        self._plot_point_indices = sel.astype(np.int32)
        plot_pts = coords[self._plot_point_indices]

        color_map = {"green": (0.20, 0.75, 0.36), "yellow": (0.94, 0.77, 0.13), "red": (0.98, 0.18, 0.18)}
        colors = []
        for idx in self._plot_point_indices.tolist():
            level = str(self._point_links[int(idx)].get("risk_level", "green")).lower()
            colors.append(color_map.get(level, (0.20, 0.75, 0.36)))
        colors_np = np.asarray(colors, dtype=np.float32)

        self._scatter_artist = self.ax.scatter(
            plot_pts[:, 0],
            plot_pts[:, 1],
            plot_pts[:, 2],
            c=colors_np,
            s=6,
            marker="o",
            picker=True,
            depthshade=False,
        )
        self.ax.set_title("3D risk cloud (click to inspect)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw_idle()

    def _on_pick_point(self, event):
        if self._scatter_artist is None or event.artist is not self._scatter_artist:
            return
        if self._plot_point_indices.size == 0:
            return
        inds = np.asarray(getattr(event, "ind", []), dtype=np.int64).reshape(-1)
        if inds.size == 0:
            return
        plot_idx = int(inds[0])
        if plot_idx < 0 or plot_idx >= len(self._plot_point_indices):
            return
        point_idx = int(self._plot_point_indices[plot_idx])
        self._select_point(point_idx)

    def _on_critical_selected(self):
        item = self.critical_list.currentItem()
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        try:
            point_idx = int(idx)
        except Exception:
            return
        self._select_point(point_idx)

    def _analysis_for_image(self, image_path: str) -> dict[str, Any]:
        key = str(image_path or "").strip()
        cached = self._analysis_cache.get(key)
        if cached is not None:
            return cached
        image = load_image(key)
        crack = detect_cracks(image)
        metal = detect_metal_defects(image)
        structural = detect_structural_defects(
            image,
            model_key="structural_multiclass_detector",
            use_model=True,
        )
        solar = detect_solar_defects(
            image,
            model_key="solar_defect_detector",
            use_model=True,
        )
        payload = {
            "image": image,
            "crack": crack,
            "metal": metal,
            "structural": structural,
            "solar": solar,
        }
        self._analysis_cache[key] = payload
        return payload

    @staticmethod
    def _local_ratio(mask: np.ndarray, x: int, y: int, radius: int = 70) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        h, w = mask.shape[:2]
        x0 = max(0, int(x - radius))
        x1 = min(w, int(x + radius + 1))
        y0 = max(0, int(y - radius))
        y1 = min(h, int(y + radius + 1))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = mask[y0:y1, x0:x1]
        return float(np.mean(roi > 0))

    def _select_point(self, point_idx: int):
        if point_idx < 0 or point_idx >= len(self._point_links):
            return
        self._selected_point_index = int(point_idx)
        link = self._point_links[point_idx]

        if self.ax is not None and self.canvas is not None:
            if self._selected_artist is not None:
                try:
                    self._selected_artist.remove()
                except Exception:
                    pass
                self._selected_artist = None
            xyz = np.asarray(link.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
            if xyz.size == 3:
                self._selected_artist = self.ax.scatter(
                    [float(xyz[0])],
                    [float(xyz[1])],
                    [float(xyz[2])],
                    c=[[1.0, 1.0, 1.0]],
                    s=90,
                    marker="o",
                    edgecolors="black",
                    linewidths=1.2,
                    depthshade=False,
                )
                self.canvas.draw_idle()
        else:
            self._refresh_fallback_plot()

        source = str(link.get("source_image", "")).strip()
        if source and self.session is not None and hasattr(self.session, "set_active_image"):
            self.session.set_active_image(source)
        self._render_point_analysis(link)

    def _render_point_analysis(self, link: dict[str, Any]):
        source = str(link.get("source_image", "")).strip()
        if not source or not Path(source).exists():
            self.selected_point_info.setPlainText(
                "\n".join(
                    [
                        f"Point index: {int(link.get('point_index', -1))}",
                        "No linked source image available for this point.",
                        "This usually means the cloud was imported from PLY without point_link_map.json.",
                        f"Risk: {str(link.get('risk_level', 'green')).upper()} ({float(link.get('risk_score', 0.0)):.3f})",
                        f"Criticality: {float(link.get('criticality_score', 0.0)):.3f}",
                    ]
                )
            )
            self.selected_image_preview.setText("No linked image metadata")
            self.selected_image_preview.setPixmap(QPixmap())
            return
        try:
            analysis = self._analysis_for_image(source)
        except Exception as exc:
            self.selected_point_info.setPlainText(f"Analysis failed for source image:\n{exc}")
            return

        image = analysis["image"].copy()
        h, w = image.shape[:2]
        pix = link.get("pixel_xy", [0, 0]) if isinstance(link.get("pixel_xy", [0, 0]), list) else [0, 0]
        if len(pix) < 2:
            pix = [0, 0]
        px = int(np.clip(int(pix[0]), 0, max(0, w - 1)))
        py = int(np.clip(int(pix[1]), 0, max(0, h - 1)))

        crack = analysis["crack"]
        metal = analysis["metal"]
        structural = analysis["structural"]
        solar = analysis["solar"]

        overlay = cv2.addWeighted(image, 0.72, structural.overlay, 0.20, 0.0)
        overlay = cv2.addWeighted(overlay, 0.86, solar.overlay, 0.14, 0.0)
        cv2.circle(overlay, (px, py), 12, (0, 255, 255), 2)
        cv2.rectangle(
            overlay,
            (max(0, px - 70), max(0, py - 70)),
            (min(w - 1, px + 70), min(h - 1, py + 70)),
            (255, 255, 255),
            1,
        )

        self.selected_image_preview.setPixmap(
            _pixmap_from_bgr(overlay).scaled(
                self.selected_image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        struct_hits = []
        for label, mask in structural.mask_by_class.items():
            if mask is not None and mask.size > 0 and bool(mask[py, px] > 0):
                struct_hits.append(str(label))
        solar_hits = []
        for label, mask in solar.mask_by_class.items():
            if mask is not None and mask.size > 0 and bool(mask[py, px] > 0):
                solar_hits.append(str(label))

        struct_local = {
            label: self._local_ratio(mask, px, py)
            for label, mask in structural.mask_by_class.items()
            if mask is not None and mask.size > 0
        }
        solar_local = {
            label: self._local_ratio(mask, px, py)
            for label, mask in solar.mask_by_class.items()
            if mask is not None and mask.size > 0
        }

        struct_nonzero = {k: float(v) for k, v in struct_local.items() if v > 0.0001}
        solar_nonzero = {k: float(v) for k, v in solar_local.items() if v > 0.0001}

        lines = [
            f"Point index: {int(link.get('point_index', -1))}",
            f"Source image: {source}",
            f"Pixel: ({px}, {py})",
            f"Risk: {str(link.get('risk_level', 'green')).upper()} ({float(link.get('risk_score', 0.0)):.3f})",
            f"Criticality: {float(link.get('criticality_score', 0.0)):.3f}",
            f"Crack at point: {bool(crack.mask[py, px] > 0)}",
            f"Metal anomaly at point: {bool(metal.mask[py, px] > 0)}",
            f"Structural labels at point: {', '.join(struct_hits) if struct_hits else '[none]'}",
            f"Solar labels at point: {', '.join(solar_hits) if solar_hits else '[none]'}",
            f"Local crack ratio: {self._local_ratio(crack.mask, px, py):.4f}",
            f"Local metal ratio: {self._local_ratio(metal.mask, px, py):.4f}",
            f"Local structural ratios: {json.dumps(struct_nonzero, sort_keys=True)}",
            f"Local solar ratios: {json.dumps(solar_nonzero, sort_keys=True)}",
            f"Structural model used: {structural.model_used} (available={structural.model_available})",
            f"Solar model used: {solar.model_used} (available={solar.model_available})",
        ]
        self.selected_point_info.setPlainText("\n".join(lines))

    def _run_selected_propagation(self):
        if self._selected_point_index is None or self._selected_point_index < 0:
            QMessageBox.warning(self, "No Selection", "Select a point in the 3D viewer first.")
            return
        if self._selected_point_index >= len(self._point_links):
            QMessageBox.warning(self, "Invalid Selection", "Selected point index is out of range.")
            return
        link = self._point_links[int(self._selected_point_index)]
        source = str(link.get("source_image", "")).strip()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Missing Source", "Selected point has no valid source image.")
            return

        try:
            analysis = self._analysis_for_image(source)
        except Exception as exc:
            QMessageBox.warning(self, "Analysis Error", str(exc))
            return

        crack_mask = analysis["crack"].mask
        image = analysis["image"]
        h, w = crack_mask.shape[:2]
        pix = link.get("pixel_xy", [0, 0]) if isinstance(link.get("pixel_xy", [0, 0]), list) else [0, 0]
        if len(pix) < 2:
            pix = [0, 0]
        px = int(np.clip(int(pix[0]), 0, max(0, w - 1)))
        py = int(np.clip(int(pix[1]), 0, max(0, h - 1)))

        radius = 120
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)

        roi_mask = np.zeros_like(crack_mask, dtype=np.uint8)
        roi_mask[y0:y1, x0:x1] = crack_mask[y0:y1, x0:x1]
        if int(np.sum(roi_mask > 0)) == 0:
            QMessageBox.information(
                self,
                "No Crack In ROI",
                "No crack pixels near this point. Select a crack-heavy point to run propagation.",
            )
            return

        try:
            mode = str(self.propagation_mode.currentData() or "geometric")
            physics_cfg = None
            if mode == "physics_informed":
                physics_cfg = PropagationPhysicsConfig.from_context(
                    structure_type="generic",
                    material_type="concrete",
                )
            seq, metrics = self.forecaster.forecast(
                roi_mask,
                steps=int(self.propagation_steps.value()),
                mode=mode,
                physics=physics_cfg,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Propagation Error", str(exc))
            return

        out_root = Path(self.output_dir.text().strip() or "final_toolkit_outputs/ui_reconstruction")
        out_dir = out_root / "point_propagation" / f"point_{int(self._selected_point_index):06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.forecaster.save_sequence(seq, out_dir, prefix="step")

        final_mask = seq[-1]
        vis = image.copy()
        heat = np.zeros_like(vis)
        heat[final_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.74, heat, 0.26, 0.0)
        cv2.circle(vis, (px, py), 12, (0, 255, 255), 2)
        self.selected_image_preview.setPixmap(
            _pixmap_from_bgr(vis).scaled(
                self.selected_image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        start = metrics[0]
        end = metrics[-1]
        lines = [
            f"Selected point: {int(self._selected_point_index)}",
            f"Source image: {source}",
            f"ROI bounds: x=({x0},{x1}), y=({y0},{y1})",
            f"Steps: {int(self.propagation_steps.value())}",
            f"Mode: {mode}",
            f"Initial crack ratio: {start.crack_ratio:.6f}",
            f"Final crack ratio: {end.crack_ratio:.6f}",
            f"Initial est. length px: {start.estimated_length_px:.2f}",
            f"Final est. length px: {end.estimated_length_px:.2f}",
            f"Initial max width px: {start.estimated_max_width_px:.2f}",
            f"Final max width px: {end.estimated_max_width_px:.2f}",
            f"Final crack length m: {float(end.crack_length_m):.6f}",
            f"Final K_I MPa*sqrt(m): {float(end.stress_intensity_mpa_sqrt_m):.4f}",
            f"Final DeltaK MPa*sqrt(m): {float(end.delta_k_mpa_sqrt_m):.4f}",
            f"Final da/dN m/cycle: {float(end.fatigue_da_dn_m_per_cycle):.3e}",
            f"Final factor of safety: {float(end.factor_of_safety):.3f}",
            f"Final failure prob (horizon): {100.0 * float(end.failure_probability_horizon):.2f}%",
            f"Saved sequence: {out_dir}",
        ]
        self.propagation_log.setPlainText("\n".join(lines))

    def _run(self):
        image_dir = self.image_dir.text().strip()
        if not image_dir:
            QMessageBox.warning(self, "Missing Input", "Select an image folder first.")
            return

        if self._recon_thread is not None and self._recon_thread.isRunning():
            QMessageBox.information(self, "Reconstruction", "A reconstruction run is already in progress.")
            return

        output_dir = self.output_dir.text().strip() or "final_toolkit_outputs/ui_reconstruction"
        self.log.setPlainText(
            "\n".join(
                [
                    "Reconstruction started in background worker.",
                    "The UI should stay responsive while processing.",
                    f"Input: {image_dir}",
                    f"Output: {output_dir}",
                ]
            )
        )
        self.recon_progress.setRange(0, 100)
        self.recon_progress.setValue(0)
        self.recon_progress_label.setText("Starting...")
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Running...")

        self._recon_thread = QThread(self)
        self._recon_worker = ReconstructionRunWorker(
            image_dir=image_dir,
            output_dir=output_dir,
            crack_mask_dir=self.mask_dir.text().strip() or None,
            profile=str(self.profile.currentData()),
            execution_mode=str(self.execution_mode.currentData()),
            use_cache=self.use_cache.isChecked(),
        )
        self._recon_worker.moveToThread(self._recon_thread)
        self._recon_thread.started.connect(self._recon_worker.run)
        self._recon_worker.progress.connect(self._on_reconstruction_progress)
        self._recon_worker.finished.connect(self._on_reconstruction_finished)
        self._recon_worker.finished.connect(self._recon_thread.quit)
        self._recon_worker.finished.connect(self._recon_worker.deleteLater)
        self._recon_thread.finished.connect(self._recon_thread.deleteLater)
        self._recon_thread.start()

    def _on_reconstruction_progress(self, percent: int, message: str):
        p = int(max(0, min(100, int(percent))))
        self.recon_progress.setRange(0, 100)
        self.recon_progress.setValue(p)
        self.recon_progress_label.setText(str(message or "Processing..."))

    def _on_reconstruction_finished(self, recon_payload: object, error_text: str):
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Reconstruction")

        self._recon_worker = None
        self._recon_thread = None

        if error_text:
            self.recon_progress.setRange(0, 100)
            self.recon_progress.setValue(0)
            self.recon_progress_label.setText("Failed")
            self.log.setPlainText(error_text)
            QMessageBox.critical(self, "Reconstruction Error", error_text.splitlines()[-1] if error_text else "Unknown error")
            return

        recon = recon_payload if isinstance(recon_payload, dict) else {}
        if not recon:
            self.recon_progress.setRange(0, 100)
            self.recon_progress.setValue(0)
            self.recon_progress_label.setText("Failed")
            QMessageBox.critical(self, "Reconstruction Error", "No reconstruction output returned.")
            return

        self.recon_progress.setRange(0, 100)
        self.recon_progress.setValue(100)
        self.recon_progress_label.setText("Completed")

        lines = [
            "Reconstruction completed.",
            f"Profile: {recon.get('processing_profile', '')}",
            f"Mode: {recon.get('execution_mode_requested', '')} -> {recon.get('execution_mode_used', '')}",
            f"Frames: {int(recon.get('frame_count', 0) or 0)}",
            f"Processed pairs: {int(recon.get('processed_pairs', 0) or 0)}",
            f"Failed pairs: {int(recon.get('failed_pairs', 0) or 0)}",
            f"Total points: {int(recon.get('total_points', 0) or 0)}",
            f"Crack points: {int(recon.get('crack_points', 0) or 0)}",
            f"Point cloud: {recon.get('point_cloud_path', '')}",
            f"Camera poses: {recon.get('camera_pose_path', '')}",
            f"Orthomosaic: {recon.get('orthomosaic_path', '') or '[none]'}",
            f"DSM: {recon.get('dsm_path', '') or '[none]'}",
            f"DTM: {recon.get('dtm_path', '') or '[none]'}",
            f"Mesh: {recon.get('mesh_path', '') or '[none]'}",
            f"Textured mesh: {recon.get('textured_mesh_obj_path', '') or '[none]'}",
            f"Digital twin: {recon.get('digital_twin_path', '') or '[none]'}",
            f"Cache hits/misses: {int(recon.get('cache_hits', 0) or 0)}/{int(recon.get('cache_misses', 0) or 0)}",
        ]
        if recon.get("risk_point_cloud_path", ""):
            lines.append(f"Risk cloud: {recon.get('risk_point_cloud_path', '')}")
        if recon.get("point_link_path", ""):
            lines.append(f"Point-link map: {recon.get('point_link_path', '')}")
        if recon.get("critical_points_path", ""):
            lines.append(f"Critical points: {recon.get('critical_points_path', '')}")
        if recon.get("crack_cloud_path", ""):
            lines.append(f"Crack cloud: {recon.get('crack_cloud_path', '')}")
        if recon.get("cloud_request_path", ""):
            lines.append(f"Cloud request: {recon.get('cloud_request_path', '')}")
        if recon.get("cloud_response_path", ""):
            lines.append(f"Cloud response: {recon.get('cloud_response_path', '')}")
        warnings = recon.get("warnings", [])
        if isinstance(warnings, list) and warnings:
            lines.append(f"Warnings: {'; '.join(str(w) for w in warnings)}")
        self.log.setPlainText("\n".join(lines))

        point_link = str(recon.get("point_link_path", "") or "").strip()
        critical = str(recon.get("critical_points_path", "") or "").strip()
        if point_link:
            self._load_point_link_artifacts(
                point_link_path=point_link,
                critical_points_path=critical,
            )
        if self.session is not None and hasattr(self.session, "publish_run_artifacts"):
            self.session.publish_run_artifacts(
                {
                    "run_dir": str(self.output_dir.text().strip() or ""),
                    "summary_path": "",
                    "report_path": "",
                    "point_cloud_path": str(recon.get("point_cloud_path", "") or ""),
                    "risk_point_cloud_path": str(recon.get("risk_point_cloud_path", "") or ""),
                    "point_link_path": str(point_link),
                    "critical_points_path": str(critical),
                    "digital_twin_path": str(recon.get("digital_twin_path", "") or ""),
                    "source": "reconstruction_tab",
                }
            )


class FullPipelineTab(QWidget):
    def __init__(self, session: Any = None):
        super().__init__()
        self.session = session
        self._thread: QThread | None = None
        self._worker: PipelineRunWorker | None = None
        self._build_ui()
        self._load_legacy_info()
        self._bind_session()

    def _bind_session(self):
        if self.session is None:
            return
        if hasattr(self.session, "activeDatasetChanged"):
            self.session.activeDatasetChanged.connect(self._on_active_dataset_changed)
        if hasattr(self.session, "projectChanged"):
            self.session.projectChanged.connect(self._on_project_changed)
        if hasattr(self.session, "active_dataset_dir"):
            dataset = str(getattr(self.session, "active_dataset_dir", "") or "").strip()
            if dataset:
                self.image_dir.setText(dataset)
                if not self.asset_id.text().strip():
                    self.asset_id.setText(Path(dataset).name)
        if hasattr(self.session, "active_project") and isinstance(self.session.active_project, dict):
            self._on_project_changed(self.session.active_project)

    def _on_active_dataset_changed(self, dataset_dir: str):
        path = str(dataset_dir or "").strip()
        if path:
            self.image_dir.setText(path)
            if not self.asset_id.text().strip():
                self.asset_id.setText(Path(path).name)

    def _on_project_changed(self, project: dict):
        if not isinstance(project, dict):
            return
        root = Path(str(project.get("root_dir", "") or "")).expanduser()
        if not root:
            return
        pipeline_root = root / "pipeline_runs"
        pipeline_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.setText(str(pipeline_root))

    def _build_ui(self):
        from PyQt6.QtWidgets import QScrollArea, QFrame
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        outer = QSplitter(Qt.Orientation.Horizontal)
        outer.setContentsMargins(12, 10, 12, 10)

        # Left: grouped config (scrollable)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(_section_lbl("Full Inspection Pipeline"))

        # Input & target card
        input_card = QGroupBox("Input & Target")
        input_form = QFormLayout(input_card)
        input_form.setSpacing(10)

        img_row = QHBoxLayout()
        self.image_dir = QLineEdit()
        self.image_dir.setPlaceholderText("Folder with drone images...")
        self.image_dir.setMinimumHeight(32)
        img_btn = QPushButton("Browse")
        img_btn.setObjectName("ghost")
        img_btn.setFixedWidth(72)
        img_btn.setMinimumHeight(32)
        img_btn.clicked.connect(self._browse_images)
        img_row.addWidget(self.image_dir)
        img_row.addWidget(img_btn)
        input_form.addRow("Images:", img_row)

        out_row = QHBoxLayout()
        self.output_dir = QLineEdit("final_toolkit_outputs")
        self.output_dir.setMinimumHeight(32)
        out_btn = QPushButton("Browse")
        out_btn.setObjectName("ghost")
        out_btn.setFixedWidth(72)
        out_btn.setMinimumHeight(32)
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_dir)
        out_row.addWidget(out_btn)
        input_form.addRow("Output:", out_row)

        self.asset_id = QLineEdit()
        self.asset_id.setPlaceholderText("Defaults to dataset folder name")
        self.asset_id.setMinimumHeight(32)
        input_form.addRow("Asset ID:", self.asset_id)

        self.structure_type = QComboBox()
        self.structure_type.setMinimumHeight(32)
        for label, value in (
            ("Generic", "generic"), ("Bridge", "bridge"), ("Building", "building"),
            ("Tower", "tower"), ("Pipeline", "pipeline"), ("Solar", "solar"),
        ):
            self.structure_type.addItem(label, value)
        input_form.addRow("Structure:", self.structure_type)

        self.material_type = QComboBox()
        self.material_type.setMinimumHeight(32)
        for label, value in (
            ("Concrete", "concrete"), ("Steel", "steel"), ("Masonry", "masonry"),
            ("Composite", "composite"), ("Mixed", "mixed"),
        ):
            self.material_type.addItem(label, value)
        input_form.addRow("Material:", self.material_type)

        bundle_row = QHBoxLayout()
        self.capture_bundle_path = QLineEdit()
        self.capture_bundle_path.setPlaceholderText("Optional multi-sensor bundle JSON")
        self.capture_bundle_path.setMinimumHeight(32)
        bundle_btn = QPushButton("Browse")
        bundle_btn.setObjectName("ghost")
        bundle_btn.setFixedWidth(72)
        bundle_btn.setMinimumHeight(32)
        bundle_btn.clicked.connect(self._browse_capture_bundle)
        bundle_row.addWidget(self.capture_bundle_path)
        bundle_row.addWidget(bundle_btn)
        input_form.addRow("Bundle (opt.):", bundle_row)

        calib_row = QHBoxLayout()
        self.calibration_profile_path = QLineEdit()
        self.calibration_profile_path.setPlaceholderText("Optional calibration JSON")
        self.calibration_profile_path.setMinimumHeight(32)
        calib_btn = QPushButton("Browse")
        calib_btn.setObjectName("ghost")
        calib_btn.setFixedWidth(72)
        calib_btn.setMinimumHeight(32)
        calib_btn.clicked.connect(self._browse_calibration_profile)
        calib_row.addWidget(self.calibration_profile_path)
        calib_row.addWidget(calib_btn)
        input_form.addRow("Calibration (opt.):", calib_row)
        left_layout.addWidget(input_card)

        # Propagation card
        prop_card = QGroupBox("Crack Propagation")
        prop_form = QFormLayout(prop_card)
        prop_form.setSpacing(10)

        self.steps = QSpinBox()
        self.steps.setRange(1, 30)
        self.steps.setValue(6)
        self.steps.setMinimumHeight(32)
        prop_form.addRow("Forecast steps:", self.steps)

        self.propagation_mode = QComboBox()
        self.propagation_mode.setMinimumHeight(32)
        self.propagation_mode.addItem("Geometric (fast)", "geometric")
        self.propagation_mode.addItem("Physics Informed", "physics_informed")
        prop_form.addRow("Mode:", self.propagation_mode)

        self.prop_pixel_size_mm = QDoubleSpinBox()
        self.prop_pixel_size_mm.setRange(0.01, 20.0)
        self.prop_pixel_size_mm.setDecimals(3)
        self.prop_pixel_size_mm.setValue(0.5)
        self.prop_pixel_size_mm.setSingleStep(0.05)
        self.prop_pixel_size_mm.setSuffix(" mm/px")
        self.prop_pixel_size_mm.setMinimumHeight(32)
        prop_form.addRow("Pixel size:", self.prop_pixel_size_mm)

        self.prop_sigma = QDoubleSpinBox()
        self.prop_sigma.setRange(0.1, 2000.0)
        self.prop_sigma.setDecimals(2)
        self.prop_sigma.setValue(35.0)
        self.prop_sigma.setSuffix(" MPa")
        self.prop_sigma.setMinimumHeight(32)
        prop_form.addRow("Nominal stress σ:", self.prop_sigma)

        self.prop_delta_sigma = QDoubleSpinBox()
        self.prop_delta_sigma.setRange(0.01, 2000.0)
        self.prop_delta_sigma.setDecimals(2)
        self.prop_delta_sigma.setValue(12.0)
        self.prop_delta_sigma.setSuffix(" MPa")
        self.prop_delta_sigma.setMinimumHeight(32)
        prop_form.addRow("Stress range Δσ:", self.prop_delta_sigma)

        self.prop_cycles_year = QDoubleSpinBox()
        self.prop_cycles_year.setRange(1.0, 1.0e9)
        self.prop_cycles_year.setDecimals(0)
        self.prop_cycles_year.setValue(1_000_000.0)
        self.prop_cycles_year.setSingleStep(10000.0)
        self.prop_cycles_year.setMinimumHeight(32)
        prop_form.addRow("Cycles / year:", self.prop_cycles_year)

        self.prop_horizon_years = QDoubleSpinBox()
        self.prop_horizon_years.setRange(0.01, 20.0)
        self.prop_horizon_years.setDecimals(1)
        self.prop_horizon_years.setValue(1.0)
        self.prop_horizon_years.setSuffix(" years")
        self.prop_horizon_years.setMinimumHeight(32)
        prop_form.addRow("Forecast horizon:", self.prop_horizon_years)

        self.prop_cycles_step = QDoubleSpinBox()
        self.prop_cycles_step.setRange(0.0, 1.0e12)
        self.prop_cycles_step.setDecimals(0)
        self.prop_cycles_step.setValue(0.0)
        self.prop_cycles_step.setSingleStep(10000.0)
        self.prop_cycles_step.setMinimumHeight(32)
        prop_form.addRow("Cycles/step (0=auto):", self.prop_cycles_step)

        self.prop_kic = QDoubleSpinBox()
        self.prop_kic.setRange(0.0, 500.0)
        self.prop_kic.setDecimals(2)
        self.prop_kic.setValue(0.0)
        self.prop_kic.setMinimumHeight(32)
        prop_form.addRow("Fracture toughness KIC:", self.prop_kic)

        self.prop_paris_c = QDoubleSpinBox()
        self.prop_paris_c.setRange(0.0, 1.0)
        self.prop_paris_c.setDecimals(16)
        self.prop_paris_c.setValue(0.0)
        self.prop_paris_c.setSingleStep(1e-11)
        self.prop_paris_c.setMinimumHeight(32)
        prop_form.addRow("Paris law C (0=default):", self.prop_paris_c)

        self.prop_paris_m = QDoubleSpinBox()
        self.prop_paris_m.setRange(0.0, 10.0)
        self.prop_paris_m.setDecimals(3)
        self.prop_paris_m.setValue(0.0)
        self.prop_paris_m.setMinimumHeight(32)
        prop_form.addRow("Paris law m (0=default):", self.prop_paris_m)
        left_layout.addWidget(prop_card)

        # 3D Reconstruction card
        recon_card = QGroupBox("3D Reconstruction")
        recon_form = QFormLayout(recon_card)
        recon_form.setSpacing(10)

        self.recon_profile = QComboBox()
        self.recon_profile.setMinimumHeight(32)
        for name in available_reconstruction_profiles():
            label = {
                "fast_preview": "Fast Preview",
                "standard": "Standard",
                "inspection_high_accuracy": "Inspection — High Accuracy",
            }.get(name, name)
            self.recon_profile.addItem(label, name)
        self.recon_profile.setCurrentIndex(max(0, self.recon_profile.findData("standard")))
        recon_form.addRow("Profile:", self.recon_profile)

        self.recon_mode = QComboBox()
        self.recon_mode.setMinimumHeight(32)
        self.recon_mode.addItem("Local", "local")
        self.recon_mode.addItem("Cloud (fallback local)", "cloud")
        recon_form.addRow("Execution:", self.recon_mode)

        self.recon_use_cache = QCheckBox("Reuse cached features and matches")
        self.recon_use_cache.setChecked(True)
        recon_form.addRow("", self.recon_use_cache)
        left_layout.addWidget(recon_card)

        # Models & options card
        models_card = QGroupBox("Detection Models & Options")
        models_form = QFormLayout(models_card)
        models_form.setSpacing(10)

        self.use_multisensor = QCheckBox("Enable multi-sensor bundle processing")
        self.use_multisensor.setChecked(True)
        models_form.addRow("", self.use_multisensor)

        self.use_structural_models = QCheckBox("Enable structural defect model")
        self.use_structural_models.setChecked(True)
        models_form.addRow("", self.use_structural_models)

        self.structural_model_key = QLineEdit("structural_multiclass_detector")
        self.structural_model_key.setMinimumHeight(32)
        models_form.addRow("Structural model:", self.structural_model_key)

        self.use_solar_models = QCheckBox("Enable solar defect model")
        self.use_solar_models.setChecked(True)
        models_form.addRow("", self.use_solar_models)

        self.solar_model_key = QLineEdit("solar_defect_detector")
        self.solar_model_key.setMinimumHeight(32)
        models_form.addRow("Solar model:", self.solar_model_key)

        self.use_fenicsx = QCheckBox("Use legacy FEniCSx bridge (cracksim.py)")
        models_form.addRow("", self.use_fenicsx)
        left_layout.addWidget(models_card)

        left_layout.addStretch(1)

        self.run_btn = _primary_btn("Run Full Inspection Pipeline")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run)
        left_layout.addWidget(self.run_btn)

        left_sa = QScrollArea()
        left_sa.setWidget(left_content)
        left_sa.setWidgetResizable(True)
        left_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_sa.setFrameShape(QFrame.Shape.NoFrame)
        left_sa.setMinimumWidth(300)
        left_sa.setMaximumWidth(400)
        outer.addWidget(left_sa)

        # Right: progress + log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_layout.addWidget(_section_lbl("Pipeline Status"))

        status_card = QGroupBox("Progress")
        status_layout = QVBoxLayout(status_card)
        progress_row = QHBoxLayout()
        self.pipeline_status_label = QLabel("Ready")
        self.pipeline_status_label.setObjectName("muted")
        self.pipeline_progress = QProgressBar()
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(0)
        self.pipeline_progress.setMinimumHeight(20)
        progress_row.addWidget(self.pipeline_status_label, stretch=1)
        progress_row.addWidget(self.pipeline_progress, stretch=2)
        status_layout.addLayout(progress_row)
        right_layout.addWidget(status_card)

        self.legacy_info = QTextEdit()
        self.legacy_info.setReadOnly(True)
        self.legacy_info.setMaximumHeight(80)
        self.legacy_info.setPlaceholderText("System asset info...")
        right_layout.addWidget(self.legacy_info)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline output will appear here after running.")
        right_layout.addWidget(self.log, stretch=1)

        outer.addWidget(right)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 2)
        root.addWidget(outer, stretch=1)

    def _load_legacy_info(self):
        summary = legacy_summary()
        lines = [
            "Legacy asset discovery:",
            f"SwinUNet checkpoint: {summary.get('swin_unet_checkpoint') or '[missing]'}",
            f"cracksim.py: {summary.get('cracksim') or '[missing]'}",
            f"viewer.py: {summary.get('viewer') or '[missing]'}",
        ]
        self.legacy_info.setPlainText("\n".join(lines))

    def _browse_images(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self.image_dir.setText(path)
            if not self.asset_id.text().strip():
                self.asset_id.setText(Path(path).name)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output root")
        if path:
            self.output_dir.setText(path)

    def _browse_capture_bundle(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Capture Bundle",
            "",
            "JSON (*.json);;All Files (*)",
        )
        if path:
            self.capture_bundle_path.setText(path)

    def _browse_calibration_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Calibration Profile",
            "",
            "JSON (*.json);;All Files (*)",
        )
        if path:
            self.calibration_profile_path.setText(path)

    def _run(self):
        image_dir = self.image_dir.text().strip()
        output_dir = self.output_dir.text().strip() or "final_toolkit_outputs"
        if not image_dir:
            QMessageBox.warning(self, "Missing Input", "Select an image folder first.")
            return
        if self.session is not None and hasattr(self.session, "set_active_dataset"):
            self.session.set_active_dataset(image_dir)
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Pipeline", "A pipeline run is already in progress.")
            return

        config = PipelineConfig(
            forecast_steps=self.steps.value(),
            propagation_mode=str(self.propagation_mode.currentData() or "geometric"),
            propagation_pixel_size_mm=float(self.prop_pixel_size_mm.value()),
            propagation_sigma_nominal_mpa=float(self.prop_sigma.value()),
            propagation_delta_sigma_mpa=float(self.prop_delta_sigma.value()),
            propagation_cycles_per_year=float(self.prop_cycles_year.value()),
            propagation_horizon_years=float(self.prop_horizon_years.value()),
            propagation_cycles_per_step=float(self.prop_cycles_step.value()),
            propagation_fracture_toughness_mpa_sqrt_m=float(self.prop_kic.value()),
            propagation_paris_c=float(self.prop_paris_c.value()),
            propagation_paris_m=float(self.prop_paris_m.value()),
            use_fenicsx=self.use_fenicsx.isChecked(),
            run_multisensor_processing=self.use_multisensor.isChecked(),
            multi_sensor_bundle_path=self.capture_bundle_path.text().strip(),
            multi_sensor_calibration_profile_path=self.calibration_profile_path.text().strip(),
            reconstruction_profile=str(self.recon_profile.currentData()),
            reconstruction_execution_mode=str(self.recon_mode.currentData()),
            reconstruction_use_cache=self.recon_use_cache.isChecked(),
            run_structural_models=self.use_structural_models.isChecked(),
            run_solar_models=self.use_solar_models.isChecked(),
            structural_model_key=self.structural_model_key.text().strip() or "structural_multiclass_detector",
            solar_model_key=self.solar_model_key.text().strip() or "solar_defect_detector",
            asset_id=self.asset_id.text().strip(),
            structure_type=str(self.structure_type.currentData() or "generic"),
            material_type=str(self.material_type.currentData() or "concrete"),
        )

        self.run_btn.setEnabled(False)
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(0)
        self.pipeline_status_label.setText("Starting pipeline...")
        self.log.setPlainText("Pipeline started in background worker.")

        self._thread = QThread(self)
        self._worker = PipelineRunWorker(image_dir=image_dir, output_dir=output_dir, config=config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_pipeline_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.finished.connect(lambda _result, _error: self._thread.quit())
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_pipeline_progress(self, percent: int, message: str):
        p = int(max(0, min(100, int(percent))))
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(p)
        self.pipeline_status_label.setText(str(message or "Running pipeline..."))

    def _on_pipeline_finished(self, result: dict, error: str):
        self.run_btn.setEnabled(True)
        self._worker = None
        self._thread = None
        if error:
            self.pipeline_progress.setRange(0, 100)
            self.pipeline_progress.setValue(0)
            self.pipeline_status_label.setText("Pipeline failed")
            QMessageBox.critical(self, "Pipeline Error", error.splitlines()[-1] if error else "Unknown error")
            return

        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(100)
        self.pipeline_status_label.setText("Pipeline completed")

        result_dict = result if isinstance(result, dict) else {}
        self.log.setPlainText(
            "\n".join(
                [
                    "Pipeline completed.",
                    f"Run dir: {result_dict.get('run_dir', '')}",
                    f"Report: {result_dict.get('report_markdown_path', '')}",
                    f"Summary: {result_dict.get('summary_json_path', '')}",
                    f"Point cloud: {result_dict.get('point_cloud_path', '')}",
                    f"Crack cloud: {result_dict.get('crack_point_cloud_path', '') or '[none]'}",
                    f"Asset integrity score: {float(result_dict.get('asset_integrity_score', 0.0)):.2f}",
                    f"Structural risk score: {float(result_dict.get('structural_risk_score', 0.0)):.2f}",
                    f"Condition level: {int(result_dict.get('condition_level', 0) or 0)}",
                    f"Health scoring JSON: {result_dict.get('health_scoring_path', '')}",
                    f"Forecast dir: {result_dict.get('forecast_dir', '')}",
                    f"Coverage dir: {result_dict.get('coverage_dir', '')}",
                    f"Multi-sensor dir: {result_dict.get('multisensor_dir', '')}",
                ]
            )
        )
        if self.session is not None and hasattr(self.session, "publish_run_artifacts"):
            self.session.publish_run_artifacts(
                {
                    "run_dir": str(result_dict.get("run_dir", "")),
                    "summary_path": str(result_dict.get("summary_json_path", "")),
                    "report_path": str(result_dict.get("report_markdown_path", "")),
                    "point_cloud_path": str(result_dict.get("point_cloud_path", "")),
                    "health_scoring_path": str(result_dict.get("health_scoring_path", "")),
                    "source": "full_pipeline_tab",
                }
            )


class OperationsCenterTab(QWidget):
    """All-in-one UI workflow: dataset import, pipeline run, inspection review, and report view."""

    def __init__(self, session: Any = None):
        super().__init__()
        self.session = session
        self.dataset_files: list[str] = []
        self.current_summary: dict = {}
        self.current_run_dir: str = ""
        self._thread: QThread | None = None
        self._worker: PipelineRunWorker | None = None
        self._build_ui()
        self._load_legacy_info()
        self._bind_session()

    def _bind_session(self):
        if self.session is None:
            return
        if hasattr(self.session, "activeDatasetChanged"):
            self.session.activeDatasetChanged.connect(self._on_active_dataset_changed)
        if hasattr(self.session, "activeImageChanged"):
            self.session.activeImageChanged.connect(self._on_active_image_changed)
        if hasattr(self.session, "projectChanged"):
            self.session.projectChanged.connect(self._on_project_changed)
        if hasattr(self.session, "active_dataset_dir"):
            dataset = str(getattr(self.session, "active_dataset_dir", "") or "").strip()
            if dataset:
                self.image_dir.setText(dataset)
        if hasattr(self.session, "active_project") and isinstance(self.session.active_project, dict):
            self._on_project_changed(self.session.active_project)

    def _on_active_dataset_changed(self, dataset_dir: str):
        path = str(dataset_dir or "").strip()
        if not path:
            return
        current = self.image_dir.text().strip()
        if current:
            try:
                if Path(current).resolve() == Path(path).resolve() and self.dataset_files:
                    return
            except Exception:
                if current == path and self.dataset_files:
                    return
        self.image_dir.setText(path)
        if hasattr(self, "asset_id") and not self.asset_id.text().strip():
            self.asset_id.setText(Path(path).name)
        self._import_dataset()

    def _on_active_image_changed(self, image_path: str):
        p = str(image_path or "").strip()
        if not p:
            return
        image_dir = self.image_dir.text().strip()
        if not image_dir:
            return
        try:
            rel = Path(p).name
        except Exception:
            return
        matches = self.dataset_list.findItems(rel, Qt.MatchFlag.MatchExactly)
        if matches:
            self.dataset_list.setCurrentItem(matches[0])

    def _on_project_changed(self, project: dict):
        if not isinstance(project, dict):
            return
        root = Path(str(project.get("root_dir", "") or "")).expanduser()
        if not root:
            return
        output_root = root / "pipeline_runs"
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.setText(str(output_root))

    def _build_ui(self):
        from PyQt6.QtWidgets import QScrollArea, QFrame
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top toolbar ──────────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setObjectName("appBar")
        toolbar.setFixedHeight(54)
        toolbar_row = QHBoxLayout(toolbar)
        toolbar_row.setContentsMargins(14, 0, 14, 0)
        toolbar_row.setSpacing(8)

        self.image_dir = QLineEdit()
        self.image_dir.setPlaceholderText("Dataset folder with drone images...")
        self.image_dir.setMinimumHeight(32)
        toolbar_row.addWidget(self.image_dir, stretch=2)

        browse_btn = _ghost_btn("Browse")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(self._browse_images)
        toolbar_row.addWidget(browse_btn)

        import_btn = _ghost_btn("Import")
        import_btn.setFixedWidth(72)
        import_btn.clicked.connect(self._import_dataset)
        toolbar_row.addWidget(import_btn)

        toolbar_row.addSpacing(8)

        self.run_btn = _primary_btn("Run Inspection")
        self.run_btn.setMinimumWidth(140)
        self.run_btn.clicked.connect(self._run)
        toolbar_row.addWidget(self.run_btn)

        self.load_run_btn = _ghost_btn("Load Run")
        self.load_run_btn.clicked.connect(self._load_existing_run)
        toolbar_row.addWidget(self.load_run_btn)

        self.open_run_btn = _ghost_btn("Open Folder")
        self.open_run_btn.clicked.connect(self._open_run_folder)
        toolbar_row.addWidget(self.open_run_btn)

        self.open_report_btn = _ghost_btn("Open Report")
        self.open_report_btn.clicked.connect(self._open_report_file)
        toolbar_row.addWidget(self.open_report_btn)

        toolbar_row.addSpacing(16)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(120)
        self.progress.setFixedHeight(16)
        toolbar_row.addWidget(self.status_label)
        toolbar_row.addWidget(self.progress)
        root.addWidget(toolbar)

        # ── Config strip (compact, collapsible feel) ──────────────────────────
        from PyQt6.QtWidgets import QScrollArea as _QSA
        cfg_widget = QWidget()
        cfg_widget.setFixedHeight(54)
        cfg_row = QHBoxLayout(cfg_widget)
        cfg_row.setContentsMargins(14, 6, 14, 6)
        cfg_row.setSpacing(10)

        self.output_dir = QLineEdit("final_toolkit_outputs")
        self.output_dir.setPlaceholderText("Output root...")
        self.output_dir.setMinimumHeight(32)
        out_browse = _ghost_btn("…")
        out_browse.setFixedWidth(36)
        out_browse.clicked.connect(self._browse_output)

        self.asset_id = QLineEdit()
        self.asset_id.setPlaceholderText("Asset ID (opt.)")
        self.asset_id.setMinimumHeight(32)
        self.asset_id.setMaximumWidth(160)

        self.structure_type = QComboBox()
        self.structure_type.setMinimumHeight(32)
        for label, value in (
            ("Generic", "generic"), ("Bridge", "bridge"), ("Building", "building"),
            ("Tower", "tower"), ("Pipeline", "pipeline"), ("Solar", "solar"),
        ):
            self.structure_type.addItem(label, value)

        self.material_type = QComboBox()
        self.material_type.setMinimumHeight(32)
        for label, value in (
            ("Concrete", "concrete"), ("Steel", "steel"), ("Masonry", "masonry"),
            ("Composite", "composite"), ("Mixed", "mixed"),
        ):
            self.material_type.addItem(label, value)

        cfg_row.addWidget(QLabel("Output:"))
        cfg_row.addWidget(self.output_dir, stretch=1)
        cfg_row.addWidget(out_browse)
        cfg_row.addWidget(QLabel("Asset ID:"))
        cfg_row.addWidget(self.asset_id)
        cfg_row.addWidget(QLabel("Structure:"))
        cfg_row.addWidget(self.structure_type)
        cfg_row.addWidget(QLabel("Material:"))
        cfg_row.addWidget(self.material_type)
        root.addWidget(cfg_widget)

        # Hidden advanced params (preserved as attributes for _run())
        self.capture_bundle_path = QLineEdit()
        self.calibration_profile_path = QLineEdit()
        self.steps = QSpinBox()
        self.steps.setRange(1, 30)
        self.steps.setValue(6)
        self.propagation_mode = QComboBox()
        self.propagation_mode.addItem("Geometric", "geometric")
        self.propagation_mode.addItem("Physics Informed", "physics_informed")
        self.prop_pixel_size_mm = QDoubleSpinBox()
        self.prop_pixel_size_mm.setRange(0.01, 20.0)
        self.prop_pixel_size_mm.setDecimals(3)
        self.prop_pixel_size_mm.setValue(0.5)
        self.prop_pixel_size_mm.setSingleStep(0.05)
        self.prop_sigma = QDoubleSpinBox()
        self.prop_sigma.setRange(0.1, 2000.0)
        self.prop_sigma.setDecimals(3)
        self.prop_sigma.setValue(35.0)
        self.prop_delta_sigma = QDoubleSpinBox()
        self.prop_delta_sigma.setRange(0.01, 2000.0)
        self.prop_delta_sigma.setDecimals(3)
        self.prop_delta_sigma.setValue(12.0)
        self.prop_cycles_year = QDoubleSpinBox()
        self.prop_cycles_year.setRange(1.0, 1.0e9)
        self.prop_cycles_year.setDecimals(0)
        self.prop_cycles_year.setValue(1_000_000.0)
        self.prop_horizon_years = QDoubleSpinBox()
        self.prop_horizon_years.setRange(0.01, 20.0)
        self.prop_horizon_years.setDecimals(3)
        self.prop_horizon_years.setValue(1.0)
        self.prop_cycles_step = QDoubleSpinBox()
        self.prop_cycles_step.setRange(0.0, 1.0e12)
        self.prop_cycles_step.setDecimals(0)
        self.prop_cycles_step.setValue(0.0)
        self.prop_kic = QDoubleSpinBox()
        self.prop_kic.setRange(0.0, 500.0)
        self.prop_kic.setDecimals(3)
        self.prop_kic.setValue(0.0)
        self.prop_paris_c = QDoubleSpinBox()
        self.prop_paris_c.setRange(0.0, 1.0)
        self.prop_paris_c.setDecimals(16)
        self.prop_paris_c.setValue(0.0)
        self.prop_paris_m = QDoubleSpinBox()
        self.prop_paris_m.setRange(0.0, 10.0)
        self.prop_paris_m.setDecimals(3)
        self.prop_paris_m.setValue(0.0)
        self.recon_profile = QComboBox()
        for name in available_reconstruction_profiles():
            label = {"fast_preview": "Fast Preview", "standard": "Standard",
                     "inspection_high_accuracy": "Inspection High Accuracy"}.get(name, name)
            self.recon_profile.addItem(label, name)
        self.recon_profile.setCurrentIndex(max(0, self.recon_profile.findData("standard")))
        self.recon_mode = QComboBox()
        self.recon_mode.addItem("Local", "local")
        self.recon_mode.addItem("Cloud (Fallback Local)", "cloud")
        self.recon_use_cache = QCheckBox()
        self.recon_use_cache.setChecked(True)
        self.use_fenicsx = QCheckBox()
        self.use_multisensor = QCheckBox()
        self.use_multisensor.setChecked(True)
        self.use_structural_models = QCheckBox()
        self.use_structural_models.setChecked(True)
        self.structural_model_key = QLineEdit("structural_multiclass_detector")
        self.use_solar_models = QCheckBox()
        self.use_solar_models.setChecked(True)
        self.solar_model_key = QLineEdit("solar_defect_detector")

        # ── Main splitter: dataset inspector | results ─────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 6, 12, 10)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        dataset_group = QGroupBox("Dataset Inspector")
        dataset_layout = QVBoxLayout(dataset_group)
        self.dataset_count_label = QLabel("0 images loaded")
        self.dataset_count_label.setObjectName("muted")
        dataset_layout.addWidget(self.dataset_count_label)
        self.dataset_list = QListWidget()
        self.dataset_list.currentTextChanged.connect(self._on_dataset_image_selected)
        dataset_layout.addWidget(self.dataset_list)
        self.raw_preview = QLabel()
        self.raw_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_preview.setMinimumHeight(200)
        self.raw_preview.setText("Select a dataset to preview images")
        self.raw_preview.setStyleSheet(
            "QLabel { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px; color: #5a7898; }"
        )
        dataset_layout.addWidget(self.raw_preview)
        left_layout.addWidget(dataset_group)

        legacy_group = QGroupBox("Environment")
        legacy_layout = QVBoxLayout(legacy_group)
        self.legacy_info = QPlainTextEdit()
        self.legacy_info.setReadOnly(True)
        self.legacy_info.setMaximumHeight(90)
        legacy_layout.addWidget(self.legacy_info)
        left_layout.addWidget(legacy_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        inspection_group = QGroupBox("Inspection Results")
        inspection_layout = QVBoxLayout(inspection_group)
        top = QHBoxLayout()
        self.result_image_combo = QComboBox()
        self.result_image_combo.setMinimumHeight(30)
        self.result_image_combo.currentIndexChanged.connect(self._on_result_image_changed)
        top.addWidget(QLabel("Result image:"))
        top.addWidget(self.result_image_combo, stretch=1)
        inspection_layout.addLayout(top)

        preview_row = QHBoxLayout()
        self.crack_preview = QLabel()
        self.crack_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crack_preview.setMinimumHeight(200)
        self.crack_preview.setText("Crack overlay")
        self.crack_preview.setStyleSheet(
            "QLabel { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px; color: #5a7898; }"
        )
        self.metal_preview = QLabel()
        self.metal_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metal_preview.setMinimumHeight(200)
        self.metal_preview.setText("Metal/defect overlay")
        self.metal_preview.setStyleSheet(
            "QLabel { background: #0f1c2e; border: 1px solid #192535; border-radius: 6px; color: #5a7898; }"
        )
        preview_row.addWidget(self.crack_preview)
        preview_row.addWidget(self.metal_preview)
        inspection_layout.addLayout(preview_row)

        self.image_metrics = QPlainTextEdit()
        self.image_metrics.setReadOnly(True)
        self.image_metrics.setMaximumHeight(100)
        self.image_metrics.setPlaceholderText("Per-image metrics after run.")
        inspection_layout.addWidget(self.image_metrics)
        right_layout.addWidget(inspection_group, stretch=2)

        report_group = QGroupBox("Report & Summary")
        report_layout = QVBoxLayout(report_group)
        self.report_box = QPlainTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setPlaceholderText("Report markdown will appear here after a run.")
        report_layout.addWidget(self.report_box)
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMaximumHeight(100)
        self.summary_box.setPlaceholderText("Summary JSON will appear here after a run.")
        report_layout.addWidget(self.summary_box)
        right_layout.addWidget(report_group, stretch=1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _load_legacy_info(self):
        summary = legacy_summary()
        lines = [
            "Environment info:",
            f"SwinUNet checkpoint: {summary.get('swin_unet_checkpoint') or '[missing]'}",
            f"cracksim.py: {summary.get('cracksim') or '[missing]'}",
            f"viewer.py: {summary.get('viewer') or '[missing]'}",
            "",
            "Use the 'Mission Planner' tab to create/export flight missions.",
        ]
        self.legacy_info.setPlainText("\n".join(lines))

    def _browse_images(self):
        path = QFileDialog.getExistingDirectory(self, "Select image dataset folder")
        if path:
            self.image_dir.setText(path)
            if not self.asset_id.text().strip():
                self.asset_id.setText(Path(path).name)
            self._import_dataset()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output root")
        if path:
            self.output_dir.setText(path)

    def _browse_capture_bundle(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Capture Bundle",
            "",
            "JSON (*.json);;All Files (*)",
        )
        if path:
            self.capture_bundle_path.setText(path)

    def _browse_calibration_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Calibration Profile",
            "",
            "JSON (*.json);;All Files (*)",
        )
        if path:
            self.calibration_profile_path.setText(path)

    def _import_dataset(self):
        image_dir = self.image_dir.text().strip()
        if not image_dir:
            QMessageBox.warning(self, "Missing Dataset", "Select an image folder first.")
            return
        root = Path(image_dir)
        if not root.exists():
            QMessageBox.warning(self, "Invalid Dataset", f"Folder does not exist:\n{image_dir}")
            return

        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        files.sort()
        self.dataset_files = [str(p) for p in files]
        self.dataset_list.clear()
        self.dataset_list.addItems([Path(p).name for p in self.dataset_files])
        self.dataset_count_label.setText(f"{len(self.dataset_files)} images loaded")
        self.status_label.setText("Dataset imported")
        if self.session is not None and hasattr(self.session, "set_active_dataset"):
            should_emit = True
            current_active = str(getattr(self.session, "active_dataset_dir", "") or "").strip()
            if current_active:
                try:
                    should_emit = Path(current_active).resolve() != Path(image_dir).resolve()
                except Exception:
                    should_emit = current_active != image_dir
            if should_emit:
                self.session.set_active_dataset(image_dir)
        if self.dataset_files:
            self.dataset_list.setCurrentRow(0)
            if self.session is not None and hasattr(self.session, "set_active_image"):
                self.session.set_active_image(self.dataset_files[0])
        else:
            self.raw_preview.setText("No supported images in folder")
            self.raw_preview.setPixmap(QPixmap())

    def _on_dataset_image_selected(self, item_name: str):
        if not item_name:
            return
        image_dir = self.image_dir.text().strip()
        p = Path(image_dir) / item_name
        _set_preview_from_path(self.raw_preview, str(p), "Dataset preview unavailable")
        if self.session is not None and hasattr(self.session, "set_active_image"):
            self.session.set_active_image(str(p))

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.load_run_btn.setEnabled(not running)
        if running:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.status_label.setText("Starting pipeline...")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)

    def _run(self):
        image_dir = self.image_dir.text().strip()
        output_dir = self.output_dir.text().strip() or "final_toolkit_outputs"
        if not image_dir:
            QMessageBox.warning(self, "Missing Input", "Select and import a dataset first.")
            return
        if self.session is not None and hasattr(self.session, "set_active_dataset"):
            self.session.set_active_dataset(image_dir)
        if not self.dataset_files:
            self._import_dataset()
            if not self.dataset_files:
                return

        config = PipelineConfig(
            forecast_steps=self.steps.value(),
            propagation_mode=str(self.propagation_mode.currentData() or "geometric"),
            propagation_pixel_size_mm=float(self.prop_pixel_size_mm.value()),
            propagation_sigma_nominal_mpa=float(self.prop_sigma.value()),
            propagation_delta_sigma_mpa=float(self.prop_delta_sigma.value()),
            propagation_cycles_per_year=float(self.prop_cycles_year.value()),
            propagation_horizon_years=float(self.prop_horizon_years.value()),
            propagation_cycles_per_step=float(self.prop_cycles_step.value()),
            propagation_fracture_toughness_mpa_sqrt_m=float(self.prop_kic.value()),
            propagation_paris_c=float(self.prop_paris_c.value()),
            propagation_paris_m=float(self.prop_paris_m.value()),
            use_fenicsx=self.use_fenicsx.isChecked(),
            run_multisensor_processing=self.use_multisensor.isChecked(),
            multi_sensor_bundle_path=self.capture_bundle_path.text().strip(),
            multi_sensor_calibration_profile_path=self.calibration_profile_path.text().strip(),
            reconstruction_profile=str(self.recon_profile.currentData()),
            reconstruction_execution_mode=str(self.recon_mode.currentData()),
            reconstruction_use_cache=self.recon_use_cache.isChecked(),
            run_structural_models=self.use_structural_models.isChecked(),
            run_solar_models=self.use_solar_models.isChecked(),
            structural_model_key=self.structural_model_key.text().strip() or "structural_multiclass_detector",
            solar_model_key=self.solar_model_key.text().strip() or "solar_defect_detector",
            asset_id=self.asset_id.text().strip(),
            structure_type=str(self.structure_type.currentData() or "generic"),
            material_type=str(self.material_type.currentData() or "concrete"),
        )

        self._set_running(True)
        self._worker = PipelineRunWorker(image_dir=image_dir, output_dir=output_dir, config=config)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_pipeline_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.finished.connect(lambda _result, _error: self._thread.quit())
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_pipeline_progress(self, percent: int, message: str):
        p = int(max(0, min(100, int(percent))))
        self.progress.setRange(0, 100)
        self.progress.setValue(p)
        self.status_label.setText(str(message or "Running pipeline..."))

    def _on_pipeline_finished(self, result: dict, error: str):
        self._set_running(False)
        if error:
            self.status_label.setText("Pipeline failed")
            self.progress.setValue(0)
            QMessageBox.critical(self, "Pipeline Error", error)
            return

        integrity = float(result.get("asset_integrity_score", 0.0) or 0.0)
        risk = float(result.get("structural_risk_score", 0.0) or 0.0)
        level = int(result.get("condition_level", 0) or 0)
        self.status_label.setText(f"Pipeline completed | Integrity {integrity:.1f} | Risk {risk:.1f} | L{level}")
        self.progress.setValue(100)
        self.current_run_dir = str(result.get("run_dir", ""))
        if self.session is not None and hasattr(self.session, "publish_run_artifacts"):
            self.session.publish_run_artifacts(
                {
                    "run_dir": str(result.get("run_dir", "")),
                    "summary_path": str(result.get("summary_json_path", "")),
                    "report_path": str(result.get("report_markdown_path", "")),
                    "point_cloud_path": str(result.get("point_cloud_path", "")),
                    "health_scoring_path": str(result.get("health_scoring_path", "")),
                    "source": "operations_center_run",
                }
            )
        self._load_run_artifacts(self.current_run_dir)

    def _load_existing_run(self):
        path = QFileDialog.getExistingDirectory(self, "Select a run directory")
        if not path:
            return
        self._load_run_artifacts(path)

    def _load_run_artifacts(self, run_dir: str):
        root = Path(run_dir)
        summary_path = root / "summary.json"
        report_path = root / "inspection_report.md"
        if not summary_path.exists():
            QMessageBox.warning(self, "Invalid Run", f"summary.json not found in:\n{run_dir}")
            return

        try:
            self.current_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.current_run_dir = str(root)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load summary.json:\n{exc}")
            return

        self.summary_box.setPlainText(json.dumps(self.current_summary, indent=2))
        if report_path.exists():
            self.report_box.setPlainText(report_path.read_text(encoding="utf-8"))
        else:
            self.report_box.setPlainText("Report file not found.")

        health = self.current_summary.get("health_scoring", {})
        if isinstance(health, dict) and health:
            integrity = float(health.get("integrity_score", 0.0) or 0.0)
            risk = float(health.get("structural_risk_score", 0.0) or 0.0)
            level = int(health.get("condition_level", 0) or 0)
            self.status_label.setText(f"Loaded run | Integrity {integrity:.1f} | Risk {risk:.1f} | L{level}")

        rows = self.current_summary.get("images", [])
        self.result_image_combo.blockSignals(True)
        self.result_image_combo.clear()
        self.result_image_combo.addItems([r.get("image", "") for r in rows])
        self.result_image_combo.blockSignals(False)
        if rows:
            self.result_image_combo.setCurrentIndex(0)
            self._on_result_image_changed(0)
        if self.session is not None and hasattr(self.session, "publish_run_artifacts"):
            self.session.publish_run_artifacts(
                {
                    "run_dir": str(root),
                    "summary_path": str(summary_path),
                    "report_path": str(report_path if report_path.exists() else ""),
                    "point_cloud_path": str(self.current_summary.get("reconstruction", {}).get("point_cloud_path", "")),
                    "health_scoring_path": str(self.current_summary.get("artifacts", {}).get("health_scoring_path", "")),
                    "source": "operations_center_load",
                }
            )

    def _on_result_image_changed(self, index: int):
        rows = self.current_summary.get("images", [])
        if index < 0 or index >= len(rows):
            return
        row = rows[index]
        crack_overlay = str(row.get("crack_overlay_path", ""))
        metal_overlay = str(row.get("metal_overlay_path", ""))
        structural_overlay = str(row.get("structural_overlay_path", ""))
        solar_overlay = str(row.get("solar_overlay_path", ""))

        _set_preview_from_path(self.crack_preview, crack_overlay, "Crack overlay unavailable")
        _set_preview_from_path(self.metal_preview, metal_overlay, "Metal overlay unavailable")

        metrics = [
            f"Image: {row.get('image', '')}",
            f"Crack ratio: {float(row.get('crack_ratio', 0.0)):.5f}",
            f"Crack length (px): {float(row.get('crack_length_px', 0.0)):.1f}",
            f"Crack max width (px): {float(row.get('crack_max_width_px', 0.0)):.1f}",
            f"Metal defect ratio: {float(row.get('metal_ratio', 0.0)):.5f}",
            f"Metal defect regions: {int(row.get('metal_regions', 0))}",
            f"Structural defect ratio: {float(row.get('structural_ratio', 0.0)):.5f}",
            f"Structural detections: {int(row.get('structural_detection_count', 0))}",
            f"Structural model used: {row.get('structural_model_used', 'heuristic')}",
            f"Solar defect ratio: {float(row.get('solar_ratio', 0.0)):.5f}",
            f"Solar detections: {int(row.get('solar_detection_count', 0))}",
            f"Solar model used: {row.get('solar_model_used', 'heuristic')}",
            f"Solar hotspot max temp C: {row.get('solar_hotspot_max_temp_c', '[n/a]')}",
            f"Sharpness: {float(row.get('sharpness', 0.0)):.2f}",
            f"Motion blur score: {float(row.get('motion_blur_score', 0.0)):.3f}",
            f"Lens blur ratio: {float(row.get('lens_blur_ratio', 0.0)):.3f}",
            f"Dark clip: {100.0 * float(row.get('dark_clip_pct', 0.0)):.1f}%",
            f"Bright clip: {100.0 * float(row.get('bright_clip_pct', 0.0)):.1f}%",
            f"Overlap prev/next: {float(row.get('overlap_prev', 0.0)):.3f}/{float(row.get('overlap_next', 0.0)):.3f}",
            f"Quality OK: {'yes' if bool(row.get('quality_ok', True)) else 'no'}",
            f"Quality flags: {', '.join(row.get('quality_flags', [])) or '[none]'}",
            "",
            f"Crack overlay: {crack_overlay}",
            f"Metal overlay: {metal_overlay}",
            f"Structural overlay: {structural_overlay}",
            f"Solar overlay: {solar_overlay}",
        ]
        self.image_metrics.setPlainText("\n".join(metrics))

    def _open_run_folder(self):
        if not self.current_run_dir:
            QMessageBox.information(self, "No Run Loaded", "Run the pipeline or load a run first.")
            return
        run_dir = Path(self.current_run_dir).resolve()
        QDesktopServices.openUrl(run_dir.as_uri())

    def _open_report_file(self):
        if not self.current_run_dir:
            QMessageBox.information(self, "No Run Loaded", "Run the pipeline or load a run first.")
            return
        report = (Path(self.current_run_dir) / "inspection_report.md").resolve()
        if not report.exists():
            QMessageBox.warning(self, "Missing Report", f"Report not found:\n{report}")
            return
        QDesktopServices.openUrl(report.as_uri())
