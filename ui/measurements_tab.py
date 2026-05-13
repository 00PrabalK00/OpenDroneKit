"""Measurements Tab — UI for core/measurements.py."""

import json
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QSplitter,
    QListWidget,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QDoubleSpinBox,
    QComboBox,
    QTextEdit,
    QPushButton,
    QMessageBox
)
from .workspace import AppSession, _primary_btn, _ghost_btn, _section_label

class MeasurementsTab(QWidget):
    def __init__(self, session: AppSession):
        super().__init__()
        self.session = session
        self._measurements = []
        self._build_ui()
        self.session.projectChanged.connect(lambda _p: self._refresh())
        
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("appBar")
        header.setFixedHeight(54)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(16, 0, 16, 0)
        hb.setSpacing(12)
        lbl = QLabel("Measurements")
        lbl.setObjectName("secondary")
        hb.addWidget(lbl)
        hb.addStretch(1)
        self.btn_refresh = _ghost_btn("Refresh")
        self.btn_refresh.clicked.connect(self._refresh)
        hb.addWidget(self.btn_refresh)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 10, 12, 10)

        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(6)
        ll.addWidget(_section_label("Measurements"))
        self.meas_list = QListWidget()
        self.meas_list.currentRowChanged.connect(self._on_selected)
        ll.addWidget(self.meas_list, stretch=1)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        rl.addWidget(_section_label("Details"))

        detail_card = QGroupBox("Selected Measurement")
        dc = QFormLayout(detail_card)
        self.lbl_id = QLabel("—")
        self.lbl_type = QLabel("—")
        self.lbl_val = QLabel("—")
        self.lbl_lbl = QLabel("—")
        self.lbl_notes = QLabel("—")
        dc.addRow("ID:", self.lbl_id)
        dc.addRow("Type:", self.lbl_type)
        dc.addRow("Value:", self.lbl_val)
        dc.addRow("Label:", self.lbl_lbl)
        dc.addRow("Notes:", self.lbl_notes)
        rl.addWidget(detail_card)

        create_card = QGroupBox("Create Manual Measurement")
        cc = QFormLayout(create_card)
        self.fld_type = QComboBox()
        self.fld_type.addItems(["distance", "area", "crack_length", "volume"])
        self.fld_val = QDoubleSpinBox()
        self.fld_val.setRange(0, 999999)
        self.fld_unit = QComboBox()
        self.fld_unit.addItems(["m", "m2", "px", "m3"])
        self.fld_lbl = QLineEdit()
        self.btn_create = _primary_btn("Create")
        self.btn_create.clicked.connect(self._create)
        cc.addRow("Type:", self.fld_type)
        cc.addRow("Value:", self.fld_val)
        cc.addRow("Unit:", self.fld_unit)
        cc.addRow("Label:", self.fld_lbl)
        cc.addRow("", self.btn_create)
        rl.addWidget(create_card)
        
        rl.addStretch(1)
        splitter.addWidget(right)
        root.addWidget(splitter, stretch=1)
        
    def _refresh(self):
        self.meas_list.clear()
        self._measurements = []
        try:
            from core.measurements import _load_store
            project = self.session.active_project
            if not project:
                return
            from pathlib import Path
            root = Path(str(project.get("root_dir", "")))
            if root.exists():
                self._measurements = _load_store(root)
                for m in self._measurements:
                    self.meas_list.addItem(f"{m.get('label', m.get('id', '?'))} ({m.get('value', 0)} {m.get('unit', '')})")
        except Exception:
            pass

    def _on_selected(self, row):
        if row < 0 or row >= len(self._measurements):
            return
        m = self._measurements[row]
        self.lbl_id.setText(m.get("id", "—"))
        self.lbl_type.setText(m.get("measurement_type", "—"))
        self.lbl_val.setText(f"{m.get('value', '—')} {m.get('unit', '')}")
        self.lbl_lbl.setText(m.get("label", "—"))
        self.lbl_notes.setText(m.get("notes", "—"))

    def _create(self):
        try:
            from core.measurements import create_measurement
            project = self.session.active_project
            if not project:
                QMessageBox.warning(self, "Error", "No active project")
                return
            from pathlib import Path
            root = Path(str(project.get("root_dir", "")))
            m = create_measurement(
                project_root=root,
                project_id=str(project.get("id")),
                source_type="manual",
                source_id="ui",
                measurement_type=self.fld_type.currentText(),
                geometry={},
                value=self.fld_val.value(),
                unit=self.fld_unit.currentText(),
                label=self.fld_lbl.text()
            )
            self._refresh()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
