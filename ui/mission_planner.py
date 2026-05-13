"""Map-based mission planner tab for UAV inspections."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PyQt6.QtCore import QObject, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mission import (
    AssetReferenceFrame,
    FlightRecipe,
    MissionConstraints,
    MissionPlan,
    MissionPlanner,
    export_flight_recipe,
    export_geojson,
    export_qgc_wpl,
    load_flight_recipe,
)
from .theme import standard_icon


class _MapBridge(QObject):
    def __init__(self, tab: "MissionPlannerTab"):
        super().__init__(tab)
        self._tab = tab

    @pyqtSlot(str)
    def receivePolygon(self, json_coords: str):
        try:
            coords = json.loads(json_coords)
        except Exception:
            return
        self._tab.set_polygon(coords)

    @pyqtSlot(str)
    def receiveLine(self, json_coords: str):
        try:
            coords = json.loads(json_coords)
        except Exception:
            return
        self._tab.set_line(coords)

    @pyqtSlot(str)
    def receiveRows(self, json_rows: str):
        try:
            rows = json.loads(json_rows)
        except Exception:
            return
        self._tab.set_rows(rows)

    @pyqtSlot(str)
    def receivePoint(self, json_coords: str):
        try:
            coords = json.loads(json_coords)
        except Exception:
            return
        self._tab.set_point(coords)

    @pyqtSlot(str)
    def receiveNoFly(self, json_polygons: str):
        try:
            polygons = json.loads(json_polygons)
        except Exception:
            return
        self._tab.set_no_fly_polygons(polygons)

    @pyqtSlot()
    def clearNoFly(self):
        self._tab.clear_no_fly_polygons()

    @pyqtSlot(str)
    def receiveTelemetry(self, json_payload: str):
        try:
            payload = json.loads(json_payload)
        except Exception:
            return
        self._tab.record_measurement_from_payload(payload)

    @pyqtSlot()
    def clearPolygon(self):
        self._tab.clear_polygon()

    @pyqtSlot()
    def clearArea(self):
        self._tab.clear_area()


class MissionPlannerTab(QWidget):
    planGenerated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.planner = MissionPlanner()
        self.survey_polygon: list[list[float]] = []
        self.no_fly_polygons: list[list[list[float]]] = []
        self.linear_path: list[list[float]] = []
        self.solar_rows: list[list[list[float]]] = []
        self.tower_center: list[float] = []
        self.measurement_samples: list[dict] = []
        self.linked_segments: list[FlightRecipe] = []
        self.current_plan: MissionPlan | None = None
        self.loaded_repeat_recipe: FlightRecipe | None = None
        self.repeat_recipe_path: str = ""
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        self.map = QWebEngineView()
        for attr in (
            "LocalContentCanAccessFileUrls",
            "LocalContentCanAccessRemoteUrls",
            "ErrorPageEnabled",
        ):
            try:
                web_attr = getattr(QWebEngineSettings.WebAttribute, attr)
                self.map.settings().setAttribute(web_attr, True)
            except Exception:
                pass

        html_path = Path(__file__).with_name("mission_map.html")
        self.map.load(QUrl.fromLocalFile(str(html_path)))
        self.map.setMinimumSize(520, 360)
        self.map.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.channel = QWebChannel(self.map.page())
        self.bridge = _MapBridge(self)
        self.channel.registerObject("bridge", self.bridge)
        self.map.page().setWebChannel(self.channel)

        panel = QWidget()
        panel.setMinimumWidth(360)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        params = QGroupBox("Mission Parameters")
        form = QFormLayout(params)

        self.camera = QComboBox()
        self.camera.addItem("Mavic 2 Pro", "mavic2pro")
        self.camera.addItem("Phantom 4 RTK", "phantom4rtk")
        self.camera.addItem("Custom", "custom")
        form.addRow("Camera:", self.camera)

        self.mode = QComboBox()
        self.mode.addItem("Grid", "grid")
        self.mode.addItem("Corridor", "corridor")
        self.mode.addItem("Facade Vertical", "facade")
        self.mode.addItem("Facade Mapping (3D)", "facade_mapping")
        self.mode.addItem("Linear Inspection", "linear_inspection")
        self.mode.addItem("Lateral Capture", "lateral_capture")
        self.mode.addItem("Advanced Waypoints", "waypoints")
        self.mode.addItem("Tower Mapping", "tower_mapping")
        self.mode.addItem("Solar Inspection", "solar_inspection")
        self.mode.addItem("Magnetic Mapping", "magnetic_mapping")
        self.mode.addItem("Orbit", "orbit")
        self.mode.addItem("Panorama", "panorama")
        self.mode.addItem("360 Bubble", "bubble_360")
        self.mode.addItem("Smart Adaptive", "smart_adaptive")
        self.mode.addItem("Double Grid", "double_grid")
        self.mode.addItem("Roof Inspection", "roof_inspection")
        form.addRow("Template:", self.mode)

        self.altitude = QDoubleSpinBox()
        self.altitude.setRange(5.0, 300.0)
        self.altitude.setValue(60.0)
        self.altitude.setSuffix(" m")
        form.addRow("Target Altitude:", self.altitude)

        self.front = QSpinBox()
        self.front.setRange(20, 95)
        self.front.setValue(80)
        self.front.setSuffix(" %")
        form.addRow("Front Overlap:", self.front)

        self.side = QSpinBox()
        self.side.setRange(20, 95)
        self.side.setValue(70)
        self.side.setSuffix(" %")
        form.addRow("Side Overlap:", self.side)

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 20.0)
        self.speed.setDecimals(1)
        self.speed.setValue(5.0)
        self.speed.setSuffix(" m/s")
        form.addRow("Speed:", self.speed)

        self.flight_direction = QDoubleSpinBox()
        self.flight_direction.setRange(-180.0, 180.0)
        self.flight_direction.setDecimals(1)
        self.flight_direction.setValue(0.0)
        self.flight_direction.setSuffix(" deg")
        form.addRow("Flight Direction:", self.flight_direction)

        self.lock_camera_direction = QCheckBox("Lock camera direction")
        self.lock_camera_direction.setChecked(False)
        form.addRow("", self.lock_camera_direction)

        self.camera_direction = QDoubleSpinBox()
        self.camera_direction.setRange(-180.0, 180.0)
        self.camera_direction.setDecimals(1)
        self.camera_direction.setValue(0.0)
        self.camera_direction.setSuffix(" deg")
        form.addRow("Camera Direction:", self.camera_direction)

        self.gimbal_tilt = QDoubleSpinBox()
        self.gimbal_tilt.setRange(-120.0, 30.0)
        self.gimbal_tilt.setDecimals(1)
        self.gimbal_tilt.setValue(-90.0)
        self.gimbal_tilt.setSuffix(" deg")
        form.addRow("Gimbal Tilt:", self.gimbal_tilt)

        self.inspection_dwell = QDoubleSpinBox()
        self.inspection_dwell.setRange(0.0, 30.0)
        self.inspection_dwell.setDecimals(1)
        self.inspection_dwell.setValue(1.5)
        self.inspection_dwell.setSuffix(" s")
        form.addRow("Inspection Dwell:", self.inspection_dwell)

        self.linear_segment_enabled = QCheckBox("Battery-aware split")
        self.linear_segment_enabled.setChecked(True)
        form.addRow("", self.linear_segment_enabled)

        self.linear_segment_length = QDoubleSpinBox()
        self.linear_segment_length.setRange(100.0, 20000.0)
        self.linear_segment_length.setDecimals(0)
        self.linear_segment_length.setValue(1500.0)
        self.linear_segment_length.setSuffix(" m")
        form.addRow("Linear Segment Max:", self.linear_segment_length)

        self.lateral_standoff = QDoubleSpinBox()
        self.lateral_standoff.setRange(0.5, 200.0)
        self.lateral_standoff.setDecimals(1)
        self.lateral_standoff.setValue(10.0)
        self.lateral_standoff.setSuffix(" m")
        form.addRow("Lateral Standoff:", self.lateral_standoff)

        self.lateral_target_side = QComboBox()
        self.lateral_target_side.addItem("Target On Right", "right")
        self.lateral_target_side.addItem("Target On Left", "left")
        form.addRow("Lateral Target Side:", self.lateral_target_side)

        self.waypoint_heading_mode = QComboBox()
        self.waypoint_heading_mode.addItem("Tangent To Path", "tangent")
        self.waypoint_heading_mode.addItem("Fixed Yaw", "fixed")
        self.waypoint_heading_mode.addItem("Point Of Interest", "poi")
        form.addRow("Waypoint Heading:", self.waypoint_heading_mode)

        self.waypoint_fixed_yaw = QDoubleSpinBox()
        self.waypoint_fixed_yaw.setRange(-180.0, 180.0)
        self.waypoint_fixed_yaw.setDecimals(1)
        self.waypoint_fixed_yaw.setValue(0.0)
        self.waypoint_fixed_yaw.setSuffix(" deg")
        form.addRow("Waypoint Fixed Yaw:", self.waypoint_fixed_yaw)

        self.waypoint_capture_enabled = QCheckBox("Capture at waypoints")
        self.waypoint_capture_enabled.setChecked(True)
        form.addRow("", self.waypoint_capture_enabled)

        self.waypoint_smoothing = QCheckBox("Curve path between points")
        self.waypoint_smoothing.setChecked(True)
        form.addRow("", self.waypoint_smoothing)

        self.waypoint_turn_radius = QDoubleSpinBox()
        self.waypoint_turn_radius.setRange(0.0, 80.0)
        self.waypoint_turn_radius.setDecimals(1)
        self.waypoint_turn_radius.setValue(6.0)
        self.waypoint_turn_radius.setSuffix(" m")
        form.addRow("Waypoint Turn Radius:", self.waypoint_turn_radius)

        self.orbit_radius = QDoubleSpinBox()
        self.orbit_radius.setRange(1.0, 500.0)
        self.orbit_radius.setDecimals(1)
        self.orbit_radius.setValue(20.0)
        self.orbit_radius.setSuffix(" m")
        form.addRow("Orbit Radius:", self.orbit_radius)

        self.orbit_levels = QSpinBox()
        self.orbit_levels.setRange(1, 8)
        self.orbit_levels.setValue(2)
        form.addRow("Orbit Levels:", self.orbit_levels)

        self.orbit_vertical_step = QDoubleSpinBox()
        self.orbit_vertical_step.setRange(0.5, 100.0)
        self.orbit_vertical_step.setDecimals(1)
        self.orbit_vertical_step.setValue(8.0)
        self.orbit_vertical_step.setSuffix(" m")
        form.addRow("Orbit Vertical Step:", self.orbit_vertical_step)

        self.orbit_poi_lock = QCheckBox("POI yaw lock")
        self.orbit_poi_lock.setChecked(True)
        form.addRow("", self.orbit_poi_lock)

        self.panorama_overlap = QSpinBox()
        self.panorama_overlap.setRange(5, 90)
        self.panorama_overlap.setValue(35)
        self.panorama_overlap.setSuffix(" %")
        form.addRow("Panorama Overlap:", self.panorama_overlap)

        self.panorama_multi_row = QCheckBox("Multi-row panorama")
        self.panorama_multi_row.setChecked(False)
        form.addRow("", self.panorama_multi_row)

        self.panorama_row_count = QSpinBox()
        self.panorama_row_count.setRange(1, 8)
        self.panorama_row_count.setValue(1)
        form.addRow("Panorama Rows:", self.panorama_row_count)

        self.panorama_pitch_step = QDoubleSpinBox()
        self.panorama_pitch_step.setRange(1.0, 45.0)
        self.panorama_pitch_step.setDecimals(1)
        self.panorama_pitch_step.setValue(12.0)
        self.panorama_pitch_step.setSuffix(" deg")
        form.addRow("Panorama Pitch Step:", self.panorama_pitch_step)

        self.bubble_overlap = QSpinBox()
        self.bubble_overlap.setRange(5, 90)
        self.bubble_overlap.setValue(35)
        self.bubble_overlap.setSuffix(" %")
        form.addRow("Bubble Overlap:", self.bubble_overlap)

        self.bubble_pitch_step = QDoubleSpinBox()
        self.bubble_pitch_step.setRange(1.0, 45.0)
        self.bubble_pitch_step.setDecimals(1)
        self.bubble_pitch_step.setValue(12.0)
        self.bubble_pitch_step.setSuffix(" deg")
        form.addRow("Bubble Pitch Step:", self.bubble_pitch_step)

        self.bubble_top_pitch = QDoubleSpinBox()
        self.bubble_top_pitch.setRange(-120.0, 30.0)
        self.bubble_top_pitch.setDecimals(1)
        self.bubble_top_pitch.setValue(20.0)
        self.bubble_top_pitch.setSuffix(" deg")
        form.addRow("Bubble Top Pitch:", self.bubble_top_pitch)

        self.bubble_bottom_pitch = QDoubleSpinBox()
        self.bubble_bottom_pitch.setRange(-120.0, 30.0)
        self.bubble_bottom_pitch.setDecimals(1)
        self.bubble_bottom_pitch.setValue(-90.0)
        self.bubble_bottom_pitch.setSuffix(" deg")
        form.addRow("Bubble Bottom Pitch:", self.bubble_bottom_pitch)

        self.tower_top_alt = QDoubleSpinBox()
        self.tower_top_alt.setRange(5.0, 500.0)
        self.tower_top_alt.setDecimals(1)
        self.tower_top_alt.setValue(110.0)
        self.tower_top_alt.setSuffix(" m")
        form.addRow("Tower Top Alt:", self.tower_top_alt)

        self.tower_bottom_alt = QDoubleSpinBox()
        self.tower_bottom_alt.setRange(5.0, 500.0)
        self.tower_bottom_alt.setDecimals(1)
        self.tower_bottom_alt.setValue(35.0)
        self.tower_bottom_alt.setSuffix(" m")
        form.addRow("Tower Bottom Alt:", self.tower_bottom_alt)

        self.tower_object_radius = QDoubleSpinBox()
        self.tower_object_radius.setRange(0.5, 100.0)
        self.tower_object_radius.setDecimals(1)
        self.tower_object_radius.setValue(3.0)
        self.tower_object_radius.setSuffix(" m")
        form.addRow("Tower Object Radius:", self.tower_object_radius)

        self.tower_flight_radius = QDoubleSpinBox()
        self.tower_flight_radius.setRange(2.0, 250.0)
        self.tower_flight_radius.setDecimals(1)
        self.tower_flight_radius.setValue(12.0)
        self.tower_flight_radius.setSuffix(" m")
        form.addRow("Tower Flight Radius:", self.tower_flight_radius)

        self.tower_resume_enabled = QCheckBox("Battery-change resume by orbit level")
        self.tower_resume_enabled.setChecked(True)
        form.addRow("", self.tower_resume_enabled)

        self.solar_row_angle = QDoubleSpinBox()
        self.solar_row_angle.setRange(-180.0, 180.0)
        self.solar_row_angle.setDecimals(1)
        self.solar_row_angle.setValue(0.0)
        self.solar_row_angle.setSuffix(" deg")
        form.addRow("Solar Row Angle:", self.solar_row_angle)

        self.solar_sensor_profile = QComboBox()
        self.solar_sensor_profile.addItem("RGB", "rgb")
        self.solar_sensor_profile.addItem("Thermal", "thermal")
        form.addRow("Solar Sensor Mode:", self.solar_sensor_profile)

        self.solar_orientation = QComboBox()
        self.solar_orientation.addItem("Row Aligned (Locked)", "row_aligned")
        self.solar_orientation.addItem("Path Aligned", "path_aligned")
        form.addRow("Solar Orientation:", self.solar_orientation)

        self.magnetic_tie_spacing = QDoubleSpinBox()
        self.magnetic_tie_spacing.setRange(5.0, 1000.0)
        self.magnetic_tie_spacing.setDecimals(1)
        self.magnetic_tie_spacing.setValue(60.0)
        self.magnetic_tie_spacing.setSuffix(" m")
        form.addRow("Tie-line Spacing:", self.magnetic_tie_spacing)

        self.magnetic_turn_radius = QDoubleSpinBox()
        self.magnetic_turn_radius.setRange(0.0, 80.0)
        self.magnetic_turn_radius.setDecimals(1)
        self.magnetic_turn_radius.setValue(8.0)
        self.magnetic_turn_radius.setSuffix(" m")
        form.addRow("Turn Smooth Radius:", self.magnetic_turn_radius)

        self.facade_top_alt = QDoubleSpinBox()
        self.facade_top_alt.setRange(5.0, 500.0)
        self.facade_top_alt.setDecimals(1)
        self.facade_top_alt.setValue(80.0)
        self.facade_top_alt.setSuffix(" m")
        form.addRow("Facade Top Alt:", self.facade_top_alt)

        self.facade_bottom_alt = QDoubleSpinBox()
        self.facade_bottom_alt.setRange(5.0, 500.0)
        self.facade_bottom_alt.setDecimals(1)
        self.facade_bottom_alt.setValue(40.0)
        self.facade_bottom_alt.setSuffix(" m")
        form.addRow("Facade Bottom Alt:", self.facade_bottom_alt)

        self.facade_distance = QDoubleSpinBox()
        self.facade_distance.setRange(0.5, 200.0)
        self.facade_distance.setDecimals(1)
        self.facade_distance.setValue(8.0)
        self.facade_distance.setSuffix(" m")
        form.addRow("Facade Distance:", self.facade_distance)

        self.facade_rotate = QCheckBox("Rotate points 180 deg")
        form.addRow("", self.facade_rotate)

        self.facade_capture_profile = QComboBox()
        self.facade_capture_profile.addItem("Normal (0 deg)", "normal")
        self.facade_capture_profile.addItem("Oblique (-20 deg)", "oblique")
        self.facade_capture_profile.addItem("Custom (Use Gimbal)", "custom")
        form.addRow("Facade Capture:", self.facade_capture_profile)

        self.ground_offset = QDoubleSpinBox()
        self.ground_offset.setRange(-100.0, 300.0)
        self.ground_offset.setDecimals(1)
        self.ground_offset.setValue(0.0)
        self.ground_offset.setSuffix(" m")
        form.addRow("Ground Offset:", self.ground_offset)

        self.terrain_follow = QCheckBox("Terrain adjustment (uses polygon elevation if available)")
        form.addRow("", self.terrain_follow)

        self.terrain_follow_mode = QComboBox()
        self.terrain_follow_mode.addItem("AGL (Terrain Follow)", "agl")
        self.terrain_follow_mode.addItem("AMSL (Fixed Altitude)", "amsl")
        form.addRow("Terrain Mode:", self.terrain_follow_mode)

        terrain_row = QHBoxLayout()
        self.terrain_source_path = QLineEdit()
        self.terrain_source_path.setPlaceholderText("Optional DEM/DSM source (.json/.tif/.tiff)")
        self.btn_pick_terrain = QPushButton("Browse")
        self.btn_pick_terrain.clicked.connect(self.pick_terrain_source)
        terrain_row.addWidget(self.terrain_source_path, stretch=1)
        terrain_row.addWidget(self.btn_pick_terrain)
        form.addRow("Terrain Source:", terrain_row)

        self.terrain_normal_enable = QCheckBox("Slope-normal camera compensation")
        self.terrain_normal_enable.setChecked(False)
        form.addRow("", self.terrain_normal_enable)

        self.terrain_normal_gain = QDoubleSpinBox()
        self.terrain_normal_gain.setRange(0.0, 3.0)
        self.terrain_normal_gain.setDecimals(2)
        self.terrain_normal_gain.setSingleStep(0.1)
        self.terrain_normal_gain.setValue(1.0)
        form.addRow("Slope Gain:", self.terrain_normal_gain)

        self.terrain_normal_yaw_align = QCheckBox("Align yaw to slope direction")
        self.terrain_normal_yaw_align.setChecked(False)
        form.addRow("", self.terrain_normal_yaw_align)

        self.wind_speed = QDoubleSpinBox()
        self.wind_speed.setRange(0.0, 40.0)
        self.wind_speed.setDecimals(1)
        self.wind_speed.setValue(0.0)
        self.wind_speed.setSuffix(" m/s")
        form.addRow("Wind Speed:", self.wind_speed)

        self.wind_direction = QDoubleSpinBox()
        self.wind_direction.setRange(-180.0, 180.0)
        self.wind_direction.setDecimals(1)
        self.wind_direction.setValue(0.0)
        self.wind_direction.setSuffix(" deg")
        form.addRow("Wind Direction:", self.wind_direction)

        self.wind_gust = QDoubleSpinBox()
        self.wind_gust.setRange(0.0, 60.0)
        self.wind_gust.setDecimals(1)
        self.wind_gust.setValue(0.0)
        self.wind_gust.setSuffix(" m/s")
        form.addRow("Wind Gust:", self.wind_gust)

        self.facade_curvature_alignment = QCheckBox("Curvature-aware facade alignment")
        self.facade_curvature_alignment.setChecked(False)
        form.addRow("", self.facade_curvature_alignment)

        self.auto_update = QCheckBox("Auto-generate when params change")
        self.auto_update.setChecked(True)
        form.addRow("", self.auto_update)

        self.btn_import_kml = QPushButton("Import KML Geometry")
        self.btn_import_kml.setIcon(standard_icon(self, "SP_DirOpenIcon", "SP_FileDialogStart"))
        self.btn_import_kml.clicked.connect(self.import_kml_geometry)
        form.addRow("", self.btn_import_kml)

        self.btn_generate = QPushButton("Generate Mission")
        self.btn_generate.setIcon(standard_icon(self, "SP_MediaPlay", "SP_ArrowForward"))
        self.btn_generate.clicked.connect(self.generate_plan)
        form.addRow("", self.btn_generate)

        self.btn_export_qgc = QPushButton("Export QGC WPL")
        self.btn_export_qgc.setIcon(standard_icon(self, "SP_DialogSaveButton", "SP_DriveFDIcon"))
        self.btn_export_qgc.clicked.connect(self.export_qgc)
        form.addRow("", self.btn_export_qgc)

        self.btn_export_geojson = QPushButton("Export GeoJSON")
        self.btn_export_geojson.setIcon(standard_icon(self, "SP_DialogSaveButton", "SP_DriveFDIcon"))
        self.btn_export_geojson.clicked.connect(self.export_geojson_file)
        form.addRow("", self.btn_export_geojson)

        self.btn_export_recipe = QPushButton("Export Flight Recipe")
        self.btn_export_recipe.setIcon(standard_icon(self, "SP_DialogSaveButton", "SP_DriveFDIcon"))
        self.btn_export_recipe.clicked.connect(self.export_recipe_file)
        form.addRow("", self.btn_export_recipe)

        panel_layout.addWidget(params)

        safety = QGroupBox("Safety Constraints")
        safety_form = QFormLayout(safety)

        self.min_alt = QDoubleSpinBox()
        self.min_alt.setRange(5.0, 300.0)
        self.min_alt.setValue(30.0)
        self.min_alt.setSuffix(" m")
        safety_form.addRow("Min Altitude:", self.min_alt)

        self.max_alt = QDoubleSpinBox()
        self.max_alt.setRange(6.0, 500.0)
        self.max_alt.setValue(120.0)
        self.max_alt.setSuffix(" m")
        safety_form.addRow("Max Altitude:", self.max_alt)

        self.standoff = QDoubleSpinBox()
        self.standoff.setRange(0.0, 200.0)
        self.standoff.setValue(8.0)
        self.standoff.setSuffix(" m")
        safety_form.addRow("Standoff:", self.standoff)

        self.rth_alt = QDoubleSpinBox()
        self.rth_alt.setRange(10.0, 500.0)
        self.rth_alt.setValue(140.0)
        self.rth_alt.setSuffix(" m")
        safety_form.addRow("RTH Altitude:", self.rth_alt)

        self.oa_profile = QComboBox()
        self.oa_profile.addItem("Balanced", "balanced")
        self.oa_profile.addItem("Conservative", "conservative")
        self.oa_profile.addItem("Aggressive", "aggressive")
        safety_form.addRow("Obstacle Avoidance:", self.oa_profile)

        self.no_fly_count_label = QLabel("No-fly polygons: 0")
        safety_form.addRow("", self.no_fly_count_label)
        self.btn_clear_no_fly = QPushButton("Clear No-Fly Polygons")
        self.btn_clear_no_fly.setIcon(standard_icon(self, "SP_TrashIcon", "SP_DialogResetButton"))
        self.btn_clear_no_fly.clicked.connect(self.clear_no_fly_polygons)
        safety_form.addRow("", self.btn_clear_no_fly)

        panel_layout.addWidget(safety)

        repeat = QGroupBox("Repeat Inspection")
        repeat_form = QFormLayout(repeat)

        self.repeat_mode = QCheckBox("Enable repeat mode from baseline recipe")
        repeat_form.addRow("", self.repeat_mode)

        recipe_row = QHBoxLayout()
        self.btn_load_recipe = QPushButton("Load Baseline Recipe")
        self.btn_load_recipe.setIcon(standard_icon(self, "SP_DialogOpenButton", "SP_DirOpenIcon"))
        self.btn_load_recipe.clicked.connect(self.load_repeat_recipe)
        self.btn_clear_recipe = QPushButton("Clear")
        self.btn_clear_recipe.setIcon(standard_icon(self, "SP_DialogDiscardButton", "SP_TrashIcon"))
        self.btn_clear_recipe.clicked.connect(self.clear_repeat_recipe)
        recipe_row.addWidget(self.btn_load_recipe)
        recipe_row.addWidget(self.btn_clear_recipe)
        repeat_form.addRow("Baseline:", recipe_row)

        self.recipe_label = QLabel("No baseline recipe loaded.")
        self.recipe_label.setWordWrap(True)
        repeat_form.addRow("", self.recipe_label)

        panel_layout.addWidget(repeat)

        linked = QGroupBox("Linked Missions")
        linked_form = QFormLayout(linked)
        self.linked_count_label = QLabel("Segments queued: 0")
        linked_form.addRow("", self.linked_count_label)
        self.linked_optimize_order = QCheckBox("Optimize segment order (nearest transition)")
        self.linked_optimize_order.setChecked(True)
        linked_form.addRow("", self.linked_optimize_order)
        self.linked_dry_run = QCheckBox("Simulate dry run checks")
        self.linked_dry_run.setChecked(True)
        linked_form.addRow("", self.linked_dry_run)

        linked_btn_row = QHBoxLayout()
        self.btn_add_segment = QPushButton("Add Current Segment")
        self.btn_add_segment.setIcon(standard_icon(self, "SP_DialogApplyButton", "SP_DialogYesButton"))
        self.btn_add_segment.clicked.connect(self.add_current_segment)
        self.btn_pop_segment = QPushButton("Remove Last")
        self.btn_pop_segment.setIcon(standard_icon(self, "SP_ArrowBack", "SP_ArrowLeft"))
        self.btn_pop_segment.clicked.connect(self.remove_last_segment)
        self.btn_clear_segments = QPushButton("Clear Segments")
        self.btn_clear_segments.setIcon(standard_icon(self, "SP_TrashIcon", "SP_DialogDiscardButton"))
        self.btn_clear_segments.clicked.connect(self.clear_linked_segments)
        linked_btn_row.addWidget(self.btn_add_segment)
        linked_btn_row.addWidget(self.btn_pop_segment)
        linked_btn_row.addWidget(self.btn_clear_segments)
        linked_form.addRow("", linked_btn_row)

        self.btn_generate_linked = QPushButton("Generate Linked Mission")
        self.btn_generate_linked.setIcon(standard_icon(self, "SP_MediaPlay", "SP_ArrowForward"))
        self.btn_generate_linked.clicked.connect(self.generate_linked_plan)
        linked_form.addRow("", self.btn_generate_linked)
        panel_layout.addWidget(linked)

        fly = QGroupBox("Fly-To-Draw")
        fly_form = QFormLayout(fly)
        self.measure_lon = QDoubleSpinBox()
        self.measure_lon.setRange(-180.0, 180.0)
        self.measure_lon.setDecimals(7)
        self.measure_lon.setValue(0.0)
        fly_form.addRow("Lon:", self.measure_lon)
        self.measure_lat = QDoubleSpinBox()
        self.measure_lat.setRange(-90.0, 90.0)
        self.measure_lat.setDecimals(7)
        self.measure_lat.setValue(0.0)
        fly_form.addRow("Lat:", self.measure_lat)
        self.measure_alt = QDoubleSpinBox()
        self.measure_alt.setRange(-500.0, 5000.0)
        self.measure_alt.setDecimals(2)
        self.measure_alt.setValue(60.0)
        self.measure_alt.setSuffix(" m")
        fly_form.addRow("Altitude:", self.measure_alt)
        self.measure_yaw = QDoubleSpinBox()
        self.measure_yaw.setRange(-180.0, 180.0)
        self.measure_yaw.setDecimals(1)
        self.measure_yaw.setValue(0.0)
        self.measure_yaw.setSuffix(" deg")
        fly_form.addRow("Yaw:", self.measure_yaw)
        self.measure_count_label = QLabel("Measurement points: 0")
        fly_form.addRow("", self.measure_count_label)

        fly_btn_row = QHBoxLayout()
        self.btn_drop_measure = QPushButton("Drop Point")
        self.btn_drop_measure.setIcon(standard_icon(self, "SP_FileDialogNewFolder", "SP_DialogApplyButton"))
        self.btn_drop_measure.clicked.connect(self.drop_measurement_point)
        self.btn_apply_measure = QPushButton("Apply Measured Geometry")
        self.btn_apply_measure.setIcon(standard_icon(self, "SP_DialogApplyButton", "SP_DialogYesButton"))
        self.btn_apply_measure.clicked.connect(self.apply_measured_geometry)
        self.btn_clear_measure = QPushButton("Clear Points")
        self.btn_clear_measure.setIcon(standard_icon(self, "SP_TrashIcon", "SP_DialogDiscardButton"))
        self.btn_clear_measure.clicked.connect(self.clear_measurement_points)
        fly_btn_row.addWidget(self.btn_drop_measure)
        fly_btn_row.addWidget(self.btn_apply_measure)
        fly_btn_row.addWidget(self.btn_clear_measure)
        fly_form.addRow("", fly_btn_row)
        panel_layout.addWidget(fly)

        self.summary_label = QLabel("Draw a polygon/line/point on the map to start.")
        self.summary_label.setWordWrap(True)
        panel_layout.addWidget(self.summary_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel_layout.addWidget(self.log, stretch=1)

        panel_scroll = QScrollArea()
        panel_scroll.setWidget(panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel_scroll.setMinimumWidth(380)
        panel_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        root.addWidget(self.map, stretch=3)
        root.addWidget(panel_scroll, stretch=2)

        for widget in (
            self.camera,
            self.mode,
            self.altitude,
            self.front,
            self.side,
            self.speed,
            self.flight_direction,
            self.camera_direction,
            self.gimbal_tilt,
            self.inspection_dwell,
            self.linear_segment_length,
            self.lateral_standoff,
            self.lateral_target_side,
            self.waypoint_heading_mode,
            self.waypoint_fixed_yaw,
            self.waypoint_turn_radius,
            self.orbit_radius,
            self.orbit_levels,
            self.orbit_vertical_step,
            self.panorama_overlap,
            self.panorama_row_count,
            self.panorama_pitch_step,
            self.bubble_overlap,
            self.bubble_pitch_step,
            self.bubble_top_pitch,
            self.bubble_bottom_pitch,
            self.tower_top_alt,
            self.tower_bottom_alt,
            self.tower_object_radius,
            self.tower_flight_radius,
            self.solar_row_angle,
            self.solar_sensor_profile,
            self.solar_orientation,
            self.magnetic_tie_spacing,
            self.magnetic_turn_radius,
            self.facade_top_alt,
            self.facade_bottom_alt,
            self.facade_distance,
            self.facade_capture_profile,
            self.ground_offset,
            self.terrain_follow_mode,
            self.terrain_normal_gain,
            self.wind_speed,
            self.wind_direction,
            self.wind_gust,
            self.min_alt,
            self.max_alt,
            self.standoff,
            self.rth_alt,
            self.oa_profile,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_param_change)
            else:
                widget.currentIndexChanged.connect(self._on_param_change)

        self.repeat_mode.toggled.connect(self._on_param_change)
        self.terrain_follow.toggled.connect(self._on_param_change)
        self.terrain_normal_enable.toggled.connect(self._on_param_change)
        self.terrain_normal_yaw_align.toggled.connect(self._on_param_change)
        self.facade_curvature_alignment.toggled.connect(self._on_param_change)
        self.terrain_source_path.textChanged.connect(self._on_param_change)
        self.lock_camera_direction.toggled.connect(self._on_lock_camera_direction_toggled)
        self.facade_rotate.toggled.connect(self._on_param_change)
        self.linear_segment_enabled.toggled.connect(self._on_param_change)
        self.waypoint_capture_enabled.toggled.connect(self._on_param_change)
        self.waypoint_smoothing.toggled.connect(self._on_param_change)
        self.orbit_poi_lock.toggled.connect(self._on_param_change)
        self.panorama_multi_row.toggled.connect(self._on_param_change)
        self.tower_resume_enabled.toggled.connect(self._on_param_change)
        self.mode.currentIndexChanged.connect(self._sync_mode_state)
        self.facade_capture_profile.currentIndexChanged.connect(self._sync_mode_state)
        self.waypoint_heading_mode.currentIndexChanged.connect(self._sync_mode_state)
        self._sync_mode_state()

    def _can_generate(self) -> bool:
        if self.repeat_mode.isChecked() and self.loaded_repeat_recipe is not None:
            return True
        template = str(self.mode.currentData())
        if template == "linear_inspection":
            return len(self.linear_path) >= 2 or len(self.survey_polygon) >= 2
        if template == "lateral_capture":
            return len(self.linear_path) >= 2 or len(self.survey_polygon) >= 2
        if template == "waypoints":
            return len(self.linear_path) >= 2 or len(self.survey_polygon) >= 2
        if template == "orbit":
            return len(self.tower_center) >= 2 or len(self.survey_polygon) >= 3
        if template == "panorama":
            return len(self.tower_center) >= 2 or len(self.survey_polygon) >= 3
        if template == "bubble_360":
            return len(self.tower_center) >= 2 or len(self.survey_polygon) >= 3
        if template == "tower_mapping":
            return len(self.tower_center) >= 2 or len(self.survey_polygon) >= 3
        if template == "solar_inspection":
            has_rows = bool(self.solar_rows) or len(self.linear_path) >= 2
            return len(self.survey_polygon) >= 3 and has_rows
        return len(self.survey_polygon) >= 3

    def _on_param_change(self, *_):
        if self.auto_update.isChecked() and self._can_generate():
            self.generate_plan()

    def _on_lock_camera_direction_toggled(self, *_):
        self._sync_mode_state()
        self._on_param_change()

    def _sync_mode_state(self):
        template = str(self.mode.currentData())
        roof_mode = template == "roof_inspection"
        facade_mode = template in {"facade", "facade_mapping"}
        facade_mapping_mode = template == "facade_mapping"
        linear_mode = template == "linear_inspection"
        lateral_mode = template == "lateral_capture"
        waypoint_mode = template == "waypoints"
        orbit_mode = template == "orbit"
        panorama_mode = template == "panorama"
        bubble_mode = template == "bubble_360"
        tower_mode = template == "tower_mapping"
        solar_mode = template == "solar_inspection"
        magnetic_mode = template == "magnetic_mapping"
        self.inspection_dwell.setEnabled(roof_mode or linear_mode or waypoint_mode or panorama_mode or bubble_mode)
        self.lock_camera_direction.setEnabled(roof_mode)
        self.camera_direction.setEnabled(roof_mode and self.lock_camera_direction.isChecked())
        self.linear_segment_enabled.setEnabled(linear_mode)
        self.linear_segment_length.setEnabled(linear_mode and self.linear_segment_enabled.isChecked())
        self.lateral_standoff.setEnabled(lateral_mode)
        self.lateral_target_side.setEnabled(lateral_mode)
        self.waypoint_heading_mode.setEnabled(waypoint_mode)
        fixed_mode = waypoint_mode and str(self.waypoint_heading_mode.currentData()) == "fixed"
        self.waypoint_fixed_yaw.setEnabled(fixed_mode)
        self.waypoint_capture_enabled.setEnabled(waypoint_mode)
        self.waypoint_smoothing.setEnabled(waypoint_mode)
        self.waypoint_turn_radius.setEnabled(waypoint_mode and self.waypoint_smoothing.isChecked())
        self.orbit_radius.setEnabled(orbit_mode)
        self.orbit_levels.setEnabled(orbit_mode)
        self.orbit_vertical_step.setEnabled(orbit_mode and self.orbit_levels.value() > 1)
        self.orbit_poi_lock.setEnabled(orbit_mode)
        self.panorama_overlap.setEnabled(panorama_mode)
        self.panorama_multi_row.setEnabled(panorama_mode)
        self.panorama_row_count.setEnabled(panorama_mode and self.panorama_multi_row.isChecked())
        self.panorama_pitch_step.setEnabled(panorama_mode and self.panorama_multi_row.isChecked())
        self.bubble_overlap.setEnabled(bubble_mode)
        self.bubble_pitch_step.setEnabled(bubble_mode)
        self.bubble_top_pitch.setEnabled(bubble_mode)
        self.bubble_bottom_pitch.setEnabled(bubble_mode)
        self.tower_top_alt.setEnabled(tower_mode)
        self.tower_bottom_alt.setEnabled(tower_mode)
        self.tower_object_radius.setEnabled(tower_mode)
        self.tower_flight_radius.setEnabled(tower_mode)
        self.tower_resume_enabled.setEnabled(tower_mode)
        self.solar_row_angle.setEnabled(solar_mode)
        self.solar_sensor_profile.setEnabled(solar_mode)
        self.solar_orientation.setEnabled(solar_mode)
        self.magnetic_tie_spacing.setEnabled(magnetic_mode)
        self.magnetic_turn_radius.setEnabled(magnetic_mode)
        self.facade_top_alt.setEnabled(facade_mode)
        self.facade_bottom_alt.setEnabled(facade_mode)
        self.facade_distance.setEnabled(facade_mode)
        self.facade_rotate.setEnabled(facade_mode)
        self.facade_capture_profile.setEnabled(facade_mode)
        self.facade_curvature_alignment.setEnabled(facade_mode)

        terrain_enabled = self.terrain_follow.isChecked()
        terrain_normal_enabled = terrain_enabled and self.terrain_normal_enable.isChecked()
        self.terrain_follow_mode.setEnabled(terrain_enabled)
        self.terrain_source_path.setEnabled(terrain_enabled)
        self.btn_pick_terrain.setEnabled(terrain_enabled)
        self.terrain_normal_enable.setEnabled(terrain_enabled)
        self.terrain_normal_gain.setEnabled(terrain_normal_enabled)
        self.terrain_normal_yaw_align.setEnabled(terrain_normal_enabled)

        if facade_mapping_mode:
            profile = str(self.facade_capture_profile.currentData())
            self.front.blockSignals(True)
            self.side.blockSignals(True)
            self.front.setValue(max(self.front.value(), 85))
            self.side.setValue(max(self.side.value(), 80))
            self.front.blockSignals(False)
            self.side.blockSignals(False)

            if profile == "normal":
                self.gimbal_tilt.blockSignals(True)
                self.gimbal_tilt.setValue(0.0)
                self.gimbal_tilt.blockSignals(False)
            elif profile == "oblique":
                self.gimbal_tilt.blockSignals(True)
                self.gimbal_tilt.setValue(-20.0)
                self.gimbal_tilt.blockSignals(False)

    def set_polygon(self, coords: list[list[float]]):
        if not isinstance(coords, list) or len(coords) < 3:
            return
        normalized: list[list[float]] = []
        for c in coords:
            if not isinstance(c, list) or len(c) < 2:
                continue
            row = [float(c[0]), float(c[1])]
            if len(c) >= 3:
                row.append(float(c[2]))
            normalized.append(row)
        if len(normalized) < 3:
            return
        if str(self.mode.currentData()) not in {"solar_inspection", "waypoints"}:
            self.linear_path = []
            self.solar_rows = []
        self.survey_polygon = normalized
        self.summary_label.setText(f"Polygon loaded: {len(self.survey_polygon)} vertices")
        if self.auto_update.isChecked():
            self.generate_plan()

    def set_line(self, coords: list[list[float]]):
        if not isinstance(coords, list) or len(coords) < 2:
            return
        normalized: list[list[float]] = []
        for c in coords:
            if not isinstance(c, list) or len(c) < 2:
                continue
            normalized.append([float(c[0]), float(c[1])])
        if len(normalized) < 2:
            return
        current_template = str(self.mode.currentData())
        if current_template == "solar_inspection":
            self.solar_rows.append(normalized)
            self.linear_path = normalized
            self.summary_label.setText(
                f"Solar row marked: {len(normalized)} points (rows: {len(self.solar_rows)})"
            )
        elif current_template == "lateral_capture":
            self.solar_rows = []
            self.linear_path = normalized
            self.summary_label.setText(f"Lateral target line loaded: {len(self.linear_path)} points")
        elif current_template == "waypoints":
            self.solar_rows = []
            self.linear_path = normalized
            self.summary_label.setText(f"Waypoint path loaded: {len(self.linear_path)} points")
        elif current_template in {"facade", "facade_mapping"}:
            self.solar_rows = []
            self.linear_path = normalized
            self.summary_label.setText(f"Facade curve guide loaded: {len(self.linear_path)} points")
        else:
            self.solar_rows = []
            self.survey_polygon = []
            self.linear_path = normalized
            self.summary_label.setText(f"Line loaded: {len(self.linear_path)} points")
        if self.auto_update.isChecked():
            self.generate_plan()

    def set_rows(self, rows: list[list[list[float]]]):
        if not isinstance(rows, list):
            return
        template = str(self.mode.currentData())
        if template not in {"solar_inspection", "linear_inspection", "lateral_capture", "waypoints"}:
            self.solar_rows = []
            self.linear_path = []
            return

        normalized_rows: list[list[list[float]]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 2:
                continue
            clean_row: list[list[float]] = []
            for c in row:
                if not isinstance(c, list) or len(c) < 2:
                    continue
                clean_row.append([float(c[0]), float(c[1])])
            if len(clean_row) >= 2:
                normalized_rows.append(clean_row)

        self.solar_rows = normalized_rows
        self.linear_path = normalized_rows[-1] if normalized_rows else []

        if template == "solar_inspection":
            self.summary_label.setText(f"Solar rows loaded: {len(self.solar_rows)}")
        elif template == "lateral_capture" and self.linear_path:
            self.summary_label.setText(f"Lateral target line loaded: {len(self.linear_path)} points")
        elif template == "waypoints" and self.linear_path:
            self.summary_label.setText(f"Waypoint path loaded: {len(self.linear_path)} points")
        elif self.linear_path:
            self.summary_label.setText(f"Line loaded: {len(self.linear_path)} points")

        if self.auto_update.isChecked() and self._can_generate():
            self.generate_plan()

    def set_point(self, coords: list[float]):
        if not isinstance(coords, list) or len(coords) < 2:
            return
        self.tower_center = [float(coords[0]), float(coords[1])]
        current_template = str(self.mode.currentData())
        if current_template == "waypoints":
            self.summary_label.setText(
                f"Waypoint POI set: lon {self.tower_center[0]:.6f}, lat {self.tower_center[1]:.6f}"
            )
        elif current_template == "orbit":
            self.summary_label.setText(
                f"Orbit center set: lon {self.tower_center[0]:.6f}, lat {self.tower_center[1]:.6f}"
            )
        elif current_template == "panorama":
            self.summary_label.setText(
                f"Panorama center set: lon {self.tower_center[0]:.6f}, lat {self.tower_center[1]:.6f}"
            )
        elif current_template == "bubble_360":
            self.summary_label.setText(
                f"360 bubble center set: lon {self.tower_center[0]:.6f}, lat {self.tower_center[1]:.6f}"
            )
        else:
            self.summary_label.setText(
                f"Tower center set: lon {self.tower_center[0]:.6f}, lat {self.tower_center[1]:.6f}"
            )
        if self.auto_update.isChecked():
            self.generate_plan()

    def set_no_fly_polygons(self, polygons: list[list[list[float]]]):
        if not isinstance(polygons, list):
            return
        normalized: list[list[list[float]]] = []
        for poly in polygons:
            if not isinstance(poly, list) or len(poly) < 3:
                continue
            clean: list[list[float]] = []
            for c in poly:
                if not isinstance(c, list) or len(c) < 2:
                    continue
                clean.append([float(c[0]), float(c[1])])
            if len(clean) < 3:
                continue
            if clean[0] != clean[-1]:
                clean.append(list(clean[0]))
            normalized.append(clean)
        self.no_fly_polygons = normalized
        self.no_fly_count_label.setText(f"No-fly polygons: {len(self.no_fly_polygons)}")
        if self.auto_update.isChecked() and self._can_generate():
            self.generate_plan()

    def clear_no_fly_polygons(self):
        self.no_fly_polygons = []
        self.no_fly_count_label.setText("No-fly polygons: 0")
        if self.auto_update.isChecked() and self._can_generate():
            self.generate_plan()

    def record_measurement_from_payload(self, payload: dict):
        if not isinstance(payload, dict):
            return
        try:
            lon = float(payload.get("lon", payload.get("longitude")))
            lat = float(payload.get("lat", payload.get("latitude")))
        except Exception:
            return
        alt = float(payload.get("alt_m", payload.get("altitude_m", payload.get("alt", self.altitude.value()))))
        yaw = float(payload.get("yaw_deg", payload.get("yaw", 0.0)))
        self.measure_lon.setValue(lon)
        self.measure_lat.setValue(lat)
        self.measure_alt.setValue(alt)
        self.measure_yaw.setValue(yaw)
        self.drop_measurement_point()

    def drop_measurement_point(self):
        sample = {
            "lon": float(self.measure_lon.value()),
            "lat": float(self.measure_lat.value()),
            "alt_m": float(self.measure_alt.value()),
            "yaw_deg": float(self.measure_yaw.value()),
        }
        self.measurement_samples.append(sample)
        self.measure_count_label.setText(f"Measurement points: {len(self.measurement_samples)}")
        self.summary_label.setText(
            f"Measurement dropped: lon {sample['lon']:.6f}, lat {sample['lat']:.6f}, alt {sample['alt_m']:.1f} m"
        )

    def clear_measurement_points(self):
        self.measurement_samples = []
        self.measure_count_label.setText("Measurement points: 0")

    def apply_measured_geometry(self):
        if not self.measurement_samples:
            QMessageBox.warning(self, "No Measurements", "Drop at least one measurement point first.")
            return
        pts = [[float(p["lon"]), float(p["lat"])] for p in self.measurement_samples]
        alts = [float(p["alt_m"]) for p in self.measurement_samples]
        avg_alt = sum(alts) / len(alts)
        self.altitude.blockSignals(True)
        self.altitude.setValue(avg_alt)
        self.altitude.blockSignals(False)

        template = str(self.mode.currentData())
        center_modes = {"tower_mapping", "orbit", "panorama", "bubble_360"}
        line_modes = {"linear_inspection", "lateral_capture", "waypoints"}
        if template in center_modes:
            lon = sum(p[0] for p in pts) / len(pts)
            lat = sum(p[1] for p in pts) / len(pts)
            self.set_point([lon, lat])
            if template == "tower_mapping" and alts:
                self.tower_top_alt.setValue(max(alts))
                self.tower_bottom_alt.setValue(min(alts))
        elif template in line_modes:
            if len(pts) < 2:
                QMessageBox.warning(self, "Insufficient Measurements", "Line-based missions need at least 2 points.")
                return
            self.set_line(pts)
        else:
            if len(pts) >= 3:
                self.set_polygon(pts)
            elif len(pts) >= 2:
                self.set_line(pts)
            else:
                self.set_point(pts[0])
            if template in {"facade", "facade_mapping"} and alts:
                self.facade_top_alt.setValue(max(alts))
                self.facade_bottom_alt.setValue(min(alts))
            if template == "solar_inspection" and len(pts) >= 2 and not self.solar_rows:
                self.set_rows([pts])

        self.summary_label.setText(
            f"Applied measured geometry ({len(self.measurement_samples)} points), default altitude set to {avg_alt:.1f} m"
        )
        if not self.auto_update.isChecked() and self._can_generate():
            self.generate_plan()

    def _update_linked_count_label(self):
        self.linked_count_label.setText(f"Segments queued: {len(self.linked_segments)}")

    def add_current_segment(self):
        if self.current_plan is None:
            QMessageBox.warning(self, "No Mission", "Generate a mission segment first.")
            return
        try:
            recipe = self.planner._coerce_recipe(self.current_plan.flight_recipe)  # noqa: SLF001
        except Exception as exc:
            QMessageBox.critical(self, "Segment Error", f"Unable to queue current mission:\n{exc}")
            return
        self.linked_segments.append(recipe)
        self._update_linked_count_label()
        self.summary_label.setText(f"Queued linked segment: {recipe.recipe_id} ({len(self.linked_segments)} total)")

    def remove_last_segment(self):
        if self.linked_segments:
            self.linked_segments.pop()
        self._update_linked_count_label()

    def clear_linked_segments(self):
        self.linked_segments = []
        self._update_linked_count_label()

    def generate_linked_plan(self):
        if len(self.linked_segments) < 2:
            QMessageBox.warning(self, "Linked Mission", "Queue at least 2 mission segments first.")
            return
        constraint_override = {
            "min_altitude_m": float(self.min_alt.value()),
            "max_altitude_m": float(self.max_alt.value()),
            "standoff_m": float(self.standoff.value()),
            "rth_altitude_m": float(max(self.rth_alt.value(), self.max_alt.value() + 5.0)),
            "rth_action": "return_home",
            "obstacle_avoidance_profile": str(self.oa_profile.currentData()),
            "no_fly_polygons": self.no_fly_polygons,
        }
        try:
            plan = self.planner.generate_linked_mission(
                recipes=self.linked_segments,
                speed_m_s=float(self.speed.value()),
                camera=str(self.camera.currentData()),
                optimize_order=bool(self.linked_optimize_order.isChecked()),
                constraints=constraint_override,
                simulate_dry_run=bool(self.linked_dry_run.isChecked()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Linked Mission Error", str(exc))
            return

        self.current_plan = plan
        payload = json.dumps(plan.geojson)
        self.map.page().runJavaScript(f"showMission({payload});")

        meta = plan.flight_recipe.get("metadata", {}) if isinstance(plan.flight_recipe, dict) else {}
        lines = [
            f"Planner source: {plan.source}",
            f"Template: {plan.template}",
            f"Linked segments: {plan.linked_segment_count}",
            f"Transitions: {plan.linked_transition_count}",
            f"No-fly polygons: {plan.no_fly_polygon_count}",
            f"Distance: {plan.path_distance_m:.1f} m",
            f"Estimated flight time: {plan.estimated_time_min:.2f} min",
            f"Autopilot commands: {len(plan.autopilot_commands)}",
            f"Dry run: {'ok' if plan.linked_dry_run_ok else 'attention needed'} "
            f"(altitude_violations={int(meta.get('linked_dry_run_altitude_violations', 0))}, "
            f"obstacle_hits={int(meta.get('linked_dry_run_obstacle_hits', 0))})",
        ]
        adjustments = plan.safety_adjustments or {}
        if adjustments:
            lines.append(
                "Safety adjustments: "
                f"alt={int(adjustments.get('altitude_clamps', 0))}, "
                f"standoff={int(adjustments.get('standoff_adjustments', 0))}, "
                f"geofence={int(adjustments.get('geofence_projections', 0))}, "
                f"no_fly={int(adjustments.get('no_fly_projections', 0))}, "
                f"detours={int(adjustments.get('obstacle_detours', 0))}, "
                f"terrain_normal={int(adjustments.get('terrain_normal_adjustments', 0))}"
            )
        self.summary_label.setText(lines[0])
        self.log.setPlainText("\n".join(lines))
        self.planGenerated.emit(plan)

    def clear_polygon(self):
        self.survey_polygon = []
        self.no_fly_polygons = []
        self.linear_path = []
        self.solar_rows = []
        self.tower_center = []
        self.current_plan = None
        self.no_fly_count_label.setText("No-fly polygons: 0")
        self.summary_label.setText("Geometry cleared. Draw a new area/line/point.")
        self.log.clear()

    def clear_area(self):
        self.survey_polygon = []

    def _build_constraints(self, geofence: list[list[float]]) -> MissionConstraints:
        min_alt = float(self.min_alt.value())
        max_alt = float(self.max_alt.value())
        if max_alt < min_alt + 1.0:
            max_alt = min_alt + 1.0
            self.max_alt.blockSignals(True)
            self.max_alt.setValue(max_alt)
            self.max_alt.blockSignals(False)

        rth = float(max(self.rth_alt.value(), max_alt + 5.0))
        if rth != float(self.rth_alt.value()):
            self.rth_alt.blockSignals(True)
            self.rth_alt.setValue(rth)
            self.rth_alt.blockSignals(False)

        return MissionConstraints(
            geofence=geofence,
            min_altitude_m=min_alt,
            max_altitude_m=max_alt,
            standoff_m=float(self.standoff.value()),
            rth_altitude_m=rth,
            no_fly_polygons=self.no_fly_polygons,
            rth_action="return_home",
            obstacle_avoidance_profile=str(self.oa_profile.currentData()),
        )

    def _repeat_asset_override(self) -> AssetReferenceFrame | None:
        if self.loaded_repeat_recipe is None or len(self.survey_polygon) < 3:
            return None
        base = self.loaded_repeat_recipe.asset_frame
        frame = self.planner.derive_asset_frame(
            polygon_lonlat=self.survey_polygon,
            asset_id=base.asset_id,
            coordinate_source="repeat_reanchor",
        )
        frame.reference_note = "Re-anchored using current polygon."
        return frame

    def generate_plan(self):
        repeat_enabled = self.repeat_mode.isChecked()
        template = str(self.mode.currentData())
        linear_mode = template == "linear_inspection"
        lateral_mode = template == "lateral_capture"
        waypoint_mode = template == "waypoints"
        orbit_mode = template == "orbit"
        panorama_mode = template == "panorama"
        bubble_mode = template == "bubble_360"
        tower_mode = template == "tower_mapping"
        solar_mode = template == "solar_inspection"
        magnetic_mode = template == "magnetic_mapping"
        if repeat_enabled and self.loaded_repeat_recipe is None:
            QMessageBox.warning(self, "Missing Baseline", "Load a baseline flight recipe for repeat mode.")
            return

        if linear_mode or lateral_mode or waypoint_mode:
            if len(self.linear_path) < 2 and len(self.survey_polygon) < 2 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Line", "Draw a line on the map first (inspection/lateral/waypoints).")
                return
        elif orbit_mode:
            if len(self.tower_center) < 2 and len(self.survey_polygon) < 3 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Orbit Center", "Place a center marker or draw an orbit area polygon.")
                return
        elif panorama_mode:
            if len(self.tower_center) < 2 and len(self.survey_polygon) < 3 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Panorama Center", "Place a center marker or draw a panorama area polygon.")
                return
        elif bubble_mode:
            if len(self.tower_center) < 2 and len(self.survey_polygon) < 3 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Bubble Center", "Place a center marker or draw a bubble area polygon.")
                return
        elif tower_mode:
            if len(self.tower_center) < 2 and len(self.survey_polygon) < 3 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Tower Center", "Mark a tower center point or draw a tower boundary polygon.")
                return
        elif solar_mode:
            if len(self.survey_polygon) < 3 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Polygon", "Draw a farm polygon on the map first.")
                return
            if not self.solar_rows and len(self.linear_path) < 2 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Rows", "Mark at least one solar row line on the map.")
                return
        else:
            if len(self.survey_polygon) < 3 and not repeat_enabled:
                QMessageBox.warning(self, "Missing Polygon", "Draw a survey polygon on the map first.")
                return

        base_polygon = self.survey_polygon
        base_line = self.linear_path
        base_rows = list(self.solar_rows)
        tower_center = list(self.tower_center)
        if (linear_mode or lateral_mode or waypoint_mode) and len(base_line) < 2 and len(self.survey_polygon) >= 2:
            base_line = self.survey_polygon[:2]
        if (linear_mode or lateral_mode or waypoint_mode) and len(base_line) < 2 and self.loaded_repeat_recipe is not None:
            candidate = self.loaded_repeat_recipe.metadata.get("linear_path_points", [])
            if isinstance(candidate, list) and len(candidate) >= 2:
                base_line = candidate
        if lateral_mode and len(base_line) < 2 and self.loaded_repeat_recipe is not None:
            lateral_candidate = self.loaded_repeat_recipe.metadata.get("lateral_path_points", [])
            if isinstance(lateral_candidate, list) and len(lateral_candidate) >= 2:
                base_line = lateral_candidate
        if waypoint_mode and len(base_line) < 2 and self.loaded_repeat_recipe is not None:
            candidate_wp = self.loaded_repeat_recipe.metadata.get("waypoint_path_points", [])
            if isinstance(candidate_wp, list) and len(candidate_wp) >= 2:
                base_line = candidate_wp
        if solar_mode and not base_rows and len(base_line) >= 2:
            base_rows = [base_line]
        if solar_mode and not base_rows and self.loaded_repeat_recipe is not None:
            candidate_rows = self.loaded_repeat_recipe.metadata.get("solar_rows_lonlat", [])
            if isinstance(candidate_rows, list):
                clean_rows: list[list[list[float]]] = []
                for row in candidate_rows:
                    if isinstance(row, list) and len(row) >= 2:
                        clean_rows.append(row)
                if clean_rows:
                    base_rows = clean_rows
        if tower_mode and len(tower_center) < 2 and len(base_polygon) >= 3:
            lon = sum(float(p[0]) for p in base_polygon) / len(base_polygon)
            lat = sum(float(p[1]) for p in base_polygon) / len(base_polygon)
            tower_center = [lon, lat]
        if orbit_mode and len(tower_center) < 2 and len(base_polygon) >= 3:
            lon = sum(float(p[0]) for p in base_polygon) / len(base_polygon)
            lat = sum(float(p[1]) for p in base_polygon) / len(base_polygon)
            tower_center = [lon, lat]
        if panorama_mode and len(tower_center) < 2 and len(base_polygon) >= 3:
            lon = sum(float(p[0]) for p in base_polygon) / len(base_polygon)
            lat = sum(float(p[1]) for p in base_polygon) / len(base_polygon)
            tower_center = [lon, lat]
        if bubble_mode and len(tower_center) < 2 and len(base_polygon) >= 3:
            lon = sum(float(p[0]) for p in base_polygon) / len(base_polygon)
            lat = sum(float(p[1]) for p in base_polygon) / len(base_polygon)
            tower_center = [lon, lat]
        if tower_mode and len(tower_center) < 2 and self.loaded_repeat_recipe is not None:
            candidate_center = self.loaded_repeat_recipe.metadata.get("tower_center_lonlat", [])
            if isinstance(candidate_center, list) and len(candidate_center) >= 2:
                tower_center = [float(candidate_center[0]), float(candidate_center[1])]
        if orbit_mode and len(tower_center) < 2 and self.loaded_repeat_recipe is not None:
            candidate_center = self.loaded_repeat_recipe.metadata.get("orbit_center_lonlat", [])
            if isinstance(candidate_center, list) and len(candidate_center) >= 2:
                tower_center = [float(candidate_center[0]), float(candidate_center[1])]
        if panorama_mode and len(tower_center) < 2 and self.loaded_repeat_recipe is not None:
            candidate_center = self.loaded_repeat_recipe.metadata.get("panorama_center_lonlat", [])
            if isinstance(candidate_center, list) and len(candidate_center) >= 2:
                tower_center = [float(candidate_center[0]), float(candidate_center[1])]
        if bubble_mode and len(tower_center) < 2 and self.loaded_repeat_recipe is not None:
            candidate_center = self.loaded_repeat_recipe.metadata.get("bubble_center_lonlat", [])
            if isinstance(candidate_center, list) and len(candidate_center) >= 2:
                tower_center = [float(candidate_center[0]), float(candidate_center[1])]

        if (linear_mode or lateral_mode or waypoint_mode) and len(base_polygon) < 3 and len(base_line) >= 2:
            line_buffer_m = max(15.0, float(self.standoff.value()))
            if lateral_mode:
                line_buffer_m = max(line_buffer_m, float(self.lateral_standoff.value()) + 10.0)
            base_polygon = self.planner.line_buffer_geofence(
                line_lonlat=base_line,
                buffer_m=line_buffer_m,
            )
        if tower_mode and len(base_polygon) < 3 and len(tower_center) >= 2:
            base_polygon = self.planner.tower_buffer_geofence(
                center_lonlat=tower_center,
                flight_radius_m=float(self.tower_flight_radius.value()),
                padding_m=max(15.0, float(self.standoff.value())),
            )
        if orbit_mode and len(base_polygon) < 3 and len(tower_center) >= 2:
            base_polygon = self.planner.tower_buffer_geofence(
                center_lonlat=tower_center,
                flight_radius_m=float(self.orbit_radius.value()),
                padding_m=max(15.0, float(self.standoff.value())),
            )
        if panorama_mode and len(base_polygon) < 3 and len(tower_center) >= 2:
            base_polygon = self.planner.tower_buffer_geofence(
                center_lonlat=tower_center,
                flight_radius_m=max(12.0, float(self.standoff.value()) + 6.0),
                padding_m=max(10.0, float(self.standoff.value())),
            )
        if bubble_mode and len(base_polygon) < 3 and len(tower_center) >= 2:
            base_polygon = self.planner.tower_buffer_geofence(
                center_lonlat=tower_center,
                flight_radius_m=max(12.0, float(self.standoff.value()) + 6.0),
                padding_m=max(10.0, float(self.standoff.value())),
            )

        if len(base_polygon) < 3 and self.loaded_repeat_recipe is not None:
            base_polygon = self.loaded_repeat_recipe.constraints.geofence

        if len(base_polygon) < 3:
            QMessageBox.warning(
                self,
                "Missing Geofence",
                "Provide a survey polygon/line or use a baseline recipe with geofence data.",
            )
            return

        if (linear_mode or lateral_mode or waypoint_mode) and len(base_line) < 2:
            QMessageBox.warning(
                self,
                "Missing Line",
                "Provide a line for mission planning.",
            )
            return
        if waypoint_mode and str(self.waypoint_heading_mode.currentData()) == "poi" and len(tower_center) < 2:
            QMessageBox.warning(
                self,
                "Missing POI",
                "Place a marker on the map for point-of-interest heading mode.",
            )
            return
        if orbit_mode and len(tower_center) < 2:
            QMessageBox.warning(
                self,
                "Missing Orbit Center",
                "Place a center marker for orbit planning.",
            )
            return
        if panorama_mode and len(tower_center) < 2:
            QMessageBox.warning(
                self,
                "Missing Panorama Center",
                "Place a center marker for panorama planning.",
            )
            return
        if bubble_mode and len(tower_center) < 2:
            QMessageBox.warning(
                self,
                "Missing Bubble Center",
                "Place a center marker for 360 bubble planning.",
            )
            return
        if tower_mode and len(tower_center) < 2:
            QMessageBox.warning(
                self,
                "Missing Tower Center",
                "Provide a tower center point for tower mapping.",
            )
            return
        if solar_mode and not base_rows:
            QMessageBox.warning(
                self,
                "Missing Rows",
                "Provide at least one solar row line for row-aligned planning.",
            )
            return

        constraints = self._build_constraints(base_polygon)
        asset_override = self._repeat_asset_override() if repeat_enabled else None

        try:
            roof_mode = template == "roof_inspection"
            facade_mode = template in {"facade", "facade_mapping"}
            plan = self.planner.generate(
                polygon_lonlat=base_polygon,
                altitude_m=float(self.altitude.value()),
                front_overlap_pct=float(self.front.value()),
                side_overlap_pct=float(self.side.value()),
                speed_m_s=float(self.speed.value()),
                mode=template,
                camera=str(self.camera.currentData()),
                flight_direction_deg=float(self.flight_direction.value()),
                camera_direction_deg=float(self.camera_direction.value()) if (roof_mode and self.lock_camera_direction.isChecked()) else None,
                gimbal_tilt_deg=float(self.gimbal_tilt.value()),
                inspection_dwell_s=float(self.inspection_dwell.value()),
                facade_top_altitude_m=float(self.facade_top_alt.value()) if facade_mode else None,
                facade_bottom_altitude_m=float(self.facade_bottom_alt.value()) if facade_mode else None,
                facade_standoff_m=float(self.facade_distance.value()) if facade_mode else None,
                facade_rotate_points_180=bool(self.facade_rotate.isChecked()) if facade_mode else False,
                facade_capture_profile=str(self.facade_capture_profile.currentData()) if facade_mode else "custom",
                linear_path_lonlat=base_line if linear_mode else None,
                linear_segmentation_enabled=bool(self.linear_segment_enabled.isChecked()) if linear_mode else False,
                linear_max_segment_length_m=float(self.linear_segment_length.value()) if linear_mode else 1500.0,
                lateral_target_path_lonlat=base_line if lateral_mode else None,
                lateral_standoff_m=float(self.lateral_standoff.value()) if lateral_mode else 10.0,
                lateral_target_side=str(self.lateral_target_side.currentData()) if lateral_mode else "right",
                waypoint_path_lonlat=base_line if waypoint_mode else None,
                waypoint_heading_mode=str(self.waypoint_heading_mode.currentData()) if waypoint_mode else "tangent",
                waypoint_fixed_yaw_deg=float(self.waypoint_fixed_yaw.value()) if waypoint_mode else 0.0,
                waypoint_poi_lonlat=tower_center if (waypoint_mode and str(self.waypoint_heading_mode.currentData()) == "poi" and len(tower_center) >= 2) else None,
                waypoint_enable_smoothing=bool(self.waypoint_smoothing.isChecked()) if waypoint_mode else False,
                waypoint_turn_radius_m=float(self.waypoint_turn_radius.value()) if waypoint_mode else 6.0,
                waypoint_capture_enabled=bool(self.waypoint_capture_enabled.isChecked()) if waypoint_mode else True,
                orbit_center_lonlat=tower_center if orbit_mode else None,
                orbit_radius_m=float(self.orbit_radius.value()) if orbit_mode else None,
                orbit_level_count=int(self.orbit_levels.value()) if orbit_mode else None,
                orbit_vertical_step_m=float(self.orbit_vertical_step.value()) if orbit_mode else None,
                orbit_poi_yaw_lock=bool(self.orbit_poi_lock.isChecked()) if orbit_mode else True,
                orbit_poi_lonlat=tower_center if (orbit_mode and self.orbit_poi_lock.isChecked() and len(tower_center) >= 2) else None,
                panorama_center_lonlat=tower_center if panorama_mode else None,
                panorama_overlap_pct=float(self.panorama_overlap.value()) if panorama_mode else 35.0,
                panorama_multi_row_enabled=bool(self.panorama_multi_row.isChecked()) if panorama_mode else False,
                panorama_row_count=int(self.panorama_row_count.value()) if panorama_mode else 1,
                panorama_pitch_step_deg=float(self.panorama_pitch_step.value()) if panorama_mode else 12.0,
                bubble_center_lonlat=tower_center if bubble_mode else None,
                bubble_overlap_pct=float(self.bubble_overlap.value()) if bubble_mode else 35.0,
                bubble_pitch_step_deg=float(self.bubble_pitch_step.value()) if bubble_mode else 12.0,
                bubble_top_pitch_deg=float(self.bubble_top_pitch.value()) if bubble_mode else 20.0,
                bubble_bottom_pitch_deg=float(self.bubble_bottom_pitch.value()) if bubble_mode else -90.0,
                tower_center_lonlat=tower_center if tower_mode else None,
                tower_top_altitude_m=float(self.tower_top_alt.value()) if tower_mode else None,
                tower_bottom_altitude_m=float(self.tower_bottom_alt.value()) if tower_mode else None,
                tower_object_radius_m=float(self.tower_object_radius.value()) if tower_mode else 2.0,
                tower_flight_radius_m=float(self.tower_flight_radius.value()) if tower_mode else None,
                tower_resume_enabled=bool(self.tower_resume_enabled.isChecked()) if tower_mode else True,
                solar_row_angle_deg=float(self.solar_row_angle.value()) if solar_mode else None,
                solar_sensor_profile=str(self.solar_sensor_profile.currentData()) if solar_mode else "rgb",
                solar_orientation_mode=str(self.solar_orientation.currentData()) if solar_mode else "row_aligned",
                solar_rows_lonlat=base_rows if solar_mode else None,
                magnetic_tie_line_spacing_m=float(self.magnetic_tie_spacing.value()) if magnetic_mode else 50.0,
                magnetic_smoothing_radius_m=float(self.magnetic_turn_radius.value()) if magnetic_mode else 8.0,
                ground_offset_m=float(self.ground_offset.value()),
                terrain_follow_enabled=bool(self.terrain_follow.isChecked()),
                terrain_source_path=self.terrain_source_path.text().strip(),
                terrain_follow_mode=str(self.terrain_follow_mode.currentData()),
                terrain_normal_camera_enabled=bool(self.terrain_normal_enable.isChecked()),
                terrain_normal_gain=float(self.terrain_normal_gain.value()),
                terrain_normal_yaw_align=bool(self.terrain_normal_yaw_align.isChecked()),
                wind_speed_m_s=float(self.wind_speed.value()),
                wind_direction_deg=float(self.wind_direction.value()),
                wind_gust_m_s=float(self.wind_gust.value()),
                facade_curvature_alignment=bool(self.facade_curvature_alignment.isChecked()) if facade_mode else False,
                facade_curve_path_lonlat=self.linear_path if (facade_mode and len(self.linear_path) >= 2) else None,
                repeat_recipe=self.loaded_repeat_recipe if repeat_enabled else None,
                enable_repeat=repeat_enabled,
                constraints=constraints,
                asset_frame=asset_override,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Mission Planning Error", str(exc))
            return

        self.current_plan = plan
        payload = json.dumps(plan.geojson)
        self.map.page().runJavaScript(f"showMission({payload});")

        recipe = plan.flight_recipe if isinstance(plan.flight_recipe, dict) else {}
        recipe_id = str(recipe.get("recipe_id", "[unknown]"))
        recipe_version = int(recipe.get("version", 1)) if recipe else 1

        lines = [
            f"Planner source: {plan.source}",
            f"Template: {plan.template}",
            f"Recipe: {recipe_id} (v{recipe_version})",
            f"Repeat mode: {'enabled' if plan.repeat_enabled else 'disabled'}",
            f"Waypoints: {len(plan.waypoints)}",
            f"Autopilot commands: {len(plan.autopilot_commands)}",
            f"Distance: {plan.path_distance_m:.1f} m",
            f"Estimated flight time: {plan.estimated_time_min:.2f} min",
            f"Estimated GSD: {plan.estimated_gsd_cm:.2f} cm/px",
            f"Mapping: direction {plan.flight_direction_deg:.1f} deg, gimbal {plan.gimbal_tilt_deg:.1f} deg, ground offset {plan.ground_offset_m:.1f} m",
            f"Camera direction: {'locked' if plan.camera_direction_locked else 'path-aligned'} at {plan.camera_direction_deg:.1f} deg",
            f"Capture spacing/interval: {plan.capture_spacing_m:.2f} m / {plan.capture_interval_s:.2f} s (line spacing {plan.line_spacing_m:.2f} m)",
            f"Terrain follow: {'enabled' if plan.terrain_follow_enabled else 'disabled'}",
            f"Terrain model: {plan.terrain_model_type} ({plan.terrain_model_source}) mode {plan.terrain_follow_mode.upper()}",
            f"Slope-normal camera: {'on' if plan.terrain_normal_camera_enabled else 'off'} (gain {plan.terrain_normal_gain:.2f}, yaw align {'on' if plan.terrain_normal_yaw_align else 'off'})",
            f"Wind: {plan.wind_speed_m_s:.1f} m/s @ {plan.wind_direction_deg:.1f} deg, gust {plan.wind_gust_m_s:.1f} m/s, adjusted speed {plan.wind_adjusted_speed_m_s:.2f} m/s ({plan.wind_penalty_pct:.1f}% penalty)",
            f"Safety: alt {constraints.min_altitude_m:.1f}-{constraints.max_altitude_m:.1f} m, standoff {constraints.standoff_m:.1f} m, OA {constraints.obstacle_avoidance_profile}",
        ]

        if plan.template == "double_grid":
            lines.append(
                f"3D modelling: cross angle {plan.double_grid_cross_angle_deg:.1f} deg, overlap F/S {plan.front_overlap_pct:.1f}%/{plan.side_overlap_pct:.1f}%"
            )
            if plan.camera_policy:
                iso_max = plan.camera_policy.get("iso_max", "[n/a]")
                shutter = float(plan.camera_policy.get("min_shutter_s", 0.0))
                if shutter > 0.0:
                    lines.append(
                        f"Camera policy: locked exposure, ISO <= {iso_max}, min shutter {shutter:.4f} s ({(1.0 / shutter):.0f} fps equivalent)"
                    )
                else:
                    lines.append(f"Camera policy: locked exposure, ISO <= {iso_max}")

        if plan.template == "roof_inspection":
            lines.append(
                f"Roof inspection: stop-and-capture with dwell {plan.inspection_dwell_s:.1f} s at each point"
            )

        if plan.template == "facade":
            lines.append(
                f"Facade inspection: top/bottom {plan.facade_top_altitude_m:.1f}/{plan.facade_bottom_altitude_m:.1f} m, standoff {plan.facade_standoff_m:.1f} m, rotate {'on' if plan.facade_rotate_points_180 else 'off'}"
            )
            lines.append(
                f"Facade curvature alignment: {'enabled' if plan.facade_curvature_alignment else 'disabled'}"
            )
            lines.append("Facade baseline: first polygon edge (draw first segment along building face).")

        if plan.template == "facade_mapping":
            lines.append(
                f"Facade mapping: profile {plan.facade_capture_profile}, top/bottom {plan.facade_top_altitude_m:.1f}/{plan.facade_bottom_altitude_m:.1f} m, standoff {plan.facade_standoff_m:.1f} m"
            )
            lines.append(
                f"Facade curvature alignment: {'enabled' if plan.facade_curvature_alignment else 'disabled'}"
            )
            lines.append(
                f"Facade mapping tune: overlap F/S {plan.front_overlap_pct:.1f}%/{plan.side_overlap_pct:.1f}%, smooth profile {plan.smooth_motion_profile or 'default'}"
            )
            lines.append("Facade baseline: first polygon edge (draw first segment along building face).")
            if plan.camera_policy:
                shutter = float(plan.camera_policy.get("min_shutter_s", 0.0))
                iso_max = plan.camera_policy.get("iso_max", "[n/a]")
                if shutter > 0.0:
                    lines.append(
                        f"Camera policy: locked exposure, ISO <= {iso_max}, min shutter {shutter:.4f} s ({(1.0 / shutter):.0f} fps equivalent)"
                    )
                else:
                    lines.append(f"Camera policy: locked exposure, ISO <= {iso_max}")

        if plan.template == "linear_inspection":
            lines.append(
                f"Linear inspection: path {plan.linear_path_length_m:.1f} m, segments {plan.linear_segment_count}, split {'on' if plan.linear_segmentation_enabled else 'off'}"
            )

        if plan.template == "lateral_capture":
            lines.append(
                f"Lateral capture: standoff {plan.lateral_standoff_m:.1f} m, target side {plan.lateral_target_side}, yaw offset {plan.lateral_yaw_offset_deg:.1f} deg"
            )
            lines.append(
                f"Lateral profile path: {plan.lateral_path_length_m:.1f} m (continuous capture while moving)"
            )

        if plan.template == "waypoints":
            lines.append(
                f"Advanced waypoints: heading {plan.waypoint_heading_mode}, fixed yaw {plan.waypoint_fixed_yaw_deg:.1f} deg, path {plan.waypoint_path_length_m:.1f} m"
            )
            lines.append(
                f"Waypoint actions: capture {'on' if plan.waypoint_capture_enabled else 'off'}, dwell {plan.inspection_dwell_s:.1f} s, smoothing {'on' if plan.waypoint_smoothing_enabled else 'off'} ({plan.waypoint_turn_radius_m:.1f} m)"
            )

        if plan.template == "orbit":
            lines.append(
                f"Orbit: radius {plan.orbit_radius_m:.1f} m, levels {plan.orbit_level_count}, vertical step {plan.orbit_vertical_step_m:.1f} m"
            )
            lines.append(
                f"Orbit yaw mode: {'POI lock' if plan.orbit_poi_yaw_lock else 'tangent-to-path'}"
            )

        if plan.template == "panorama":
            lines.append(
                f"Panorama: overlap {plan.panorama_overlap_pct:.1f}%, rows {max(1, plan.panorama_row_count)}, pitch step {plan.panorama_pitch_step_deg:.1f} deg"
            )
            lines.append(
                f"Panorama sweep: yaw step {plan.panorama_yaw_step_deg:.1f} deg, frames/row {max(1, plan.panorama_yaw_count)}"
            )

        if plan.template == "bubble_360":
            lines.append(
                f"360 bubble: overlap {plan.bubble_overlap_pct:.1f}%, pitch {plan.bubble_top_pitch_deg:.1f} to {plan.bubble_bottom_pitch_deg:.1f} deg, step {plan.bubble_pitch_step_deg:.1f} deg"
            )
            lines.append(
                f"360 bubble sweep: yaw step {plan.bubble_yaw_step_deg:.1f} deg, yaw frames {max(1, plan.bubble_yaw_count)}, pitch layers {max(1, plan.bubble_pitch_count)}"
            )

        if plan.template == "tower_mapping":
            lines.append(
                f"Tower mapping: top/bottom {plan.tower_top_altitude_m:.1f}/{plan.tower_bottom_altitude_m:.1f} m, object radius {plan.tower_object_radius_m:.1f} m, flight radius {plan.tower_flight_radius_m:.1f} m"
            )
            lines.append(
                f"Tower resume: {'enabled' if plan.tower_resume_enabled else 'disabled'}, orbit levels {plan.tower_orbit_count}, safe RTH {plan.tower_safe_rth_altitude_m:.1f} m"
            )

        if plan.template == "solar_inspection":
            lines.append(
                f"Solar inspection: row angle {plan.solar_row_angle_deg:.1f} deg, sensor {plan.solar_sensor_profile}, orientation {plan.solar_orientation_mode}, marked rows {len(base_rows)}"
            )

        if plan.template == "magnetic_mapping":
            lines.append(
                f"Magnetic mapping: tie-line spacing {plan.magnetic_tie_line_spacing_m:.1f} m, turn smooth radius {plan.magnetic_smoothing_radius_m:.1f} m"
            )

        coverage = plan.expected_coverage or {}
        if coverage:
            lines.append(
                "Coverage: "
                f"{coverage.get('achieved_viewpoints', 0)}/{coverage.get('required_viewpoints', 0)} viewpoints "
                f"({float(coverage.get('achieved_coverage_pct', 0.0)):.1f}%)"
            )

        adjustments = plan.safety_adjustments or {}
        if adjustments:
            lines.append(
                "Safety adjustments: "
                f"alt={int(adjustments.get('altitude_clamps', 0))}, "
                f"standoff={int(adjustments.get('standoff_adjustments', 0))}, "
                f"geofence={int(adjustments.get('geofence_projections', 0))}, "
                f"no_fly={int(adjustments.get('no_fly_projections', 0))}, "
                f"detours={int(adjustments.get('obstacle_detours', 0))}"
            )

        self.summary_label.setText(lines[0])
        self.log.setPlainText("\n".join(lines))
        self.planGenerated.emit(plan)

    def pick_terrain_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Terrain Source",
            self.terrain_source_path.text().strip() or "",
            "Terrain Files (*.json *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        self.terrain_source_path.setText(path)

    def import_kml_geometry(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import KML Geometry",
            "",
            "KML Files (*.kml *.xml);;All Files (*)",
        )
        if not path:
            return

        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            QMessageBox.critical(self, "KML Import Error", f"Unable to parse KML:\n{exc}")
            return

        def _parse_coords(text: str) -> list[list[float]]:
            out: list[list[float]] = []
            for token in text.replace("\n", " ").replace("\t", " ").split():
                parts = token.split(",")
                if len(parts) < 2:
                    continue
                try:
                    out.append([float(parts[0]), float(parts[1])])
                except Exception:
                    continue
            return out

        polygon: list[list[float]] = []
        for node in root.findall(".//{*}Polygon//{*}outerBoundaryIs//{*}LinearRing//{*}coordinates"):
            if node.text:
                polygon = _parse_coords(node.text)
                if len(polygon) >= 3:
                    break

        if len(polygon) >= 3:
            self.set_polygon(polygon)
            return

        line: list[list[float]] = []
        for node in root.findall(".//{*}LineString//{*}coordinates"):
            if node.text:
                line = _parse_coords(node.text)
                if len(line) >= 2:
                    break

        if len(line) >= 2:
            self.set_line(line)
            return

        QMessageBox.warning(self, "KML Import", "No usable Polygon or LineString geometry found in file.")

    def load_repeat_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Baseline Flight Recipe",
            "",
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            recipe = load_flight_recipe(path)
        except Exception as exc:
            QMessageBox.critical(self, "Recipe Load Error", str(exc))
            return
        self.loaded_repeat_recipe = recipe
        self.repeat_recipe_path = path
        self.recipe_label.setText(f"Loaded: {Path(path).name} [{recipe.recipe_id} v{recipe.version}]")
        self.repeat_mode.setChecked(True)
        if self.auto_update.isChecked() and self._can_generate():
            self.generate_plan()

    def clear_repeat_recipe(self):
        self.loaded_repeat_recipe = None
        self.repeat_recipe_path = ""
        self.recipe_label.setText("No baseline recipe loaded.")
        if self.repeat_mode.isChecked():
            self.repeat_mode.setChecked(False)

    def export_qgc(self):
        if self.current_plan is None:
            QMessageBox.warning(self, "No Mission", "Generate a mission first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export QGC WPL",
            "mission.waypoints",
            "QGC Waypoints (*.waypoints *.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            out = export_qgc_wpl(path, self.current_plan)
            QMessageBox.information(self, "Export Complete", f"Saved QGC mission:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def export_geojson_file(self):
        if self.current_plan is None:
            QMessageBox.warning(self, "No Mission", "Generate a mission first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export GeoJSON",
            "mission.geojson",
            "GeoJSON (*.geojson *.json);;All Files (*)",
        )
        if not path:
            return
        try:
            out = export_geojson(path, self.current_plan)
            QMessageBox.information(self, "Export Complete", f"Saved GeoJSON:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def export_recipe_file(self):
        if self.current_plan is None:
            QMessageBox.warning(self, "No Mission", "Generate a mission first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Flight Recipe",
            "flight_recipe.json",
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            out = export_flight_recipe(path, self.current_plan)
            QMessageBox.information(self, "Export Complete", f"Saved Flight Recipe:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
