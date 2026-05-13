# OpenDroneKit Open Source Inspection Toolkit Specification

## 1. Product Definition

OpenDroneKit is an open source inspection operations toolkit for drone based asset inspection. It is not only a flight controller. It is a full workbench for planning missions, capturing structured datasets, validating data quality, processing visual outputs, running defect analysis, creating measurements, producing inspection reports and maintaining a clear project audit trail.

The product must feel like a reliable engineering tool. It must not feel like a script launcher or a debug panel. Every operator facing screen must answer three questions.

1. What workflow am I in
2. What step am I currently on
3. What do I need to do next

The software must be local first. Cloud execution can be optional later, but the core product must run without depending on paid services.

## 2. Main User Types

### 2.1 Drone Operator

The drone operator plans missions, checks flight readiness, flies the drone and imports captured data.

Core needs:

1. Clear map based mission planning
2. Safe preflight checks
3. Drone connection status
4. Battery and GPS visibility
5. Mission upload and execution controls
6. Flight logs
7. Dataset import after capture

### 2.2 Inspection Engineer

The inspection engineer validates datasets, runs analysis, reviews defects, annotates findings and prepares reports.

Core needs:

1. Dataset quality checks
2. Image preview
3. 2D inspection outputs
4. 3D reconstruction outputs
5. Defect detection overlays
6. Crack propagation analysis
7. Measurements and annotations
8. Report builder

### 2.3 Developer or Research User

The developer configures model paths, processing engines, experimental algorithms and external integrations.

Core needs:

1. Model manager
2. Processing logs
3. Script diagnostics
4. Module registry
5. Plugin configuration
6. Local environment validation
7. Open source extension points

## 3. Recommended Technology Stack

### 3.1 Desktop Application

Use Python with PySide6.

Reason:

1. The current interface appears to already be a Python desktop app
2. PySide6 has suitable licensing for an open source toolkit
3. Python integrates well with computer vision, photogrammetry, AI models, drone SDKs and reporting
4. Qt gives strong desktop widgets, docking panels, native file dialogs and threaded workers

Primary libraries:

1. PySide6 for desktop UI
2. Qt WebEngine for embedded map and HTML previews
3. SQLite for local project index
4. SQLModel or SQLAlchemy for database models
5. Pydantic for config validation
6. OpenCV for image processing
7. NumPy for numerical operations
8. Pillow for image loading and thumbnails
9. PyTorch for AI models
10. ONNX Runtime for optional fast model inference
11. Open3D for point cloud and mesh viewing
12. trimesh for mesh operations
13. pyproj for coordinate conversions
14. rasterio for geospatial raster handling if needed
15. shapely for geometry
16. reportlab or WeasyPrint for reports
17. Jinja2 for HTML report templates
18. MAVSDK Python for drone control
19. pymavlink for lower level MAVLink support
20. loguru or structlog for clean logging

### 3.2 Map and Geospatial UI

Recommended approach:

1. Use Leaflet inside Qt WebEngine for the mission planner map
2. Use local offline tile support through MBTiles
3. Use a JavaScript bridge between PySide6 and Leaflet
4. Store geometry as GeoJSON
5. Store mission plans as structured JSON
6. Export compatible mission files later if needed

Key technologies:

1. Leaflet for map rendering
2. Leaflet Draw or custom drawing tools for waypoint and polygon editing
3. MBTiles for offline maps
4. GeoJSON for mission geometry
5. pyproj for coordinate transforms
6. shapely for polygon, line and area calculations

### 3.3 Processing and Analysis

Recommended architecture:

1. Each processing capability must be a module
2. Every module must define inputs, outputs, readiness checks and run status
3. Modules must be callable from the UI and from a command line interface
4. Long running tasks must run in worker threads or subprocesses
5. The UI must never freeze during processing

Core processing engines:

1. Dataset validator
2. Image quality checker
3. Defect detection engine
4. Crack propagation engine
5. 3D reconstruction engine
6. Measurement engine
7. Annotation engine
8. Report engine

### 3.4 Database and Project Storage

Use a hybrid structure.

SQLite database stores searchable metadata.

Project folder stores files and artifacts.

Suggested project folder structure:

```text
project_root/
  project.json
  audit_log.jsonl
  missions/
    mission_001.json
    mission_versions/
  datasets/
    dataset_001/
      images/
      masks/
      thumbnails/
      metadata.json
      qa_results.json
  processing/
    run_001/
      run_config.json
      status.json
      logs.txt
      outputs/
  analysis/
    defects/
    crack_growth/
    reconstruction/
    measurements/
    annotations/
  reports/
    report_001/
      report.html
      report.pdf
      assets/
  cache/
    map_tiles/
    temp/
```

### 3.5 Configuration

Use layered configuration.

1. Application defaults
2. User settings
3. Project settings
4. Workflow template settings
5. Mission settings
6. Processing run settings

Use Pydantic models for all config objects.

Example config classes:

```python
class AppSettings(BaseModel):
    theme: str
    default_units: str
    workspace_root: Path
    offline_mode: bool
    log_level: str

class DroneProfile(BaseModel):
    name: str
    model: str
    payload: str
    max_altitude_m: float
    min_altitude_m: float
    rth_altitude_m: float
    cruise_speed_mps: float

class ProcessingSettings(BaseModel):
    use_gpu: bool
    device: str
    model_registry_path: Path
    max_workers: int
```

## 4. Core Product Modules

### 4.1 Project Manager

Purpose:

Manage inspection projects and provide one consistent source of truth for missions, datasets, processing outputs, reports and audit events.

Main functions:

```python
def create_project(name: str, root_dir: Path | None, description: str) -> Project:
    """
    Create a project folder, project metadata file and database entry.
    Validate project name.
    Create required folder structure.
    Write initial audit event.
    Return project object.
    """

def load_project(project_id: str) -> Project:
    """
    Load project metadata from SQLite and project.json.
    Validate that project folder exists.
    Validate folder structure.
    Return project object.
    """

def set_active_project(project_id: str) -> None:
    """
    Set active project in application state.
    Notify UI panels.
    Refresh project dashboard.
    """

def update_project(project_id: str, patch: ProjectPatch) -> Project:
    """
    Update editable project fields.
    Preserve created timestamp.
    Write audit event.
    Return updated project.
    """

def list_projects() -> list[ProjectSummary]:
    """
    Return all known projects with readiness summaries.
    Include mission count, dataset count, report count and sync state.
    """

def archive_project(project_id: str) -> None:
    """
    Mark project as archived.
    Do not delete files.
    Hide from active project list unless archive view is enabled.
    """

def validate_project(project_id: str) -> ProjectReadiness:
    """
    Check folder availability, database consistency, active workflow, mission count, dataset count and report readiness.
    Return a structured readiness object.
    """
```

Implementation notes:

1. Store project metadata in project.json
2. Store project index in SQLite
3. Store audit events in audit_log.jsonl
4. Never silently overwrite an existing project folder
5. Every project changing action must write an audit event

Important UI components:

1. ProjectCard
2. ProjectReadinessPanel
3. AuditTimeline
4. CreateProjectDialog
5. ProjectHealthBanner

### 4.2 Workflow Template Manager

Purpose:

Provide reusable inspection workflows for different asset types.

Required workflow templates:

1. General Inspection
2. 3D Mapping
3. Facade Inspection
4. Roof Inspection
5. Bridge Inspection
6. Solar Inspection
7. Tower Inspection
8. Wind Turbine Inspection
9. Stockpile Measurement
10. Linear Corridor Inspection
11. Custom Workflow

Workflow model:

```python
class WorkflowTemplate(BaseModel):
    id: str
    name: str
    description: str
    asset_type: str
    required_inputs: list[str]
    optional_inputs: list[str]
    mission_modes: list[str]
    processing_stages: list[str]
    report_sections: list[str]
    default_parameters: dict[str, Any]
```

Main functions:

```python
def list_workflow_templates() -> list[WorkflowTemplate]:
    """
    Load bundled workflow templates from app resources.
    Load user defined templates from workspace.
    Return templates sorted by asset type.
    """

def get_workflow_template(template_id: str) -> WorkflowTemplate:
    """
    Load one workflow template by id.
    Raise a clear error if template does not exist.
    """

def assign_workflow_to_project(project_id: str, template_id: str) -> Project:
    """
    Attach workflow template to project.
    Copy default workflow settings into project settings.
    Write audit event.
    """

def create_custom_workflow(template: WorkflowTemplate) -> WorkflowTemplate:
    """
    Validate a custom workflow.
    Save it into the user template directory.
    Return saved template.
    """

def validate_workflow_readiness(project_id: str) -> WorkflowReadiness:
    """
    Check whether the current project has the data required by its selected workflow.
    Return missing inputs, optional improvements and next recommended action.
    """
```

Implementation notes:

1. Store default workflow templates as YAML files
2. Store user templates in workspace config
3. Use a visual workflow card gallery in the UI
4. Each workflow must define required processing stages
5. Each workflow must define report sections

### 4.3 Mission Planner

Purpose:

Create mission plans using map geometry and inspection settings.

Mission concepts:

1. Waypoint mission
2. Polygon survey mission
3. Facade grid mission
4. Orbit mission
5. Linear corridor mission
6. Repeat inspection mission
7. Manual capture mission

Core data models:

```python
class MissionPlan(BaseModel):
    id: str
    project_id: str
    name: str
    workflow_id: str
    geometry: dict
    items: list[MissionItem]
    parameters: MissionParameters
    safety_constraints: SafetyConstraints
    created_at: datetime
    updated_at: datetime
    version: int

class MissionItem(BaseModel):
    id: str
    item_type: str
    coordinates: list[float]
    altitude_m: float
    speed_mps: float
    camera_action: str | None
    metadata: dict[str, Any]

class SafetyConstraints(BaseModel):
    min_altitude_m: float
    max_altitude_m: float
    rth_altitude_m: float
    geofence: dict | None
    no_fly_zones: list[dict]
```

Main functions:

```python
def create_mission(project_id: str, name: str, workflow_id: str) -> MissionPlan:
    """
    Create a new empty mission linked to a project and workflow.
    Save version 1.
    Return mission plan.
    """

def save_mission_version(mission_id: str, note: str) -> MissionVersion:
    """
    Snapshot current mission plan.
    Store version metadata.
    Write project audit event.
    """

def load_mission_version(mission_id: str, version_id: str) -> MissionPlan:
    """
    Load a previous mission version.
    Do not overwrite current mission until user confirms restore.
    """

def restore_mission_version(mission_id: str, version_id: str) -> MissionPlan:
    """
    Restore selected version as the current mission.
    Create a new version after restore.
    """

def add_waypoint(mission_id: str, lat: float, lon: float, altitude_m: float) -> MissionPlan:
    """
    Add a waypoint to the mission.
    Recalculate mission metrics.
    Validate safety constraints.
    """

def add_polygon_survey(mission_id: str, polygon_geojson: dict, parameters: SurveyParameters) -> MissionPlan:
    """
    Add a polygon survey area.
    Generate coverage path.
    Recalculate image count, distance, duration and battery estimate.
    """

def add_linear_corridor(mission_id: str, line_geojson: dict, width_m: float, parameters: CorridorParameters) -> MissionPlan:
    """
    Generate a corridor inspection path around a line.
    Useful for roads, pipelines, bridges and rails.
    """

def generate_facade_grid(mission_id: str, facade_line: dict, height_m: float, distance_m: float, overlap: float) -> MissionPlan:
    """
    Generate a vertical or stepped facade inspection mission.
    Store camera angle and standoff distance.
    """

def calculate_mission_metrics(mission: MissionPlan) -> MissionMetrics:
    """
    Calculate total distance, estimated duration, image count, area coverage, battery usage and warnings.
    """

def validate_mission(mission: MissionPlan, drone_profile: DroneProfile) -> MissionValidationResult:
    """
    Validate altitude limits, RTH altitude, geofence, no fly zones, path continuity, battery estimate and camera actions.
    """

def export_mission(mission_id: str, export_format: str, output_path: Path) -> Path:
    """
    Export mission to supported external format.
    Supported initial formats can include JSON, GeoJSON and CSV.
    MAVLink compatible export can be added later.
    """
```

Mission planner UI:

1. Large map canvas
2. Left floating tool rail
3. Right inspector panel
4. Bottom mission metrics bar
5. Mission validation drawer
6. Mission version history drawer

Implementation notes:

1. Store geometries as GeoJSON
2. Keep all map edits undoable
3. Use JavaScript bridge to pass map edits to Python
4. Validate mission after every major edit
5. Never allow hidden invalid flight settings

### 4.4 Preflight Manager

Purpose:

Verify that mission, drone, battery, GPS, camera and safety settings are ready before flight.

Preflight checks:

1. Drone connected
2. GPS or RTK lock available
3. Home position set
4. Battery healthy
5. Camera ready
6. Storage ready
7. Mission valid
8. Mission uploaded
9. Geofence valid
10. No fly zone validation complete
11. RTH altitude valid
12. Max altitude valid
13. Weather manually acknowledged if weather module is not available
14. Obstacle avoidance profile selected
15. Operator confirmation complete

Main functions:

```python
def run_preflight(project_id: str, mission_id: str, drone_id: str) -> PreflightReport:
    """
    Run all automatic preflight checks.
    Combine automatic checks with pending manual confirmations.
    Return a report with blocking issues, warnings and passed checks.
    """

def check_drone_connection(drone: DroneClient) -> CheckResult:
    """
    Verify drone heartbeat and telemetry stream.
    """

def check_gps_status(telemetry: DroneTelemetry) -> CheckResult:
    """
    Verify GPS fix type, satellite count and horizontal accuracy.
    """

def check_battery_status(telemetry: DroneTelemetry, mission_metrics: MissionMetrics) -> CheckResult:
    """
    Verify battery percentage and estimated mission reserve.
    """

def check_rth_altitude(safety: SafetyConstraints, drone_profile: DroneProfile) -> CheckResult:
    """
    Block if RTH altitude conflicts with maximum altitude unless explicit override is enabled.
    """

def check_geofence(mission: MissionPlan, geofence: dict) -> CheckResult:
    """
    Confirm mission path is inside allowed geofence.
    """

def confirm_manual_check(check_id: str, operator_note: str) -> None:
    """
    Save manual confirmation for checks that require human review.
    Write audit event.
    """

def can_start_mission(preflight_report: PreflightReport) -> bool:
    """
    Return true only when all blocking checks pass.
    """
```

UI behavior:

1. Start Mission disabled until blocking checks pass
2. Failed checks show Fix action
3. Warnings must be amber
4. Blocking safety issues must be red
5. Manual confirmations must be clearly marked as manual
6. Automatic checks must show live status

### 4.5 Drone Connection Layer

Purpose:

Abstract drone communication so the UI does not depend directly on one SDK.

Recommended initial support:

1. MAVSDK for PX4 and ArduPilot style vehicles
2. pymavlink for lower level MAVLink control and diagnostics
3. MockDroneClient for offline testing

Core interface:

```python
class DroneClient(Protocol):
    def connect(self, connection_uri: str) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def get_telemetry(self) -> DroneTelemetry: ...
    def upload_mission(self, mission: MissionPlan) -> None: ...
    def start_mission(self) -> None: ...
    def pause_mission(self) -> None: ...
    def resume_mission(self) -> None: ...
    def return_to_home(self) -> None: ...
    def abort_mission(self) -> None: ...
```

Main functions:

```python
def create_drone_client(driver: str) -> DroneClient:
    """
    Create a drone client based on selected driver.
    Supported initial drivers are mavsdk, pymavlink and mock.
    """

def connect_drone(connection_uri: str, driver: str) -> DroneConnectionState:
    """
    Connect to drone.
    Start telemetry polling.
    Update app state.
    """

def poll_telemetry(drone: DroneClient) -> DroneTelemetry:
    """
    Read latest telemetry.
    Publish telemetry update to the UI event bus.
    """

def upload_active_mission(project_id: str, mission_id: str, drone: DroneClient) -> UploadResult:
    """
    Convert mission plan into drone mission format.
    Upload it to the drone.
    Return upload status.
    """

def execute_command(command: FlightCommand, drone: DroneClient) -> CommandResult:
    """
    Execute a flight command with safety checks.
    Commands include start, pause, resume, RTH and abort.
    """
```

Implementation notes:

1. Use a worker thread for drone telemetry
2. Never block UI on drone communication
3. All flight commands must pass through one command gateway
4. Every command must be logged
5. Provide mock mode for development without drone hardware

### 4.6 Live Flight Manager

Purpose:

Monitor and control active mission execution.

Main functions:

```python
def start_mission_execution(project_id: str, mission_id: str, drone_id: str) -> FlightSession:
    """
    Verify preflight report.
    Upload mission if required.
    Start mission.
    Create flight session record.
    """

def pause_flight(session_id: str) -> CommandResult:
    """
    Pause active mission.
    Log command.
    Update flight state.
    """

def resume_flight(session_id: str) -> CommandResult:
    """
    Resume paused mission.
    Log command.
    Update flight state.
    """

def trigger_rth(session_id: str, reason: str) -> CommandResult:
    """
    Command return to home.
    Require confirmation if not emergency.
    Log reason.
    """

def abort_flight(session_id: str, reason: str) -> CommandResult:
    """
    Abort mission.
    Use strongest safety handling supported by drone client.
    Log reason.
    """

def update_flight_state(session_id: str, telemetry: DroneTelemetry) -> FlightState:
    """
    Update current waypoint, progress, battery, position and warning state.
    """

def record_flight_log(session_id: str, telemetry: DroneTelemetry) -> None:
    """
    Append telemetry to flight log.
    Store time, coordinates, altitude, speed, battery and mode.
    """
```

Live flight UI:

1. Large map or video view
2. Active drone marker
3. Mission path
4. Home point
5. RTH path
6. Geofence
7. No fly zones
8. Current waypoint
9. Telemetry cards
10. Command log
11. Emergency controls

Safety rules:

1. Abort is red and visually separated
2. RTH is safety styled
3. Pause only enabled during active mission
4. Resume only enabled during paused mission
5. Start mission only visible before execution
6. Do not show inactive commands as normal blue buttons

### 4.7 Data Library

Purpose:

Import, manage, validate and preview datasets.

Dataset types:

1. RGB image dataset
2. Thermal image dataset
3. Mask dataset
4. Video dataset
5. Metadata file bundle
6. Calibration profile
7. Reconstruction output
8. Analysis artifact

Dataset model:

```python
class Dataset(BaseModel):
    id: str
    project_id: str
    name: str
    dataset_type: str
    root_dir: Path
    image_count: int
    has_gps_metadata: bool
    has_camera_metadata: bool
    linked_mission_id: str | None
    qa_status: str
    created_at: datetime

class ImageAsset(BaseModel):
    id: str
    dataset_id: str
    file_path: Path
    thumbnail_path: Path | None
    width: int
    height: int
    gps_lat: float | None
    gps_lon: float | None
    captured_at: datetime | None
    qa_flags: list[str]
```

Main functions:

```python
def import_image_dataset(project_id: str, folder_path: Path, dataset_name: str | None) -> Dataset:
    """
    Scan folder.
    Register images.
    Extract metadata.
    Generate thumbnails.
    Run initial validation.
    Save dataset record.
    """

def scan_dataset_folder(folder_path: Path) -> list[Path]:
    """
    Find supported image files.
    Ignore hidden files and unsupported formats.
    Return sorted file list.
    """

def extract_image_metadata(image_path: Path) -> ImageMetadata:
    """
    Extract dimensions, EXIF, GPS, timestamp and camera details where available.
    """

def generate_thumbnail(image_path: Path, output_dir: Path, size: tuple[int, int]) -> Path:
    """
    Create thumbnail for fast UI browsing.
    """

def validate_dataset(dataset_id: str) -> DatasetValidationReport:
    """
    Check image count, corrupt files, metadata coverage, duplicate images, blurry images and missing GPS.
    """

def link_dataset_to_mission(dataset_id: str, mission_id: str) -> None:
    """
    Associate dataset with mission.
    Enable mission based filtering and reporting.
    """

def list_datasets(project_id: str, filters: DatasetFilters) -> list[DatasetSummary]:
    """
    Return datasets matching mission, date and QA tag filters.
    """

def get_image_assets(dataset_id: str, page: int, page_size: int) -> list[ImageAsset]:
    """
    Return paginated image assets for thumbnail grid.
    """

def tag_image_asset(image_id: str, tag: str) -> None:
    """
    Add QA tag or inspection tag to image.
    """
```

Quality checks:

```python
def check_blur(image_path: Path) -> QualityMetric:
    """
    Use variance of Laplacian or better focus metric.
    Return blur score and pass or fail state.
    """

def check_exposure(image_path: Path) -> QualityMetric:
    """
    Estimate overexposure and underexposure.
    """

def check_duplicate_images(image_paths: list[Path]) -> list[DuplicateGroup]:
    """
    Use perceptual hashing to find near duplicate images.
    """

def check_metadata_coverage(dataset_id: str) -> MetadataCoverage:
    """
    Calculate percentage of images with GPS, timestamp and camera metadata.
    """
```

Implementation notes:

1. Use Pillow for EXIF
2. Use OpenCV for quality metrics
3. Use imagehash for duplicate detection
4. Store thumbnails in dataset thumbnails folder
5. Use pagination to keep UI fast

### 4.8 Processing Pipeline Manager

Purpose:

Run staged processing jobs and show transparent progress.

Pipeline stages:

1. Dataset validation
2. 2D map preparation
3. Image alignment
4. 3D reconstruction
5. Defect detection
6. Crack propagation
7. Measurement extraction
8. Risk scoring
9. Report asset generation

Processing model:

```python
class PipelineStage(BaseModel):
    id: str
    name: str
    status: str
    required_inputs: list[str]
    output_artifacts: list[str]
    progress_percent: float
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None

class ProcessingRun(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    workflow_id: str
    stages: list[PipelineStage]
    status: str
    output_dir: Path
```

Main functions:

```python
def create_processing_run(project_id: str, dataset_id: str, workflow_id: str) -> ProcessingRun:
    """
    Create a new processing run folder.
    Build stages from workflow template.
    Save run configuration.
    """

def validate_pipeline_inputs(run_id: str) -> PipelineReadiness:
    """
    Verify required dataset, model files, output paths and processing settings.
    """

def run_pipeline(run_id: str, selected_stages: list[str] | None) -> ProcessingRun:
    """
    Execute selected pipeline stages in order.
    Update status after each stage.
    Stop on blocking failure.
    """

def run_pipeline_stage(run_id: str, stage_id: str) -> StageResult:
    """
    Run one stage.
    Save logs and output artifact metadata.
    """

def stop_processing_run(run_id: str) -> None:
    """
    Request cancellation.
    Allow current subprocess or worker to stop safely.
    """

def get_processing_status(run_id: str) -> ProcessingStatus:
    """
    Return current progress, active stage, logs and artifacts.
    """

def list_processing_runs(project_id: str) -> list[ProcessingRunSummary]:
    """
    Return previous processing runs for project history.
    """
```

Implementation notes:

1. Use QThreadPool for lightweight Python workers
2. Use subprocess for heavy external tools
3. Use JSON status files so failed runs can be inspected later
4. Use structured logs
5. The UI must poll or subscribe to progress updates
6. Run cancellation must be supported where possible

### 4.9 Defect Detection Engine

Purpose:

Detect visual defects using classical algorithms and optional AI models.

Detection modes:

1. Classical metal defect detection
2. Classical crack detection
3. AI model detection
4. Hybrid detection

Defect model:

```python
class DefectDetectionConfig(BaseModel):
    mode: str
    model_key: str | None
    threshold: float
    min_area_px: int
    output_masks: bool
    output_overlay: bool

class DetectedDefect(BaseModel):
    id: str
    image_id: str
    defect_type: str
    confidence: float
    severity: str
    bbox: list[int]
    mask_path: Path | None
    area_px: int
    notes: str | None
```

Main functions:

```python
def run_defect_detection(dataset_id: str, config: DefectDetectionConfig) -> DefectDetectionResult:
    """
    Run selected detection mode over dataset or selected images.
    Save masks, overlays and defect records.
    """

def detect_defects_classical(image_path: Path, config: DefectDetectionConfig) -> list[DetectedDefect]:
    """
    Run thresholding, edge detection, morphology and connected components.
    Suitable for simple visible defects.
    """

def detect_defects_ai(image_path: Path, model: DefectModel, config: DefectDetectionConfig) -> list[DetectedDefect]:
    """
    Run AI model inference.
    Convert predictions into standard defect records.
    """

def create_defect_overlay(image_path: Path, defects: list[DetectedDefect], output_path: Path) -> Path:
    """
    Render defect masks, bounding boxes and severity colors.
    """

def classify_defect_severity(defect: DetectedDefect, rules: SeverityRules) -> str:
    """
    Assign severity based on area, confidence, type and workflow rules.
    """

def export_defect_table(result_id: str, output_format: str) -> Path:
    """
    Export detected defects to CSV, JSON or report ready HTML.
    """
```

Recommended AI model handling:

1. Use PyTorch for research models
2. Use ONNX Runtime for deployment models
3. Store model registry in a model_registry.json file
4. Allow model path configuration in Developer Tools
5. Show model readiness in operator UI

Model registry functions:

```python
def register_model(model_key: str, model_path: Path, model_type: str, labels: list[str]) -> ModelRecord:
    """
    Register model in local model registry.
    Validate file exists.
    Save model metadata.
    """

def load_model(model_key: str) -> DefectModel:
    """
    Load model from registry.
    Cache loaded model where appropriate.
    """

def validate_model_registry() -> ModelRegistryStatus:
    """
    Check that configured model files exist and can be loaded.
    """
```

### 4.10 Crack Propagation Engine

Purpose:

Estimate crack growth over time from image or mask based inputs.

This module must be presented as engineering analysis, not as guaranteed physical truth. The UI must show assumptions and parameter limits.

Inputs:

1. Original image
2. Crack mask
3. Pixel size in mm per pixel
4. Material profile
5. Stress range
6. Time horizon
7. Propagation steps
8. Paris law parameters
9. Fracture toughness value

Data model:

```python
class CrackPropagationConfig(BaseModel):
    image_path: Path
    mask_path: Path | None
    pixel_size_mm_per_px: float
    sigma_nominal_mpa: float
    delta_sigma_mpa: float
    cycles_per_year: float
    horizon_years: float
    steps: int
    kic_mpa_sqrt_m: float | None
    paris_c: float | None
    paris_m: float | None
    material_profile: str | None

class CrackPropagationResult(BaseModel):
    id: str
    config: CrackPropagationConfig
    forecast_images: list[Path]
    risk_level: str
    critical_points: list[dict]
    summary: str
    assumptions: list[str]
```

Main functions:

```python
def run_crack_propagation(config: CrackPropagationConfig) -> CrackPropagationResult:
    """
    Validate inputs.
    Extract crack geometry.
    Estimate growth over configured steps.
    Save overlays and summary.
    """

def extract_crack_geometry(mask_path: Path, pixel_size_mm_per_px: float) -> CrackGeometry:
    """
    Skeletonize crack mask.
    Estimate crack length, endpoints, branches and orientation.
    """

def estimate_stress_intensity(crack_geometry: CrackGeometry, sigma_mpa: float) -> StressIntensityResult:
    """
    Estimate stress intensity using configured simplified model.
    Clearly label approximation.
    """

def calculate_crack_growth_step(current_length_m: float, delta_k: float, paris_c: float, paris_m: float, cycles: float) -> float:
    """
    Calculate one crack growth increment.
    """

def generate_growth_overlay(image_path: Path, crack_states: list[CrackState], output_path: Path) -> Path:
    """
    Render predicted crack states on image.
    """

def classify_crack_risk(result: CrackPropagationResult) -> str:
    """
    Assign Low, Medium, High or Critical risk based on growth rate, KIC margin and configured thresholds.
    """
```

Implementation notes:

1. Use OpenCV and scikit image for mask operations
2. Use NumPy for numerical calculations
3. Keep default material profiles configurable
4. Always show assumptions
5. Keep advanced parameters collapsed by default
6. Do not hide when defaults are used

UI sections:

1. Input image preview
2. Mask overlay
3. Material and stress profile
4. Time horizon
5. Advanced parameters
6. Predicted growth overlay
7. Critical points list
8. Risk summary
9. Assumptions panel

### 4.11 3D Reconstruction Engine

Purpose:

Create or import 3D reconstruction outputs and display defects in spatial context.

Initial approach:

1. Support imported reconstruction folders first
2. Add OpenSfM or COLMAP integration later as optional modules
3. Use Open3D for local point cloud and mesh handling
4. Use a custom viewer panel or embedded visualization window

Data model:

```python
class ReconstructionConfig(BaseModel):
    image_folder: Path
    mask_folder: Path | None
    output_folder: Path
    profile: str
    execution_mode: str
    reuse_cache: bool

class ReconstructionResult(BaseModel):
    id: str
    output_folder: Path
    point_cloud_path: Path | None
    mesh_path: Path | None
    camera_poses_path: Path | None
    defect_projection_path: Path | None
    quality_metrics: dict[str, Any]
```

Main functions:

```python
def run_reconstruction(config: ReconstructionConfig) -> ReconstructionResult:
    """
    Validate image folder.
    Select reconstruction backend.
    Run backend.
    Register outputs.
    """

def import_reconstruction_folder(project_id: str, folder_path: Path) -> ReconstructionResult:
    """
    Scan folder for point clouds, meshes, camera poses and metadata.
    Register as reconstruction output.
    """

def load_point_cloud(path: Path) -> PointCloudData:
    """
    Load point cloud using Open3D.
    Return points, colors and metadata.
    """

def load_mesh(path: Path) -> MeshData:
    """
    Load mesh using Open3D or trimesh.
    Return vertices, faces and metadata.
    """

def project_defects_to_3d(defect_result_id: str, reconstruction_id: str) -> DefectProjectionResult:
    """
    Link 2D defects to 3D points when camera pose data is available.
    """

def calculate_reconstruction_quality(result_id: str) -> ReconstructionQuality:
    """
    Calculate image count, sparse point count, dense point count, coverage and failed image count where data exists.
    """
```

UI requirements:

1. 3D viewer must use a dark background
2. Viewer must fill the main area
3. Defect overlay toggle must be visible
4. Selected point inspector must show linked evidence image
5. Critical points table must be visible below
6. Empty state must explain how to import or run reconstruction

### 4.12 Measurement Engine

Purpose:

Support inspection measurements on maps, images and 3D outputs.

Measurement types:

1. Distance
2. Area
3. Crack length
4. Defect area
5. Volume
6. Height
7. Angle
8. Coordinate point
9. Stockpile volume later

Data model:

```python
class Measurement(BaseModel):
    id: str
    project_id: str
    source_type: str
    source_id: str
    measurement_type: str
    geometry: dict
    value: float
    unit: str
    confidence: float | None
    created_by: str | None
    created_at: datetime
```

Main functions:

```python
def create_measurement(source_type: str, source_id: str, measurement_type: str, geometry: dict) -> Measurement:
    """
    Calculate measurement value based on source and geometry.
    Store measurement record.
    """

def calculate_image_distance(points_px: list[tuple[int, int]], scale_mm_per_px: float) -> float:
    """
    Calculate real distance from image points.
    """

def calculate_polygon_area(geometry: dict, coordinate_system: str) -> float:
    """
    Calculate area for map or image based geometry.
    """

def calculate_crack_length(mask_path: Path, scale_mm_per_px: float) -> float:
    """
    Estimate crack length using skeletonized mask.
    """

def export_measurements(project_id: str, output_format: str) -> Path:
    """
    Export measurements to CSV, JSON or report section.
    """
```

UI:

1. Measurement toolbar
2. Measurement list
3. Selected measurement inspector
4. Unit display
5. Delete and edit actions
6. Report inclusion toggle

### 4.13 Annotation Engine

Purpose:

Allow users to mark inspection findings manually.

Annotation types:

1. Point annotation
2. Rectangle annotation
3. Polygon annotation
4. Free text note
5. Severity tag
6. Defect confirmation
7. False positive mark
8. Report highlight

Data model:

```python
class Annotation(BaseModel):
    id: str
    project_id: str
    source_type: str
    source_id: str
    geometry: dict
    label: str
    severity: str | None
    note: str | None
    include_in_report: bool
    created_at: datetime
```

Main functions:

```python
def create_annotation(source_type: str, source_id: str, geometry: dict, label: str, note: str | None) -> Annotation:
    """
    Create annotation on image, map or 3D view.
    """

def update_annotation(annotation_id: str, patch: AnnotationPatch) -> Annotation:
    """
    Edit label, note, severity or report inclusion.
    """

def delete_annotation(annotation_id: str) -> None:
    """
    Delete annotation and write audit event.
    """

def list_annotations(project_id: str, source_id: str | None) -> list[Annotation]:
    """
    Return annotations for source or full project.
    """

def convert_defect_to_annotation(defect_id: str) -> Annotation:
    """
    Save detected defect as user reviewed annotation.
    """
```

### 4.14 Report Engine

Purpose:

Generate professional inspection reports from project data.

Report types:

1. Standard inspection report
2. Detailed engineering report
3. Executive summary
4. Defect only report
5. Dataset quality report
6. Mission summary report

Report sections:

1. Overview
2. Site information
3. Mission map
4. Dataset summary
5. Key photos
6. Defect summary
7. Crack propagation forecast
8. 3D reconstruction
9. Measurements
10. Annotations
11. Recommendations
12. Appendix
13. Audit trail

Data model:

```python
class ReportConfig(BaseModel):
    project_id: str
    title: str
    report_type: str
    sections: list[str]
    include_images: bool
    include_measurements: bool
    include_defects: bool
    include_audit_trail: bool

class ReportResult(BaseModel):
    id: str
    project_id: str
    title: str
    html_path: Path
    pdf_path: Path | None
    created_at: datetime
```

Main functions:

```python
def validate_report_readiness(config: ReportConfig) -> ReportReadiness:
    """
    Check whether required project, mission, dataset and analysis data exists.
    Return missing sections and warnings.
    """

def build_report_context(config: ReportConfig) -> dict:
    """
    Collect project metadata, mission metrics, dataset summaries, defects, measurements, annotations and images.
    """

def render_report_html(context: dict, template_path: Path, output_path: Path) -> Path:
    """
    Render report HTML using Jinja2 template.
    """

def export_report_pdf(html_path: Path, output_path: Path) -> Path:
    """
    Convert HTML to PDF using WeasyPrint or another local renderer.
    """

def generate_report(config: ReportConfig) -> ReportResult:
    """
    Validate readiness.
    Build context.
    Render HTML.
    Export PDF if enabled.
    Register report in project database.
    Write audit event.
    """

def list_reports(project_id: str) -> list[ReportResult]:
    """
    Return generated reports.
    """
```

Implementation notes:

1. Use HTML as primary report format
2. Generate PDF from HTML
3. Store report assets in report folder
4. Allow custom templates later
5. Show readiness checklist before generation
6. Disable report generation when minimum data is missing

### 4.15 Settings and Diagnostics

Purpose:

Keep operational settings clean and developer settings separated.

Settings groups:

1. Drone profile
2. Flight safety
3. Team workflow
4. Offline planning
5. File portability
6. Processing engines
7. Model registry
8. Developer diagnostics

Main functions:

```python
def load_app_settings() -> AppSettings:
    """
    Load user settings from config file.
    Apply defaults for missing values.
    """

def save_app_settings(settings: AppSettings) -> None:
    """
    Validate and save settings.
    Notify UI.
    """

def validate_safety_settings(drone_profile: DroneProfile) -> list[ValidationMessage]:
    """
    Check min altitude, max altitude, RTH altitude and speed limits.
    """

def configure_model_path(model_key: str, path: Path) -> ModelRecord:
    """
    Register or update a model path.
    Validate file exists.
    """

def run_system_diagnostics() -> DiagnosticReport:
    """
    Check Python environment, model files, processing scripts, writable folders and optional tools.
    """

def export_diagnostic_bundle(output_path: Path) -> Path:
    """
    Export logs, settings summary and diagnostic results for debugging.
    """
```

UI rule:

1. Normal Settings must not show raw missing scripts by default
2. Developer Tools can show missing scripts and model paths
3. Every missing dependency must have explanation and fix action

## 5. UI Architecture

### 5.1 Main Window Structure

Use a consistent shell.

Components:

1. TopAppBar
2. LeftSidebar
3. MainWorkspace
4. RightInspectorPanel
5. BottomStatusBar
6. NotificationCenter
7. ModalManager

Class skeleton:

```python
class MainWindow(QMainWindow):
    """
    Main application shell.
    Owns navigation, active project state and shared services.
    """

class AppState(QObject):
    """
    Central observable state.
    Stores active project, active mission, drone connection, current workflow and processing status.
    """

class NavigationController(QObject):
    """
    Switches pages.
    Maintains navigation history.
    Prevents invalid navigation where needed.
    """

class EventBus(QObject):
    """
    Publishes app wide events.
    Examples include project changed, dataset imported, telemetry updated and processing status changed.
    """
```

### 5.2 Required Reusable UI Components

#### StatusChip

Purpose:

Show small state labels such as Ready, Missing, Failed, Online, Offline, Running and Complete.

Properties:

```python
text: str
state: Literal["neutral", "ready", "warning", "error", "running", "disabled"]
icon: str | None
```

#### ReadinessCard

Purpose:

Show whether one part of the workflow is ready.

Fields:

```python
title: str
status: str
summary: str
missing_items: list[str]
primary_action: QAction | None
```

#### EmptyState

Purpose:

Replace useless blank panels.

Fields:

```python
title: str
message: str
primary_action_text: str | None
secondary_help: str | None
```

#### ActionButton

Purpose:

Standard action button with consistent styling.

Fields:

```python
text: str
action_type: Literal["primary", "secondary", "danger", "warning", "ghost"]
enabled_reason: str | None
```

#### PathInput

Purpose:

File or folder input with middle truncation, validation and browse button.

Fields:

```python
label: str
path: Path | None
mode: Literal["file", "folder"]
required: bool
validation_state: str
```

#### PipelineStageCard

Purpose:

Display each processing stage.

Fields:

```python
stage_name: str
status: str
progress: float
required_inputs: list[str]
outputs: list[str]
error_message: str | None
```

#### TelemetryCard

Purpose:

Display flight telemetry.

Fields:

```python
label: str
value: str
unit: str | None
state: str
```

#### InspectorPanel

Purpose:

Display selected map item, image, defect, measurement or reconstruction point.

Fields:

```python
source_type: str
source_id: str
properties: dict
actions: list[QAction]
```

### 5.3 Styling Rules

Use a design token file.

Example:

```python
class ThemeTokens:
    color_bg = "#0b1220"
    color_surface = "#121c2e"
    color_surface_raised = "#18243a"
    color_border = "#2b3d5c"
    color_text = "#e8eefc"
    color_text_muted = "#95a3b8"
    color_primary = "#3b82f6"
    color_success = "#22c55e"
    color_warning = "#f59e0b"
    color_danger = "#ef4444"
```

Rules:

1. Do not use the same blue for every action
2. Do not use borders around every object
3. Use spacing to separate sections
4. Use danger styling for Abort
5. Use warning styling for RTH and safety warnings
6. Use green only for confirmed ready states
7. Use gray for disabled or unavailable states

## 6. Page Specifications

### 6.1 Dashboard

Purpose:

Give a quick view of active work, open issues and available workflows.

Sections:

1. Active project card
2. Start new workflow
3. Recent projects
4. System readiness
5. Recent reports
6. Open issues

Functions used:

```python
list_projects()
list_workflow_templates()
validate_project()
run_system_diagnostics()
list_reports()
```

UI actions:

1. Create project
2. Open project
3. Start workflow
4. Fix system issue
5. Open recent report

### 6.2 Projects Page

Sections:

1. Project list
2. Project overview
3. Readiness cards
4. Recent activity
5. Project actions

Functions used:

```python
create_project()
load_project()
set_active_project()
update_project()
list_projects()
validate_project()
```

Implementation details:

1. Project cards must not use horizontal scroll
2. Audit trail must be a timeline
3. Empty project must show next recommended action

### 6.3 Workflow Templates Page

Sections:

1. Template gallery
2. Template details
3. Required inputs
4. Output examples
5. Start workflow button

Functions used:

```python
list_workflow_templates()
get_workflow_template()
assign_workflow_to_project()
create_custom_workflow()
```

Implementation details:

1. Each template must show required data
2. Each template must show generated outputs
3. Custom workflow creation can be added after core templates

### 6.4 Mission Planner Page

Sections:

1. Map canvas
2. Tool rail
3. Mission inspector
4. Mission metrics
5. Validation drawer
6. Version history

Functions used:

```python
create_mission()
save_mission_version()
add_waypoint()
add_polygon_survey()
add_linear_corridor()
generate_facade_grid()
calculate_mission_metrics()
validate_mission()
export_mission()
```

Implementation details:

1. Use Leaflet in Qt WebEngine
2. Use GeoJSON for all map shapes
3. Use JavaScript bridge for map events
4. Mission metrics update after every edit
5. Invalid mission state must be visible immediately

### 6.5 Preflight Page

Sections:

1. Mission summary
2. Drone status
3. Preflight checklist
4. Risk summary
5. Start mission action

Functions used:

```python
run_preflight()
check_drone_connection()
check_gps_status()
check_battery_status()
check_rth_altitude()
check_geofence()
confirm_manual_check()
can_start_mission()
```

Implementation details:

1. Automatic checks update live
2. Manual checks need operator note or confirmation
3. Start Mission remains disabled until blocking checks pass
4. RTH altitude conflict must be blocking

### 6.6 Live Flight Page

Sections:

1. Flight mode bar
2. Map or video view
3. Telemetry panel
4. Flight controls
5. Command log
6. Battery swap and resume panel

Functions used:

```python
start_mission_execution()
pause_flight()
resume_flight()
trigger_rth()
abort_flight()
update_flight_state()
record_flight_log()
```

Implementation details:

1. Drone communication runs in worker thread
2. Telemetry updates must not freeze the UI
3. Abort must require confirmation unless configured as immediate emergency action
4. Every command must be logged

### 6.7 Data Library Page

Sections:

1. Dataset import toolbar
2. Dataset tree or cards
3. Thumbnail grid
4. Image preview
5. Metadata inspector
6. QA summary

Functions used:

```python
import_image_dataset()
scan_dataset_folder()
extract_image_metadata()
generate_thumbnail()
validate_dataset()
link_dataset_to_mission()
list_datasets()
get_image_assets()
tag_image_asset()
```

Implementation details:

1. Use thumbnail caching
2. Use pagination for large datasets
3. Show metadata coverage
4. Show missing GPS warnings
5. Show dataset readiness for processing

### 6.8 Processing Workbench Page

Sections:

1. Pipeline stage cards
2. Run controls
3. Processing status
4. Logs drawer
5. Output artifacts
6. System readiness

Functions used:

```python
create_processing_run()
validate_pipeline_inputs()
run_pipeline()
run_pipeline_stage()
stop_processing_run()
get_processing_status()
list_processing_runs()
```

Implementation details:

1. Each stage shows ready, running, complete or failed
2. Missing model files appear as setup issues
3. Logs are available but not the main UI
4. Processing can be stopped safely

### 6.9 Defect Detection Page

Sections:

1. Input selector
2. Detection mode
3. Model status
4. Threshold settings
5. Image preview
6. Overlay preview
7. Defect table
8. Export actions

Functions used:

```python
run_defect_detection()
detect_defects_classical()
detect_defects_ai()
create_defect_overlay()
classify_defect_severity()
export_defect_table()
register_model()
load_model()
validate_model_registry()
```

Implementation details:

1. Model based detection disabled until model is configured
2. Classical mode explains limitations
3. Results include confidence and severity
4. Defects can be sent to report

### 6.10 Crack Propagation Page

Sections:

1. Input image
2. Mask selector
3. Material profile
4. Stress profile
5. Time horizon
6. Advanced parameters
7. Growth overlay
8. Critical points
9. Risk summary

Functions used:

```python
run_crack_propagation()
extract_crack_geometry()
estimate_stress_intensity()
calculate_crack_growth_step()
generate_growth_overlay()
classify_crack_risk()
```

Implementation details:

1. Advanced parameters are collapsed by default
2. Scientific assumptions are always visible
3. Defaults are explicitly shown
4. Bad input values show validation errors

### 6.11 3D Reconstruction Page

Sections:

1. Setup drawer
2. Large 3D viewer
3. Defect overlay controls
4. Selected point inspector
5. Evidence image preview
6. Critical points table
7. Reconstruction logs

Functions used:

```python
run_reconstruction()
import_reconstruction_folder()
load_point_cloud()
load_mesh()
project_defects_to_3d()
calculate_reconstruction_quality()
```

Implementation details:

1. Viewer must not have white background
2. Import reconstruction support should come before full reconstruction backend
3. Use Open3D for point cloud and mesh loading
4. Show quality metrics when available

### 6.12 Measurements Page

Sections:

1. Source selector
2. Measurement toolbar
3. Main image or map view
4. Measurement list
5. Inspector
6. Export actions

Functions used:

```python
create_measurement()
calculate_image_distance()
calculate_polygon_area()
calculate_crack_length()
export_measurements()
```

Implementation details:

1. Measurements must store source id
2. Measurements must include units
3. Report inclusion can be toggled

### 6.13 Reports Page

Sections:

1. Report type selector
2. Report readiness checklist
3. Section selector
4. Live preview
5. Generated report list
6. Export actions

Functions used:

```python
validate_report_readiness()
build_report_context()
render_report_html()
export_report_pdf()
generate_report()
list_reports()
```

Implementation details:

1. Report generation disabled when missing data
2. Readiness explains missing items
3. HTML report generated
4. PDF report generated
5. Report appears in history

### 6.14 Settings Page

Sections:

1. Drone profile
2. Flight safety
3. Team workflow
4. Offline planning
5. File portability
6. Processing engines
7. Developer diagnostics

Functions used:

```python
load_app_settings()
save_app_settings()
validate_safety_settings()
configure_model_path()
run_system_diagnostics()
export_diagnostic_bundle()
```

Implementation details:

1. Save button enabled only when settings change
2. Safety validation is immediate
3. Developer diagnostics are collapsed by default
4. Missing scripts and model files are shown as fixable issues

## 7. State Management

Use a central AppState object.

Required state fields:

```python
class AppStateModel(BaseModel):
    active_project_id: str | None
    active_workflow_id: str | None
    active_mission_id: str | None
    active_dataset_id: str | None
    active_processing_run_id: str | None
    drone_connection_state: str
    flight_state: str
    offline_mode: bool
    sync_state: str
    model_registry_state: str
```

State rules:

1. Run Mission disabled unless mission exists, drone connected and preflight passed
2. Generate Report disabled unless project, dataset and analysis result exist
3. Run Full Pipeline disabled unless dataset exists and output folder is writable
4. Run Crack Propagation disabled unless image or mask is valid
5. Run Reconstruction disabled unless image folder and output folder are valid
6. RTH enabled only during active or connected flight state
7. Abort enabled only during active mission or emergency state
8. Report preview visible only when report context can be built

## 8. Event System

Use Qt signals or a simple event bus.

Events:

```text
PROJECT_CREATED
PROJECT_CHANGED
WORKFLOW_ASSIGNED
MISSION_CREATED
MISSION_UPDATED
MISSION_VALIDATED
PREFLIGHT_UPDATED
DRONE_CONNECTED
DRONE_DISCONNECTED
TELEMETRY_UPDATED
FLIGHT_STARTED
FLIGHT_PAUSED
FLIGHT_RESUMED
FLIGHT_ABORTED
DATASET_IMPORTED
DATASET_VALIDATED
PROCESSING_STARTED
PROCESSING_PROGRESS
PROCESSING_COMPLETED
PROCESSING_FAILED
DEFECTS_DETECTED
REPORT_GENERATED
SETTINGS_CHANGED
DIAGNOSTICS_UPDATED
```

Event bus functions:

```python
def publish_event(event_type: str, payload: dict) -> None:
    """
    Publish event to all subscribers.
    """

def subscribe_event(event_type: str, callback: Callable) -> None:
    """
    Register callback for event type.
    """

def unsubscribe_event(event_type: str, callback: Callable) -> None:
    """
    Remove callback.
    """
```

## 9. Validation System

Every user action must validate required inputs.

Validation object:

```python
class ValidationMessage(BaseModel):
    field: str | None
    severity: str
    message: str
    fix_action: str | None
```

Common validation functions:

```python
def validate_path_exists(path: Path, expected_type: str) -> ValidationMessage | None:
    """
    Check file or folder exists.
    """

def validate_numeric_range(field: str, value: float, min_value: float, max_value: float) -> ValidationMessage | None:
    """
    Check numeric field range.
    """

def validate_required_string(field: str, value: str) -> ValidationMessage | None:
    """
    Check required text field.
    """

def validate_output_writable(path: Path) -> ValidationMessage | None:
    """
    Check output folder exists or can be created.
    """

def validate_units(value: float, unit: str, allowed_units: list[str]) -> ValidationMessage | None:
    """
    Check unit compatibility.
    """
```

UI rules:

1. Show validation near the field
2. Show blocking issues in page readiness panel
3. Disable unsafe actions
4. Explain why disabled actions are disabled

## 10. Audit Trail

Every important action must write an audit event.

Audit model:

```python
class AuditEvent(BaseModel):
    timestamp: datetime
    project_id: str
    event_type: str
    actor: str | None
    summary: str
    details: dict[str, Any]
```

Audit functions:

```python
def write_audit_event(project_id: str, event_type: str, summary: str, details: dict) -> None:
    """
    Append audit event to audit_log.jsonl and database index.
    """

def list_audit_events(project_id: str, limit: int = 100) -> list[AuditEvent]:
    """
    Return recent audit events.
    """

def export_audit_log(project_id: str, output_path: Path) -> Path:
    """
    Export audit trail for report or debugging.
    """
```

Events to audit:

1. Project created
2. Workflow assigned
3. Mission saved
4. Mission validated
5. Preflight completed
6. Flight started
7. Flight paused
8. Flight resumed
9. RTH triggered
10. Flight aborted
11. Dataset imported
12. Dataset validated
13. Processing started
14. Processing completed
15. Processing failed
16. Defects detected
17. Measurement added
18. Annotation added
19. Report generated
20. Settings changed

## 11. Error Handling

Use structured errors.

Error model:

```python
class AppError(Exception):
    code: str
    user_message: str
    technical_message: str
    severity: str
    recovery_action: str | None
```

Rules:

1. Operator sees user message and recovery action
2. Developer diagnostics show technical message
3. Errors are logged with stack trace
4. Blocking errors stop unsafe operations
5. Non blocking warnings allow continuation with clear warning

Example:

```python
raise AppError(
    code="MODEL_MISSING",
    user_message="Defect detection model is not configured.",
    technical_message="Model key crack_detector_v1 has no valid path in model registry.",
    severity="error",
    recovery_action="Open Model Manager and select a valid model file."
)
```

## 12. Background Workers

Use worker classes for long tasks.

Worker pattern:

```python
class WorkerSignals(QObject):
    progress = Signal(float, str)
    result = Signal(object)
    error = Signal(object)
    finished = Signal()

class ProcessingWorker(QRunnable):
    """
    Runs processing stage without freezing UI.
    Emits progress, result and error signals.
    """
```

Worker requirements:

1. Dataset import runs in worker
2. Thumbnail generation runs in worker
3. Dataset validation runs in worker
4. Defect detection runs in worker
5. Crack propagation runs in worker
6. Reconstruction runs in subprocess or worker
7. Report generation runs in worker
8. Drone telemetry runs in dedicated thread

## 13. Implementation Roadmap

### Phase 1: Foundation

Build:

1. Project manager
2. AppState
3. EventBus
4. Design tokens
5. Main window shell
6. Left navigation
7. Reusable components
8. SQLite project index
9. Audit logging

Acceptance criteria:

1. App opens cleanly
2. Project can be created
3. Project can be selected
4. Audit events are written
5. No clipped text
6. Navigation works

### Phase 2: Workflow and Mission Planning

Build:

1. Workflow template manager
2. Template gallery
3. Mission data model
4. Map canvas
5. Waypoint editing
6. Polygon survey editing
7. Mission metrics
8. Mission validation
9. Version saving

Acceptance criteria:

1. User can select workflow
2. User can create mission
3. User can draw on map
4. User can validate mission
5. Mission can be saved and reloaded

### Phase 3: Data Library

Build:

1. Dataset import
2. Metadata extraction
3. Thumbnail generation
4. Dataset validation
5. Image grid
6. Image preview
7. QA tags

Acceptance criteria:

1. User can import image folder
2. Thumbnails appear
3. Metadata summary appears
4. Dataset readiness appears
5. Processing button becomes available only when valid

### Phase 4: Processing Workbench

Build:

1. Processing run model
2. Pipeline stage cards
3. Worker execution
4. Logs drawer
5. Status updates
6. Output artifact registration

Acceptance criteria:

1. Pipeline stages show status
2. Missing inputs are shown clearly
3. Processing runs without freezing UI
4. Failed stages show useful errors

### Phase 5: Defect Detection and Crack Propagation

Build:

1. Classical defect detection
2. Model registry
3. AI model loading
4. Overlay generation
5. Crack mask handling
6. Crack propagation config
7. Forecast overlay
8. Risk summary

Acceptance criteria:

1. User can run classical detection
2. User can configure model path
3. Missing model disables AI mode
4. Defect table appears
5. Crack propagation produces overlay and summary

### Phase 6: 3D Reconstruction

Build:

1. Import reconstruction folder
2. Point cloud loading
3. Mesh loading
4. 3D viewer
5. Selected point inspector
6. Defect projection placeholder with clear unavailable state

Acceptance criteria:

1. User can import reconstruction output
2. Viewer displays 3D result
3. Viewer uses dark background
4. Selected point details appear

### Phase 7: Reports

Build:

1. Report readiness checklist
2. Jinja2 report templates
3. HTML report generation
4. PDF export
5. Report preview
6. Report history

Acceptance criteria:

1. Report disabled when missing data
2. Readiness explains missing items
3. HTML report generated
4. PDF report generated
5. Report appears in history

### Phase 8: Drone Connection and Live Flight

Build after mission, data and workflow foundation are stable.

1. Mock drone client
2. MAVSDK drone client
3. Telemetry polling
4. Preflight checklist
5. Mission upload
6. Live flight page
7. Command gateway
8. Flight log

Acceptance criteria:

1. Mock mode supports full flow
2. Telemetry updates UI
3. Commands are logged
4. Unsafe commands are disabled
5. Preflight blocks invalid mission

## 14. Testing Plan

Unit tests:

1. Project manager
2. Workflow template manager
3. Mission metrics
4. Mission validation
5. Dataset scanning
6. Metadata extraction
7. Dataset validation
8. Defect detection helpers
9. Crack propagation calculations
10. Report context building

Integration tests:

1. Create project and workflow
2. Create mission and save version
3. Import dataset and validate
4. Run processing stage
5. Generate report
6. Write audit log

UI tests:

1. Page loads without clipped text
2. Disabled buttons show reason
3. Empty states appear correctly
4. Long paths are truncated
5. Worker progress updates
6. Error messages display correctly

Recommended tools:

1. pytest
2. pytest qt
3. mypy
4. ruff
5. coverage
6. pre commit

## 15. Packaging

Recommended packaging:

1. Use uv for Python environment management
2. Use pyproject.toml
3. Use PyInstaller for desktop executable
4. Keep models external to the main binary
5. Provide sample project data
6. Provide mock drone mode

Suggested commands:

```text
uv sync
uv run python main.py
uv run pytest
uv run ruff check .
uv run mypy .
```

## 16. Suggested Repository Structure

```text
opendronekit/
  app/
    main.py
    app_state.py
    event_bus.py
    navigation.py
  ui/
    shell/
    components/
    pages/
    themes/
  core/
    projects/
    workflows/
    missions/
    preflight/
    drone/
    data_library/
    processing/
    defects/
    crack_growth/
    reconstruction/
    measurements/
    annotations/
    reports/
    settings/
    audit/
    validation/
  resources/
    workflows/
    report_templates/
    icons/
    map/
  tests/
  pyproject.toml
  README.md
```

## 17. Final Product Rules

1. The app must be workflow first
2. The app must be project based
3. The app must be local first
4. The app must separate operator UI from developer diagnostics
5. The app must show readiness before actions
6. The app must disable unsafe or impossible actions
7. The app must explain missing data
8. The app must never show clipped text
9. The app must keep long processing tasks off the UI thread
10. The app must audit important actions
11. The app must support mock mode for development
12. The app must use reusable UI components
13. The app must make reports a first class output
14. The app must be extensible for new inspection workflows
15. The app must feel like a professional inspection toolkit
