"""The OpenDroneKit feature registry: the specification, as data.

Every capability in the platform specification is one `Feature` here. The point of
this file is that a feature's status is **not** a claim anyone writes down -- it is
computed by `tools/feature_status.py` from whether the named tests actually pass.

Status ladder, and what each level costs to reach:

``not_started``
    No implementation.

``in_progress``
    Code exists but does not satisfy the acceptance criteria yet.

``implemented``
    The code is believed complete, but no test proves it. This level is deliberately
    uncomfortable: it means "someone says so".

``verified``
    Every test listed in ``tests`` passes, and those tests check the acceptance
    criteria against real inputs -- not mocks of the thing under test.

A feature may only be marked ``verified`` by the tooling, never by hand. If the tests
are missing or failing, the tool downgrades whatever was claimed. That is the whole
mechanism: a feature cannot be ticked off by asserting it.

Test naming: entries in ``tests`` are pytest node ids or prefixes, e.g.
``tests/test_geo.py::TestUmeyamaSimilarity`` or a specific
``tests/test_jobs.py::test_cooperative_cancel_stops_the_job``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["not_started", "in_progress", "implemented", "verified"]

# Which of the six products a feature belongs to.
PRODUCTS = {
    "hub": "OpenDroneKit Hub (browser command centre)",
    "app": "OpenDroneKit App (Android field application)",
    "workers": "OpenDroneKit Processing Workers",
    "vision": "OpenDroneKit Vision (AI subsystem)",
    "sdk": "OpenDroneKit SDK",
    "plugins": "OpenDroneKit Plugin System",
    "core": "Shared libraries (mission engine, geospatial, storage)",
    "infra": "Deployment, security, observability",
}


@dataclass(frozen=True)
class Feature:
    """One specified capability, with what it would take to call it done."""

    id: str
    title: str
    product: str
    category: str
    # What must be demonstrably true. Written so a reviewer can disagree with a claim.
    criteria: str
    claimed: Status = "not_started"
    tests: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


def F(id, title, product, category, criteria, claimed="not_started", tests=(), notes=""):
    return Feature(id, title, product, category, criteria, claimed, tuple(tests), notes)


# ---------------------------------------------------------------------------
# Mission planning
# ---------------------------------------------------------------------------

MISSION_PLANNING = [
    F("mp.engine.independent", "Mission engine is an independent library", "core", "Mission engine",
      "Mission geometry importable without the web UI, drone SDK, or any manufacturer dependency.",
      "implemented", ["tests/test_mission_constraints.py"],
      "mission/planner.py imports no UI or vendor code."),
    F("mp.output.contract", "Mission output carries the full capture contract", "core", "Mission engine",
      "Every generated mission yields waypoints, altitude, speed, heading, yaw, gimbal pitch, "
      "trigger timing, estimated time, distance, image count and a return path.",
      "in_progress", ["tests/test_exporters.py::test_capture_commands_survive_export"],
      "Storage and battery estimates not yet emitted."),
    F("mp.type.mapping_2d", "2D mapping mission", "core", "Mission types",
      "Single and double grid over a drawn polygon with configurable direction, overlap and GSD.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.mapping_3d", "3D modelling mission", "core", "Mission types",
      "Cross-hatch plus oblique capture at multiple gimbal angles and altitude bands.",
      "in_progress", [], "double_grid gives cross-hatch; oblique bands not implemented."),
    F("mp.type.roof_mapping", "Roof mapping", "core", "Mission types",
      "Nadir and oblique capture over a roof outline with GSD-driven altitude.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.roof_inspection", "Roof inspection", "core", "Mission types",
      "Stop-and-capture at computed picture points with stand-off and gimbal control.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.facade_mapping", "Facade mapping", "core", "Mission types",
      "Vertical capture paths along an elevation with stand-off and vertical overlap.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.facade_inspection", "Facade inspection", "core", "Mission types",
      "Inspection rows and columns at configurable GSD and stand-off.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.multi_facade", "Multi-facade missions", "core", "Mission types",
      "One pass per building face with per-face stand-off, altitude band and spacing; "
      "capture points sit outside the footprint with the camera facing the wall.",
      "implemented", ["tests/test_mission_types.py::TestMultiFacade"]),
    F("mp.type.complex_facade", "Complex facade planning", "core", "Mission types",
      "Balconies, recesses, courtyards, overhangs and non-rectangular footprints.",
      "not_started", []),
    F("mp.type.closed_loop", "Closed-loop structure capture", "core", "Mission types",
      "Continuous loops at multiple radii and altitude rings, either direction, with "
      "the camera always facing the structure.",
      "implemented", ["tests/test_mission_types.py::TestClosedLoop"]),
    F("mp.type.l_shaped", "L-shaped and irregular buildings", "core", "Mission types",
      "Facade planning maintains coverage on non-rectangular footprints.",
      "not_started", []),
    F("mp.type.linear_mapping", "Linear mapping", "core", "Mission types",
      "Corridor following a polyline with configurable width and parallel passes.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.linear_inspection", "Linear inspection", "core", "Mission types",
      "Asset-following capture at inspection distance with camera facing control.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.tower_mapping", "Tower mapping", "core", "Mission types",
      "Circular and spiral paths, stacked orbits at multiple radii and heights.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.pylon_inspection", "Utility pylon inspection", "core", "Mission types",
      "Body, crossarm, insulator and conductor capture at multiple elevations.",
      "not_started", []),
    F("mp.type.orbit", "Orbit missions", "core", "Mission types",
      "Radius, altitude, rotations, direction and gimbal pitch with target lock.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.waypoints", "Waypoint missions", "core", "Mission types",
      "Per-waypoint altitude, speed, heading, gimbal, hover, capture and payload commands.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.panorama", "Panorama missions", "core", "Mission types",
      "360, 180, horizontal, vertical and spherical capture patterns.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.solar", "Solar inspection", "core", "Mission types",
      "Row-aligned capture over panels with RGB and thermal support.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.wind_turbine", "Wind turbine inspection", "core", "Mission types",
      "Tower stack, four nacelle aspects, and per-blade stations captured from both "
      "faces at the parked rotor angles, skipping points below a safe altitude.",
      "implemented", ["tests/test_mission_types.py::TestWindTurbine"]),
    F("mp.type.magnetic", "Magnetic mapping", "core", "Mission types",
      "Constant ground clearance survey grid with payload trigger and time sync.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.thermal", "Thermal missions", "core", "Mission types",
      "Thermal mapping and inspection with paired RGB capture.",
      "not_started", []),
    F("mp.type.multispectral", "Multispectral missions", "core", "Mission types",
      "Synchronised band capture for agricultural and vegetation survey.",
      "not_started", []),
    F("mp.linking", "Mission linking", "core", "Mission engine",
      "Several mission types execute as one sequence with per-segment completion tracking.",
      "in_progress", [], "generate_linked_mission exists; segment tracking unproven."),
    F("mp.geometry_3d", "3D geometry based planning", "core", "Mission engine",
      "Paths generated from imported OBJ/GLB/PLY/LAS/IFC surfaces.",
      "not_started", []),
    F("mp.standoff", "Stand-off distance planning", "core", "Mission engine",
      "Fixed, per-surface and adaptive stand-off with a minimum clearance guarantee.",
      "in_progress", [], "Fixed stand-off only."),
    F("mp.gsd", "GSD planning and calculator", "core", "Mission engine",
      "Ground sample distance from sensor, focal length and distance; and the inverse.",
      "implemented", ["tests/test_mission_constraints.py::TestPhotogrammetry"]),
    F("mp.overlap", "Image overlap planning", "core", "Mission engine",
      "Front, side, vertical and horizontal overlap drive spacing.",
      "implemented", ["tests/test_mission_constraints.py::TestPhotogrammetry::test_more_overlap_costs_more_waypoints"]),
    F("mp.terrain_follow", "Terrain following", "core", "Mission engine",
      "AGL/AMSL follow from GeoTIFF, ASC, CSV or fitted plane, with an explicit warning "
      "whenever the plan degrades to flat earth.",
      "implemented", [], "Warning surfaced in app/api.py; needs a test."),
    F("mp.terrain_offline", "Offline terrain cache", "core", "Mission engine",
      "Terrain tiles cached so terrain following works with no connectivity.",
      "not_started", []),
    F("mp.obstacles", "Obstacle planning", "core", "Mission engine",
      "Point, circular, polygon and 3D obstacles with route intersection detection and detours.",
      "implemented", ["tests/test_mission_constraints.py::TestNoFlyZones"]),
    F("mp.geofence", "Geofence containment", "core", "Mission engine",
      "No waypoint outside the inclusion fence or inside an exclusion zone.",
      "verified", ["tests/test_mission_constraints.py::TestNoFlyZones",
                   "tests/test_mission_constraints.py::TestAltitudeBand"]),
    F("mp.estimates", "Mission estimates", "core", "Mission engine",
      "Duration, distance, image count, storage, battery count and reserve before flight.",
      "in_progress", ["tests/test_mission_constraints.py::TestPhotogrammetry::test_duration_estimate_is_positive_and_finite"],
      "Storage and battery estimates missing."),
    F("mp.simulation", "3D mission simulation", "hub", "Mission engine",
      "Timeline playback of trajectory, gimbal, capture points, terrain and battery.",
      "not_started", []),
    F("mp.versioning", "Mission versioning", "hub", "Mission engine",
      "Version, author, timestamp, diff against previous and restore.",
      "in_progress", [], "save_mission_version stores versions; diff and restore missing."),
    F("mp.sharing", "Mission preview sharing", "hub", "Mission engine",
      "Secure link showing path, area, altitude, duration, drone and safety areas.",
      "not_started", []),
    F("mp.repeatable", "Repeatable missions", "hub", "Mission engine",
      "Repeat exactly, with updated terrain, modified boundary, or a different aircraft.",
      "not_started", []),
    F("mp.import", "Mission import", "hub", "Mission engine",
      "KML, KMZ, GeoJSON, GPX and CSV waypoint import.",
      "in_progress", [], "GeoJSON import works; KML/KMZ/GPX/CSV missing."),
    F("mp.fly_to_draw", "Fly to draw", "app", "Mission engine",
      "Boundary defined by flying the aircraft to mark positions.",
      "not_started", []),
    F("mp.camera_db", "Camera database", "core", "Mission engine",
      "Sensor dimensions, resolution, focal length, pixel size and thermal capability, "
      "with user-defined cameras.",
      "in_progress", [], "Sensor width DB exists in core/geo.py; not a full profile store."),
    F("mp.payload_db", "Payload database", "core", "Mission engine",
      "RGB, thermal, multispectral, LiDAR, magnetometer and custom payload commands.",
      "not_started", []),
]

# ---------------------------------------------------------------------------
# Flight execution
# ---------------------------------------------------------------------------

FLIGHT = [
    F("fl.abstraction", "Drone abstraction layer", "app", "Flight",
      "A generic interface with adapters; no core code depends on one manufacturer.",
      "in_progress", [], "core/drone.py defines the protocol; only MAVLink and mock exist."),
    F("fl.mavlink.upload", "MAVLink mission upload", "core", "Flight",
      "Request/ack transfer protocol; gimbal, yaw, dwell and trigger items survive a round trip.",
      "verified", ["tests/test_mavlink_transfer.py::test_mission_upload_is_acknowledged",
                   "tests/test_mavlink_transfer.py::test_capture_commands_reach_the_vehicle",
                   "tests/test_mavlink_transfer.py::test_download_round_trips_the_uploaded_mission"]),
    F("fl.mavlink.fence", "Geofence and rally upload", "core", "Flight",
      "Fence and rally land in their own MAV_MISSION_TYPE without overwriting the flight plan.",
      "verified", ["tests/test_mavlink_transfer.py::test_each_list_lands_in_its_own_slot"]),
    F("fl.sitl", "SITL verified flight", "core", "Flight",
      "A mission uploads, arms and flies to completion in ArduPilot or PX4 SITL.",
      "not_started", [], "Loopback peer is tested; a real SITL flight is not."),
    F("fl.preflight", "Preflight checks", "app", "Flight",
      "Connection, battery, GPS, compass, IMU, home point, storage, camera, gimbal and "
      "mission-vs-geofence conflicts checked before arming.",
      "in_progress", [], "core/preflight.py exists; not wired to a real vehicle."),
    F("fl.telemetry", "Live telemetry", "core", "Flight",
      "Position, battery, GPS, RC link and mission progress streamed to subscribers.",
      "implemented", [], "subscribe() added to the bridge; no UI consumer test."),
    F("fl.manual_override", "Manual override", "app", "Flight",
      "The pilot can interrupt autonomy at any time, and the control state is displayed.",
      "not_started", []),
    F("fl.battery_swap", "Battery swap and resume", "app", "Flight",
      "Completed segments recorded; resume continues without duplicate capture.",
      "not_started", []),
    F("fl.crash_recovery", "Crash recovery", "app", "Flight",
      "Mission state and telemetry persist; the app never silently restarts a mission.",
      "not_started", []),
    F("fl.camera_control", "Camera control", "app", "Flight",
      "Photo, video, RAW, ISO, shutter, exposure, white balance, focus and zoom.",
      "not_started", []),
    F("fl.gimbal_control", "Gimbal control", "core", "Flight",
      "Pitch, yaw, centre and look-at commands issued as mission items.",
      "verified", ["tests/test_exporters.py::test_gimbal_pitch_and_mount_mode_land_in_the_right_slots"]),
    F("fl.logging", "Flight logging", "app", "Flight",
      "Full telemetry log exported to CSV, JSON, GPX and KML.",
      "not_started", []),
    F("fl.data_verification", "On-site data verification", "app", "Flight",
      "Image count, blur, exposure, corrupt files, missing GPS and coverage gaps checked "
      "before leaving the site.",
      "in_progress", [], "core/coverage_validation.py covers gaps; blur/exposure checks missing."),
]

# ---------------------------------------------------------------------------
# Processing and reconstruction
# ---------------------------------------------------------------------------

PROCESSING = [
    F("pr.sfm", "Structure from motion", "workers", "Processing",
      "COLMAP SfM with bundle adjustment; reprojection error under 1.5 px on a real survey.",
      "implemented", [], "77/77 registered at 1.27 px on Aukerman; needs a committed regression test."),
    F("pr.georeference", "Georeferencing", "workers", "Processing",
      "RANSAC Helmert similarity between camera centres and geotags, into an auto-selected UTM zone.",
      "verified", ["tests/test_geo.py::TestUmeyamaSimilarity", "tests/test_geo.py::TestUtmZoneSelection"]),
    F("pr.ortho", "Orthomosaic generation", "workers", "Processing",
      "True orthophoto by DSM back-projection, written as a Cloud-Optimized GeoTIFF.",
      "implemented", [], "COLMAP engine only; needs an output regression test."),
    F("pr.dsm_dtm", "DSM and DTM", "workers", "Processing",
      "Metric elevation rasters with a ground filter, in a stated CRS.",
      "implemented", ["tests/test_dsm_analysis.py"]),
    F("pr.dense", "Dense point cloud", "workers", "Processing",
      "Patch-match multi-view stereo. Never synthesised from the sparse cloud.",
      "not_started", ["tests/test_honesty.py::TestNoSyntheticDensification"],
      "Blocked: pycolmap wheels are CPU-only. The fake was removed."),
    F("pr.mesh", "Textured mesh", "workers", "Processing",
      "Poisson surface reconstruction with density trimming and orthophoto texture.",
      "implemented", []),
    F("pr.gps_denied", "GPS denied reconstruction", "workers", "Processing",
      "Indoor, handheld and ground-robot imagery reconstructed without geotags.",
      "not_started", []),
    F("pr.gcp", "Ground control points", "workers", "Processing",
      "GCP import, image marking, and a reprojection error report.",
      "not_started", []),
    F("pr.rtk_ppk", "RTK and PPK", "workers", "Processing",
      "RTK/PPK metadata and RINEX base station data with timestamp alignment.",
      "not_started", []),
    F("pr.distributed", "Distributed processing", "workers", "Processing",
      "Job queue with priorities, retries, cancellation, progress and resource limits.",
      "in_progress", ["tests/test_jobs.py", "tests/test_processing.py"],
      "Submit/poll/cancel over the API with cooperative cancellation and honest failure "
      "reporting. Still single-process: no multi-worker queue, priorities or retries."),
    F("pr.large_datasets", "Large dataset processing", "workers", "Processing",
      "Thousands of images via chunking, resumable jobs and memory-aware scheduling.",
      "not_started", []),
    F("pr.provenance", "Derived file provenance", "workers", "Processing",
      "Every derived artifact records its source, engine, CRS and parameters.",
      "in_progress", [], "Digital twin records some of this."),
]

# ---------------------------------------------------------------------------
# Vision / AI
# ---------------------------------------------------------------------------

VISION = [
    F("ai.registry", "AI model registry", "vision", "AI",
      "Register, version, deploy and roll back models, with metrics and a real checksum.",
      "implemented", [], "training/register.py; refuses an export that failed parity."),
    F("ai.parity", "ONNX export parity", "vision", "AI",
      "An export is rejected unless it matches torch within tolerance and loads in the "
      "runtime the pipeline actually uses.",
      "implemented", [], "Verified on two models; needs a committed test."),
    F("ai.crack", "Crack detection", "vision", "AI",
      "A trained model measurably better than the classical baseline on a held-out split, "
      "installed with a real checksum and reported as the model actually used.",
      "implemented", ["tests/test_honesty.py::TestDetectionReportsWhatItActuallyUsed"],
      "SegFormer-B2 installed: test-split IoU 0.515 vs heuristic 0.045 (11x), precision 0.540 vs 0.046."),
    F("ai.spalling", "Spalling detection", "vision", "AI",
      "Trained detector for spallation with published validation metrics.",
      "in_progress", [], "CODEBRIM data prepared; not yet trained."),
    F("ai.corrosion", "Corrosion detection", "vision", "AI",
      "Trained detector for corrosion and rust with published validation metrics.",
      "in_progress", [], "Data prepared; not yet trained."),
    F("ai.solar", "Solar defect detection", "vision", "AI",
      "Trained detector for panel defects with published validation metrics.",
      "in_progress", [], "Data prepared; not yet trained."),
    F("ai.water_ponding", "Water ponding detection", "vision", "AI",
      "Trained detector for standing water on roofs.",
      "not_started", []),
    F("ai.deformation", "Surface deformation detection", "vision", "AI",
      "Trained detector for visible deformation.",
      "not_started", []),
    F("ai.custom_training", "Custom defect training", "vision", "AI",
      "Users label, split, train, review metrics and deploy their own model.",
      "in_progress", [], "Trainers exist; no labelling UI or user-facing dataset builder."),
    F("ai.assisted_annotation", "AI assisted annotation", "hub", "AI",
      "Model pre-labels imagery; a reviewer accepts, edits, merges, splits or reclassifies.",
      "not_started", []),
    F("ai.human_validation", "Human validation record", "hub", "AI",
      "A model prediction is never stored as verified. The model's claim (key, sha256, "
      "confidence) and the reviewer's decision are separate fields, so reviewing never "
      "erases what the model asserted.",
      "implemented", ["tests/test_inspection.py::TestProvenance",
                      "tests/test_inspection.py::TestReview"]),
    F("ai.quantification", "Defect quantification", "vision", "AI",
      "Counts, total length, total area and percentage of surface affected, in metric "
      "units, with unmeasured defects declared rather than counted as zero.",
      "verified", ["tests/test_dsm_analysis.py::test_measurements_sum_defect_geometry",
                   "tests/test_risk_scoring.py", "tests/test_inspection.py::TestSummary"]),
    F("ai.projection", "Defect back-projection to 3D", "vision", "AI",
      "2-D masks projected onto the reconstructed surface as georeferenced polygons in m2.",
      "verified", ["tests/test_defect_projection.py"]),
    F("ai.change_detection", "Change detection", "vision", "AI",
      "New, resolved, grown and shrunk defects between two surveys, matched by type "
      "within a radius; a defect that moved beyond the radius is reported as one "
      "resolved plus one new rather than as the same defect having moved.",
      "implemented", ["tests/test_change_detection.py::TestDefectComparison"]),
    F("ai.progress_tracking", "Construction progress tracking", "vision", "AI",
      "Baseline versus current model with progress percentage and change regions.",
      "in_progress", ["tests/test_change_detection.py::TestSurfaceComparison"],
      "Volume added/removed and changed area between two surveys are computed; "
      "progress percentage against a design model is not."),
    F("ai.model_version_recorded", "Every AI result stores model version and confidence", "vision", "AI",
      "No detection is stored without its model identity and confidence.",
      "in_progress", ["tests/test_honesty.py::TestDetectionReportsWhatItActuallyUsed"],
      "model_used is reported; per-detection model version is not yet persisted."),
]

# ---------------------------------------------------------------------------
# Inspection, measurement, thermal
# ---------------------------------------------------------------------------

INSPECTION = [
    F("in.annotations", "Annotations", "hub", "Inspection",
      "Point, line, polygon, rectangle, circle, freehand and text with severity and status.",
      "in_progress", [], "core/annotations.py exists; no UI."),
    F("in.defect_library", "Defect library", "hub", "Inspection",
      "Default categories offered, with any organisation-defined category accepted; "
      "the default list is not a whitelist.",
      "implemented", ["tests/test_inspection.py::TestDefectLibrary"]),
    F("in.defect_record", "Defect record", "hub", "Inspection",
      "ID, category, severity, confidence, location, 3D position, measurements and history.",
      "implemented", ["tests/test_defect_projection.py"]),
    F("me.2d", "2D measurements", "hub", "Measurement",
      "Distance, area and perimeter directly on an orthomosaic.",
      "in_progress", [], "core/measurements.py exists; no viewer integration."),
    F("me.3d", "3D measurements", "hub", "Measurement",
      "Length, height, area and volume inside the 3D model.",
      "not_started", []),
    F("me.volume", "Volume and stockpile", "core", "Measurement",
      "Cut/fill against DTM, plane and lowest-point references, exact on a known surface.",
      "verified", ["tests/test_dsm_analysis.py::test_volume_against_dtm_is_exact",
                   "tests/test_dsm_analysis.py::test_polygon_clip_halves_the_volume"]),
    F("me.slope", "Slope measurement", "core", "Measurement",
      "Gradient for pitched roofs, pavements and ramps.",
      "not_started", []),
    F("th.radiometric", "Radiometric thermal processing", "workers", "Thermal",
      "Raw counts converted to temperature through the camera's Planck constants with "
      "emissivity and reflected-temperature correction, verified by round trip within "
      "0.1 K, and a file carrying no radiometric data refused rather than read from "
      "its palette.",
      "implemented", ["tests/test_thermal.py::TestConversion",
                      "tests/test_thermal.py::TestEmissivity",
                      "tests/test_thermal.py::TestRefusal"],
      "Sidecar counts supported; extracting the embedded APP1 payload from a thermal "
      "JPEG is not implemented and says so."),
    F("th.map_2d", "2D thermal map", "workers", "Thermal",
      "Temperature field written as a georeferenced raster in Celsius, so its units "
      "are unambiguous in GIS.",
      "in_progress", ["tests/test_thermal.py::TestGeoreferencedOutput"],
      "Single-frame georeferenced output done; mosaicking many frames is not."),
    F("th.model_3d", "3D thermal model", "workers", "Thermal",
      "Thermal values projected onto reconstructed geometry.",
      "not_started", []),
    F("th.comparison", "RGB and thermal comparison", "hub", "Thermal",
      "Side by side, swipe and opacity overlay with linked zoom.",
      "not_started", []),
]

# ---------------------------------------------------------------------------
# Hub, platform, infrastructure
# ---------------------------------------------------------------------------

PLATFORM = [
    F("hub.map.basemaps", "Map basemaps and layers", "hub", "Hub",
      "OSM, satellite, terrain and offline basemaps with XYZ/WMS/WMTS and vector overlays.",
      "in_progress", [], "MapLibre with four basemaps; no WMS/WMTS."),
    F("hub.map.geocoding", "Address search", "hub", "Hub",
      "Search an address, place name or typed coordinates and fly the map to the result, "
      "through a replaceable provider that can be self-hosted or fully offline.",
      "implemented", ["tests/test_geocoding.py"],
      "Nominatim (self-hostable) and offline providers; a typed coordinate never leaves the machine."),
    F("hub.map.offline_tiles", "Offline tile cache", "hub", "Hub",
      "Tiles actually downloaded and served without connectivity.",
      "not_started", []),
    F("hub.projects", "Project management", "hub", "Hub",
      "Projects with client, site, type, tags, members, history and activity timeline.",
      "in_progress", [], "SQLite project store; most fields missing."),
    F("hub.assets", "Asset management", "hub", "Hub",
      "Persistent assets with geometry, inspection history and a timeline.",
      "not_started", []),
    F("hub.orgs", "Organizations and roles", "hub", "Hub",
      "Multiple organizations with eight ranked roles, member invite/remove, and an "
      "organization that can never lose its last owner.",
      "implemented", ["tests/test_api.py::TestAuthorization"]),
    F("hub.auth", "Authentication", "hub", "Security",
      "Local auth with bcrypt, JWT sessions, and long-lived API tokens stored only as "
      "hashes with the secret shown once.",
      "implemented", ["tests/test_api.py::TestAuthentication", "tests/test_api.py::TestApiTokens"],
      "MFA and OAuth/OIDC still outstanding."),
    F("hub.viewer_2d", "2D viewer", "hub", "Hub",
      "Tiled imagery with layers, measurements, annotations and comparison mode.",
      "in_progress", [], "Raster preview only."),
    F("hub.viewer_3d", "3D viewer", "hub", "Hub",
      "Orbit, fly navigation, WASDQE keys, clipping and overlays.",
      "not_started", []),
    F("hub.point_cloud", "Point cloud viewer", "hub", "Hub",
      "Large point clouds streamed progressively.",
      "not_started", []),
    F("hub.digital_twin", "Digital twin", "hub", "Hub",
      "Model, cloud, ortho, images, annotations, defects and history as one navigable object.",
      "in_progress", [], "digital_twin.json artifact index exists; no UI."),
    F("hub.timeline", "Time based comparison / 4D", "hub", "Hub",
      "Elevation difference and volume added/removed between two surveys, refusing to "
      "compare surveys in different coordinate systems or at different resolutions "
      "rather than resampling them into apparent agreement.",
      "implemented", ["tests/test_change_detection.py::TestSurfaceComparison",
                      "tests/test_change_detection.py::TestIncomparableSurveys"],
      "Core comparison done; no timeline UI yet."),
    F("rp.reports", "Automated inspection reports", "hub", "Reporting",
      "Structured report with findings, images, measurements and AI results.",
      "implemented", [], "Markdown and HTML; PDF path unverified."),
    F("rp.formats", "Report formats", "hub", "Reporting",
      "PDF and DOCX generated directly from the report payload, each verified by "
      "opening the result with its own reader rather than checking it is non-empty.",
      "implemented", ["tests/test_report_formats.py::TestPdf",
                      "tests/test_report_formats.py::TestDocx"],
      "PDF, DOCX, HTML and Markdown done; CSV and JSON exports still to come."),
    F("rp.templates", "Report templates", "hub", "Reporting",
      "Reusable organisation templates controlling title, organisation, client, "
      "section selection, severity ordering and whether unreviewed findings appear.",
      "implemented", ["tests/test_report_formats.py::TestTemplates",
                      "tests/test_report_formats.py::TestFindingOrder"]),
    F("sh.links", "Shareable project links", "hub", "Sharing",
      "A high-entropy link opens one project read-only with no account. Only the token "
      "hash is stored, and the response states plainly that it grants no write access.",
      "implemented", ["tests/test_sharing.py::TestCreation",
                      "tests/test_sharing.py::TestPublicAccess"]),
    F("sh.security", "Sharing security", "hub", "Sharing",
      "Password protection, expiry and revocation checked on every access, with every "
      "attempt logged including failures, and a revoked link indistinguishable from an "
      "unknown one so a probe cannot learn which tokens existed.",
      "implemented", ["tests/test_sharing.py::TestRevocationAndExpiry",
                      "tests/test_sharing.py::TestPassword",
                      "tests/test_sharing.py::TestAccessLog"]),
    F("ex.geotiff", "GeoTIFF export", "core", "Export",
      "Cloud-Optimized GeoTIFF readable by QGIS with correct CRS.",
      "verified", ["tests/test_geo.py::TestGeoTiffRoundTrip"]),
    F("ex.mission_formats", "Mission export formats", "core", "Export",
      "QGC .plan, QGC WPL, DJI WPML KMZ, Litchi CSV and KML.",
      "verified", ["tests/test_exporters.py::test_every_registered_format_writes_a_file",
                   "tests/test_exporters.py::test_dji_kmz_has_the_required_wpml_members"]),
    F("ex.model_formats", "3D export formats", "core", "Export",
      "OBJ, PLY, LAS/LAZ, GLB and GLTF.",
      "in_progress", [], "OBJ and PLY only."),
    F("api.rest", "REST API", "hub", "API",
      "Documented endpoints for every resource in the specification.",
      "in_progress", ["tests/test_api.py::TestProjectsAndMissions", "tests/test_api.py::TestAssets",
                      "tests/test_uploads.py", "tests/test_processing.py"],
      "Auth, orgs, projects, assets, missions, export, datasets, resumable upload and "
      "processing jobs done. Defects, measurements, reports and AI jobs still to come."),
    F("api.uploads", "Resumable dataset upload", "hub", "API",
      "A client declares a file with its size and sha256, sends chunks in any order, "
      "queries what is missing, and finalises; assembly is refused unless both the byte "
      "count and the checksum match, and a rejected file is not left on disk.",
      "implemented", ["tests/test_uploads.py::TestResumableUpload"]),
    F("api.upload_safety", "Upload path containment", "hub", "Security",
      "A client-supplied filename can never place a file outside the storage root.",
      "implemented", ["tests/test_uploads.py::TestUploadPathSafety"]),
    F("api.webhooks", "Webhooks", "hub", "API",
      "Events for mission, flight, dataset, processing, AI, defect and report lifecycle.",
      "not_started", []),
    F("api.realtime", "Real time communication", "hub", "API",
      "WebSocket telemetry, processing progress and notifications.",
      "not_started", []),
    F("fm.fleet", "Fleet management", "hub", "Fleet",
      "Drones, controllers, payloads, batteries and firmware with flight hours.",
      "not_started", []),
    F("fm.battery", "Battery management", "hub", "Fleet",
      "Cycle count, health, capacity and retirement state.",
      "not_started", []),
    F("fm.pilots", "Pilot management", "hub", "Fleet",
      "Certifications, licences, currency and expiry tracking.",
      "not_started", []),
    F("sec.audit", "Audit log", "hub", "Security",
      "Every state-changing request recorded with actor, resource and timestamp, "
      "committed in the same transaction as the action it describes.",
      "implemented", ["tests/test_api.py::TestAudit"]),
    F("sec.rbac", "Role based access control", "hub", "Security",
      "Every endpoint resolves the caller's role in the owning organization; a non-member "
      "gets 404 rather than 403, and nobody may grant a role above their own.",
      "implemented", ["tests/test_api.py::TestAuthorization"]),
    F("inf.docker", "Docker deployment", "infra", "Deployment",
      "docker compose up brings PostGIS, MinIO, the API and a worker online, with "
      "healthchecks so the API waits for a database that actually accepts connections, "
      "and no credential committed to the repository.",
      "implemented", ["tests/test_compose.py"],
      "Definition verified as data; booting the stack needs a Docker daemon."),
    F("inf.k8s", "Kubernetes and Helm", "infra", "Deployment",
      "Helm chart deploys the platform to a cluster.",
      "not_started", []),
    F("inf.postgis", "PostgreSQL and PostGIS", "infra", "Deployment",
      "Spatial data in native PostGIS types, with the API reporting which backend is "
      "actually live so a development database is never mistaken for a deployment.",
      "in_progress", ["tests/test_api.py::TestHealth"],
      "Schema and CRS columns are in place and the backend is reported honestly; native "
      "geometry columns and spatial indexes still to come."),
    F("inf.storage", "Storage abstraction", "infra", "Deployment",
      "Local filesystem and S3-compatible backends behind one interface, with keys "
      "that cannot escape the storage root and an unknown backend refused rather than "
      "silently falling back.",
      "implemented", ["tests/test_storage.py"]),
    F("inf.offline_first", "No external telemetry by default", "infra", "Privacy",
      "A self-hosted install sends nothing outward unless explicitly configured.",
      "verified", ["tests/test_honesty.py::TestNoSilentNetworkCall"]),
    F("inf.observability", "Observability", "infra", "Deployment",
      "Structured logs, metrics, tracing and health endpoints.",
      "not_started", []),
    F("sdk.plugin_system", "Plugin system", "plugins", "SDK",
      "Documented plugin points for drones, cameras, payloads, mission types, engines, "
      "models, exporters, report templates and map providers.",
      "in_progress", [], "Reconstruction engines and exporters are pluggable; nothing else."),
    F("sdk.libraries", "Developer SDK", "sdk", "SDK",
      "Libraries for mission generation, drone adapters, telemetry and job APIs.",
      "not_started", []),
    F("doc.guides", "Documentation set", "infra", "Docs",
      "Installation, architecture, user, pilot, plugin, API and deployment guides.",
      "in_progress", [], "README and training/cloud docs only."),
    F("demo.mode", "Demo mode", "hub", "Docs",
      "Full workflow explorable with no hardware.",
      "not_started", []),
]

ALL_FEATURES: list[Feature] = [
    *MISSION_PLANNING,
    *FLIGHT,
    *PROCESSING,
    *VISION,
    *INSPECTION,
    *PLATFORM,
]


def by_id() -> dict[str, Feature]:
    index: dict[str, Feature] = {}
    for feature in ALL_FEATURES:
        if feature.id in index:
            raise ValueError(f"Duplicate feature id: {feature.id}")
        index[feature.id] = feature
    return index
