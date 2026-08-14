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
      "implemented", ["tests/test_exporters.py::test_capture_commands_survive_export",
                      "tests/test_estimates.py::TestWholeMission"],
      "Storage and battery now emitted by MissionPlan.estimates()."),
    F("mp.type.mapping_2d", "2D mapping mission", "core", "Mission types",
      "Single and double grid over a drawn polygon with configurable direction, overlap and GSD.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.mapping_3d", "3D modelling mission", "core", "Mission types",
      "Cross-hatch plus oblique capture at multiple gimbal angles and altitude bands.",
      "in_progress", [], "double_grid gives cross-hatch; oblique bands not implemented."),
    F("mp.type.roof_mapping", "Roof mapping", "core", "Mission types",
      "Nadir and oblique capture over a roof outline with GSD-driven altitude.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates",
                      "tests/test_standoff.py::TestAgainstARealPlan::test_every_named_mission_type_compiles_to_its_own_template"],
      "Resolves to grid deliberately -- a roof map is a nadir grid -- rather than by "
      "falling through the alias default, which is how it used to arrive."),
    F("mp.type.roof_inspection", "Roof inspection", "core", "Mission types",
      "Stop-and-capture at computed picture points with stand-off and gimbal control.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.facade_mapping", "Facade mapping", "core", "Mission types",
      "Vertical capture paths along an elevation with stand-off and vertical overlap.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates"]),
    F("mp.type.facade_inspection", "Facade inspection", "core", "Mission types",
      "Inspection rows and columns at configurable GSD and stand-off, compiling to the "
      "facade primitive rather than a nadir grid, and keeping the stand-off it declares.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates",
                      "tests/test_standoff.py::TestAgainstARealPlan::test_every_named_mission_type_compiles_to_its_own_template",
                      "tests/test_standoff.py::TestAgainstARealPlan::test_facade_inspection_is_not_silently_a_nadir_grid"],
      "Was silently aliased to grid; the alias and the geofence clipping are fixed."),
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
      "Corridor following a polyline with configurable width and parallel passes, "
      "compiling to the corridor primitive rather than gridding the bounding area.",
      "implemented", ["tests/test_mission_constraints.py::TestTemplates",
                      "tests/test_standoff.py::TestAgainstARealPlan::test_every_named_mission_type_compiles_to_its_own_template"],
      "Was silently aliased to grid."),
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
      "Fixed, per-surface and adaptive stand-off with a minimum clearance guarantee, "
      "where a resolution that would require flying inside the clearance is reported as "
      "a conflict rather than silently resolved either way, and compiled missions are "
      "measured against the structure to prove the stand-off survived compilation.",
      "implemented", ["tests/test_standoff.py::TestFixedPolicy",
                      "tests/test_standoff.py::TestPerSurfacePolicy",
                      "tests/test_standoff.py::TestAdaptivePolicy",
                      "tests/test_standoff.py::TestTheConflict",
                      "tests/test_standoff.py::TestTheFloor",
                      "tests/test_standoff.py::TestGeometry",
                      "tests/test_standoff.py::TestVerification",
                      "tests/test_standoff.py::TestAgainstARealPlan"],
      "mission/standoff.py. Verification found facade_inspection compiling to a nadir "
      "grid and the geofence projecting capture points onto the wall; both fixed."),
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
      "Duration, distance, image count, storage, battery count and reserve before flight, "
      "with a mission too long for one battery reported as such rather than rounded down, "
      "and an unknown camera reported as a guess rather than sized silently.",
      "implemented", ["tests/test_mission_constraints.py::TestPhotogrammetry::test_duration_estimate_is_positive_and_finite",
                      "tests/test_estimates.py::TestStorage",
                      "tests/test_estimates.py::TestBatteries",
                      "tests/test_estimates.py::TestWholeMission"],
      "mission/estimates.py; MissionPlan.estimates() and Api.mission_estimates."),
    F("mp.simulation", "3D mission simulation", "hub", "Mission engine",
      "Timeline playback of trajectory, gimbal, capture points, terrain and battery.",
      "not_started", []),
    F("mp.versioning", "Mission versioning", "hub", "Mission engine",
      "Version, author, timestamp, diff against previous and restore, with the diff "
      "stated in operator terms rather than as a raw field dump, and restore appending "
      "a new version so the record of what was flown in between survives.",
      "implemented", ["tests/test_versioning.py::TestDiff",
                      "tests/test_versioning.py::TestHistory",
                      "tests/test_versioning.py::TestRestore",
                      "tests/test_versioning.py::TestAgainstRealStorage"],
      "mission/versioning.py; Api.diff_mission_versions and restore_mission_version."),
    F("mp.sharing", "Mission preview sharing", "hub", "Mission engine",
      "Secure link showing path, area, altitude, duration, drone and safety areas.",
      "not_started", []),
    F("mp.repeatable", "Repeatable missions", "hub", "Mission engine",
      "Repeat exactly, with updated terrain, modified boundary, or a different aircraft.",
      "not_started", []),
    F("mp.import", "Mission import", "hub", "Mission engine",
      "KML, KMZ, GeoJSON, GPX and CSV boundary import, each read in its own coordinate "
      "order, with out-of-range coordinates and headerless CSVs refused rather than "
      "guessed at, and the imported boundary usable directly for planning.",
      "implemented", ["tests/test_boundary_import.py::TestKML",
                      "tests/test_boundary_import.py::TestKMZ",
                      "tests/test_boundary_import.py::TestGeoJSON",
                      "tests/test_boundary_import.py::TestGPX",
                      "tests/test_boundary_import.py::TestCSV",
                      "tests/test_boundary_import.py::TestValidation",
                      "tests/test_boundary_import.py::TestDispatch"],
      "mission/boundary_import.py; Api.import_boundary sets the session AOI."),
    F("mp.fly_to_draw", "Fly to draw", "app", "Mission engine",
      "Boundary defined by flying the aircraft to mark positions.",
      "not_started", []),
    F("mp.camera_db", "Camera database", "core", "Mission engine",
      "Sensor dimensions, resolution, focal length, pixel size and thermal capability, "
      "with user-defined cameras, geometry validated as physically possible, and an "
      "unrecognised camera reported as unrecognised rather than resolved to a default "
      "that would yield a confident but wrong GSD.",
      "implemented", ["tests/test_cameras.py::TestGeometry",
                      "tests/test_cameras.py::TestValidation",
                      "tests/test_cameras.py::TestResolution",
                      "tests/test_cameras.py::TestThermal",
                      "tests/test_cameras.py::TestUserProfiles",
                      "tests/test_cameras.py::TestAgreementWithThePlanner"],
      "mission/cameras.py; 11 published profiles, planner reads them, "
      "Api.list_cameras/describe_camera/add_camera/altitude_for_gsd."),
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
    F("fl.capture.match", "Captured images matched to planned capture points", "core", "Flight verification",
      "Every planned capture point is paired with at most one image, and every image with at "
      "most one point, within a stated match radius. A planned point with no image is reported "
      "by index rather than absorbed, and an image with no GPS is set aside rather than "
      "guessed at from filename order.",
      "implemented", ["tests/test_capture_matching.py::TestMatching",
                      "tests/test_capture_matching.py::TestGeotagging",
                      "tests/test_capture_matching.py::TestPlanExtraction"],
      "core/capture_matching.py; greedy nearest-first, one-to-one."),
    F("fl.capture.deviation", "Flight deviation from the plan is quantified", "core", "Flight verification",
      "Per-image and aggregate distance between the planned capture point and where the "
      "photograph was actually taken, with altitude difference where known, and an operator "
      "warning when the plan was flown but loosely.",
      "implemented", ["tests/test_capture_matching.py::TestDeviation",
                      "tests/test_capture_matching.py::TestReportShape"],
      "Warning threshold 8 m, match radius 15 m."),
    F("fl.capture.priors", "Planned poses seed reconstruction", "workers", "Flight verification",
      "Matched images carry their planned position, yaw and gimbal pitch into reconstruction "
      "as priors; unmatched images are left for SfM to place rather than given a position "
      "they were not observed at.",
      "implemented", ["tests/test_capture_matching.py::TestReconstructionPriors"],
      "pose_priors_for_reconstruction; consumed by the COLMAP engine."),
]

# ---------------------------------------------------------------------------
# Processing and reconstruction
# ---------------------------------------------------------------------------

PROCESSING = [
    F("pr.sfm", "Structure from motion", "workers", "Processing",
      "COLMAP SfM with bundle adjustment; reprojection error under 1.5 px on a real "
      "survey, with images that failed to register declared rather than hidden.",
      "implemented", ["tests/test_reconstruction_colmap.py::TestStructureFromMotion",
                      "tests/test_reconstruction_colmap.py::TestNoFabrication"],
      "Regression test runs the real engine on 8 Aukerman frames in ~15 s: "
      "0.81 px, 6/8 registered and warned about."),
    F("pr.georeference", "Georeferencing", "workers", "Processing",
      "RANSAC Helmert similarity between camera centres and geotags, into an auto-selected UTM zone.",
      "verified", ["tests/test_geo.py::TestUmeyamaSimilarity", "tests/test_geo.py::TestUtmZoneSelection"]),
    F("pr.ortho", "Orthomosaic generation", "workers", "Processing",
      "True orthophoto by DSM back-projection, written as a Cloud-Optimized GeoTIFF "
      "carrying the survey's CRS, not a PNG that would leave measurements in pixels.",
      "implemented", ["tests/test_reconstruction_colmap.py::TestGeoreferencing"],
      "COLMAP engine; CRS asserted against the auto-selected UTM zone."),
    F("pr.dsm_dtm", "DSM and DTM", "workers", "Processing",
      "Metric elevation rasters with a ground filter, in a stated CRS, coarsened openly "
      "when the cloud is too sparse to fill a finer grid rather than interpolated into "
      "detail that was never measured.",
      "implemented", ["tests/test_dsm_analysis.py",
                      "tests/test_reconstruction_colmap.py::TestGeoreferencing",
                      "tests/test_reconstruction_colmap.py::TestNoFabrication"]),
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
      "Every derived artifact records its source, engine, CRS and parameters in a "
      "sidecar that travels with the file, and the record can be checked against the "
      "artifact: a file modified since it was recorded is reported as such rather than "
      "passed off as attributed.",
      "implemented", ["tests/test_provenance.py::TestRecording",
                      "tests/test_provenance.py::TestVerification",
                      "tests/test_provenance.py::TestReconstructionOutputs",
                      "tests/test_provenance.py::TestAudit"],
      "core/provenance.py; sha256 sidecars, verified end to end on a real COLMAP run."),
]

# ---------------------------------------------------------------------------
# Vision / AI
# ---------------------------------------------------------------------------

VISION = [
    F("ai.registry", "AI model registry", "vision", "AI",
      "Register, version, deploy and roll back models, with metrics and a real checksum.",
      "implemented", [], "training/register.py; refuses an export that failed parity."),
    F("ai.parity", "ONNX export parity", "vision", "AI",
      "An export is rejected unless it matches torch within a tolerance scaled to each "
      "value's magnitude -- absolute at probability scale, relative at pixel-coordinate "
      "scale -- and loads in the runtime the pipeline actually uses. A graph computing "
      "something else must still fail the gate.",
      "implemented", ["tests/test_export_parity.py::TestAgreementPasses",
                      "tests/test_export_parity.py::TestDisagreementFails",
                      "tests/test_export_parity.py::TestScaling"],
      "training/export_onnx.py::parity_violation; two models pass at 0.08x tolerance."),
    F("ai.crack", "Crack detection", "vision", "AI",
      "A trained model measurably better than the classical baseline on a held-out split, "
      "installed with a real checksum and reported as the model actually used.",
      "implemented", ["tests/test_honesty.py::TestDetectionReportsWhatItActuallyUsed"],
      "SegFormer-B2 installed: test-split IoU 0.515 vs heuristic 0.045 (11x), precision 0.540 vs 0.046."),
    F("ai.spalling", "Spalling detection", "vision", "AI",
      "Trained detector for spallation with published validation metrics, used in "
      "preference to the heuristic, and reporting an empty result as an empty result "
      "rather than falling back to invented findings.",
      "implemented", ["tests/test_honesty.py::TestFindingNothingIsAnAnswer"],
      "YOLO11x on CODEBRIM, installed as structural_multiclass_detector: "
      "mAP50 0.417, mAP50-95 0.201; Spallation 0.306, ExposedBars 0.330, "
      "CorrosionStain 0.254, Efflorescence 0.193, Crack 0.124."),
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
# India-first survey intelligence roadmap
# ---------------------------------------------------------------------------

INDIA_FIRST = [
    # Shared engines
    F("eng.semantic", "Semantic understanding engine", "vision", "India: shared engines",
      "Versioned class schemas run through overlap-blended tiled inference and emit "
      "georeferenced class/confidence rasters, polygons and model provenance.",
      "in_progress", ["tests/test_semantic_engine.py"],
      "Runtime and DINOv2/UPerNet architecture are implemented; the shared production "
      "head still needs licence-filtered training and site/date holdout evaluation."),
    F("eng.assets", "Object and asset detection engine", "vision", "India: shared engines",
      "Count and individually locate assets such as trees, modules, equipment, poles "
      "and insulators with model provenance and confidence.",
      "in_progress", [],
      "Defect-specific detection exists; a shared asset taxonomy and geospatial output do not."),
    F("eng.change", "Survey change intelligence engine", "core", "India: shared engines",
      "Aligned T1/T2 DSMs produce a georeferenced difference raster, contiguous change "
      "polygons, exact rise/fall volumes and an interpretation-safe report.",
      "implemented", ["tests/test_survey_intelligence.py::TestSurfaceChangePackage"]),
    F("eng.anomaly", "Anomaly intelligence engine", "vision", "India: shared engines",
      "Find deviations from a validated normal baseline without assigning an unsupported "
      "named defect class.",
      "not_started", []),

    # Photogrammetry and geometry foundation
    F("india.foundation.orthomosaic", "Survey orthomosaic", "workers", "India: foundation",
      "A georeferenced site orthomosaic suitable for GIS and downstream analysis.",
      "in_progress", [], "Writer and reconstruction path exist; end-to-end survey evidence is not automated."),
    F("india.foundation.dsm_dtm", "Survey DSM and DTM", "workers", "India: foundation",
      "Metric surface and terrain rasters in an explicit projected CRS.",
      "in_progress", [], "Reconstruction path exists; end-to-end survey evidence is not automated."),
    F("india.foundation.reconstruction", "Survey 3D reconstruction", "workers", "India: foundation",
      "Georeferenced point cloud and model from a real drone survey.",
      "in_progress", [], "Verified manually on Aukerman; the registry requires an automated real-input test."),
    F("india.foundation.area", "Survey area and distance", "core", "India: foundation",
      "Metric area and distance measured from georeferenced survey products.",
      "in_progress", [], "Core measurement functions exist; the mapped client workflow is incomplete."),
    F("india.foundation.volume", "Stockpile volume", "core", "India: foundation",
      "Volume against an explicit reference surface, exact on a known metric DSM/DTM.",
      "implemented", ["tests/test_dsm_analysis.py::test_volume_against_dtm_is_exact",
                      "tests/test_dsm_analysis.py::test_polygon_clip_halves_the_volume"]),
    F("india.foundation.cut_fill", "Cut and fill", "core", "India: foundation",
      "Measured surface rise and fall between aligned surveys with area and volume in "
      "metric units.",
      "implemented", ["tests/test_change_detection.py::TestSurfaceComparison",
                      "tests/test_survey_intelligence.py::TestSurfaceChangePackage"]),
    F("india.foundation.deliverable", "Georeferenced client deliverable", "core", "India: foundation",
      "A customer receives mapped vectors/rasters, quantities, method and honest limits "
      "rather than an isolated model prediction.",
      "in_progress", ["tests/test_survey_intelligence.py::TestSurfaceChangePackage"],
      "Complete for surface-change comparisons; each remaining mission pack needs its own deliverable."),

    # Construction
    F("pack.construction.change", "Construction survey comparison", "core",
      "India pack: Construction",
      "Two survey DSMs produce mapped rise/fall regions, changed area, added/removed "
      "volume and a client-readable report without inventing a semantic cause.",
      "implemented", ["tests/test_survey_intelligence.py"]),
    F("pack.construction.segmentation", "Construction site segmentation", "vision",
      "India pack: Construction",
      "Building, unfinished building, road, bare soil, vegetation, water, concrete, "
      "excavation, stockpile, construction material and equipment classes.",
      "in_progress", ["tests/test_india_construction_pack.py::test_construction_schema_covers_the_registry_contract"],
      "Versioned schema and task-trained runtime gate exist; no production construction head is trained."),
    F("pack.construction.progress", "Progress against approved design", "core",
      "India pack: Construction",
      "Measured percentage and location of progress against an explicit approved design model.",
      "implemented", ["tests/test_india_construction_pack.py::test_approved_design_progress_measures_observed_surface_not_contract_completion"]),

    # Mining and quarry
    F("pack.mining.stockpile", "Stockpile selection and measurement", "core",
      "India pack: Mining",
      "Select or segment a pile, state the base-surface method, calculate volume and "
      "produce a mapped client result.",
      "implemented", ["tests/test_stockpile_intelligence.py"]),
    F("pack.mining.stockpile_segmentation", "Automatic stockpile segmentation", "vision",
      "India pack: Mining",
      "Automatically propose georeferenced stockpile boundaries for human review before "
      "the geometry engine measures them.",
      "not_started", []),
    F("pack.mining.scene", "Mine and quarry segmentation", "vision",
      "India pack: Mining",
      "Pit, bench, haul road, stockpile, water, vegetation, excavated region and "
      "restricted-boundary layers.",
      "not_started", []),
    F("pack.mining.change", "Mine and stockpile change", "core",
      "India pack: Mining",
      "Per-pile and per-pit area/volume change between dated surveys.",
      "implemented", ["tests/test_survey_intelligence.py::TestSelectedROIChangePackage",
                      "tests/test_survey_intelligence.py::TestSelectedROIChangeWorkflow"]),

    # Solar
    F("pack.solar.inventory", "Solar array and module inventory", "vision",
      "India pack: Solar",
      "Geolocated arrays and individual modules with missing/damaged/obstructed module findings.",
      "implemented", ["tests/test_india_assets_pack.py::test_solar_inventory_counts_modules_and_only_calls_layout_gaps_missing"]),
    F("pack.solar.thermal", "Solar RGB and thermal inspection", "vision",
      "India pack: Solar",
      "Aligned RGB/radiometric thermal imagery with module-level hotspots, hot cells, "
      "string/module anomalies and temperature-backed severity.",
      "in_progress", ["tests/test_thermal.py"],
      "Radiometric conversion and single-frame mapping work; alignment and module association do not."),

    # Land and property
    F("pack.land.gis", "Land-survey GIS extraction", "vision",
      "India pack: Land",
      "Georeferenced building, road/path, water and vegetation polygons extracted from "
      "an orthomosaic.",
      "implemented", ["tests/test_india_land_pack.py::test_land_gis_extracts_real_georeferenced_semantic_classes",
                      "tests/test_india_land_pack.py::test_metric_land_analysis_refuses_unreferenced_raster"]),
    F("pack.land.encroachment", "Property and encroachment change", "core",
      "India pack: Land",
      "New buildings, boundary encroachment, new roads and structure expansion between surveys.",
      "implemented", ["tests/test_india_land_pack.py::test_encroachment_uses_imported_boundary_and_aligned_previous_survey"]),

    # Agriculture
    F("pack.agriculture.canopy", "Crop canopy segmentation", "vision",
      "India pack: Agriculture",
      "Crop, soil, unwanted vegetation and water masks with canopy cover and bare/missing regions.",
      "implemented", ["tests/test_india_agriculture_pack.py::test_canopy_cover_uses_real_semantic_raster"]),
    F("pack.agriculture.indices", "Multispectral crop indices", "core",
      "India pack: Agriculture",
      "NDVI, NDRE and GNDVI from calibrated bands, reported as spectral indices rather than AI.",
      "implemented", ["tests/test_india_agriculture_pack.py::test_indices_use_calibrated_bands_and_mark_missing_band_unavailable"]),
    F("pack.agriculture.stress", "Crop stress and anomaly map", "vision",
      "India pack: Agriculture",
      "Georeferenced stress/anomaly regions with the sensor, crop and validation scope stated.",
      "implemented", ["tests/test_india_agriculture_pack.py::test_stress_zones_require_and_preserve_crop_sensor_scope"]),
    F("pack.agriculture.count", "Plant and tree counting", "vision",
      "India pack: Agriculture",
      "Geolocated plant/tree counts with missing and health categories only where validated.",
      "implemented", ["tests/test_india_agriculture_pack.py::test_plant_count_counts_connected_instances_without_inventing_missing_or_health"]),

    # Roads, power and rail
    F("pack.roads.condition", "Mapped road condition", "vision", "India pack: Roads",
      "Road/edge segmentation plus geolocated pothole, crack, waterlogging and debris "
      "counts, severity and surveyed distance.",
      "implemented", ["tests/test_india_assets_pack.py::test_road_condition_uses_explicit_centerline_for_metric_distance"]),
    F("pack.power.inspection", "Power-line asset inspection", "vision",
      "India pack: Power and rail",
      "Close-range, geolocated towers/poles, crossarms, insulators, conductors, "
      "transformers, vegetation and validated component findings.",
      "implemented", ["tests/test_india_assets_pack.py::test_power_and_rail_enforce_capture_geometry_and_map_real_vectors"]),
    F("pack.rail.inspection", "Railway corridor inspection", "vision",
      "India pack: Power and rail",
      "Mapped railway track, bridge, overhead equipment and corridor findings from "
      "capture geometry appropriate to the component.",
      "implemented", ["tests/test_india_assets_pack.py::test_power_and_rail_enforce_capture_geometry_and_map_real_vectors"]),
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
      "Distance, area and perimeter directly on an orthomosaic, taking the scale from "
      "the raster's own transform rather than an operator-typed value, and refusing a "
      "raster with no CRS instead of reporting pixels as metres.",
      "implemented", ["tests/test_raster_measurement.py::TestDistance",
                      "tests/test_raster_measurement.py::TestArea",
                      "tests/test_raster_measurement.py::TestPerimeter",
                      "tests/test_raster_measurement.py::TestRefusals",
                      "tests/test_raster_measurement.py::TestHonestReporting"],
      "core/raster_measurement.py; Api.measure_on_raster."),
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
      "implemented", [
          "tests/test_hub_web.py::TestHubPanels::test_viewer_module_parses_real_scene_cloud_twin_and_map_sources",
          "tests/test_hub_web.py::TestHubRealBrowser::test_2d_viewer_loads_real_geojson_layers_and_measurement_tool",
      ], "MapLibre standard sources, local cache, XYZ/WMS/WMTS adapters and vector overlays."),
    F("hub.map.geocoding", "Address search", "hub", "Hub",
      "Search an address, place name or typed coordinates and fly the map to the result, "
      "through a replaceable provider that can be self-hosted or fully offline.",
      "implemented", ["tests/test_geocoding.py"],
      "Nominatim (self-hostable) and offline providers; a typed coordinate never leaves the machine."),
    F("hub.map.offline_tiles", "Offline tile cache", "hub", "Hub",
      "Tiles actually downloaded and served without connectivity.",
      "implemented", ["tests/test_hub_web.py::TestHubOfflineTiles",
                      "tests/test_hub_web.py::TestHubRestClient"],
      "The UI reports real cache progress; evidence downloads from a local XYZ server, stops it, then serves the disk copy."),
    F("hub.projects", "Project management", "hub", "Hub",
      "Projects with client, site, type, tags, members, history and activity timeline.",
      "in_progress", ["tests/test_hub_web.py::TestHubPanels",
                      "tests/test_hub_web.py::TestHubRestClient",
                      "tests/test_api.py::TestAudit"],
      "Persistent project fields and activity are exposed; membership is still organization-wide rather than project-scoped."),
    F("hub.assets", "Asset management", "hub", "Hub",
      "Persistent assets with geometry, inspection history and a timeline.",
      "in_progress", ["tests/test_api.py::TestAssets",
                      "tests/test_hub_web.py::TestHubPanels",
                      "tests/test_hub_web.py::TestHubRestClient"],
      "Persistent asset geometry is exposed; inspection records are not yet linked to an asset timeline."),
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
      "implemented", [
          "tests/test_hub_web.py::TestHubRealBrowser::test_2d_viewer_loads_real_geojson_layers_and_measurement_tool",
          "tests/test_hub_web.py::TestHubPanels",
      ], "MapLibre layer comparison plus distance, area and annotation tools."),
    F("hub.viewer_3d", "3D viewer", "hub", "Hub",
      "Orbit, fly navigation, WASDQE keys, clipping and overlays.",
      "implemented", [
          "tests/test_hub_web.py::TestHubRealBrowser::test_webgl_scene_and_progressive_point_chunks_render_in_edge",
          "tests/test_hub_web.py::TestHubPanels",
      ], "Local WebGL2 scene viewer with orbit, wheel/fly controls, clipping and overlay descriptors."),
    F("hub.point_cloud", "Point cloud viewer", "hub", "Hub",
      "Large point clouds streamed progressively.",
      "implemented", [
          "tests/test_hub_web.py::TestHubRealBrowser::test_webgl_scene_and_progressive_point_chunks_render_in_edge",
          "tests/test_hub_web.py::TestHubPanels::test_viewer_module_parses_real_scene_cloud_twin_and_map_sources",
      ], "Manifest chunks are fetched and rendered sequentially with visible progress."),
    F("hub.digital_twin", "Digital twin", "hub", "Hub",
      "Model, cloud, ortho, images, annotations, defects and history as one navigable object.",
      "implemented", [
          "tests/test_hub_web.py::TestHubPanels::test_viewer_module_parses_real_scene_cloud_twin_and_map_sources",
          "tests/test_hub_web.py::TestHubPanels::test_hub_exposes_every_registry_panel_and_local_viewer_script",
      ], "The local digital_twin.json loader indexes artifacts, surveys, annotations, defects and measured-change history."),
    F("hub.timeline", "Time based comparison / 4D", "hub", "Hub",
      "Elevation difference and volume added/removed between two surveys, refusing to "
      "compare surveys in different coordinate systems or at different resolutions "
      "rather than resampling them into apparent agreement.",
      "implemented", ["tests/test_change_detection.py::TestSurfaceComparison",
                      "tests/test_change_detection.py::TestIncomparableSurveys",
                      "tests/test_hub_web.py::TestHubPanels"],
      "Core comparison and a dated digital-twin survey timeline are available."),
    F("rp.reports", "Automated inspection reports", "hub", "Reporting",
      "Structured report with findings, images, measurements and AI results, generated "
      "from real project data, listed and persisted per project, refused outright when "
      "the project is missing, and reporting an absent defect run as absent rather than "
      "as zero defects found.",
      "implemented", ["tests/test_report_engine.py::TestReadiness",
                      "tests/test_report_engine.py::TestGeneration",
                      "tests/test_report_engine.py::TestContent"],
      "core/report_engine.py; each report gets its own directory and a manifest."),
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
      "Signed HMAC-SHA256 deliveries with the timestamp inside the signed material, "
      "subscriptions refused for unknown events, and every delivery recorded including "
      "failures. Best-effort delivery is declared rather than implied otherwise.",
      "implemented", ["tests/test_events.py::TestSubscription",
                      "tests/test_events.py::TestDelivery",
                      "tests/test_events.py::TestHonestyAboutGuarantees"]),
    F("api.realtime", "Real time communication", "hub", "API",
      "Per-organization WebSocket stream with the token verified as a normal request "
      "would be, declaring that in-process fan-out only reaches clients on the same "
      "worker.",
      "in_progress", ["tests/test_events.py::TestLiveStream"],
      "Authenticated stream works; a shared broker is needed for multi-worker "
      "deployments and telemetry is not yet published into it."),
    F("fm.fleet", "Fleet management", "hub", "Fleet",
      "Airframes with accumulating flight hours and a service interval that reports "
      "when maintenance is due, plus a fleet status view answering what needs "
      "attention before the next job.",
      "implemented", ["tests/test_fleet.py::TestAircraftService",
                      "tests/test_fleet.py::TestFleetStatus"]),
    F("fm.battery", "Battery management", "hub", "Fleet",
      "Cycle count against a per-battery limit, measured health, and retirement, each "
      "surfaced as a warning rather than left for the reader to compute.",
      "implemented", ["tests/test_fleet.py::TestBatteries"]),
    F("fm.pilots", "Pilot management", "hub", "Fleet",
      "Licence and medical expiry with currency computed, warning a month ahead so a "
      "lapse is not discovered on the morning of a job.",
      "implemented", ["tests/test_fleet.py::TestPilotCurrency"]),
    F("fm.maintenance", "Maintenance records", "hub", "Fleet",
      "Service history per airframe, where recording a service resets the interval "
      "from the hours at which it happened without altering the airframe's total.",
      "implemented", ["tests/test_fleet.py::TestAircraftService::test_recording_maintenance_resets_the_interval",
                      "tests/test_fleet.py::TestAircraftService::test_maintenance_history_is_kept"]),
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
      "implemented", ["tests/test_sdk.py::TestPluginSystem"],
      "Versioned direct, manifest and opt-in entry-point registration cover all nine documented plugin kinds."),
    F("sdk.libraries", "Developer SDK", "sdk", "SDK",
      "Libraries for mission generation, drone adapters, telemetry and job APIs.",
      "implemented", ["tests/test_sdk.py::TestDeveloperLibraries"],
      "The SDK delegates planning to the shared core, validates real drone adapters and provides authenticated REST job helpers."),
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
    *INDIA_FIRST,
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
