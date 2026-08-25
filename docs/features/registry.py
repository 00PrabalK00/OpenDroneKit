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
      "verified", ["tests/test_mapping_3d_oblique.py::TestTheBandsAreOblique",
                   "tests/test_mapping_3d_oblique.py::TestOnlyAskedForMissionsPayForIt",
                   "tests/test_mapping_3d_oblique.py::TestTheCostStaysProportionate",
                   "tests/test_mapping_3d_oblique.py::TestRefusals"],
      "mapping_3d compiles as a double grid plus oblique perimeter rings at -45 and -60 "
      "degrees, standing off by altitude/tan(tilt) so the camera looks across the site "
      "rather than into it. The bands are opt-in: a plain double_grid stays nadir and "
      "does not grow. Cost on a 400 m site is 64.5 to 73.1 minutes. "
      "The bands first compiled as a SECOND FULL NADIR GRID -- 6,120 poses instead of "
      "48, tripling the mission to 179 minutes -- because the primitive dispatcher "
      "normalises its kind through the template alias table, which maps anything "
      "unrecognised to grid rather than failing. Primitive kinds that are not templates "
      "are now matched before normalisation, and a test asserts the pose count."),
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
      "Balconies, recesses, courtyards, overhangs and non-rectangular footprints, with "
      "occlusion reported rather than silently omitted.",
      "verified", ["tests/test_footprints.py",
                   "tests/test_api_measurements.py::TestComplexFacadeCapability"],
      "A courtyard INVERTS the standoff: outside the building the aircraft offsets "
      "outward, inside a courtyard it offsets inward, and a planner treating them alike "
      "flies into masonry. A standoff wider than half the courtyard is refused rather "
      "than planned. Overhangs are assessed against the lens: wall hidden under a "
      "balcony is never photographed and the reconstruction renders it as smooth "
      "surface rather than a hole, so the gap is invisible in the deliverable unless "
      "reported. Counterintuitively, flying CLOSER sees less under a projection, which "
      "a test pins."),
    F("mp.type.closed_loop", "Closed-loop structure capture", "core", "Mission types",
      "Continuous loops at multiple radii and altitude rings, either direction, with "
      "the camera always facing the structure.",
      "implemented", ["tests/test_mission_types.py::TestClosedLoop"]),
    F("mp.type.l_shaped", "L-shaped and irregular buildings", "core", "Mission types",
      "Facade planning maintains coverage on non-rectangular footprints. Reflex corners "
      "are reported rather than smoothed, and any pass falling inside the building is "
      "dropped.",
      "verified", ["tests/test_footprints.py",
                   "tests/test_api_measurements.py::TestIrregularFacadeCapability"],
      "A naive offset ring folds back THROUGH the structure at a concave corner, "
      "silently: the mission uploads and the aircraft flies it. Passes are built per "
      "wall and checked against the footprint by ray casting."),
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
      "Body, crossarm, insulator and conductor capture at multiple elevations, each "
      "flown at the height that element sits at with a gimbal angle framing it rather "
      "than a nadir pass, and a structure whose element heights were not supplied "
      "refused rather than approximated near energised conductors.",
      "implemented", ["tests/test_special_mission_types.py::TestPylonInspection"],
      "mission/mission_types.py; plan_pylon_inspection compiles one real orbit per "
      "named element through the mission engine."),
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
      "Thermal mapping and inspection with paired RGB capture, flown at the altitude "
      "the thermal sensor needs for the requested GSD rather than the RGB one, with a "
      "non-radiometric camera refused instead of producing ordinary photographs "
      "labelled as a thermal survey.",
      "implemented", ["tests/test_special_mission_types.py::TestThermalMission"],
      "mission/mission_types.py; plan_thermal_mission."),
    F("mp.type.multispectral", "Multispectral missions", "core", "Mission types",
      "Synchronised band capture for agricultural and vegetation survey, with the band "
      "centres carried into the mission and reflectance-panel captures planned before "
      "and after the flight, since indices from a flight without them cannot be "
      "compared with any other survey.",
      "implemented", ["tests/test_special_mission_types.py::TestMultispectralMission"],
      "mission/mission_types.py; plan_multispectral_mission, reading the band set from "
      "the payload database."),
    F("mp.linking", "Mission linking", "core", "Mission engine",
      "Several mission types execute as one sequence with per-segment completion "
      "tracking: each capture point is attributed to the survey that produced it from "
      "the stamp the compiler wrote at link time, a segment counts as complete only "
      "when every one of its own points matched an image, completed segments are not "
      "re-flown, and an attribution that cannot be trusted -- an unstamped pose, or a "
      "declared segment count disagreeing with the poses -- is refused rather than "
      "reported partially.",
      "implemented", ["tests/test_linking.py::TestAttribution",
                      "tests/test_linking.py::TestProgress",
                      "tests/test_linking.py::TestResumingASortie",
                      "tests/test_linking.py::TestRefusals"],
      "mission/linking.py; Api.linked_mission_progress. Completion comes from the same "
      "image matcher the single-mission resume uses, so an ambiguous point is re-flown "
      "rather than assumed done."),
    F("mp.geometry_3d", "3D geometry based planning", "core", "Mission engine",
      "Paths generated from imported OBJ/GLB/PLY/LAS/IFC surfaces: each capture point "
      "stands off a face along that face's own normal and looks back at it, so a wall "
      "is photographed rather than the roof above it; an unscaled surface is refused "
      "because a stand-off in structure-from-motion units flies the wrong distance from "
      "the structure; and a point cloud is refused because it has no normal to stand "
      "off along.",
      "verified", ["tests/test_geometry_3d.py::TestReadingSurfaces",
                   "tests/test_geometry_3d.py::TestStandOffGeometry",
                   "tests/test_geometry_3d.py::TestFiltering",
                   "tests/test_geometry_3d.py::TestRefusals",
                   "tests/test_las_surface.py::TestReadingTheFormat",
                   "tests/test_las_surface.py::TestRefusals",
                   "tests/test_las_surface.py::TestPlanningFromACloud"],
      "mission/geometry_3d.py plans from OBJ, ASCII PLY, GLB/glTF and now LAS. LAS is "
      "read directly rather than through a package, because a survey should not need a "
      "package index to open a cloud already on its disk, and the tests build the files "
      "byte by byte -- a reader tested only against its own writer agrees with itself "
      "about a format it may have misunderstood. "
      "The old refusal said a cloud has no normals to stand off along. That was right "
      "about the physics and wrong about the conclusion: orientation is recovered from "
      "the data by meshing rather than assumed, so a cloud now plans. "
      "STILL REFUSED, deliberately: LAZ needs laszip vendored or implemented, and is a "
      "conversion away; IFC is a building-information schema whose walls are "
      "parameterised objects, so producing triangles from it is a modelling decision "
      "rather than a file conversion. Both say so and say what to do instead."),
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
      "verified", ["tests/test_terrain_follow_warnings.py::TestNoTerrainLoaded",
                   "tests/test_terrain_follow_warnings.py::TestUnreadableTerrainSource",
                   "tests/test_terrain_follow_warnings.py::TestTerrainActuallyFollowed"],
      "Tests plan real missions through the Api and assert on the returned warnings, "
      "because the warning had already failed silently once: the resolved terrain model "
      "lives under the recipe's metadata and reading it from the root returned None."),
    F("mp.terrain_offline", "Offline terrain cache", "core", "Mission engine",
      "Terrain cached per project so terrain following works with no connectivity, with "
      "the extent recorded so an area the cache does not fully contain is reported as "
      "uncovered rather than quietly planned flat, and nothing cached, cached elsewhere "
      "and cached-but-not-reaching distinguished.",
      "implemented", ["tests/test_terrain_cache.py::TestReadingRasters",
                      "tests/test_terrain_cache.py::TestCaching",
                      "tests/test_terrain_cache.py::TestCoverage",
                      "tests/test_terrain_cache.py::TestMissingFiles",
                      "tests/test_terrain_cache.py::TestUnreadableIndex",
                      "tests/test_terrain_cache.py::TestSourceCoversTheArea",
                      "tests/test_terrain_cache.py::TestBounds",
                      "tests/test_terrain_source_coverage.py::TestTerrainSourceIsCheckedAgainstTheArea",
                      "tests/test_terrain_source_coverage.py::TestChoosingTheSourceAnswersImmediately"],
      "core/terrain_cache.py; Api.cache_terrain, terrain_coverage, "
      "describe_terrain_cache. Planning consults the chosen source: a DEM that stops "
      "short of the area, or belongs to another site, is named at plan time instead of "
      "silently degrading to flat earth. An unreadable index reports not knowing rather "
      "than an empty cache."),
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
      "implemented", ["tests/test_mission_hub.py::TestCompiledMissionSimulation",
                      "tests/test_mission_hub.py::TestMissionHubBrowser",
                      "tests/test_mission_hub.py::TestMissionHubWiring"],
      "Playback reads the persisted compiled plan. Missing terrain remains unavailable "
      "with no synthetic flat surface; battery is labelled as an operator-input estimate."),
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
      "implemented", ["tests/test_mission_hub.py::TestMissionPreviewSharing"],
      "Reuses the verified password, expiry, revocation and access-log ShareLink path; "
      "preview fields are derived from the persisted compiled plan."),
    F("mp.repeatable", "Repeatable missions", "core", "Mission engine",
      "Repeat exactly, with updated terrain, a modified boundary, or a different "
      "aircraft, where a camera change adjusts altitude to hold ground resolution "
      "rather than holding altitude, and every repeat states whether it can honestly be "
      "compared against the original.",
      "implemented", ["tests/test_repeat.py::TestExactRepeat",
                      "tests/test_repeat.py::TestDifferentAircraft",
                      "tests/test_repeat.py::TestUpdatedTerrain",
                      "tests/test_repeat.py::TestModifiedBoundary",
                      "tests/test_repeat.py::TestComparability",
                      "tests/test_repeat.py::TestAgainstARealPlan"],
      "mission/repeat.py; Api.repeat_mission and compare_survey_specifications."),
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
      "Boundary defined by flying the aircraft to mark positions, with a corner marked "
      "without a 3D fix refused rather than recorded as a position the receiver itself "
      "does not trust, a double press refused rather than becoming an edge, a crossed "
      "outline refused even though its area still computes, and the weakest corner's fix "
      "quality reported rather than an average.",
      "implemented", ["tests/test_fly_to_draw.py::TestMarking",
                      "tests/test_fly_to_draw.py::TestBuildingTheBoundary",
                      "tests/test_fly_to_draw.py::TestRefusals",
                      "tests/test_fly_to_draw.py::TestThroughTheApi"],
      "mission/fly_to_draw.py; Api.mark_boundary_corner, boundary_from_marks, "
      "clear_boundary_marks. The result never claims to be a surveyed boundary."),
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
      "RGB, thermal, multispectral, LiDAR, magnetometer and custom payload commands, "
      "with a command the instrument does not declare refused rather than exported, and "
      "an undescribed payload refused rather than planned as a generic camera.",
      "implemented", ["tests/test_payloads.py::TestTheDatabaseItself",
                      "tests/test_payloads.py::TestRefusals",
                      "tests/test_payloads.py::TestCaptureCommand",
                      "tests/test_payloads.py::TestPlanNotes",
                      "tests/test_payloads.py::TestOperatorPayloads",
                      "tests/test_payloads.py::TestPlanningWithAPayload"],
      "mission/payloads.py; Api.list_payloads/describe_payload/add_payload and the "
      "payload block on Api.plan_mission. Streaming payloads start a run where framing "
      "ones trigger; mass and power are recorded as unknown rather than invented, so no "
      "endurance penalty is claimed from a guessed figure."),
]

# ---------------------------------------------------------------------------
# Flight execution
# ---------------------------------------------------------------------------

FLIGHT = [
    F("fl.abstraction", "Drone abstraction layer", "core", "Flight",
      "A generic interface with adapters; no core code depends on one manufacturer, "
      "enforced by reading the source rather than by convention, and the mission engine "
      "imports no drone SDK at all.",
      "implemented", ["tests/test_drone_abstraction.py::TestLayering",
                      "tests/test_drone_abstraction.py::TestProtocolConformance",
                      "tests/test_drone_abstraction.py::TestFactory",
                      "tests/test_drone_abstraction.py::TestHonestyAtTheBoundary",
                      "tests/test_drone_abstraction.py::TestMockBehaviour"],
      "Three drivers: mock, MAVSDK, pymavlink. pymavlink is confined to "
      "mission_planner_bridge plus one declared capability probe."),
    F("fl.mavlink.upload", "MAVLink mission upload", "core", "Flight",
      "Request/ack transfer protocol; gimbal, yaw, dwell and trigger items survive a round trip.",
      "verified", ["tests/test_mavlink_transfer.py::test_mission_upload_is_acknowledged",
                   "tests/test_mavlink_transfer.py::test_capture_commands_reach_the_vehicle",
                   "tests/test_mavlink_transfer.py::test_download_round_trips_the_uploaded_mission"]),
    F("fl.mavlink.fence", "Geofence and rally upload", "core", "Flight",
      "Fence and rally land in their own MAV_MISSION_TYPE without overwriting the flight plan.",
      "verified", ["tests/test_mavlink_transfer.py::test_each_list_lands_in_its_own_slot"]),
    F("fl.sitl", "SITL verified flight", "core", "Flight",
      "The flight path exercised against a real ArduPilot autopilot rather than a mock.",
      "verified", ["tests/sitl/test_flight_lifecycle.py",
                   "tests/sitl/test_mission_upload.py"],
      "VERIFIED IN CI, AND ONLY THERE -- earned by run 32853156668, whose junit report "
      "records 2 tests, 0 skipped, 0 failures against ArduPilot Copter-4.5.7. "
      "Both tests go green against ArduPilot "
      "Copter-4.5.7 in infrastructure/docker/Dockerfile.sitl -- and they SKIP under a "
      "plain pytest, because SITL needs the container. Status is computed from passing "
      "tests and a skip is not a pass, so this row reads implemented rather than "
      "verified on any laptop no matter how many times the container succeeds. That is "
      "the honest reading, not a bookkeeping problem to argue away. "
      "The CI status job closes the gap without weakening it: the sitl job publishes the "
      "container's junit report and tools/feature_status.py merges it with "
      "--extra-report, so the row is earned by a flight run that actually happened "
      "rather than by anyone's belief about what the container would do. CI FAILS if "
      "those tests skipped rather than ran, and fails again if the report is missing or "
      "empty -- a green tick over a skip would hide this permanently. "
      "Worth recording what SITL already caught, since it is the argument for the whole "
      "harness: our missions put NAV_TAKEOFF at sequence 0, which MAVLink reserves for "
      "home, so ArduPilot silently overwrote it and the aircraft would never have taken "
      "off. Every mock-based test passed throughout -- a mock stores what it is given, "
      "and only a real autopilot has an opinion about sequence 0."),
    F("fl.preflight", "Preflight checks", "app", "Flight",
      "Connection, battery, GPS, compass, IMU, home point, storage, camera, gimbal and "
      "mission-vs-geofence conflicts checked before arming, with a sensor the vehicle "
      "has not reported treated as unestablished rather than healthy.",
      "implemented", ["tests/test_preflight.py::TestCompass",
                      "tests/test_preflight.py::TestImu",
                      "tests/test_preflight.py::TestGimbal",
                      "tests/test_preflight.py::TestGps",
                      "tests/test_preflight.py::TestBattery",
                      "tests/test_preflight.py::TestConnection",
                      "tests/test_preflight.py::TestWholeReport"],
      "core/preflight.py; compass and IMU read the SYS_STATUS sensor health bitmask, "
      "which the bridge now keeps instead of discarding."),
    F("fl.telemetry", "Live telemetry", "core", "Flight",
      "Position, battery, GPS, RC link and mission progress streamed to subscribers.",
      "verified", ["tests/test_telemetry_subscribers.py::TestDelivery",
                   "tests/test_telemetry_subscribers.py::TestOneBadSubscriberCannotStopTheFeed",
                   "tests/test_telemetry_subscribers.py::TestTelemetrySnapshot"],
      "Callbacks run on the MAVLink listener thread, so a subscriber that raises must "
      "not take the feed down: a dead listener looks like a quiet aircraft, not a fault."),
    F("fl.manual_override", "Manual override", "app", "Flight",
      "The pilot can interrupt autonomy at any time, and the control state is displayed, "
      "with a mode change confirmed against the vehicle's own heartbeat rather than "
      "against having sent the command, and an unrecognised mode treated as autonomous.",
      "implemented", ["tests/test_flight_control.py::TestClassification",
                      "tests/test_flight_control.py::TestControlState",
                      "tests/test_flight_control.py::TestTakeManualControl",
                      "tests/test_flight_control.py::TestCommandedIsNotConfirmed"],
      "core/flight_control.py; Api.control_state and take_manual_control. The bridge "
      "now verifies mode changes and resolves custom_mode to a name."),
    F("fl.battery_swap", "Battery swap and resume", "app", "Flight",
      "Completed segments recorded; resume continues without duplicate capture and "
      "without dropping a point, with what was flown determined from the imagery on the "
      "card rather than a progress counter, and every ambiguous case re-flown.",
      "implemented", ["tests/test_resume.py::TestStateFromSegments",
                      "tests/test_resume.py::TestStateFromImages",
                      "tests/test_resume.py::TestResumePlan",
                      "tests/test_resume.py::TestBatterySegments",
                      "tests/test_resume.py::TestAgainstARealPlan"],
      "mission/resume.py; Api.plan_battery_segments and resume_from_images."),
    F("fl.crash_recovery", "Crash recovery", "app", "Flight",
      "Mission state and telemetry persist across a crash of the ground station, and "
      "the app never silently restarts a mission: recovery reports what was recorded, "
      "states that the aircraft may still be airborne, and leaves the decision to the "
      "operator.",
      "implemented", ["tests/test_flight_state.py::TestPersistence",
                      "tests/test_flight_state.py::TestCleanShutdown",
                      "tests/test_flight_state.py::TestRecovery",
                      "tests/test_flight_state.py::TestCorruptState"],
      "core/flight_state.py; atomic writes, Api.check_interrupted_flight. A test asserts "
      "recover() calls no command that could re-fly a mission."),
    F("fl.camera_control", "Camera control", "app", "Flight",
      "Photo, video, mode, focus and zoom over the standard MAVLink camera commands, "
      "gated on the capabilities the payload declares, with exposure settings that are "
      "extended parameters refused with the reason rather than transmitted into silence "
      "and reported as success.",
      "implemented", ["tests/test_camera_control.py::TestCapture",
                      "tests/test_camera_control.py::TestModes",
                      "tests/test_camera_control.py::TestZoomAndFocus",
                      "tests/test_camera_control.py::TestExtendedSettings",
                      "tests/test_camera_control.py::TestUndeclaredCapabilities",
                      "tests/test_camera_control.py::TestCapabilityParsing",
                      "tests/test_camera_control.py::TestTransportFailures"],
      "core/camera_control.py; the bridge keeps CAMERA_INFORMATION and exposes camera(). "
      "ISO, shutter and white balance are not standard MAVLink commands and are not "
      "claimed as implemented."),
    F("fl.gimbal_control", "Gimbal control", "core", "Flight",
      "Pitch, yaw, centre and look-at commands issued as mission items.",
      "verified", ["tests/test_exporters.py::test_gimbal_pitch_and_mount_mode_land_in_the_right_slots"]),
    F("fl.logging", "Flight logging", "app", "Flight",
      "Full telemetry log exported to CSV, JSON, GPX and KML, with samples that had no "
      "GPS fix kept in the record but omitted from the tracks, and gaps left as gaps "
      "rather than interpolated into a flight nobody observed.",
      "implemented", ["tests/test_flight_log.py::TestRecording",
                      "tests/test_flight_log.py::TestPositionValidity",
                      "tests/test_flight_log.py::TestCsv",
                      "tests/test_flight_log.py::TestJson",
                      "tests/test_flight_log.py::TestGpx",
                      "tests/test_flight_log.py::TestKml",
                      "tests/test_flight_log.py::TestExportDispatch",
                      "tests/test_flight_log.py::TestEmptyAndDegenerate"],
      "core/flight_log.py; AppSession records every telemetry read, "
      "Api.export_flight_log writes it out."),
    F("fl.data_verification", "On-site data verification", "app", "Flight",
      "Image count, blur, exposure, corrupt files, missing GPS and coverage gaps "
      "checked before leaving the site, with a verdict that distinguishes what blocks "
      "departure from what merely warrants knowing, and reports anything it could not "
      "check as unchecked rather than as passed.",
      "implemented", ["tests/test_site_verification.py::TestUnreadableFiles",
                      "tests/test_site_verification.py::TestQuality",
                      "tests/test_site_verification.py::TestVerdict",
                      "tests/test_site_verification.py::TestUnchecked",
                      "tests/test_site_verification.py::TestUngeotagged"],
      "core/site_verification.py joins capture_matching and coverage_validation; "
      "Api.verify_site."),
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
      "Patch-match multi-view stereo. Never synthesised from the sparse cloud, and the "
      "environment is asked before a job starts rather than discovered mid-run.",
      "implemented", ["tests/test_honesty.py::TestNoSyntheticDensification",
                      "tests/test_api_measurements.py::TestReconstructionCapabilities"],
      "The densification fake -- cloning sparse points with Gaussian jitter and calling "
      "it MVS -- was removed and a test guards its absence. api.reconstruction_"
      "capabilities() now reports dense_available honestly with the specific missing "
      "piece named. Reaching verified needs a CUDA COLMAP binary or CUDA-enabled "
      "pycolmap in the environment; that is a dependency, not code to write, and no "
      "post-processing turns a sparse cloud into a dense one."),
    F("pr.mesh", "Textured mesh", "workers", "Processing",
      "Poisson surface reconstruction with density trimming and orthophoto texture.",
      "verified", ["tests/test_mesh_reconstruction.py::TestRefusals",
                   "tests/test_mesh_reconstruction.py::TestRealMesh",
                   "tests/test_mesh_reconstruction.py::TestDensityTrimming"],
      "The trim test is measured against the untrimmed pipeline: without it 0.42 per "
      "cent of vertices land outside the surveyed patch, so the assertion is zero "
      "rather than a tolerance. A tolerant version passed on trimmed and untrimmed "
      "meshes alike, which is to say it tested nothing."),
    F("pr.gps_denied", "GPS denied reconstruction", "workers", "Processing",
      "Indoor, handheld and ground-robot imagery reconstructed without geotags, with "
      "the resulting model's spatial validity stated rather than assumed.",
      "verified", ["tests/test_spatial_reference.py",
                   "tests/test_api_measurements.py::TestSpatialReferenceCapability"],
      "Structure-from-motion recovers geometry only up to a similarity transform, so "
      "without geotags or control the model has arbitrary position, rotation and SCALE. "
      "It still renders and meshes convincingly, which is exactly the trap: every "
      "distance in it is wrong by an unknown factor. api.check_spatial_reference() "
      "reports the mode up front, no CRS is carried through for an arbitrary model, and "
      "require_measurable() refuses distance, area and volume rather than returning "
      "them in model units a reader would treat as metres. Three or more GCPs, or "
      "enough geotags, restore measurability."),
    F("pr.gcp", "Ground control points", "workers", "Processing",
      "GCP import, image marking, and a reprojection error report that states per-point "
      "residuals rather than a verdict, refuses to quote an accuracy when no control was "
      "checked, and flags points marked in too few images or on the wrong target.",
      "implemented", ["tests/test_gcp.py::TestReading",
                      "tests/test_gcp.py::TestMarking",
                      "tests/test_gcp.py::TestResiduals",
                      "tests/test_gcp.py::TestAccuracyReport",
                      "tests/test_gcp.py::TestReportOutput"],
      "core/gcp.py; Api.import_gcps, mark_gcp, gcp_accuracy_report. Fitting the "
      "transform itself belongs to the reconstruction engine."),
    F("pr.rtk_ppk", "RTK and PPK", "workers", "Processing",
      "RTK/PPK metadata and RINEX base station data with timestamp alignment: camera "
      "events read from the aircraft's own event file with corrections in the units "
      "they were written in, the base session window read from the RINEX header, every "
      "event shown to be inside that window or named as outside it, the leap-second "
      "offset stated because a wrong one shifts every event by a second, and an "
      "accuracy claim gated on the recorded solution flag so a float flight is never "
      "described as centimetre RTK.",
      "implemented", ["tests/test_rtk.py::TestGpsTime",
                      "tests/test_rtk.py::TestCameraEvents",
                      "tests/test_rtk.py::TestBaseStation",
                      "tests/test_rtk.py::TestAlignment",
                      "tests/test_rtk.py::TestPositioningReport",
                      "tests/test_rtk.py::TestThroughTheApi"],
      "core/rtk.py; Api.check_ppk_inputs. Checks and describes the inputs; it does not "
      "compute the PPK solution or rewrite a coordinate, and says so in every report."),
    F("pr.distributed", "Distributed processing", "workers", "Processing",
      "Jobs submitted, polled and cancelled across workers, with cooperative "
      "cancellation and honest failure reporting.",
      "verified", ["tests/test_jobs.py", "tests/test_processing.py",
                   "tests/test_job_queue.py", "tests/test_celery_broker.py",
                   "tests/test_worker_tasks.py"],
      "Three layers. core/job_queue.py bounds concurrency in one process with strict "
      "priority and retries that keep every attempt's own error. "
      "services/worker/celery_app.py puts the queue in Redis so it outlives the process "
      "that filled it. services/worker/tasks.py runs the real pipeline on a worker. "
      "The part that made this actually distributed rather than a thread pool with extra "
      "steps is CANCELLATION. core/processing_runs.py cancels through a module-level "
      "dict of threading.Events keyed by run id -- correct while the process calling "
      "stop_processing_run is the process running the pipeline, and silently wrong the "
      "moment a worker is elsewhere: the API sets an Event in its own memory, reports "
      "the run cancelled, and the worker reconstructs for another forty minutes with "
      "nobody told. The request now travels through Redis and is checked at each "
      "progress callback, which is a stage boundary, which is where the recorded state "
      "and the files on disk agree. "
      "An unreachable broker answers 'no cancel pending' rather than 'cancel' -- the "
      "safe direction, since treating a network blip as a stop would discard an hour of "
      "reconstruction that was going fine. Cancel keys are cleared after a run so a "
      "stale one cannot kill a later run that reuses the id. "
      "Worker concurrency defaults to 1: reconstruction is memory-bound, so two on one "
      "host is usually slower and occasionally fatal."),
    F("pr.large_datasets", "Large dataset processing", "workers", "Processing",
      "Thousands of images via chunking and memory-aware scheduling, with the job sized "
      "against the machine before it starts.",
      "implemented", ["tests/test_job_sizing.py",
                      "tests/test_api_measurements.py::TestLargeDatasetCapability"],
      "api.size_reconstruction_job() estimates peak memory and disk against what is "
      "actually free and recommends a chunk size; api.plan_job_chunks() splits a "
      "capture with mandatory overlap, because chunks reconstructed independently share "
      "no geometry and without shared views the merge produces several disconnected "
      "models rather than one survey. Feature matching grows with the SQUARE of the "
      "image count, so past a few hundred images chunking is a requirement not a "
      "preference. Estimates are labelled rough. Reaching verified needs the resumable "
      "job execution and chunk merge to be built on top of this sizing."),
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
      "Register, version, deploy and roll back models, with metrics and a real checksum "
      "that is checked rather than merely stored: the installed file is hashed and "
      "compared against the digest recorded when the model was registered, a replaced "
      "file is reported as a mismatch whose published metrics describe something else, "
      "and a model with no recorded digest is reported as unrecorded rather than as "
      "verified. A registered key with no weights behind it declares itself as awaiting "
      "weights and is not counted as an available model, so a routing target is never "
      "mistaken for a trained one.",
      "implemented", ["tests/test_model_identity_verification.py::TestVerification",
                      "tests/test_model_identity_verification.py::TestWholeRegistryReport",
                      "tests/test_model_identity_verification.py::TestEntriesWithNoWeights",
                      "tests/test_model_identity_verification.py::TestTheRealRegistry"],
      "training/register.py refuses an export that failed parity; "
      "core/models.py::verify_model_identity and verify_all_models compare installed "
      "against recorded; Api.verify_models surfaces it and audits a mismatch."),
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
      "SegFormer-B5 @1024 installed, replacing B2, at decision threshold 0.85 chosen by "
      "sweep on the validation split. Held-out test split: IoU 0.637, precision 0.811, "
      "recall 0.748, against IoU 0.607 and precision 0.664 at the previous 0.25 "
      "threshold. B2 reached test-split IoU 0.515; the heuristic reaches 0.045. The "
      "threshold gained +0.030 IoU and +0.147 precision for no retraining."),
    F("ai.spalling", "Spalling detection", "vision", "AI",
      "Trained detector for spallation with published validation metrics, used in "
      "preference to the heuristic, and reporting an empty result as an empty result "
      "rather than falling back to invented findings.",
      "implemented", ["tests/test_honesty.py::TestFindingNothingIsAnAnswer"],
      "YOLO11x on CODEBRIM, installed as structural_multiclass_detector: "
      "mAP50 0.417, mAP50-95 0.201; Spallation 0.306, ExposedBars 0.330, "
      "CorrosionStain 0.254, Efflorescence 0.193, Crack 0.124."),
    F("ai.corrosion", "Corrosion detection", "vision", "AI",
      "Trained model for corrosion with published validation metrics, reporting a "
      "severity grade it measured or refusing to report one at all.",
      "verified",
      ["tests/test_corrosion_severity.py::TestItActuallyPredictsTheWorstGrade",
       "tests/test_corrosion_severity.py::TestRefusalRatherThanAGuessedGrade",
       "tests/test_corrosion_severity.py::TestTheGradeIsAnArgmaxNotAThreshold",
       "tests/test_corrosion_severity.py::TestTheScaleIsOrdinalAndSaysSo"],
      "SEGMENTATION, AFTER DETECTION FAILED. The first attempt was YOLO11l on 498 "
      "images: mAP50 0.254, recall 0.257, so three corrosion sites in four went "
      "unfound. That was rejected rather than registered. The shipping model asks a "
      "different question -- how bad is this pixel, on an ordinal scale of "
      "good/fair/poor/severe -- because that is what a maintenance decision needs and "
      "it is what the Condition State corpus actually labels. "
      "corrosion_severity_segmentation is SegFormer-B2 @512 over 440 images, mean IoU "
      "0.5769, pixel accuracy 0.8497, and it recovers 0.788 of severe pixels rather "
      "than declining to use its worst grade. Errors are almost all one step on the "
      "scale: 124,241 of the missed severe pixels are called 'poor'. "
      "REFUSES RATHER THAN GUESSES: every other detector here degrades to a heuristic "
      "when its model is absent, and this one does not, because colour and texture "
      "rules can find rust but nothing outside the corpus separates poor from severe. "
      "440 images from one source means these figures are a first indication, and it "
      "has not been tested on Indian infrastructure imagery."),
    F("ai.solar", "Solar defect detection", "vision", "AI",
      "Trained detector for panel defects with published validation metrics.",
      "verified", ["tests/test_trained_defect_models.py::TestTheWeightsAreReallyThere",
                   "tests/test_trained_defect_models.py::TestTheNumbersArePublished",
                   "tests/test_trained_defect_models.py::TestLabelsMatchTheModel"],
      "Two models, answering different questions. solar_cell_defect_detector reads "
      "electroluminescence imagery of individual cells at mAP50 0.884; "
      "solar_thermal_anomaly_classifier reads aerial infrared of whole modules across 12 "
      "classes at balanced accuracy 0.724, and is the drone-capturable one. Soiling is "
      "its weakest class at 0.367 recall and the entry says so. "
      "A third attempt, an RGB detector over panel condition (Clear/Dusty/Damage/Snow), "
      "reached only mAP50 0.318 with Dusty its WORST class at 0.073 -- the one class "
      "Indian sites most need -- and was deliberately NOT registered. Soiling looks like "
      "a radiometric comparison problem rather than a detection one."),
    F("ai.water_ponding", "Water ponding detection", "vision", "AI",
      "Closed depressions measured from the DSM, reported as area, depth and volume "
      "with the survey's vertical accuracy carried through. Refuses without an accuracy "
      "estimate and reports nothing shallower than twice it.",
      "verified", ["tests/test_ponding.py",
                   "tests/test_api_measurements.py::TestPondingCapability"],
      "Deliberately not a trained detector. A model shown a photograph would infer "
      "ponding from colour and specularity and answer confidently on wet-but-not-ponded "
      "membrane, shadow and glare. This measures where water CAN collect; whether water "
      "is present now is a separate claim and is never merged into it."),
    F("ai.deformation", "Surface deformation detection", "vision", "AI",
      "Vertical displacement between two surveys of the same ground, gated on a "
      "detection floor built from both surveys' vertical accuracies and the "
      "co-registration residual.",
      "verified", ["tests/test_deformation.py",
                   "tests/test_api_measurements.py::TestDeformationCapability"],
      "Deliberately not a trained detector: a model shown one survey has nothing to "
      "compare against. Needs two flights and cannot say anything from one. Absence of "
      "a finding means no movement was RESOLVABLE, not that none occurred."),
    F("ai.custom_training", "Custom defect training", "vision", "AI",
      "Users label, split, train, review metrics and deploy their own model.",
      "implemented", ["tests/test_custom_training.py",
                      "tests/test_label_sets.py",
                      "tests/test_label_ui.py"],
      "Trainers exist, and core/custom_training.py is now the dataset builder they were "
      "missing: labelled images become splits or the corpus is REFUSED with the reason. "
      "Refused rather than warned, because a corpus is built once and its metric read "
      "many times -- a class too thin to learn, a class with too few held-out examples "
      "to measure (one gives a recall of 0.0 or 1.0), a single-class corpus that scores "
      "perfectly by answering the only thing it knows, the same image under two labels, "
      "and leakage detected by content digest rather than filename since a renamed copy "
      "is the usual case. "
      "Splitting is stratified per class after a whole-corpus hash was found handing a "
      "30-image class ONE validation example -- which would have been refused as 'label "
      "more' when the images were sufficient and the split was not. "
      "A USER CAN NOW DRAW THE BOX. app/web/js/label-box.js is a canvas labeller, and "
      "the boxes it produces go through core/label_sets.py into a YOLO-layout corpus the "
      "existing detection trainer reads. The drawing is verified by executing it in "
      "headless Chromium and reading the geometry back, not by asserting about it in "
      "Python: two drags produce two boxes, a one-pixel drag is discarded rather than "
      "stored as a target no model can match, and coordinates are normalised per axis "
      "(0.25 wide over a 400px canvas, 0.5 tall over 200) so a corpus does not depend on "
      "the window it was drawn in. "
      "The box builder refuses what the class builder cannot see: geometry off the edge "
      "of the image, and an image carrying no boxes at all. That last one is accepted "
      "ONLY when the user marks it deliberately empty, because a confirmed negative is "
      "evidence and a forgotten image is an accident, and training on the second teaches "
      "the model the defect is absent. "
      "WHAT IS STILL NOT HERE: training is launched from the CLI or the job queue rather "
      "than from the labelling screen, so label-to-model is two steps and not one."),
    F("ai.assisted_annotation", "AI assisted annotation", "hub", "AI",
      "Model pre-labels imagery; a reviewer accepts, edits, merges, splits or reclassifies.",
      "implemented", ["tests/test_assisted_annotations.py"],
      "A public-domain concrete photograph exercises installed YOLO ONNX inference and "
      "every review action. Original model geometry, label, confidence, key and sha256 "
      "remain immutable. SegFormer pre-label persistence is refused because its current "
      "result contract has no per-region confidence; no score is synthesised."),
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
      "verified", ["tests/test_change_detection.py::TestSurfaceComparison",
                   "tests/test_india_construction_pack.py::"
                   "test_approved_design_progress_measures_observed_surface_not_contract_completion"],
      "Two halves, both measured: volume added/removed and changed area between surveys, "
      "and observed coverage inside each approved design element. The note that design "
      "progress was not computed went stale when core/india_construction.py landed. "
      "What is deliberately NOT produced is a contractual percentage complete -- the "
      "summary reports that as unavailable, because schedule, quantities, hidden work "
      "and sign-off are not visible from the air and a number that implied otherwise "
      "would be read as one."),
    F("ai.model_version_recorded", "Every AI result stores model version and confidence",
      "vision", "AI",
      "No detection is stored without its model identity and confidence, where identity "
      "is the digest of the file that actually ran rather than a registry entry, and the "
      "heuristic path claims no identity at all.",
      "implemented", ["tests/test_model_provenance.py::TestModelIdentity",
                      "tests/test_model_provenance.py::TestDetectionCarriesIdentity",
                      "tests/test_model_provenance.py::TestApiRefusesUnattributableFindings"],
      "core/models.py::model_identity hashes the installed file; detection results carry "
      "it; the API refuses a model-sourced defect missing key, digest or confidence."),
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
      "Runtime and DINOv2/UPerNet architecture are implemented, and a head has now been "
      "TRAINED AND MEASURED AND IS NOT SHIPPING. shared_semantic_dinov2_vitb14 reached "
      "mean IoU 0.6128 in validation (building 0.890, road 0.782, vegetation 0.848, "
      "water 0.661, bare_land 0.422) and its ONNX export agrees with torch on all "
      "1,073,296 pixel decisions. On the India holdout it predicts BUILDING ON EVERY "
      "PIXEL of all four Indian tiles -- precision 0.092, recall 1.0, IoU 0.092, "
      "docs/holdout/shared_semantic_india_holdout.json. "
      "The cause is in the corpus, not the code: SpaceNet 7 labels buildings and leaves "
      "96.7 per cent of each tile at IGNORE_INDEX, so training never once penalised "
      "predicting building on an unlabelled pixel, and on imagery that resembles that "
      "source the cheapest answer is 'all building'. It scores normally on "
      "OpenEarthMap tiles, which is why validation looked healthy. "
      "Fixing this needs a corpus where non-building pixels are labelled or at least "
      "counted as negatives, not more epochs. Until then the row stays in_progress and "
      "nothing routes to the head."),
    F("eng.assets", "Object and asset detection engine", "vision", "India: shared engines",
      "Count and individually locate assets such as trees, modules, equipment, poles "
      "and insulators with model provenance and confidence.",
      "verified", ["tests/test_asset_taxonomy.py::TestTheTaxonomyIsShared",
                   "tests/test_asset_taxonomy.py::TestProvenanceIsMandatory",
                   "tests/test_asset_taxonomy.py::TestLocationAndConfidence",
                   "tests/test_asset_taxonomy.py::TestGeometryMatchesTheAssetKind",
                   "tests/test_asset_taxonomy.py::TestCountingIsHonest",
                   "tests/test_asset_taxonomy.py::TestThresholdingStaysVisible",
                   "tests/test_asset_taxonomy.py::TestTheApiExposesIt"],
      "core/asset_taxonomy.py holds one vocabulary across power, rail, solar, "
      "vegetation, agriculture and the built environment, grounded in the class sets "
      "the packs already use rather than invented. Output is a single GeoJSON form "
      "carrying model key, sha256 and per-instance confidence, reachable through "
      "Api.build_asset_inventory and Api.asset_taxonomy. "
      "An instance with no geometry, no confidence or no model digest is REFUSED rather "
      "than counted -- a count is something a crew is dispatched on. Continuous types "
      "(conductor, track, road) are never counted and appear under "
      "present_but_not_counted, so their absence from the counts does not read as their "
      "absence from the site. Point assets cannot arrive as polygons, which is how a "
      "report ends up quoting a footprint for a pole."),
    F("eng.change", "Survey change intelligence engine", "core", "India: shared engines",
      "Aligned T1/T2 DSMs produce a georeferenced difference raster, contiguous change "
      "polygons, exact rise/fall volumes and an interpretation-safe report.",
      "implemented", ["tests/test_survey_intelligence.py::TestSurfaceChangePackage"]),
    F("eng.anomaly", "Anomaly intelligence engine", "vision", "India: shared engines",
      "Find deviations from a validated normal baseline without assigning an unsupported "
      "named defect class.",
      "implemented", ["tests/test_india_anomaly.py::TestValidatedBaseline",
                      "tests/test_india_anomaly.py::TestGeospatialRefusals"],
      "Robust per-band baseline statistics require named validation scope; outputs are georeferenced deviation candidates, never inferred defect names."),

    # Photogrammetry and geometry foundation
    F("india.foundation.orthomosaic", "Survey orthomosaic", "workers", "India: foundation",
      "A georeferenced site orthomosaic suitable for GIS and downstream analysis.",
      "implemented", ["tests/test_reconstruction_colmap.py::TestGeoreferencing::test_rasters_are_written_as_georeferenced_geotiffs"],
      "Real eight-frame Aukerman survey produces a CRS-bearing orthomosaic GeoTIFF."),
    F("india.foundation.dsm_dtm", "Survey DSM and DTM", "workers", "India: foundation",
      "Metric surface and terrain rasters in an explicit projected CRS.",
      "implemented", ["tests/test_reconstruction_colmap.py::TestGeoreferencing::test_rasters_are_written_as_georeferenced_geotiffs",
                      "tests/test_reconstruction_colmap.py::TestGeoreferencing::test_the_dsm_holds_elevations_in_metres",
                      "tests/test_reconstruction_colmap.py::TestGeoreferencing::test_the_dtm_holds_ground_elevations_in_metres",
                      "tests/test_reconstruction_colmap.py::TestNoFabrication"],
      "Real survey DSM and ground-filtered DTM are metric GeoTIFFs; sparse resolution is relaxed and disclosed rather than invented."),
    F("india.foundation.reconstruction", "Survey 3D reconstruction", "workers", "India: foundation",
      "Georeferenced point cloud and model from a real drone survey.",
      "implemented", ["tests/test_reconstruction_colmap.py::TestStructureFromMotion",
                      "tests/test_reconstruction_colmap.py::TestGeoreferencing",
                      "tests/test_reconstruction_colmap.py::TestNoFabrication"],
      "Automated COLMAP reconstruction runs on eight real Aukerman drone frames and reports registration, residuals, point count and limitations."),
    F("india.foundation.area", "Survey area and distance", "core", "India: foundation",
      "Metric area and distance measured from georeferenced survey products.",
      "implemented", ["tests/test_reconstruction_colmap.py::TestGeoreferencing::test_the_real_orthomosaic_supports_metric_area_and_distance",
                      "tests/test_raster_measurement.py::TestDistance",
                      "tests/test_raster_measurement.py::TestArea",
                      "tests/test_raster_measurement.py::TestRefusals"],
      "Measured directly on projected rasters, including the real COLMAP orthomosaic; unreferenced and geographic inputs are refused."),
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
      "implemented", ["tests/test_provenance.py::TestRealReconstruction",
                      "tests/test_survey_intelligence.py::TestSurfaceChangePackage",
                      "tests/test_reconstruction_colmap.py::TestGeoreferencing"],
      "Real reconstruction artifacts carry verifiable lineage; survey-change packages add mapped quantities, method and explicit interpretation limits."),

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
      "not_started", [],
      "OUT OF v1 SCOPE by decision, 2026-08-17. No corpus exists at drone resolution. "
      "Note that stockpile MEASUREMENT already ships and is verified -- me.volume and "
      "india.foundation.volume measure a stockpile the operator outlines, with the "
      "survey's vertical accuracy carried through. What is dropped is proposing the "
      "outline automatically, which is a convenience on top of a capability that works."),
    F("pack.mining.scene", "Mine and quarry segmentation", "vision",
      "India pack: Mining",
      "Pit, bench, haul road, stockpile, water, vegetation, excavated region and "
      "restricted-boundary layers.",
      "not_started", [],
      "OUT OF v1 SCOPE by decision, 2026-08-17, rather than shipping something weak. "
      "Two attempts, both measured and both rejected. MineNetCD change detection reached "
      "IoU 0.2955 on ~15 held-out mines: 60 per cent of what it flagged was wrong and it "
      "missed 47 per cent of real change, which is a coin flip an operator would dig "
      "against. The largest open alternative -- a global mining dataset of 1,210 sites -- "
      "is Sentinel-2 at 10 m per pixel, and this product surveys in centimetres, so a "
      "model trained on it cannot be pointed at drone imagery at all. "
      "The blocker is a corpus at drone resolution, not epochs or architecture. Reopen "
      "when one exists; retraining on what is available would only produce the same "
      "number with more confidence."),
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
      "implemented", ["tests/test_thermal.py", "tests/test_hub_thermal.py",
                      "tests/test_solar_thermal.py::TestRgbThermalRegistration",
                      "tests/test_solar_thermal.py::TestModuleTemperatureAssociation",
                      "tests/test_solar_thermal.py::TestSolarThermalRefusals"],
      "At least six operator-supplied correspondences produce separate residual, inlier "
      "and coverage components; this checks tie-point self-consistency, not image feature "
      "matching. Radiometric cells are associated only where registered RGB module "
      "polygons and projected inventory polygons agree, with a required named severity "
      "convention. Tests use constructed RGB/radiometric fixtures, not a field capture, "
      "so this is not field validation; all outputs remain review candidates."),

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
      "implemented", ["tests/test_annotations.py::TestAnnotationGeometry",
                      "tests/test_annotations.py::TestAnnotationValidation",
                      "tests/test_annotations.py::TestAnnotationApi",
                      "tests/test_hub_web.py::TestHubRealBrowser::test_annotation_draw_events_cover_every_shape_with_metadata"],
      "All seven shapes pass through the shared geometry validator, project-contained REST "
      "storage and the MapLibre draw-event path with explicit severity and workflow status."),
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
      "Length, height, area and volume inside the 3D model, exact on a mesh of known "
      "size, with the horizontal and vertical parts of a run reported separately, "
      "surface area distinguished from planimetric footprint, a model whose provenance "
      "records no CRS refused rather than measured in structure-from-motion units, and "
      "volume refused on a surface that does not close.",
      "implemented", ["tests/test_model_measurement.py::TestScaleMustBeRecorded",
                      "tests/test_model_measurement.py::TestLength",
                      "tests/test_model_measurement.py::TestHeight",
                      "tests/test_model_measurement.py::TestArea",
                      "tests/test_model_measurement.py::TestVolume",
                      "tests/test_model_measurement.py::TestThroughTheApi",
                      "tests/test_model_measurement.py::TestMeshReading"],
      "core/model_measurement.py; Api.measure_in_model. Volume is the divergence "
      "theorem over a closed mesh, so it is origin-independent; an open exterior with "
      "no floor is refused instead of summed."),
    F("me.volume", "Volume and stockpile", "core", "Measurement",
      "Cut/fill against DTM, plane and lowest-point references, exact on a known surface.",
      "verified", ["tests/test_dsm_analysis.py::test_volume_against_dtm_is_exact",
                   "tests/test_dsm_analysis.py::test_polygon_clip_halves_the_volume"]),
    F("me.slope", "Slope measurement", "core", "Measurement",
      "Gradient for pitched roofs, pavements and ramps, exact on a plane of known "
      "pitch, with the cell size the gradient was measured across reported, a surface "
      "that is not one facet said to be so rather than averaged into a single pitch, "
      "and a geographic raster refused instead of dividing metres by degrees.",
      "implemented", ["tests/test_slope.py::TestAKnownPlane",
                      "tests/test_slope.py::TestAspect",
                      "tests/test_slope.py::TestResolutionIsStated",
                      "tests/test_slope.py::TestNonPlanarSurfaces",
                      "tests/test_slope.py::TestClipping",
                      "tests/test_slope.py::TestRefusals",
                      "tests/test_slope.py::TestThroughTheApi",
                      "tests/test_slope.py::TestGradientDescription"],
      "core/slope.py; Api.measure_slope. Reports the per-cell distribution and a "
      "fitted plane with its residual, since the residual is what says whether a "
      "single quoted pitch describes the surface at all."),
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
      "implemented", ["tests/test_thermal.py::TestGeoreferencedOutput",
                      "tests/test_hub_thermal.py::TestThermalMapArtifact",
                      "tests/test_hub_thermal.py::TestThermalRealBrowser"],
      "Measured Celsius cells are preserved in a projected GeoTIFF and rendered locally with CRS and interpolation status visible."),
    F("th.model_3d", "3D thermal model", "workers", "Thermal",
      "Thermal values projected onto reconstructed geometry.",
      "implemented", ["tests/test_hub_thermal.py::TestThermalProjectionContract",
                      "tests/test_hub_thermal.py::TestThermalRealBrowser"],
      "Nearest measured thermal cells colour overlapping vertices; missing CRS, mismatches and unknown coordinate frames are refused."),
    F("th.comparison", "RGB and thermal comparison", "hub", "Thermal",
      "Side by side, swipe and opacity overlay with linked zoom.",
      "implemented", ["tests/test_hub_thermal.py::TestThermalProjectionContract",
                      "tests/test_hub_thermal.py::TestThermalRealBrowser",
                      "tests/test_hub_web.py::TestHubPanels"],
      "All three local comparison modes share pan/zoom and refuse dimension-only matching without explicit validated registration metadata."),
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
    F("hub.projects", "Projects", "hub", "Hub",
      "Projects with client, site, type, tags, members, history and activity timeline, "
      "with project-scoped roles that add to organisation roles rather than replacing "
      "them, and opt-in restriction that hides a project from non-members without ever "
      "locking out an administrator.",
      "implemented", ["tests/test_project_scope.py::TestUnrestrictedProjects",
                      "tests/test_project_scope.py::TestProjectMembership",
                      "tests/test_project_scope.py::TestRestrictedProjects"],
      "services/api: ProjectMembership, role_on_project, require_project_role."),
    F("hub.assets", "Assets", "hub", "Hub",
      "Persistent assets with geometry, inspection history and a timeline, where "
      "confirmed findings are counted separately from unconfirmed model predictions and "
      "no trend is claimed from a single inspection.",
      "implemented", ["tests/test_project_scope.py::TestAssetTimeline"],
      "services/api: Project.asset_id and GET /assets/{id}/timeline."),
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
      "implemented", ["tests/test_model_formats.py::TestMeshFormats",
                      "tests/test_model_formats.py::TestPointCloudFormats",
                      "tests/test_model_formats.py::TestModelExportRefusals"],
      "Independent readers round-trip geometry for every format. GLB/GLTF retain the "
      "projected coordinate origin and CRS metadata; LAS/LAZ retain coordinates, colours "
      "and their standard CRS VLR, with LAZ verified as compressed."),
    F("api.rest", "REST API", "hub", "API",
      "Documented endpoints for every resource in the specification.",
      "implemented", ["tests/test_api.py::TestProjectsAndMissions", "tests/test_api.py::TestAssets",
                      "tests/test_uploads.py", "tests/test_processing.py",
                      "tests/test_rest_resources.py::TestRestDefects",
                      "tests/test_rest_resources.py::TestRestMeasurements",
                      "tests/test_rest_resources.py::TestRestReports",
                      "tests/test_rest_resources.py::TestRestAIJobs",
                      "tests/test_rest_resources.py::TestRestContainment",
                      "tests/test_annotations.py::TestAnnotationApi"],
      "Defects, source-attributed georeferenced measurements, immutable structured report "
      "snapshots, project annotations and persistent AI-job submissions are documented in "
      "OpenAPI and enforce project containment. AI jobs remain pending_worker until a real "
      "inference worker produces attributed output; submission does not claim inference."),
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
      "would be, with organization-scoped telemetry delivered through a broker shared "
      "by workers using the same database.",
      "implemented", ["tests/test_events.py::TestLiveStream",
                      "tests/test_realtime.py::TestSharedBroker",
                      "tests/test_realtime.py::TestTelemetryStream"],
      "The shared database buffer supports multi-worker fan-out. Delivery remains "
      "best effort: there are no client acknowledgements, durable offline cursors or "
      "exactly-once guarantee."),
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
      "Helm chart deploys the API, workers and PostGIS to a cluster, using the same "
      "images the compose stack runs.",
      "implemented", ["tests/test_helm_chart.py"],
      "The chart ships NO default credentials and fails to render without them: a "
      "default that works is a default that reaches production. Readiness uses "
      "/health/ready, which checks the live database and object store, rather than "
      "/health/live -- a pod that cannot reach PostGIS must not take traffic, and "
      "restarting it would not help. PostGIS is a StatefulSet with a claim because "
      "survey data is the one thing that cannot be regenerated by re-running a job, and "
      "the worker carries a memory limit because photogrammetry is memory-bound and an "
      "unlimited worker is OOM-killed mid-reconstruction. The tests are STRUCTURAL "
      "only: no helm binary is available here, so nothing has rendered the templates. "
      "Reaching verified needs helm lint and helm template against a real cluster."),
    F("inf.postgis", "PostgreSQL and PostGIS", "infra", "Deployment",
      "Spatial data in native geometry types with spatial indexes, so the database "
      "answers 'which assets are inside this polygon' rather than the application "
      "loading everything and filtering in Python.",
      "verified", ["tests/test_api.py::TestHealth",
                   "tests/test_spatial_backend_honesty.py",
                   "tests/test_postgis_spatial.py"],
      "GeoJSON text is mirrored into GIST-indexed PostGIS geometry columns on assets, "
      "defects, measurements and annotations, kept in step by a trigger. Verified "
      "against a real PostGIS 3.4 in Docker -- ST_Intersects finds the asset, a distant "
      "polygon finds nothing, ST_Area returns metres, and the migration is idempotent "
      "across restarts. The tests SKIP without a live instance rather than passing "
      "against SQLite, which would prove nothing about the thing they check. "
      "The text column stays the SOURCE OF TRUTH and that is not a shortcut: SQLite is "
      "a supported backend and cannot hold a geometry type, so making the native column "
      "authoritative would fork the schema and give the two backends different answers. "
      "The geom column is a derived index. "
      "A row whose GeoJSON will not parse is still STORED, with geom NULL -- it stays "
      "visible to every non-spatial query and falls back to the text path. Rejecting the "
      "write instead would lose data to gain an index. "
      "The health report reads the columns rather than the extension: it used to say "
      "native_geometry whenever PostGIS answered a version query, while every column was "
      "Text, which would have had an operator sizing a workload around indexes that did "
      "not exist."),
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
      "implemented", ["tests/test_observability.py"],
      "JSON request spans carry W3C trace context; Prometheus metrics explicitly declare "
      "per-process scope; readiness performs live database and storage calls and returns "
      "503 on failure. Trace export remains best-effort structured-log shipping."),
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
      "verified", ["tests/test_documentation_set.py::TestTheSetIsComplete",
                   "tests/test_documentation_set.py::TestClaimsMatchTheCode",
                   "tests/test_documentation_set.py::TestTheGuidesCarryTheProjectsRule"],
      "Seven guides: INSTALLATION, ARCHITECTURE, USER_GUIDE, PILOT_GUIDE, PLUGIN_GUIDE, "
      "API_GUIDE, DEPLOYMENT. The tests do what tests can do for prose -- check the set "
      "is present and unstubbed, then pin the specific claims that would misdirect a "
      "reader if they went stale: that failures carry `error` and not `reason`, that "
      "documented endpoints exist, that the plugin kinds listed are the real enum, that "
      "recommended mission modes actually plan, and that the deployment guide does not "
      "promise native geometry columns the schema does not have. Prose cannot be checked "
      "for truth; these are the statements that would cost someone a day."),
    F("demo.mode", "Demo mode", "hub", "Docs",
      "Full workflow explorable with no hardware, marked synthetic throughout.",
      "verified", ["tests/test_demo_mode.py",
                   "tests/test_api_measurements.py::TestDemoCapability"],
      "Every artefact carries synthetic: True recursively, so a single finding lifted "
      "out still declares itself. Sites are at Null Island and timestamps at the epoch, "
      "so a reader who misses the flag still cannot mistake it for a survey. "
      "refuse_if_demo() lets any publish or register path reject it."),
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
