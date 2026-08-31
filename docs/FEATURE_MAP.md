# Feature map: the market inventory, in OpenDroneKit's own names

A competitor inventory (Hammer Missions, as surveyed 31 August 2026) mapped onto this
project's capability registry, so every item has an **ODK name** and a state that can be
checked rather than argued about.

## How to read the state column

| State | Meaning |
|---|---|
| **registered** | A row exists in `docs/features/registry.py`. Its status is computed from passing tests by `tools/feature_status.py` — see that output for whether it is `verified`, `implemented` or `in_progress`. A row existing is **not** the same as a feature shipping. |
| **partial** | A row exists but covers less than the market feature does. The shortfall is named. |
| **gap** | No row. This is new work, and a proposed ODK id is given. |

Two registered rows are knowingly **not** shipping and are called out where they appear:
`eng.semantic` (two of six classes usable) and `pr.dense` (GPU dense stereo fails on this
machine's driver). Do not read those as done because a row exists.

Namespaces already in use: `mp` mission planning · `fl` flight · `pr` processing ·
`ai` inspection AI · `eng` shared engines · `me` measurement · `th` thermal · `in`
inspection records · `rp` reporting · `sh` sharing · `ex` export · `hub` workspace ·
`fm` fleet · `inf` infrastructure · `api`, `sdk`, `sec`, `doc`, `demo`.

---

## 1. Architecture

| Market feature | ODK name | State |
|---|---|---|
| Web workspace | `hub.projects`, `hub.viewer_2d`, `hub.viewer_3d` | registered |
| Field application | `fl.abstraction`, `fl.telemetry` | partial — the desktop app is one surface; there is no separate Android field build |
| Plan → simulate → sync → fly → verify → upload → process → inspect → report | `mp.simulation`, `fl.data_verification`, `pr.sfm`, `rp.reports` | registered end to end, verified on the Aukerman survey |

**Gap — `hub.field_app`**: a separate mobile/controller build. This is the single largest
item in the whole inventory and is a different application, not a feature.

## 2. Mission planning

| Market feature | ODK name | State |
|---|---|---|
| Folders, multiple missions per project | `hub.projects` | registered |
| Mission layers | `mp.linking` | partial — linked missions exist; visual layering of overlapping plans does not |
| Point/line/polygon/circle geometry | `mp.geometry_3d` | registered |
| KML import | `mp.import` | registered |
| Planning from an existing 3D model | `mp.geometry_3d` | registered |
| 3D building outlines | `mp.geometry_3d`, `mp.standoff` | registered |
| Site planning tools (markers, named measurements) | `me.2d` | partial — measurement exists; named non-flight site objects do not |
| Planning estimates | `mp.estimates` | registered |
| Simulation | `mp.simulation` | registered |
| Mission sharing before deployment | `mp.sharing` | registered |
| Bidirectional sync | — | **gap — `mp.sync`** (needs the field app first) |

## 3. Precision planning

| Market feature | ODK name | State |
|---|---|---|
| Fly to draw | `mp.fly_to_draw` | registered |
| Ground offset | `mp.standoff` | registered |
| GSD calculator | `mp.gsd` | registered |
| Image overlap (all axes) | `mp.overlap` | registered |
| Camera model database | `mp.camera_db`, `mp.payload_db` | registered |
| Mission reversal | `mp.versioning` | partial — reversal is a planner option, not its own row |
| Custom start point / resume | `fl.battery_swap`, `fl.crash_recovery` | registered |

## 4. Mission types

We generate **15 templates today**: `grid`, `double_grid`, `corridor`, `facade`,
`tower_mapping`, `solar_inspection`, `orbit`, `panorama`, `bubble_360`, `waypoints`,
`linear_inspection`, `lateral_capture`, `roof_inspection`, `magnetic_mapping`,
`linked_mission`.

| Market mission | ODK template | State |
|---|---|---|
| Standard mapping | `grid` | registered |
| 3D mapping | `double_grid` | registered |
| Roof inspection | `roof_inspection` | registered |
| Facade inspection | `facade` | registered |
| Facade **mapping** | `facade` | partial — inspection geometry only; the photogrammetry variant is not separated |
| Lateral capture | `lateral_capture` | registered |
| Linear mapping/inspection | `linear_inspection`, `corridor` | registered |
| Solar inspection | `solar_inspection` | registered |
| Tower mapping | `tower_mapping` | registered |
| Advanced waypoints | `waypoints` | registered |
| Orbit | `orbit` | registered |
| Panorama | `panorama` | registered |
| 360 bubble | `bubble_360` | registered |
| Magnetic mapping | `magnetic_mapping` | registered |
| Composite bridge | `linked_mission` | registered as a composition, not a named workflow |
| Full building capture | `linked_mission` | as above |
| Utility pylon | `mp.tpl_pylon` | **now reachable** — `Api.plan_pylon_mission`; one stacked orbit per named element (crossarm, insulator, conductor, body) |
| **Wind turbine** | — | **gap — `mp.tpl_turbine`** |
| Box inspection | `box_inspection` | **built this session** — offset-footprint circuit, camera on the building centre |
| Dome inspection | `dome_inspection` | **built this session** — hemispherical rings, gimbal on the surface normal; was silently aliased to `tower_mapping` |

### Correction, made while building this

Pylon was listed above as a gap. It was not. `mission/mission_types.py` had planned pylon,
thermal AND multispectral missions since that work landed, with tests covering the
geometry — but none of the three appeared in the planner's template table, in
`mission_templates()`, or on the `Api`. Nothing a user could do would reach them.

Implemented and unreachable is indistinguishable from missing to everyone outside the
repository, and the registry counted the rows as done. All three are now on the API and
verified end to end:

| Mission | Route | Verified |
|---|---|---|
| Pylon | `Api.plan_pylon_mission` | 144 waypoints, 3 stacked orbit levels, all elements covered |
| Thermal | `Api.plan_thermal_survey` | altitude solved for the THERMAL sensor: 38.2 m for a 5 cm thermal GSD |
| Multispectral | `Api.plan_multispectral_survey` | refuses without a reflectance panel; plans with one, 5+ bands |

Box and dome are now built too (see below), which leaves **one**: wind turbine. They remain the cheapest real
wins on this list: the generator, camera model, overlap solver and simulator all exist, so
each is a new geometry routine rather than new infrastructure.

## 5–6. Terrain, obstacles, linking

| Market feature | ODK name | State |
|---|---|---|
| SRTM terrain following | `mp.terrain_follow` | registered |
| Custom DEM/DSM GeoTIFF | `mp.terrain_offline` | registered |
| Obstacle polygons | `mp.obstacles` | registered |
| Linked missions | `mp.linking` | registered |
| Geofence | `mp.geofence` | registered |

## 7. Offline

| Market feature | ODK name | State |
|---|---|---|
| Cached tiles and offline planning | `mp.terrain_offline` | registered — and offline-first is the project's whole premise |

## 8. Flight execution

| Market feature | ODK name | State |
|---|---|---|
| Preflight checks | `fl.preflight` | registered |
| Start/pause/resume/RTL | `fl.abstraction` | registered, SITL verified (`fl.sitl`) |
| Manual takeover | `fl.manual_override` | registered |
| Return settings | `fl.preflight` | partial — return height/threshold are parameters, not a row |
| Battery swap and resume | `fl.battery_swap` | registered |
| Telemetry | `fl.telemetry` | registered |

## 9. Camera and sensor

| Market feature | ODK name | State |
|---|---|---|
| Exposure modes | `fl.camera_control` | registered |
| Autofocus per shot | `fl.camera_control` | registered |
| Gimbal control incl. positive pitch | `fl.gimbal_control` | registered |
| Stills and video | `fl.camera_control` | registered |
| Radiometric JPEG selection | `th.radiometric` | registered |

## 10–11. Verification and logging

| Market feature | ODK name | State |
|---|---|---|
| On-site image verification | `fl.data_verification` | registered |
| Flight log export | `fl.logging` | registered |
| Automatic log sync / AirData | — | **gap — `fl.log_sync`** |

## 12. Processing

| Market feature | ODK name | State |
|---|---|---|
| Upload and project creation | `hub.projects` | registered |
| Ground control points | `pr.gcp` | registered |
| Orthomosaic | `pr.ortho` | registered — 77/77 frames, geo RMSE 1.110 m |
| 3D reconstruction | `pr.sfm`, `pr.mesh` | registered |
| Dense cloud | `pr.dense` | **registered but NOT working on this machine** — CUDA patch-match rejects the driver's PTX; the run falls back to sparse and says so |
| Low-overlap / ignore-GPS modes | `pr.gps_denied` | registered |
| RTK/PPK | `pr.rtk_ppk` | registered |
| Large datasets | `pr.large_datasets`, `pr.distributed` | registered |
| Export TIFF/OBJ/XYZ/LAS/KML | `ex.geotiff`, `ex.model_formats` | registered |

## 13. 3D model tools

| Market feature | ODK name | State |
|---|---|---|
| Orbit/zoom/pan viewer | `hub.viewer_3d`, `hub.point_cloud` | registered |
| Camera markers → source image | `hub.viewer_3d` | registered |
| Model position locking | — | **gap — `hub.saved_views`** |
| Facade mode toggle | — | **gap — folded into `hub.saved_views`** |
| Polygon clipping | — | **gap — `hub.clipping`** |
| Plane clipping (heading/pitch/roll) | — | **gap — folded into `hub.clipping`** |
| Saved, named clips | — | **gap — folded into `hub.clipping`** |

Clipping and saved views are the clearest missing *deliverable* feature: they are what
turns a raw model into something a client can be handed.

## 14. Inspection and annotation

| Market feature | ODK name | State |
|---|---|---|
| Full-resolution image review | `hub.viewer_2d` | registered |
| Rectangle/polygon annotations, comments, tags, severity | `in.annotations`, `in.defect_record` | registered |
| Defect library | `in.defect_library` | registered |
| Filtering by tag/AI/severity | `in.annotations` | partial — the record supports it; filtering UI is thin |
| Bulk tagging | — | **gap — `in.bulk_tagging`** |
| AI-assisted comments | `ai.assisted_annotation` | registered |

## 15. Measurement

| Market feature | ODK name | State |
|---|---|---|
| Distance | `me.2d`, `me.3d` | registered |
| Area | `me.2d` | registered |
| Perimeter | `me.2d` | partial — area exists; perimeter is not exported separately |
| Volume | `me.volume` | registered |
| Cut and fill | `me.volume` | registered |
| Slope | `me.slope` | registered |
| Defect quantities into scope | `ai.quantification` | registered |
| Door/window component counts | — | **gap — `ai.components`** |

## 16. Thermal

| Market feature | ODK name | State |
|---|---|---|
| Radiometric ingest | `th.radiometric` | registered |
| 2D thermal map | `th.map_2d` | registered |
| 3D thermal model | `th.model_3d` | registered |
| RGB/thermal comparison | `th.comparison` | registered |
| Temperature scaling | `th.radiometric` | partial |

## 17. Inspection AI

| Market feature | ODK name | State |
|---|---|---|
| Crack / spalling / corrosion / staining | `ai.crack`, `ai.spalling`, `ai.corrosion` | registered, with measured numbers and named weaknesses |
| Solar defects | `ai.solar` | registered — mAP50 0.884 |
| Custom model training | `ai.custom_training` | registered |
| Automated annotation | `ai.assisted_annotation` | registered |
| Quantification | `ai.quantification` | registered |
| Back-projection to 3D | `ai.projection` | registered |
| Change monitoring | `ai.change_detection`, `eng.change` | registered |
| Colour-coded severity | `in.defect_record` | registered |
| Result notifications | — | **gap — `hub.notifications`** |
| **AI project assistant (natural language)** | — | **gap — `ai.assistant`** |
| Parking-structure workflow | — | **gap — `ai.parking`** (depends on `hub.slam_capture`) |

## 18. Interior / handheld capture

| Market feature | ODK name | State |
|---|---|---|
| Handheld SLAM LiDAR + 360 | — | **gap — `hub.slam_capture`** |

This is a second capture modality, not a feature. It implies LiDAR ingest, SLAM
trajectory handling and 360 image projection.

## 19. Time-based comparison

| Market feature | ODK name | State |
|---|---|---|
| Chronological timeline, split-screen compare | `hub.timeline` | registered |
| Carrying measurements across versions | `ai.progress_tracking` | registered |

## 20. CAD overlays

| Market feature | ODK name | State |
|---|---|---|
| DXF overlay by EPSG | — | **gap — `hub.cad_overlay`** |
| PNG/JPG overlay by bounding box | — | folded into `hub.cad_overlay` |

## 21. Reporting

| Market feature | ODK name | State |
|---|---|---|
| PDF reports | `rp.reports`, `rp.formats` | registered |
| Word output | `rp.formats` | partial — PDF ships; DOCX does not |
| Templates, logo, cover, intro | `rp.templates` | registered |
| Severity-first ordering | `rp.reports` | partial |
| Guest links per image | `sh.links` | registered |

## 22. Sharing

| Market feature | ODK name | State |
|---|---|---|
| Public/guest links | `sh.links`, `sh.security` | registered |
| Curated client view | — | **gap — `hub.saved_views`** (same row as §13) |
| Teams, roles, permissions | `hub.orgs`, `hub.auth` | registered |

## 23–24. Hardware and plans

| Market feature | ODK name | State |
|---|---|---|
| DJI aircraft families | `fl.abstraction`, `sdk.*` | partial — the abstraction layer exists and is SITL verified; the DJI SDK matrix is not certified per airframe |
| Camera/payload profiles | `mp.camera_db`, `mp.payload_db` | registered |
| Subscription tiers | — | out of scope: this project is offline-first and unlicensed |

---

## The buildable gap list

Sixteen new rows, in the order I would build them. Ranked by value per unit of work, not
by where they appear above.

| # | ODK id | What it is | Why this rank |
|---|---|---|---|
| 1 | `mp.tpl_turbine` | Wind turbine mission | Uses `mp.fly_to_draw` for the recorded points; the rest exists |
| 2 | `mp.tpl_box` | **done this session** | |
| 3 | `mp.tpl_dome` | **done this session** — the tower alias was a real geometry defect, not a naming one | |
| 4 | `mp.tpl_pylon` | **done this session** — was implemented and unreachable |
| 5 | `hub.clipping` | Polygon and plane clipping, saved named clips | The main thing standing between a raw model and a deliverable |
| 6 | `hub.saved_views` | Saved camera position, facade mode, curated share | Pairs with clipping; small once the viewer holds state |
| 7 | `in.bulk_tagging` | Multi-select and tag | Cheap; large day-to-day effect |
| 8 | `hub.cad_overlay` | DXF by EPSG, raster by bbox | Self-contained; we already handle CRS properly |
| 9 | `rp.formats` (extend) | DOCX output | PDF pipeline already exists |
| 10 | `me.2d` (extend) | Perimeter alongside area | Hours, not days |
| 11 | `hub.notifications` | Job and AI completion notices | Needed the moment jobs run long |
| 12 | `fl.log_sync` | Automatic flight-log upload | Small, regulatory value |
| 13 | `ai.components` | Door/window counts | New model + corpus |
| 14 | `ai.assistant` | Natural-language project questions | Large; needs an LLM path and a grounding story |
| 15 | `hub.slam_capture` | Handheld LiDAR + 360 ingest | A second capture modality |
| 16 | `hub.field_app` | Separate mobile/controller build | A second application |

**Items 1–3 are one focused piece of work** and would close §4 of their inventory
outright; item 4 is already done. Items 5–7 are what an inspector notices first. Items 14–16 are each a project.

## What I am not going to pretend

`pr.dense` has a row and does not work on this machine. `eng.semantic` has a row and two
of its six classes are usable. Counting either as parity with a shipped competitor feature
would be exactly the kind of claim the rest of this repository is arranged to prevent.
