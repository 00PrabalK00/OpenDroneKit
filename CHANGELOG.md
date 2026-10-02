# Changelog

All notable changes to OpenDroneKit are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). OpenDroneKit
has not made a tagged release yet, so everything is under **Unreleased** until the first
version is cut; the project intends to use [Semantic Versioning](https://semver.org/)
from then on.

## [Unreleased]

The largest round of work so far: every area of the application was audited for features
that were broken, unreachable from the UI, or producing fabricated output, and each was
fixed or made to refuse with a reason. Feature status in `docs/FEATURES.md` is computed
from passing tests, as before.

### Added

- **Licence**: OpenDroneKit is released under the MIT License (`LICENSE`).
- **Brand**: a new quad mark and wordmark (light, dark and mono SVG masters in
  `app/web/assets/brand/`), with every favicon, app icon, toolbar mark and Hub header
  generated from them by `tools/brand_assets.py`. The desktop window uses the app icon.
- **Launcher and environment**: `environment.yml` defines the reference `odk` conda
  environment (Python 3.11). `python main.py --check-deps` names missing or too-old
  packages, and `OpenDroneKit.bat` runs it before every start and offers to install them.
- **Explicit UI choice**: `--ui classic|cockpit` and `ODK_UI` (case-insensitive) choose
  the interface; an unknown value is refused visibly, even under `pythonw`.
- **Offline basemap**: the desktop shell serves cached tiles from disk, *Cache view*
  downloads the visible area and three deeper zoom levels, and the map switches to the
  cache when online basemaps are unreachable and says so on the map.
- **In-process Hub**: the desktop shell starts the Hub REST service on loopback, and the
  browser Hub gains a sign-in / create-account dialog and a Data and jobs panel.
- **Measurement from the map**: distance, area, perimeter, profile, slope, volume,
  cut/fill and stockpile measured on the project's own rasters from drawn geometry, kept
  in the project's measurement history; survey-to-survey comparison across different grids.
- **Ground control**: GCPs are stored with the project, marked on photographs, and used
  to georeference COLMAP runs (three or more marked points), with a fit/check report and
  leave-one-out residuals.
- **Custom engine**: real incremental structure-from-motion (SIFT matching, PnP
  registration, triangulation, bundle adjustment) with georeferencing and COG products,
  replacing the earlier frame strip.
- **Cloud reconstruction**: `--recon-mode cloud` uploads to a self-hosted OpenDroneKit
  server through the resumable, sha256-verified upload API, follows the job, and
  downloads its products. It needs an explicit endpoint and token.
- **Hub jobs**: detection, measurement and report job kinds have workers; an AI
  inference worker (`python -m services.worker.ai_worker`); webhooks fire; artifacts are
  served by storage reference.
- **Flight**: a MAVSDK driver with live telemetry and mission/fence/rally upload; both
  UIs fly the aircraft through the real `Api`; flight logs mark simulated samples.
- **Missions**: dome and box inspection have their own geometry; exact terrain following
  over DEM triangulation; per-template gimbal and dwell defaults; every template the
  planner accepts is listed in the application.
- **Cockpit**: the Digital Twin draws the project's model in 3D with picking, in-model
  measurement and saved views; Coverage QA runs on flown images; Temperature Range reads
  radiometric frames; the inspection report opens when a pipeline job finishes; AI
  labelled boxes can be trained into a deployed custom model from inside the app.
- **Reports**: PDF and Word deliverables built from the open project, with distinct
  report types; refused report types say what is missing.
- **India survey packs** run from the cockpit with all their inputs. Anomaly
  baseline/find accepts the app's own rasters (vegetation indices tag their band schema;
  an untagged raster can be given a declared schema, recorded as declared); power/rail
  corridors accept a WGS84 centreline and measure it in the UTM zone of its midpoint;
  land GIS marks each category validated, unvalidated or unknown.
- **Deep zoom**: project rasters are served as XYZ tiles cut from the GeoTIFF at the
  survey's native resolution (`core/raster_tiles.py`, `Api.raster_tiles`), with the PNG
  preview as fallback; both maps zoom to z24.
- **Measurement on a photograph**: Photo Distance, Photo Area and Crack Length in the
  cockpit's Measurements, in source-frame pixels, scaled by an operator mm/px or a
  reference of known length on the same photograph (never assumed)
  (`Api.measure_on_image`, `Api.measure_crack_on_image`).
- **Run AI Settings "Structural detection"** (and `run_pipeline --structural-detection`)
  switches the crack and structural models back on for a declared solar asset.
- **Live-service tests**: `tests/test_live_stack.py` runs a job through PostGIS, Redis, a
  separate worker process and the S3 store; `tests/test_compose.py` gains a live boot of
  the compose object store (`ODK_TEST_DOCKER_COMPOSE=1`).
- **Community files**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant
  2.1), `SECURITY.md`, `SUPPORT.md`, issue forms and a pull request checklist, this
  changelog and `CITATION.cff`; README badges and a Contributing section.
- **Docs**: the dense-MVS guide for RTX 50-series GPUs, measured on the development
  machine; the installation guide describes the launcher, data root and offline cache.

### Changed

- All application state lives under one absolute data root (`ODK_DATA_DIR`, default
  `final_toolkit_outputs/`), so the app finds its projects whichever folder it starts in.
  Tests run against an isolated data root.
- The Hub API allows loopback origins on any port so the desktop shell can reach it
  (`ODK_CORS_LOOPBACK=0` disables this).
- Flight commands wait for `COMMAND_ACK` and report accepted, denied or unacknowledged;
  abort holds position (BRAKE/LOITER) before falling back to pause; the flight state is
  "flying" only once telemetry shows armed and AUTO; `MissionManager` defaults to no
  drone rather than a silent simulator.
- Mission exporters carry the plan's poses, yaw, gimbal pitch, dwell, speed, fences and
  limits; the WPL file reserves sequence 0 for home; Litchi missions split at 99
  waypoints; AMSL plans export absolute altitudes.
- Structure missions (facade, wind turbine, dome and the other structure templates) keep their stand-off along the whole
  path and face the structure.
- Preflight: Start Mission is held to the checklist, and an override is recorded.
- Detection-job findings are recorded with severity `unassessed` until a reviewer grades
  them; model detections become reviewable findings, heuristics do not.
- The crack presence classifier is a recorded second opinion, not a default gate.
- Health scoring is an uncalibrated severity index; heuristic or refused solar and metal
  passes no longer move it.
- Core package exports resolve lazily, so the window opens without loading the
  processing stack.
- Object storage in the Docker Compose stack and the Helm chart is a SeaweedFS S3 gateway
  (`chrislusf/seaweedfs`, port 8333) instead of MinIO, which no longer publishes public
  images. A one-shot init creates the bucket; the in-stack endpoint is
  `http://seaweedfs:8333`; env var names and the storage API are unchanged. The Helm
  chart now deploys the store it points at, and refuses to render `storage.backend=s3`
  without S3 keys.
- The Celery worker uses the solo pool on Windows (prefork elsewhere; `ODK_WORKER_POOL`
  overrides).
- With no ceiling given, the planner's default ceiling admits its default RTH climb
  (max altitude + 10 m); an explicit ceiling is untouched.
- On solar EL cell or IR module imagery, Run AI runs no concrete surface model (crack,
  structural, metal), whether or not the imagery was declared: an undeclared grey image
  with the signs of an EL cell is recognised as one. A run with no assessed defect
  component has no health index or grade ("Not assessed") instead of a score from the
  coverage penalty alone.
- Saved mission versions record the planned speed, and a repeat re-plans at it.
- A connection that did not open is answered `ok: false` with the vehicle's error, and
  its flight-log block describes the attempted driver and link; both Connect UIs show the
  reason.
- The cockpit's Mission Properties form plans with the values it shows (untouched fields
  included), Mission Types sets Type, and the Planned message states what was sent.

### Fixed

- Run AI, Run Full Pipeline, Size Job, New Job and the Reports workspace complete end
  to end, report progress, and can be cancelled.
- Every installed model is reachable from AI Inspection; a custom model deployment no
  longer breaks grayscale image reads app-wide; the semantic map runs on machines where
  CUDA is listed but does not load.
- Dense-stereo capability is probed by running it, names the real cause (for example a
  CUDA toolkit newer than the driver), does not call a JIT-compiling COLMAP broken, and
  every capabilities screen shows the reason. Dense runs no longer label their fused
  cloud sparse.
- Offline map: tiles are labelled by their real type, a burst of tile requests on a busy
  machine is no longer refused, and zooming past the deepest cached level enlarges real
  cached tiles instead of requesting tiles that were never cached.
- Share links, webhooks and API keys work on a fresh install; New Key issues a real API
  token.
- Fleet: adding aircraft, batteries and pilots, maintenance, mission assignment and
  export all work; a failed read says so instead of showing an empty list.
- Reports: an empty project is refused a report on every path; a failed reconstruction
  is reported; the Markdown and HTML reports agree on what a run did.
- Terrain: every raster format is checked, ESRI grids are cached with their `.prj`, and
  projected-coordinate CSVs and feet DEMs are refused.
- Measurements judge true scale from the CRS itself; Web Mercator and feet CRSs are
  refused for surface analyses; over-the-ground length is withheld across DSM holes;
  defects meeting a DSM hole are placed rather than dropped.
- Coverage QA places images from EXIF GPS and camera geometry, and refuses a
  non-numeric height in words.
- The SDK's `wait_for_job` no longer hangs on finished jobs; artifact names with spaces
  download.
- The classic window no longer sticks on "Ready" with no map when the network is
  unreachable at start.
- A Compare no longer makes the open project unreadable to the UI.
- Map deep zoom: past the depth Esri World Imagery holds imagery for, its grey "Map data
  not yet available" tile is recognised and replaced by the enlarged nearest real
  ancestor, and the basemap is capped at the deepest level the service lists for the
  view; the offline cache never stores the placeholder and purges cached ones.
- Flight: the MAVLink bridge requests `HOME_POSITION` on connect and every 2 s until home
  arrives (ArduPilot does not stream it), so a lifecycle no longer waits on the ground
  for a home that never comes.
- Litchi CSV: action codes corrected (0 = stay for ms, 1 = take photo), so structure
  missions take their photographs instead of tilting the camera to 0 deg; Mission >
  Export Litchi CSV (and the other single-format items) writes only that format.
- Missions: multi-facade transits go round corners at the stand-off instead of cutting
  them; a turbine's back-face detour is chosen inside the altitude band and every leg is
  re-checked against the nacelle.
- Crack length follows the skeleton as a polyline (pixel counts made a 45 deg crack 29%
  short); photo bounds honour EXIF orientation.
- Import Terrain, Import Vector, Import Raster and Add Layer open their file chooser in
  the default window; the cockpit says when the chooser was cancelled and whether the DEM
  covers the site.
- Mission version pickers: diff/history no longer raise once two versions exist; Compare
  Versions, History and Repeat Survey show the Api's results, and a modified-boundary
  repeat re-plans the new area instead of reporting 0 capture points. The Cameras picker
  describes the chosen camera rather than a generic placeholder.
- Cockpit frame: the status bar's CRS comes from the project's layers, the clock shows no
  reading before the bridge answers, and the Simulation Timeline is drawn from the
  planned route.
- The model-weight tests and live S3 storage tests find what the application uses
  (`ODK_MODEL_WEIGHTS_DIR`, a test-owned bucket) instead of skipping or refusing.

### Removed

- Fabricated outputs, each replaced by a measurement or a refusal with a reason:
  failure probabilities from a hard-coded hazard table, forecast-driven health scores,
  the hard-coded coverage layout and ORB-ratio overlap flag, the frame-strip
  "orthomosaic" at an assumed overlap, pseudo-confidence and whole-frame boxes from the
  classical detector, hard-coded `medium` severity on model findings, preflight checks
  hard-coded to pass, and crack length guessed from mask pixels.
- The old PNG logo masters.

### Security

- Reconstruction results served as artifacts no longer disclose absolute server paths.
- Cloud reconstruction uploads nothing without an explicit endpoint and token.
- The desktop UI server and the in-process Hub bind to `127.0.0.1` only.
- Sharing records name the desktop project by id, never its folder path.

[Unreleased]: https://github.com/00PrabalK00/OpenDroneKit/commits/main
