# Baseline feature status (before the v1 interface)

Recorded in U00 so every rebuild unit can prove it downgraded nothing. A unit's definition
of done includes: `tools/feature_status.py` shows no feature lower than in this table.

## How it was run

- Branch `ship/fix-all` at `0ae10e4b3`, before any rebuild file existed in the tree (the run
  started before the U00/U1 files were copied in, so nothing in this table depends on them).
- Command, from the repository root, the same way round 1 ran it:

  ```
  set ODK_MODEL_WEIGHTS_DIR=E:/Shit was cluittering/Projects/OpenDroneKit/models
  set ODK_MODELS_DIR=E:/Shit was cluittering/Projects/OpenDroneKit/models
  C:/Users/Prabal/miniconda3/envs/odk/python.exe tools/feature_status.py --json
  ```

- Full pytest run on this laptop (Windows 11, trained weights installed), 2026-10-02 01:10 to 01:41 EDT (30 min).
  No `--extra-report`: SITL and live-PostGIS evidence from CI was not merged, so rows that
  only those environments can prove are lower here than in `docs/FEATURES.md`. Compare a
  later run against this table on the same machine and the same way, not against FEATURES.md.

## Summary

| Status | This run (laptop, no CI evidence) | docs/FEATURES.md (published, with CI evidence) |
|---|---:|---:|
| verified | 150 | 164 |
| implemented | 13 | 0 |
| in_progress | 3 | 2 |
| not_started | 2 | 2 |
| **total** | **168** | **168** |

## Rows below their claim in this run (4)

Each is the tool's own reason. These are the baseline: a later unit must not add to this
list or lower any row further.

| Feature | Claimed | This run | Reason |
|---|---|---|---|
| `fl.sitl` SITL verified flight | verified | implemented | test selector matched nothing: tests/sitl/test_flight_lifecycle.py, tests/sitl/test_mission_upload.py |
| `ai.solar` Solar defect detection | verified | implemented | test selector matched nothing: tests/test_solar_and_metal_use_real_models.py::TestSolarRunsTheModelForItsImagery |
| `hub.auth` Authentication | implemented | in_progress | 2 named test(s) failing |
| `inf.postgis` PostgreSQL and PostGIS | verified | implemented | test selector matched nothing: tests/test_postgis_spatial.py |

## Every row not verified in this run (18)

Includes rows claimed below verified. Most are evidence this laptop cannot produce (SITL,
PostGIS, a COLMAP run on the survey imagery); `hub.auth` has two failing tests on ship/fix-all.

| Feature | Claimed | This run | Reason |
|---|---|---|---|
| `fl.sitl` SITL verified flight | verified | implemented | test selector matched nothing: tests/sitl/test_flight_lifecycle.py, tests/sitl/test_mission_upload.py |
| `pr.sfm` Structure from motion | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestStructureFromMotion, tests/test_reconstruction_colmap.py::TestNoFabrication, tests/test_custom_engine_reconstructs.py |
| `pr.ortho` Orthomosaic generation | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestGeoreferencing, tests/test_reconstruction_colmap.py::TestSparseProductsAreUsableAndHonest, tests/test_custom_engine_reconstructs.py::test_the_orthomosaic_is_georeferenced_and_rectified_not_a_strip |
| `pr.dsm_dtm` DSM and DTM | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestGeoreferencing, tests/test_reconstruction_colmap.py::TestNoFabrication |
| `ai.solar` Solar defect detection | verified | implemented | test selector matched nothing: tests/test_solar_and_metal_use_real_models.py::TestSolarRunsTheModelForItsImagery |
| `ai.assisted_annotation` AI assisted annotation | implemented | implemented | test selector matched nothing: tests/test_ai_user_path.py::TestRunAIFromTheApi |
| `eng.semantic` Semantic understanding engine | in_progress | in_progress | tests pass but the feature is not claimed complete |
| `india.foundation.orthomosaic` Survey orthomosaic | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestGeoreferencing::test_rasters_are_written_as_georeferenced_geotiffs |
| `india.foundation.dsm_dtm` Survey DSM and DTM | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestGeoreferencing::test_rasters_are_written_as_georeferenced_geotiffs, tests/test_reconstruction_colmap.py::TestGeoreferencing::test_the_dsm_holds_elevations_in_metres, tests/test_reconstruction_colmap.py::TestGeoreferencing::test_the_dtm_holds_ground_elevations_in_metres, tests/test_reconstruction_colmap.py::TestNoFabrication |
| `india.foundation.reconstruction` Survey 3D reconstruction | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestStructureFromMotion, tests/test_reconstruction_colmap.py::TestGeoreferencing, tests/test_reconstruction_colmap.py::TestNoFabrication |
| `india.foundation.area` Survey area and distance | implemented | implemented | test selector matched nothing: tests/test_reconstruction_colmap.py::TestGeoreferencing::test_the_real_orthomosaic_supports_metric_area_and_distance |
| `india.foundation.deliverable` Georeferenced client deliverable | implemented | implemented | test selector matched nothing: tests/test_provenance.py::TestRealReconstruction, tests/test_reconstruction_colmap.py::TestGeoreferencing |
| `pack.construction.segmentation` Construction site segmentation | in_progress | in_progress | tests pass but the feature is not claimed complete |
| `pack.mining.stockpile_segmentation` Automatic stockpile segmentation | not_started | not_started | (claimed status, no tests named) |
| `pack.mining.scene` Mine and quarry segmentation | not_started | not_started | (claimed status, no tests named) |
| `hub.auth` Authentication | implemented | in_progress | 2 named test(s) failing |
| `rp.reports` Automated inspection reports | implemented | implemented | test selector matched nothing: tests/test_the_pipeline_report_is_reachable.py::TestTheMarkdownSaysOnlyWhatHappened, tests/test_the_pipeline_report_is_reachable.py::TestTheHtmlReportSaysTheReconstructionFailed, tests/test_the_pipeline_report_is_reachable.py::TestBothWindowsShowIt |
| `inf.postgis` PostgreSQL and PostGIS | verified | implemented | test selector matched nothing: tests/test_postgis_spatial.py |

## Every row

| Feature | Status |
|---|---|
| `mp.engine.independent` | verified |
| `mp.output.contract` | verified |
| `mp.type.mapping_2d` | verified |
| `mp.type.mapping_3d` | verified |
| `mp.type.roof_mapping` | verified |
| `mp.type.roof_inspection` | verified |
| `mp.type.facade_mapping` | verified |
| `mp.type.facade_inspection` | verified |
| `mp.type.multi_facade` | verified |
| `mp.type.complex_facade` | verified |
| `mp.type.closed_loop` | verified |
| `mp.type.l_shaped` | verified |
| `mp.type.linear_mapping` | verified |
| `mp.type.linear_inspection` | verified |
| `mp.type.tower_mapping` | verified |
| `mp.type.pylon_inspection` | verified |
| `mp.type.orbit` | verified |
| `mp.type.waypoints` | verified |
| `mp.type.panorama` | verified |
| `mp.type.solar` | verified |
| `mp.type.wind_turbine` | verified |
| `mp.type.magnetic` | verified |
| `mp.type.thermal` | verified |
| `mp.type.multispectral` | verified |
| `mp.linking` | verified |
| `mp.geometry_3d` | verified |
| `mp.standoff` | verified |
| `mp.gsd` | verified |
| `mp.overlap` | verified |
| `mp.terrain_follow` | verified |
| `mp.terrain_offline` | verified |
| `mp.obstacles` | verified |
| `mp.geofence` | verified |
| `mp.estimates` | verified |
| `mp.simulation` | verified |
| `mp.versioning` | verified |
| `mp.sharing` | verified |
| `mp.repeatable` | verified |
| `mp.import` | verified |
| `mp.fly_to_draw` | verified |
| `mp.camera_db` | verified |
| `mp.payload_db` | verified |
| `fl.abstraction` | verified |
| `fl.mavlink.upload` | verified |
| `fl.mavlink.fence` | verified |
| `fl.sitl` | implemented |
| `fl.preflight` | verified |
| `fl.telemetry` | verified |
| `fl.manual_override` | verified |
| `fl.battery_swap` | verified |
| `fl.crash_recovery` | verified |
| `fl.camera_control` | verified |
| `fl.gimbal_control` | verified |
| `fl.logging` | verified |
| `fl.data_verification` | verified |
| `fl.capture.match` | verified |
| `fl.capture.deviation` | verified |
| `fl.capture.priors` | verified |
| `pr.sfm` | implemented |
| `pr.georeference` | verified |
| `pr.ortho` | implemented |
| `pr.dsm_dtm` | implemented |
| `pr.dense` | verified |
| `pr.mesh` | verified |
| `pr.gps_denied` | verified |
| `pr.gcp` | verified |
| `pr.rtk_ppk` | verified |
| `pr.distributed` | verified |
| `pr.large_datasets` | verified |
| `pr.provenance` | verified |
| `ai.registry` | verified |
| `ai.parity` | verified |
| `ai.crack` | verified |
| `ai.spalling` | verified |
| `ai.corrosion` | verified |
| `ai.solar` | implemented |
| `ai.metal` | verified |
| `ai.water_ponding` | verified |
| `ai.deformation` | verified |
| `ai.custom_training` | verified |
| `ai.assisted_annotation` | implemented |
| `ai.human_validation` | verified |
| `ai.quantification` | verified |
| `ai.projection` | verified |
| `ai.change_detection` | verified |
| `ai.progress_tracking` | verified |
| `ai.model_version_recorded` | verified |
| `eng.semantic` | in_progress |
| `eng.assets` | verified |
| `eng.change` | verified |
| `eng.anomaly` | verified |
| `india.foundation.orthomosaic` | implemented |
| `india.foundation.dsm_dtm` | implemented |
| `india.foundation.reconstruction` | implemented |
| `india.foundation.area` | implemented |
| `india.foundation.volume` | verified |
| `india.foundation.cut_fill` | verified |
| `india.foundation.deliverable` | implemented |
| `pack.construction.change` | verified |
| `pack.construction.segmentation` | in_progress |
| `pack.construction.progress` | verified |
| `pack.mining.stockpile` | verified |
| `pack.mining.stockpile_segmentation` | not_started |
| `pack.mining.scene` | not_started |
| `pack.mining.change` | verified |
| `pack.solar.inventory` | verified |
| `pack.solar.thermal` | verified |
| `pack.land.gis` | verified |
| `pack.land.encroachment` | verified |
| `pack.agriculture.canopy` | verified |
| `pack.agriculture.indices` | verified |
| `pack.agriculture.stress` | verified |
| `pack.agriculture.count` | verified |
| `pack.roads.condition` | verified |
| `pack.power.inspection` | verified |
| `pack.rail.inspection` | verified |
| `in.annotations` | verified |
| `in.defect_library` | verified |
| `in.defect_record` | verified |
| `me.2d` | verified |
| `me.3d` | verified |
| `me.volume` | verified |
| `me.slope` | verified |
| `th.radiometric` | verified |
| `th.map_2d` | verified |
| `th.model_3d` | verified |
| `th.comparison` | verified |
| `hub.map.basemaps` | verified |
| `hub.map.geocoding` | verified |
| `hub.map.offline_tiles` | verified |
| `hub.projects` | verified |
| `hub.assets` | verified |
| `hub.orgs` | verified |
| `hub.auth` | in_progress |
| `hub.viewer_2d` | verified |
| `hub.viewer_3d` | verified |
| `hub.point_cloud` | verified |
| `hub.digital_twin` | verified |
| `hub.timeline` | verified |
| `rp.reports` | implemented |
| `rp.formats` | verified |
| `rp.templates` | verified |
| `sh.links` | verified |
| `sh.security` | verified |
| `ex.geotiff` | verified |
| `ex.mission_formats` | verified |
| `ex.model_formats` | verified |
| `api.rest` | verified |
| `api.uploads` | verified |
| `api.upload_safety` | verified |
| `api.webhooks` | verified |
| `api.realtime` | verified |
| `fm.fleet` | verified |
| `fm.battery` | verified |
| `fm.pilots` | verified |
| `fm.maintenance` | verified |
| `sec.audit` | verified |
| `sec.rbac` | verified |
| `inf.docker` | verified |
| `inf.k8s` | verified |
| `inf.postgis` | implemented |
| `inf.storage` | verified |
| `inf.offline_first` | verified |
| `inf.observability` | verified |
| `sdk.plugin_system` | verified |
| `sdk.libraries` | verified |
| `doc.guides` | verified |
| `demo.mode` | verified |
