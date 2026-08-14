# Tasks

India-first order is fixed. Details, evidence and licence gates are in
`docs/INDIA_FIRST_AI_PLAN.md`; the machine-readable plan is
`training/india_first_plan.py`.

- [x] 1. Selected-ROI stockpile/pit change between surveys
  - [x] Constrain DSM comparison and volumes to an operator polygon.
  - [x] Save ROI, difference raster, mapped regions, JSON and client report.
  - [x] Add stockpile/pit workflow integration and tests.
  - [ ] Validate against one real repeat survey and independent ground volume.

- [ ] 2. Shared semantic segmentation engine — STARTED
  - [x] Select DINOv2 ViT-B/14 plus an internal UPerNet-style decoder.
  - [x] Record SpaceNet 7 and licence-filtered OpenEarthMap data strategy.
  - [x] Download and hash the foundation checkpoint.
  - [x] Pin and retain the official DINOv2 source used to construct the encoder.
  - [x] Freeze versioned class-schema and task-trained model-manifest contracts.
  - [x] Implement the internal multiscale UPerNet decoder over four DINOv2 layers.
  - [x] Implement overlap-blended tiled inference with explicit CPU limits.
  - [x] Export georeferenced class/confidence GeoTIFFs, polygons and provenance.
  - [x] Wire semantic workflow readiness and processing-stage execution.
  - [x] Add a production-licence corpus filter and deterministic split builder.
  - [x] Keep all dates and tiles from one source/site in exactly one split.
  - [x] Index official SpaceNet monthly imagery and building GeoJSON labels.
  - [x] Rasterize vector labels and load schema-aware DINOv2 training tiles.
  - [x] Add CUDA/AMP training with separate encoder/decoder rates and mIoU evidence.
  - [x] Add measured log-inverse class balancing for rare semantic classes.
  - [x] Add cosine scheduling and atomic resumable best/last checkpoints.
  - [x] Add full-model ONNX export, parity check and hash-locked runtime manifest.
  - [x] Finish MD5-verified extraction of the official SpaceNet 7 archive.
  - [x] Add 1,423 SpaceNet building samples across 60 site-disjoint groups.
  - [x] Register the OpenEarthMap archive with its official MD5.
  - [x] Add the explicit CC BY/CC BY-SA per-region production allowlist.
  - [x] Map dense OpenEarthMap labels to the shared schema and ignore pavement.
  - [x] Mark non-building SpaceNet pixels as unknown instead of false background.
  - [x] Merge compatible source manifests without leaking sites across splits.
  - [x] Verify full 518-pixel six-class forward and backward passes on the local RTX 5060.
  - [x] Finish MD5-verified OpenEarthMap download, extraction and indexing.
  - [x] Populate the licence-filtered shared-class training corpus.
  - [x] Pass coverage/class-weight/tile-read readiness without starting training.
  - [ ] Train the complete encoder-decoder model.
  - [ ] Export and register the trained encoder-decoder ONNX.
  - [ ] Train shared classes with site/date-separated validation.

- [ ] 3. Construction site segmentation — STARTED
  - [x] Record IARPA SMART and SpaceNet 7 sources and their limits.
  - [x] Select the shared semantic backbone.
  - [ ] Freeze the ten-class annotation guide.
  - [ ] Build India drone labels for excavation, stockpile and materials.
  - [ ] Train, evaluate and register the construction head.

- [ ] 4. Progress against approved designs — STARTED
  - [x] Select geometry-first IFC/CAD comparison instead of an ML claim.
  - [ ] Add IFC/DXF/GeoJSON ingestion and coordinate registration.
  - [ ] Define planned, observed, uncertain and manually accepted states.
  - [ ] Generate element quantities and progress report.

- [ ] 5. Solar RGB/thermal alignment and module inventory — STARTED
  - [x] Select robust geometric registration plus YOLO11l-seg modules.
  - [x] Register Duke UAV PV and InfraredSolarModules sources.
  - [x] Download and hash YOLO11l-seg initialization.
  - [ ] Add calibration profile and registration quality score.
  - [ ] Train module masks, geotag inventory and temperature anomalies.

- [ ] 6. Land GIS extraction and encroachment — STARTED
  - [x] Select the shared semantic engine plus deterministic GIS overlay.
  - [x] Record SpaceNet 7 and Open Buildings validation sources.
  - [ ] Add topology-safe mask polygonization and class layers.
  - [ ] Compare survey footprints to boundaries and prior surveys.

- [ ] 7. Agriculture indices, canopy, stress and counting — STARTED
  - [x] Select indices first, WeedsGalore DeepLabv3+ MSI and YOLO11l-seg.
  - [x] Register and download the WeedsGalore domain checkpoint bundle.
  - [ ] Implement NDVI, NDRE and GNDVI with calibration/no-data handling.
  - [ ] Adapt canopy/crop/weed segmentation to India flights.
  - [ ] Validate plant/tree counts and stress anomaly maps by crop.

- [ ] 8. Roads — STARTED
  - [x] Select YOLO11x at high resolution plus shared road segmentation.
  - [x] Register RDD2022 India and China Drone sources.
  - [x] Download and hash YOLO11x initialization.
  - [ ] Convert VOC labels, train with spatial holdouts and map severity.

- [ ] 9. Specialized power-line and railway models — STARTED
  - [x] Select YOLO11x assets plus shared rail segmentation.
  - [x] Record InsPLAD as research-only and UAV-RSOD as CC BY 4.0.
  - [ ] Train and evaluate rail/gauge segmentation and obstacle detection.
  - [ ] Obtain commercial-compatible power-line labels or permission.
  - [ ] Train power assets/anomalies only after the licence gate clears.

