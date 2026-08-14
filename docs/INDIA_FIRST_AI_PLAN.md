# India-first survey intelligence plan

Updated: 2026-08-14

The product contract is survey to decision: report what changed, how much, where,
and what evidence supports the interpretation. Photogrammetry and geometry remain
the source of truth for measurements. AI supplies semantic labels, instances and
anomaly ranking; it must never manufacture a volume or a cause.

## Chosen implementation order

| # | Workstream | State | Selected approach |
|---:|---|---|---|
| 1 | Selected stockpile or pit ROI change | Complete | Polygon-constrained DSM differencing |
| 2 | Shared semantic engine | Started | DINOv2 ViT-B/14 encoder plus UPerNet-style head |
| 3 | Construction segmentation | Started | Shared engine, construction-specific class head |
| 4 | Approved-design progress | Started | IFC/CAD registration and deterministic geometry |
| 5 | Solar RGB/thermal and inventory | Started | Geometric registration plus YOLO11l-seg modules |
| 6 | Land GIS and encroachment | Started | Shared masks to GIS polygons plus overlay rules |
| 7 | Agriculture | Started | Indices first, DeepLabv3+ MSI, instance counting |
| 8 | Roads | Started | Shared road mask plus YOLO11x defects and mapping |
| 9 | Power and rail | Started | Separate close-range detectors and corridor masks |

Complete means code, tests and a client artifact contract exist. Started means the
dataset, architecture, licence boundary and next implementation action are recorded;
it does not mean a trained production model exists.

## Model decisions

### Shared semantic and anomaly backbone

Use DINOv2 ViT-B/14 as the common encoder. It is strong enough for high-resolution
transfer learning, fits a 24 GB training GPU with tiled inputs and gradient
accumulation, and the official code and standard model weights are Apache-2.0.
Build an internal UPerNet-style multiscale decoder for land, construction, road and
rail heads. DINOv2 features can also drive PatchCore-style anomaly ranking.

The shared runtime contract is implemented in `core/semantic_engine.py`: it performs
overlap-blended tiled inference, preserves the orthomosaic CRS and transform, and
emits class and confidence GeoTIFFs, polygon GeoJSON and a provenance manifest. It
rejects untrained foundation weights and bounds explicitly opted-in CPU runs. The
architecture-only training definition is
`training/configs/shared_semantic_dinov2_vitb14.yaml`; no production semantic head
is claimed until the full encoder-decoder is trained and passes the holdout gates.
`training/semantic_corpus.py` builds a deterministic corpus manifest, excludes
licences outside the production allowlist, requires capture dates and keeps every
tile and repeat flight from one source/site in a single split.
`training/datasets/spacenet7.py` pairs official monthly `images_masked` rasters with
building GeoJSON, and `training/semantic_tiles.py` reprojects/rasterizes those vectors
and maps stable schema ids to contiguous training channels.
`training/train_shared_semantic.py` provides CUDA/AMP training, separate encoder and
decoder learning rates, ignore-safe multiclass metrics and provenance-rich best
checkpoints. It is executable only after the corpus contains nonempty train and
validation site groups.
`training/export_shared_semantic.py` exports the complete trained graph, checks
PyTorch/ONNX parity and writes the hash-locked task-trained runtime manifest. The
export path exists, but no export is claimed before a real training checkpoint does.

The existing NVIDIA SegFormer experiments remain valid research baselines, but the
official SegFormer licence restricts the work to non-commercial use. Therefore
SegFormer-B5 is not the selected production foundation.

### Object and instance models

Use YOLO11x for offline high-accuracy road, power and railway object detection. Use
YOLO11l-seg for solar modules and plant or tree instances, where masks matter more
than boxes. Ultralytics provides these under AGPL-3.0 or Enterprise terms. Deployment
must be AGPL compliant or covered by an Enterprise licence; the checkpoint manifest
records this gate.

### Multispectral agriculture model

Start from the official WeedsGalore DeepLabv3+ RGB and multispectral checkpoints.
They are a domain baseline, not an India-calibrated model. Indices such as NDVI,
NDRE and GNDVI are deterministic calibrated-band calculations and run before ML.

### Geometry-only engines

No neural model is selected for DSM volume change, approved-design comparison,
RGB/thermal calibration, GIS overlay, or vegetation indices. A model would add
uncertainty without replacing the required coordinate and sensor calibration.

## Dataset decisions and licence gates

| Workstream | Primary data | Licence/use decision | Purpose |
|---|---|---|---|
| Mining change | MineNetCD | CC BY 4.0 | RGB change semantics; not volume truth |
| Shared land semantics | SpaceNet 7 plus licence-filtered OpenEarthMap regions | CC BY-SA 4.0; OpenEarthMap varies by region | Buildings and transferable overhead features |
| Construction | IARPA SMART annotations plus local drone labels | Repository MIT; source imagery terms vary | Activity chronology, then India drone class labels |
| Design progress | buildingSMART samples and customer IFC/DXF/GeoJSON | Check each supplied design | Parser fixtures and real approved designs |
| Solar inventory | Duke UAV solar PV dataset | CC BY 4.0 | Panel or module masks; verify annotation granularity |
| Solar thermal | InfraredSolarModules | MIT | Module anomaly classification baseline |
| Agriculture | WeedsGalore | CC BY 4.0 | Multispectral maize crop and weed segmentation |
| Roads | RDD2022 India plus China Drone | CC BY-SA 4.0 | India appearance plus UAV viewpoint defects |
| Power | InsPLAD | CC BY-NC 3.0 | Research benchmark only; not production training |
| Rail | UAV-RSOD | CC BY 4.0 | Indian UAV rail/gauge masks and obstacles |

SpaceNet 7 is retained from the official S3 source (about 8.7 GB compressed and
roughly 25 GB extracted). The corpus builder references source files rather than
copying them, avoiding another full duplicate of the benchmark.

The retained archive passed MD5 `6eda13b9c28f6f5cdf00a7e8e218c1b1`. Indexing found
1,423/1,423 labelled monthly images with no missing pairs. The deterministic split
contains 1,015 train, 211 validation and 197 test samples across 43/9/8 disjoint
sites. SpaceNet provides building polygons only, so all pixels outside those
polygons are ignored for shared multiclass training; they are not falsely labelled
background. Until the dense source is merged, class ids 0 and 2-5 remain uncovered
and the trainer refuses to run.

OpenEarthMap is pinned to the official Zenodo archive with MD5
64155d1dc9d3b68536063f79878e1a67. Its labels inherit per-region source licences.
The adapter admits only regions explicitly listed as CC BY 4.0 or CC BY-SA 4.0 in
the official attribution table. It excludes CC BY-NC-SA, public-domain-source and
unspecified-source regions because the project assigns non-commercial terms to the
latter labels; DL-DE-BY-2.0 remains excluded pending legal review. Source classes
map as bare land to bare land, grass/tree/cropland to vegetation, road to road,
water to water and buildings to building. Pavement is ignored rather than
misrepresented as road. Missing acquisition dates carry an explicit audit reason,
while every tile from a region remains in one spatial split.

The trainer derives log-inverse class weights from measured mask pixel counts,
uses a cosine learning-rate schedule and writes atomic resumable best/last
checkpoints guarded by the exact schema and corpus hash. A complete 518 by 518
six-class forward/backward optimizer step passed on the local RTX 5060 with about
1.85 GiB peak allocated memory.

The official 9,099,481,727-byte OpenEarthMap archive passed MD5
64155d1dc9d3b68536063f79878e1a67 and extracted successfully. Of the imagery
present in the release, 2,687 files belong to the labelled train/validation lists;
1,456 samples from the 37 approved regions were accepted with zero missing labels
(1,190 CC BY 4.0 and 266 CC BY-SA 4.0). The remaining 1,231 labelled candidates
were excluded by the region licence allowlist.

The merged data-only corpus contains 2,879 samples across 97 disjoint source/site
groups: 2,012 train, 592 validation and 275 test. Shared-class declared sample
coverage is background 1,456, building 2,833, road 1,366, vegetation 1,449, water
776 and bare land 221; all six classes have positive measured pixel counts and the
coverage gate passes. No training run was started, per the user's instruction.

Known gaps are intentional backlog items:

- No public corpus fully covers the construction classes unfinished building,
  excavation, stockpile, material and equipment from India-like drone surveys.
- InsPLAD cannot seed commercial weights under its current non-commercial licence.
- Public crop models do not generalize across Indian crops, seasons and flight
  altitudes without India-specific calibration and holdout flights.
- Satellite benchmarks are pretraining or validation sources, not substitutes for
  drone-domain acceptance data.

## Acceptance gates

1. Every trained checkpoint gets source datasets, licences, SHA-256, class schema,
   spatial holdout, metrics and inference preprocessing in model provenance.
2. Split data by site and date, never random neighboring tiles, to prevent leakage.
3. Geometry reports retain CRS, units, thresholds, selected ROI and no-data cells.
4. AI output reports confidence and model version and supports human review.
5. No research-only or non-commercial data enters a production checkpoint.
6. A downloaded generic checkpoint is labelled initialization, never task trained.

## Primary references

- DINOv2: https://github.com/facebookresearch/dinov2
- SegFormer licence: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
- YOLO11 models and licence: https://docs.ultralytics.com/models/yolo11
- MineNetCD: https://rodare.hzdr.de/record/3251
- IARPA SMART: https://github.com/pubgeo/IARPA-SMART
- SpaceNet 7: https://spacenet.ai/sn7-challenge/
- OpenEarthMap attribution: https://open-earth-map.org/attribution.html
- InfraredSolarModules: https://github.com/RaptorMaps/InfraredSolarModules
- Duke UAV PV dataset: https://figshare.com/articles/dataset/18093890
- WeedsGalore: https://github.com/GFZ/weedsgalore
- RDD2022: https://github.com/sekilab/RoadDamageDetector
- InsPLAD: https://data.mendeley.com/datasets/5n3fjgvfyz/1
- UAV-RSOD: https://zenodo.org/records/12606374
