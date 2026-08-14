'''Machine-readable execution plan for the nine India-first workstreams.

The plan separates architecture choices from trained model claims. A downloaded
foundation checkpoint is initialization only unless a task-specific metrics card
and provenance record say otherwise.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


PlanState = Literal['complete', 'started', 'planned', 'blocked']


@dataclass(frozen=True)
class Workstream:
    order: int
    id: str
    title: str
    state: PlanState
    approach: str
    model_choice: str
    dataset_ids: tuple[str, ...]
    external_datasets: tuple[str, ...]
    next_actions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


WORKSTREAMS: tuple[Workstream, ...] = (
    Workstream(
        1,
        'selected_roi_change',
        'Selected-ROI stockpile and pit change',
        'complete',
        'Aligned DSM differencing constrained to an operator polygon; no ML required.',
        'Metric raster geometry',
        ('odm_aukerman', 'minenetcd'),
        (),
        ('Validate on a real repeated quarry survey with an independent ground measurement.',),
    ),
    Workstream(
        2,
        'shared_semantic',
        'Shared semantic segmentation engine',
        'started',
        'One tiled geospatial inference contract with per-head class schemas.',
        'DINOv2 ViT-B/14 encoder plus UPerNet-style decoder',
        ('spacenet7', 'openearthmap_mixed'),
        (),
        (
            'Await explicit user approval before starting any training run.',
            'Train and export the complete encoder-decoder model.',
            'Create site/date-separated India holdout labels.',
        ),
    ),
    Workstream(
        3,
        'construction_segmentation',
        'Construction site segmentation',
        'started',
        'Fine-tune the shared engine for site surfaces and activity regions.',
        'Shared DINOv2-B/UPerNet head',
        ('iarpa_smart_annotations', 'spacenet7'),
        ('Customer-owned Indian drone labels',),
        ('Define the ten-class annotation guide.', 'Label excavation and material subclasses.'),
    ),
    Workstream(
        4,
        'approved_design_progress',
        'Progress against approved designs',
        'started',
        'Register IFC/CAD geometry to the survey and measure element-level completion evidence.',
        'Geometry first; semantic evidence is optional',
        (),
        ('buildingSMART IFC sample models', 'Customer approved IFC/DXF/GeoJSON'),
        ('Add IFC ingestion.', 'Define evidence thresholds and manual review states.'),
    ),
    Workstream(
        5,
        'solar_inventory_thermal',
        'Solar RGB/thermal alignment and module inventory',
        'started',
        'Calibrated image registration, instance inventory, then temperature anomalies.',
        'YOLO11l-seg for modules; robust geometric registration for RGB/thermal',
        ('infrared_solar_modules', 'solar_pv_uav'),
        ('PVsegmentation RGB/thermal module masks',),
        ('Implement calibration and registration quality score.', 'Train module instance head.'),
    ),
    Workstream(
        6,
        'land_gis_encroachment',
        'Land GIS extraction and encroachment detection',
        'started',
        'Polygonize semantic masks, compare footprints and boundaries between surveys/designs.',
        'Shared DINOv2-B/UPerNet head plus deterministic GIS overlay',
        ('spacenet7',),
        ('SpaceNet roads', 'Open Buildings validation polygons'),
        ('Add topology-safe polygonization.', 'Define boundary tolerance and review workflow.'),
    ),
    Workstream(
        7,
        'agriculture',
        'Agriculture indices, canopy, stress and counting',
        'started',
        'Compute calibrated indices before ML; segment canopy and detect anomalous regions.',
        'DeepLabv3+ multispectral baseline; YOLO11l-seg for instance counting',
        ('weedsgalore',),
        ('India-specific crop and orchard flights',),
        ('Implement NDVI/NDRE/GNDVI.', 'Adapt crop/weed head.', 'Add plant/tree count validation.'),
    ),
    Workstream(
        8,
        'roads',
        'Mapped road condition',
        'started',
        'Segment road surface, detect defects on tiles and aggregate by georeferenced corridor.',
        'YOLO11x detector plus shared semantic road head',
        ('rdd2022_india', 'rdd2022_china_drone'),
        ('UAV-PDD2023',),
        ('Prepare VOC-to-YOLO conversion.', 'Train India-plus-drone benchmark.', 'Map severity.'),
    ),
    Workstream(
        9,
        'power_rail',
        'Specialized power-line and railway inspection',
        'started',
        'Keep close-range asset detection separate from corridor-level segmentation.',
        'YOLO11x asset detector plus shared rail segmentation head',
        ('uav_rsod', 'uav_rsod_obstacles'),
        ('InsPLAD research-only',),
        ('Train rail/gauge segmentation.', 'Obtain commercial power-line labels or permission.'),
    ),
)


def validate_plan() -> list[str]:
    errors: list[str] = []
    orders = [item.order for item in WORKSTREAMS]
    if orders != list(range(1, 10)):
        errors.append('Workstream order must be exactly 1 through 9.')
    ids = [item.id for item in WORKSTREAMS]
    if len(ids) != len(set(ids)):
        errors.append('Workstream ids must be unique.')
    if any(not item.model_choice for item in WORKSTREAMS):
        errors.append('Every workstream needs an explicit model or algorithm choice.')
    return errors


if __name__ == '__main__':
    import json

    failures = validate_plan()
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps([item.to_dict() for item in WORKSTREAMS], indent=2))
