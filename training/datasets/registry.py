"""Catalogue of every dataset the training rig can fetch.

Each entry records where the data comes from, what licence it carries, and which model
it feeds. Licences are stated explicitly because several of these datasets are
research-only and must not be redistributed with trained weights without checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceKind = Literal["http", "kaggle", "roboflow"]
TaskKind = Literal[
    "segmentation",
    "detection",
    "classification",
    "photogrammetry",
    "change_detection",
    "multispectral",
]


@dataclass(frozen=True)
class DatasetSpec:
    """One fetchable dataset."""

    name: str
    kind: SourceKind
    task: TaskKind
    target: str
    license: str
    description: str
    approx_size_mb: int = 0
    # http
    url: str = ""
    archive_name: str = ""
    expected_md5: str = ""
    # kaggle / roboflow
    slug: str = ""
    workspace: str = ""
    project: str = ""
    version: int = 1
    export_format: str = "yolov8"
    # Which trained model this feeds. Empty means it is a support/test asset.
    feeds: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    # -- crack segmentation ------------------------------------------------
    "crackforest": DatasetSpec(
        name="crackforest",
        kind="http",
        task="segmentation",
        target="crack",
        license="Research use; see repository",
        description="CrackForest road crack dataset with hand-labelled ground truth.",
        approx_size_mb=10,
        url="https://github.com/cuilimeng/CrackForest-dataset/archive/refs/heads/master.zip",
        archive_name="crackforest.zip",
        feeds=("crack_segmentation",),
        notes="Ground truth ships as MATLAB .mat segmentation structs.",
    ),
    "deepcrack": DatasetSpec(
        name="deepcrack",
        kind="http",
        task="segmentation",
        target="crack",
        license="MIT (repository)",
        description="DeepCrack benchmark: 537 crack images with pixel-level annotations.",
        approx_size_mb=73,
        url="https://github.com/yhlleo/DeepCrack/archive/refs/heads/master.zip",
        archive_name="deepcrack.zip",
        feeds=("crack_segmentation",),
    ),
    "crack_segmentation_combined": DatasetSpec(
        name="crack_segmentation_combined",
        kind="http",
        task="segmentation",
        target="crack",
        license="MIT (repository)",
        description="khanhha/crack_segmentation source repository. Code and sanity images only.",
        approx_size_mb=85,
        url="https://github.com/khanhha/crack_segmentation/archive/refs/heads/master.zip",
        archive_name="crack_segmentation_combined.zip",
        feeds=(),
        notes=(
            "The repository does not vendor its corpus - the 11k image/mask pairs sit behind a "
            "Google Drive link. Use the 'crack_segmentation_kaggle' entry for the actual data. "
            "This checkout contributes no training samples."
        ),
    ),
    "crack_segmentation_kaggle": DatasetSpec(
        name="crack_segmentation_kaggle",
        kind="kaggle",
        task="segmentation",
        target="crack",
        license="See Kaggle dataset page; aggregates CFD, Crack500, GAPs384, DeepCrack, Rissbilder, Volker",
        description="11,298 crack image/mask pairs aggregating six public crack corpora.",
        approx_size_mb=1300,
        slug="lakshaymiddha/crack-segmentation-dataset",
        feeds=("crack_segmentation",),
        notes="The redistributed form of the khanhha corpus, with masks included.",
    ),
    "sdnet2018": DatasetSpec(
        name="sdnet2018",
        kind="kaggle",
        task="classification",
        target="crack",
        license="CC BY 4.0",
        description="SDNET2018: 56k labelled concrete crack / non-crack image tiles.",
        approx_size_mb=1900,
        slug="aniruddhsharma/structural-defects-network-concrete-crack-images",
        feeds=("crack_segmentation",),
        notes="Classification labels only; used for hard-negative mining, not masks.",
    ),
    "surface_crack": DatasetSpec(
        name="surface_crack",
        kind="kaggle",
        task="classification",
        target="crack",
        license="CC BY 4.0",
        description="Surface crack detection tiles, positive/negative balanced.",
        approx_size_mb=245,
        slug="arunrk7/surface-crack-detection",
        feeds=("crack_segmentation",),
    ),
    # -- solar -------------------------------------------------------------
    "elpv": DatasetSpec(
        name="elpv",
        kind="http",
        task="classification",
        target="solar",
        license="CC BY-NC-SA 4.0",
        description="ELPV: 2624 electroluminescence solar cell images with defect probabilities.",
        approx_size_mb=93,
        url="https://github.com/zae-bayern/elpv-dataset/archive/refs/heads/master.zip",
        archive_name="elpv.zip",
        feeds=("solar_pv_multidefect_detector",),
        notes="Non-commercial licence. Do not ship weights trained on this commercially.",
    ),
    # -- roboflow detection sets ------------------------------------------
    "codebrim_structural": DatasetSpec(
        name="codebrim_structural",
        kind="roboflow",
        task="detection",
        target="structural",
        license="CC BY 4.0 (declared in the export's data.yaml)",
        description="CODEBRIM concrete bridge defects: crack, spallation, efflorescence, exposed bars, corrosion stain.",
        approx_size_mb=400,
        workspace="defect-detection-edbnh",
        project="codebrim-lnsfg",
        version=3,
        feeds=("structural_multiclass_detector",),
        notes="Roboflow Universe mirror of the CODEBRIM benchmark (~1051 annotated images).",
    ),
    "solar_panel_defects": DatasetSpec(
        name="solar_panel_defects",
        kind="roboflow",
        task="detection",
        target="solar",
        license="CC BY 4.0 (declared in the export's data.yaml)",
        description="Solar panel surface condition: clear, dusty, snow-covered, bird-drop, physical damage.",
        approx_size_mb=180,
        workspace="defect-detection-in-solar-panels-using-thermal-imaging",
        project="solar-panel-defects-ki9pu",
        version=4,
        feeds=("solar_pv_multidefect_detector",),
        notes=(
            "Soiling/obstruction classes, not electrical faults - despite the project title "
            "these are panel surface conditions, so a model trained here detects what is ON "
            "a panel, not a cell defect. ELPV remains the electroluminescence source. The "
            "export also carries a junk class literally named '1'."
        ),
    ),
    "corrosion_detection": DatasetSpec(
        name="corrosion_detection",
        kind="roboflow",
        task="detection",
        target="metal",
        license="CC BY 4.0 (declared in the export's data.yaml)",
        description="Metal corrosion and rust detection set.",
        approx_size_mb=150,
        workspace="scaledge-ztihl",
        project="corrosion-detection-obirw",
        version=1,
        feeds=("metal_corrosion_detector",),
        notes=(
            "Carries both 'Corrosion' and 'Corrosion-detection' as separate classes, which "
            "appear to be the same label under two names; merge them before training."
        ),
    ),
    # -- photogrammetry test asset ----------------------------------------
    "odm_aukerman": DatasetSpec(
        name="odm_aukerman",
        kind="http",
        task="photogrammetry",
        target="reconstruction",
        license="CC BY-SA 4.0 (OpenDroneMap sample data)",
        description="Real geotagged UAV survey used to verify COLMAP georeferencing end to end.",
        approx_size_mb=700,
        url="https://github.com/OpenDroneMap/odm_data_aukerman/archive/refs/heads/master.zip",
        archive_name="odm_aukerman.zip",
        feeds=(),
        notes="Carries GPS EXIF, so the geo anchor and UTM outputs can be checked against truth.",
    ),
    # -- India-first mission packs ---------------------------------------
    'minenetcd': DatasetSpec(
        name='minenetcd',
        kind='http',
        task='change_detection',
        target='mining',
        license='CC BY 4.0',
        description='MineNetCD paired imagery and masks from 100 global mining sites.',
        approx_size_mb=3500,
        url='https://rodare.hzdr.de/record/3251/files/MineNetCD.zip?download=1',
        archive_name='minenetcd.zip',
        feeds=('mining_change_semantics',),
        notes='RGB change benchmark only; DSM geometry remains the source of volume truth.',
    ),
    'iarpa_smart_annotations': DatasetSpec(
        name='iarpa_smart_annotations',
        kind='http',
        task='change_detection',
        target='construction',
        license='MIT for repository; source imagery carries separate provider terms',
        description='Temporal heavy-construction site polygons and phase annotations.',
        approx_size_mb=30,
        url='https://github.com/pubgeo/IARPA-SMART/archive/refs/heads/main.zip',
        archive_name='iarpa_smart_annotations.zip',
        feeds=('construction_change_semantics',),
        notes='Annotations are not per-object site segmentation; obtain imagery per upstream instructions.',
    ),
    'spacenet7': DatasetSpec(
        name='spacenet7',
        kind='http',
        task='change_detection',
        target='land',
        license='CC BY-SA 4.0',
        description='Monthly building footprints and imagery for 101 time-series AOIs.',
        approx_size_mb=8700,
        url='https://spacenet-dataset.s3.amazonaws.com/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz',
        archive_name='spacenet7_train.tar.gz',
        expected_md5='6eda13b9c28f6f5cdf00a7e8e218c1b1',
        feeds=('shared_semantic_engine', 'land_gis_extraction', 'encroachment_change'),
        notes='About 8.7 GB compressed and roughly 25 GB extracted; drone-domain fine-tuning remains required.',
    ),
    'openearthmap_mixed': DatasetSpec(
        name='openearthmap_mixed',
        kind='http',
        task='segmentation',
        target='land',
        license='Mixed per region; sample-level allowlist required',
        description='Global 8-class aerial land-cover masks; only explicitly commercial-compatible regions are indexed.',
        approx_size_mb=9100,
        url='https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1',
        archive_name='OpenEarthMap.zip',
        expected_md5='64155d1dc9d3b68536063f79878e1a67',
        feeds=('shared_semantic_engine', 'land_gis_extraction'),
        notes=(
            'Licences vary by region. The adapter admits only explicit CC BY 4.0 or '
            'CC BY-SA 4.0 rows from the official attribution table. Public-domain or '
            'unspecified-source labels default to CC BY-NC-SA 4.0 and are excluded; '
            'DL-DE-BY-2.0 regions remain excluded pending legal review.'
        ),
    ),
    'infrared_solar_modules': DatasetSpec(
        name='infrared_solar_modules',
        kind='http',
        task='classification',
        target='solar',
        license='MIT',
        description='20,000 infrared PV-module crops across 11 anomaly classes plus normal.',
        approx_size_mb=25,
        url='https://github.com/RaptorMaps/InfraredSolarModules/archive/refs/heads/master.zip',
        archive_name='infrared_solar_modules.zip',
        feeds=('solar_thermal_anomaly',),
        notes='Very low-resolution module crops; not a panel localization dataset.',
    ),
    'solar_pv_uav': DatasetSpec(
        name='solar_pv_uav',
        kind='http',
        task='segmentation',
        target='solar',
        license='CC BY 4.0',
        description='Duke UAV imagery, masks and videos with 2,019 PV instances.',
        approx_size_mb=4140,
        # The article-bundle URL answers 202 with an empty body while figshare assembles
        # a 18.8 GB zip, of which 16 GB is video this project has no use for. This is the
        # imgs.zip file within that record, fetched directly.
        url='https://ndownloader.figshare.com/files/32825000',
        archive_name='solar_pv_uav_imgs.zip',
        feeds=('solar_module_inventory',),
        notes='Images and masks only; the 16 GB vid.zip in the same record is skipped. '
              'Validate whether each annotation is a module or an array before label conversion.',
    ),
    'pvel_ad': DatasetSpec(
        name='pvel_ad',
        kind='http',
        task='detection',
        target='solar',
        license='Apache-2.0 (repository); dataset released for research use',
        description='PVEL-AD electroluminescence cell anomalies with named defect boxes.',
        approx_size_mb=4210,
        # The project's request form and institutional-email requirement were superseded
        # by a public Drive link in its own README. The archive is RAR5, not zip.
        url='https://drive.google.com/file/d/1EtteKnLhSFQ3XMCRXt5wKY-lDkIP7299',
        archive_name='pvel_ad.rar',
        feeds=('solar_cell_defect_detector',),
        notes='Only trainval carries annotations: 4,500 images and 7,842 boxes. '
              'test/Annotations is present and empty, so its 19,150 images are '
              'unlabelled. Four classes have under 35 boxes and are not trainable.',
    ),
    'weedsgalore': DatasetSpec(
        name='weedsgalore',
        kind='http',
        task='multispectral',
        target='agriculture',
        license='CC BY 4.0',
        description='Multitemporal multispectral UAV maize crop and weed segmentation.',
        approx_size_mb=321,
        url='https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip',
        archive_name='weedsgalore-dataset.zip',
        feeds=('agriculture_canopy', 'agriculture_crop_weed'),
    ),
    'rdd2022_india': DatasetSpec(
        name='rdd2022_india',
        kind='kaggle',
        task='detection',
        target='roads',
        license='CC BY-SA 4.0',
        description='RDD2022 India road images with cracks and potholes.',
        approx_size_mb=528,
        # The CRDDC S3 bucket is gone. The maintainers now point at a single 13.2 GB
        # figshare zip holding all seven countries, which is a poor trade for one of
        # them, so this uses a mirror whose 528 MB matches the original 503 MB archive.
        slug='hafsaesam/rdd2022-india',
        feeds=('road_damage_detector',),
        notes='Mirror, not the maintainers\' copy: confirm it carries the documented 7,706 '
              'India images and the D00/D10/D20/D40 schema before training on it. '
              'India subset is mostly ground imagery; validate drone transfer separately.',
    ),
    'rdd2022_china_drone': DatasetSpec(
        name='rdd2022_china_drone',
        kind='roboflow',
        task='detection',
        target='roads',
        license='CC BY-SA 4.0',
        description='RDD2022 UAV road-damage subset with four defect classes.',
        approx_size_mb=153,
        # Same dead S3 bucket as the India subset. This mirror holds exactly the 2,401
        # images the maintainers document for China_Drone, which is the strongest
        # evidence available short of downloading both and comparing.
        workspace='image-pro',
        project='china-drone',
        version=1,
        export_format='yolov8',
        feeds=('road_damage_detector',),
        notes='Carries two classes beyond the RDD2022 D00/D10/D20/D40 schema -- '
              '"Block crack" and "Repair" -- so the adapter must map or drop them '
              'rather than let the class count silently disagree with the India set.',
    ),
    'uav_rsod': DatasetSpec(
        name='uav_rsod',
        kind='http',
        task='segmentation',
        target='rail',
        license='CC BY 4.0',
        description='Indian UAV railway rail, gauge and background segmentation.',
        approx_size_mb=810,
        # The record is alive; the filename was wrong. Zenodo names it with a "V1 "
        # prefix, and without it the request 404s -- which read as a dead record.
        url='https://zenodo.org/records/12606374/files/V1%20UAV-RSOD_Dataset%20for%20Segmentation.zip?download=1',
        archive_name='uav_rsod_segmentation.zip',
        feeds=('rail_corridor_segmentation',),
        notes='The Zenodo record separates segmentation and obstacle-detection archives.',
    ),
    'uav_rsod_obstacles': DatasetSpec(
        name='uav_rsod_obstacles',
        kind='http',
        task='detection',
        target='rail',
        license='CC BY 4.0',
        description='Indian UAV railway obstacle images covering six obstacle classes.',
        approx_size_mb=856,
        url='https://zenodo.org/records/12606374/files/V2%20UAV-RSOD_Dataset%20for%20Obstacle%20Detection.zip?download=1',
        archive_name='uav_rsod_obstacles.zip',
        feeds=('rail_obstacle_detector',),
        notes='Separate 2,002-image augmented obstacle archive from the segmentation set.',
    ),
}


GROUPS: dict[str, tuple[str, ...]] = {
    "crack": (
        "crackforest",
        "deepcrack",
        "crack_segmentation_kaggle",
        "sdnet2018",
        "surface_crack",
    ),
    "solar": ("elpv", "solar_panel_defects"),
    "structural": ("codebrim_structural",),
    "corrosion": ("corrosion_detection",),
    "recon": ("odm_aukerman",),
    "mining": ("odm_aukerman", "minenetcd"),
    "construction_india": ("iarpa_smart_annotations", "spacenet7"),
    "land": ("spacenet7", "openearthmap_mixed"),
    "agriculture": ("weedsgalore",),
    "roads": ("rdd2022_india", "rdd2022_china_drone"),
    "rail": ("uav_rsod", "uav_rsod_obstacles"),
    "india_first": (
        "minenetcd",
        "iarpa_smart_annotations",
        "spacenet7",
        "openearthmap_mixed",
        "infrared_solar_modules",
        "solar_pv_uav",
        "weedsgalore",
        "rdd2022_india",
        "rdd2022_china_drone",
        "uav_rsod",
        "uav_rsod_obstacles",
    ),
    "public": ("crackforest", "deepcrack", "crack_segmentation_combined", "elpv", "odm_aukerman"),
    "all": tuple(DATASETS.keys()),
}


def resolve(names: list[str]) -> list[DatasetSpec]:
    """Expand dataset names and group names into a deduplicated spec list."""
    selected: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key in GROUPS:
            selected.extend(GROUPS[key])
        elif key in DATASETS:
            selected.append(key)
        else:
            raise KeyError(
                f"Unknown dataset {name!r}. Known datasets: {', '.join(sorted(DATASETS))}. "
                f"Known groups: {', '.join(sorted(GROUPS))}."
            )
    seen: set[str] = set()
    ordered: list[DatasetSpec] = []
    for key in selected:
        if key not in seen:
            seen.add(key)
            ordered.append(DATASETS[key])
    return ordered
