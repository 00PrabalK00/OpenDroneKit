"""Catalogue of every dataset the training rig can fetch.

Each entry records where the data comes from, what licence it carries, and which model
it feeds. Licences are stated explicitly because several of these datasets are
research-only and must not be redistributed with trained weights without checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceKind = Literal["http", "kaggle", "roboflow"]
TaskKind = Literal["segmentation", "detection", "classification", "photogrammetry"]


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
        description="Combined crack segmentation corpus aggregating several public crack sets.",
        approx_size_mb=85,
        url="https://github.com/khanhha/crack_segmentation/archive/refs/heads/master.zip",
        archive_name="crack_segmentation_combined.zip",
        feeds=("crack_segmentation",),
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
        license="Per Roboflow Universe project",
        description="Concrete bridge defects: crack, spallation, efflorescence, exposed bars, corrosion.",
        workspace="",
        project="",
        feeds=("structural_multiclass_detector",),
        notes="Workspace/project resolved at fetch time via Roboflow Universe search.",
    ),
    "solar_panel_defects": DatasetSpec(
        name="solar_panel_defects",
        kind="roboflow",
        task="detection",
        target="solar",
        license="Per Roboflow Universe project",
        description="Aerial and thermal solar panel defect detection set.",
        workspace="",
        project="",
        feeds=("solar_pv_multidefect_detector",),
    ),
    "corrosion_detection": DatasetSpec(
        name="corrosion_detection",
        kind="roboflow",
        task="detection",
        target="metal",
        license="Per Roboflow Universe project",
        description="Metal corrosion and rust severity detection set.",
        workspace="",
        project="",
        feeds=("metal_corrosion_detector",),
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
}


GROUPS: dict[str, tuple[str, ...]] = {
    "crack": ("crackforest", "deepcrack", "crack_segmentation_combined", "sdnet2018", "surface_crack"),
    "solar": ("elpv", "solar_panel_defects"),
    "structural": ("codebrim_structural",),
    "corrosion": ("corrosion_detection",),
    "recon": ("odm_aukerman",),
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
