"""Typed results returned by the SDK.

Each result is a frozen dataclass with the fields a script normally needs, plus ``raw``:
the complete payload the underlying OpenDroneKit call returned, for anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectInfo:
    """A project as listed in a workspace: id, name, folder and description."""

    id: int
    name: str
    root: Path
    description: str = ""


@dataclass(frozen=True)
class MissionSummary:
    """The planner's figures for one mission, as the application shows them."""

    template: str
    waypoints: int
    distance_m: float
    duration_min: float
    gsd_cm: float
    altitude_m: float
    warnings: tuple[str, ...] = ()
    #: Set when a mission-template plugin produced the options: kind, name, version.
    plugin: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    """An image folder registered with a project, with the counts read from its EXIF."""

    name: str
    path: Path
    image_count: int | None
    geotagged_count: int | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Reconstruction:
    """A finished reconstruction. Paths are empty when the engine did not write them."""

    engine: str
    output_dir: Path
    registered_images: int
    frame_count: int
    reprojection_error_px: float | None
    crs_epsg: int | None
    geo_rmse_m: float | None
    orthomosaic_path: str
    dsm_path: str
    point_cloud_path: str
    warnings: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def georeferenced(self) -> bool:
        """True only when the engine anchored the model to a real CRS."""
        return self.crs_epsg is not None


@dataclass(frozen=True)
class CrackDetection:
    """A crack mask over one image, and what produced it.

    ``source`` is ``"model"`` when a trained segmentation model produced the mask and
    ``"heuristic"`` when classical image processing did; heuristic output is screening,
    not a model result, and is never labelled as one.
    """

    image: Path
    source: str
    model_used: str
    model_key: str
    model_sha256: str
    crack_pixels: int
    crack_ratio: float
    length_px: float
    max_width_px: float
    mask_path: Path
    overlay_path: Path
    #: Set when the model is installed but failed and the heuristic ran instead.
    fallback_reason: str = ""


@dataclass(frozen=True)
class Detections:
    """Boxes from a registered model or a detector plugin, with the weights' identity."""

    image: Path
    model_used: str
    model_key: str
    model_sha256: str
    detections: tuple[dict[str, Any], ...]
    scope: str = ""
    overlay_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Measurement:
    """One recorded measurement: value and unit, the caveat that limits it, and the
    full payload (``raw``) with the inputs it was made from.
    """

    id: str
    kind: str
    value: float
    unit: str
    caveat: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Report:
    """A written report: where its files are, how many findings it holds, its warnings."""

    id: str
    title: str
    report_type: str
    html_path: Path
    pdf_path: Path | None
    docx_path: Path | None
    output_dir: Path
    findings: int
    warnings: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
