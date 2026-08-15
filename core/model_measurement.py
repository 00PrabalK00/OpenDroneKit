"""Measuring inside the 3D model: length, height, area and volume.

A reconstruction is metric only if something made it metric. Structure from motion
recovers shape up to an unknown scale, so a model that has not been georeferenced is in
units that resemble metres, print like metres, and are not metres. Measuring on it
produces a defensible-looking number for a wall that could be six metres or sixty.

So nothing here measures a model that cannot show where its scale came from. The
provenance sidecar written beside every derived artefact records the CRS the
reconstruction was aligned to; that record is the licence to report a distance in
metres, and its absence is a refusal rather than an assumption.

The four measurements answer questions people ask differently:

``length``
    Straight-line distance through space, with the horizontal and vertical parts given
    separately -- a cable run and its span across a gap are not the same number.
``height``
    The vertical component alone. What a client means by "how tall is it".
``area``
    Both the true surface area of a polygon lying on a slope and its planimetric
    footprint. A pitched roof has more surface than the plan area it covers, which is
    the difference between ordering enough material and not.
``volume``
    Only from a closed mesh, by the divergence theorem, and only when the mesh really
    does close. An open surface has no interior, and a number computed as though it did
    is arbitrary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .provenance import sidecar_path


class NotMetric(ValueError):
    """The model has no recorded scale, so measurements on it are not in metres."""


class NotClosed(ValueError):
    """The mesh does not enclose a volume, so it has no interior to measure."""


@dataclass
class ModelScale:
    """Where a model's metric scale came from."""

    epsg: int
    engine: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"epsg": self.epsg, "engine": self.engine, "source": self.source}


def model_scale(model_path: str | Path) -> ModelScale:
    """The recorded CRS for this model, or a refusal explaining why there is none."""
    path = Path(model_path)
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        raise NotMetric(
            f"{path.name} has no provenance sidecar, so nothing records whether it is "
            "in metres or in structure-from-motion units. Measurements would be numbers "
            "without a unit. Reconstruct with georeferencing, or record provenance for "
            "this model stating the CRS it was aligned to."
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotMetric(
            f"The provenance sidecar for {path.name} cannot be read, so its scale is "
            "unknown. That is not the same as the model being unscaled -- check the "
            "sidecar rather than assuming either way."
        ) from exc

    epsg = payload.get("crs_epsg")
    if not epsg:
        raise NotMetric(
            f"{path.name} was produced without a CRS, so it carries structure-from-"
            "motion units of unknown size. A wall measured on it could be six metres or "
            "sixty. Georeference the reconstruction -- with GCPs or camera fixes -- "
            "before measuring."
        )
    return ModelScale(epsg=int(epsg), engine=str(payload.get("engine", "")),
                      source=str(sidecar))


def read_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles from a Wavefront OBJ.

    Written directly rather than through a mesh library because the toolkit does not
    otherwise need one, and a measurement that only works when open3d happens to be
    installed is not a measurement the field can rely on.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model not found: {source}")

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # OBJ indices are 1-based and may carry texture/normal references.
                corners = [int(chunk.split("/")[0]) for chunk in parts[1:]]
                corners = [c - 1 if c > 0 else len(vertices) + c for c in corners]
                # Fan-triangulate any polygon larger than a triangle.
                for i in range(1, len(corners) - 1):
                    faces.append([corners[0], corners[i], corners[i + 1]])

    if not vertices:
        raise ValueError(f"{source.name} contains no vertices.")
    return (np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64))


def measure_length(model_path: str | Path,
                   start_xyz: Sequence[float],
                   end_xyz: Sequence[float]) -> dict[str, Any]:
    """Straight-line distance between two points on the model, split into components."""
    scale = model_scale(model_path)
    a = np.asarray([float(v) for v in start_xyz[:3]], dtype=np.float64)
    b = np.asarray([float(v) for v in end_xyz[:3]], dtype=np.float64)
    if a.size != 3 or b.size != 3:
        raise ValueError("A length needs two three-dimensional points.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Measurement points must be finite.")

    delta = b - a
    horizontal = float(math.hypot(delta[0], delta[1]))
    vertical = float(delta[2])
    return {
        "kind": "length",
        "slope_distance_m": round(float(np.linalg.norm(delta)), 4),
        "horizontal_distance_m": round(horizontal, 4),
        "vertical_distance_m": round(vertical, 4),
        "inclination_deg": round(math.degrees(math.atan2(vertical, horizontal)), 3)
        if horizontal or vertical else 0.0,
        "crs_epsg": scale.epsg,
        "scale_source": scale.to_dict(),
        "limits": [
            "Measured between the two points given, on the surface as reconstructed; "
            "reconstruction error at those points carries straight into the result.",
        ],
    }


def measure_height(model_path: str | Path,
                   top_xyz: Sequence[float],
                   base_xyz: Sequence[float]) -> dict[str, Any]:
    """The vertical component only -- what "how tall is it" actually asks for."""
    scale = model_scale(model_path)
    top = float(top_xyz[2])
    base = float(base_xyz[2])
    if not math.isfinite(top) or not math.isfinite(base):
        raise ValueError("Measurement points must be finite.")
    return {
        "kind": "height",
        "height_m": round(abs(top - base), 4),
        "top_elevation_m": round(top, 4),
        "base_elevation_m": round(base, 4),
        "crs_epsg": scale.epsg,
        "scale_source": scale.to_dict(),
        "limits": [
            "Vertical difference between the two picked points. If the base point is "
            "not on the ground, this is not the height of the structure.",
        ],
    }


def measure_area(model_path: str | Path,
                 polygon_xyz: Sequence[Sequence[float]]) -> dict[str, Any]:
    """True surface area of a polygon on the model, and its planimetric footprint.

    The two differ by the cosine of the slope, which on a pitched roof is the difference
    between ordering enough covering and being short.
    """
    scale = model_scale(model_path)
    points = np.asarray([[float(p[0]), float(p[1]), float(p[2])]
                         for p in polygon_xyz if len(p) >= 3], dtype=np.float64)
    if len(points) < 3:
        raise ValueError("An area needs at least three points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Measurement points must be finite.")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
        if len(points) < 3:
            raise ValueError("An area needs at least three distinct points.")

    # Newell's method: the magnitude of the summed cross products is twice the area of
    # a planar polygon, and it degrades gracefully for one that is nearly planar.
    normal = np.zeros(3, dtype=np.float64)
    for i in range(len(points)):
        current, following = points[i], points[(i + 1) % len(points)]
        normal += np.cross(current, following)
    surface_area = float(np.linalg.norm(normal) / 2.0)

    planimetric = 0.0
    for i in range(len(points)):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1]
        planimetric += x1 * y2 - x2 * y1
    planimetric = abs(planimetric) / 2.0

    # Planarity: how far the vertices sit from their own best-fit plane, relative to the
    # polygon's size. A polygon drawn across two roof facets is not one area.
    centred = points - points.mean(axis=0)
    _, singular, _ = np.linalg.svd(centred, full_matrices=False)
    extent = float(singular[0]) or 1.0
    flatness = float(singular[-1]) / extent

    slope_deg = (math.degrees(math.acos(min(1.0, planimetric / surface_area)))
                 if surface_area > 0 and planimetric > 0 else 0.0)
    limits = [
        "Surface area is the area of the polygon as it lies on the model; the "
        "planimetric figure is its shadow on the ground, which is what a plan drawing "
        "shows.",
    ]
    if flatness > 0.02:
        limits.append(
            f"The vertices are {flatness:.1%} out of plane relative to the polygon's "
            "size, so this outline does not lie on one flat surface and its area is a "
            "reading across more than one facet."
        )
    return {
        "kind": "area",
        "surface_area_m2": round(surface_area, 4),
        "planimetric_area_m2": round(planimetric, 4),
        "slope_deg": round(slope_deg, 3),
        "planar": bool(flatness <= 0.02),
        "vertex_count": int(len(points)),
        "crs_epsg": scale.epsg,
        "scale_source": scale.to_dict(),
        "limits": limits,
    }


def _edges_are_shared(faces: np.ndarray) -> tuple[bool, int]:
    """Whether every triangle edge is used exactly twice, and how many are not."""
    counts: dict[tuple[int, int], int] = {}
    for triangle in faces:
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    open_edges = sum(1 for count in counts.values() if count != 2)
    return open_edges == 0, open_edges


def measure_volume(model_path: str | Path) -> dict[str, Any]:
    """Volume enclosed by a closed mesh, by the divergence theorem.

    Refused outright on an open surface. A reconstruction of a building's exterior has
    no floor, and summing signed tetrahedra over it yields a number that varies with
    where the origin happens to be -- confident, reproducible and meaningless.
    """
    scale = model_scale(model_path)
    vertices, faces = read_mesh(model_path)
    if len(faces) == 0:
        raise NotClosed(
            f"{Path(model_path).name} has no faces, so it is a point cloud rather than "
            "a surface. Volume from points needs a surface model -- measure the "
            "stockpile against a DSM instead, where the reference surface is explicit."
        )

    closed, open_edges = _edges_are_shared(faces)
    if not closed:
        raise NotClosed(
            f"{Path(model_path).name} is not closed: {open_edges} edges belong to only "
            "one triangle, so the mesh has holes or a missing base and encloses nothing. "
            "A volume computed from it would depend on where the origin sits. Close the "
            "mesh, or measure against a DSM with a stated reference surface."
        )

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(abs(signed.sum()))

    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    return {
        "kind": "volume",
        "volume_m3": round(volume, 4),
        "surface_area_m2": round(float(areas.sum()), 4),
        "triangle_count": int(len(faces)),
        "closed": True,
        "crs_epsg": scale.epsg,
        "scale_source": scale.to_dict(),
        "limits": [
            "Volume of the region the mesh encloses. It is exact for the mesh, which is "
            "not the same as exact for the object: reconstruction smooths detail and "
            "bridges gaps it could not see.",
        ],
    }
