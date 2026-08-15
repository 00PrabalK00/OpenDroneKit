"""Flight paths generated from an imported surface, rather than from an outline.

Planning from a polygon assumes the thing being inspected is flat, or at least that a
grid over its footprint will see it. That assumption fails on exactly the structures
worth inspecting: a bridge soffit faces downward, a cooling tower curves away from every
straight line, a building with a courtyard has walls that face inward. Drawing an outline
around any of them produces a plan that photographs their tops.

A surface knows which way it faces. Every triangle has a normal, so a capture point can
be placed off the face along that normal, looking back at it, and the aircraft ends up
photographing the wall rather than the roof above it.

Two things are refused rather than guessed.

Scale. A mesh from photogrammetry is in structure-from-motion units unless something made
it metric, so a stand-off of eight is eight of nothing. The same provenance gate used for
3D measurement applies here: no recorded CRS, no plan.

Faces. A point cloud has no normals, so there is no direction to stand off along.
Generating a path from one would mean inventing the orientation of every surface, and the
resulting flight would look purposeful and photograph nothing in particular.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.model_measurement import NotMetric, model_scale, read_mesh

SUPPORTED_SUFFIXES = (".obj", ".ply", ".glb", ".gltf")
# Named so the refusal can say what would be needed rather than only what failed.
UNSUPPORTED_SUFFIXES = {
    ".ifc": "IFC is a building-information schema rather than a triangulated surface; "
            "export the geometry to OBJ, PLY or GLB first.",
    ".las": "LAS and LAZ carry points with no faces, so no surface normal exists to "
            "stand off along. Mesh the cloud first, or plan from an outline.",
    ".laz": "LAS and LAZ carry points with no faces, so no surface normal exists to "
            "stand off along. Mesh the cloud first, or plan from an outline.",
}


class UnsupportedSurface(ValueError):
    """The file cannot be read as a triangulated surface."""


class NoUsableSurface(ValueError):
    """The surface carries nothing that can be photographed from a stand-off."""


@dataclass
class SurfacePose:
    """One capture point, and the face it exists to photograph."""

    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    gimbal_pitch_deg: float
    face_index: int
    face_area_m2: float
    standoff_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_m": round(self.x_m, 4), "y_m": round(self.y_m, 4),
            "z_m": round(self.z_m, 4),
            "yaw_deg": round(self.yaw_deg, 2),
            "gimbal_pitch_deg": round(self.gimbal_pitch_deg, 2),
            "face_index": self.face_index,
            "face_area_m2": round(self.face_area_m2, 4),
            "standoff_m": round(self.standoff_m, 3),
        }


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles from an ASCII or binary-little-endian PLY."""
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise UnsupportedSurface(f"{path.name} ended before its header did.")
            decoded = line.decode("ascii", errors="ignore").strip()
            header.append(decoded)
            if decoded == "end_header":
                break

        fmt = "ascii"
        vertex_count = face_count = 0
        vertex_props: list[str] = []
        element = ""
        for entry in header:
            parts = entry.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    vertex_count = int(parts[2])
                elif element == "face":
                    face_count = int(parts[2])
            elif parts[0] == "property" and element == "vertex" and parts[1] != "list":
                vertex_props.append(parts[2])

        if fmt != "ascii":
            raise UnsupportedSurface(
                f"{path.name} is a binary PLY. Convert it to ASCII PLY, or use OBJ or "
                "GLB, rather than having the geometry silently misread."
            )

        rows = handle.read().decode("ascii", errors="ignore").split()

    cursor = 0
    stride = max(3, len(vertex_props))
    vertices = []
    for _ in range(vertex_count):
        chunk = rows[cursor:cursor + stride]
        cursor += stride
        vertices.append([float(chunk[0]), float(chunk[1]), float(chunk[2])])

    faces = []
    for _ in range(face_count):
        corners = int(rows[cursor])
        indices = [int(v) for v in rows[cursor + 1:cursor + 1 + corners]]
        cursor += 1 + corners
        for i in range(1, len(indices) - 1):
            faces.append([indices[0], indices[i], indices[i + 1]])

    return (np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64))


def _read_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles from a self-contained GLB, or a glTF with an embedded buffer."""
    raw = path.read_bytes()
    if raw[:4] == b"glTF":
        _, _, _ = struct.unpack("<III", raw[:12])
        offset = 12
        gltf: dict[str, Any] | None = None
        binary = b""
        while offset < len(raw):
            length, kind = struct.unpack("<II", raw[offset:offset + 8])
            chunk = raw[offset + 8:offset + 8 + length]
            if kind == 0x4E4F534A:
                gltf = json.loads(chunk.decode("utf-8"))
            elif kind == 0x004E4942:
                binary = chunk
            offset += 8 + length + (-length % 4)
        if gltf is None:
            raise UnsupportedSurface(f"{path.name} carries no glTF JSON chunk.")
    else:
        gltf = json.loads(raw.decode("utf-8"))
        binary = b""
        for buffer in gltf.get("buffers", []):
            uri = str(buffer.get("uri", ""))
            if uri.startswith("data:"):
                import base64

                binary += base64.b64decode(uri.split(",", 1)[1])
            elif uri:
                binary += (path.parent / uri).read_bytes()

    component = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
    counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def read_accessor(index: int) -> np.ndarray:
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        width = counts[accessor["type"]]
        dtype = np.dtype("<" + component[accessor["componentType"]])
        total = int(accessor["count"]) * width
        data = np.frombuffer(binary, dtype=dtype, count=total, offset=start)
        return data.reshape(int(accessor["count"]), width).astype(np.float64)

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    base = 0
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position = primitive.get("attributes", {}).get("POSITION")
            if position is None:
                continue
            points = read_accessor(position)
            vertices.append(points)
            if "indices" in primitive:
                indices = read_accessor(primitive["indices"]).reshape(-1).astype(np.int64)
                triangles = indices.reshape(-1, 3) + base
                faces.append(triangles)
            base += len(points)

    if not vertices:
        raise UnsupportedSurface(f"{path.name} contains no mesh positions.")
    return (np.concatenate(vertices, axis=0),
            np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64))


def read_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles from a supported surface file."""
    source = Path(path)
    if not source.exists():
        raise UnsupportedSurface(f"Surface not found: {source}")

    suffix = source.suffix.lower()
    if suffix in UNSUPPORTED_SUFFIXES:
        raise UnsupportedSurface(f"{source.name}: {UNSUPPORTED_SUFFIXES[suffix]}")
    if suffix == ".obj":
        return read_mesh(source)
    if suffix == ".ply":
        return _read_ply(source)
    if suffix in {".glb", ".gltf"}:
        return _read_glb(source)
    raise UnsupportedSurface(
        f"{source.name} is not a surface this planner reads. Supported: "
        f"{', '.join(SUPPORTED_SUFFIXES)}."
    )


def _face_geometry(vertices: np.ndarray,
                   faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / np.linalg.norm(cross, axis=1)[:, None]
    centroids = (a + b + c) / 3.0
    return centroids, normals, areas


def plan_from_surface(
    surface_path: str | Path,
    *,
    standoff_m: float = 8.0,
    min_face_area_m2: float = 1.0,
    min_separation_m: float = 3.0,
    max_vertical_deg: float = 60.0,
    require_metric: bool = True,
) -> dict[str, Any]:
    """Capture poses standing off each face of the surface along its own normal.

    ``max_vertical_deg`` drops faces that point mostly up or down. On a building that
    leaves the walls, which is what a facade or structure survey is for; set it to 90 to
    keep roofs and soffits as well.
    """
    if standoff_m <= 0:
        raise ValueError("Stand-off must be positive.")
    if min_separation_m < 0:
        raise ValueError("Minimum separation cannot be negative.")

    source = Path(surface_path)
    scale = None
    if require_metric:
        try:
            scale = model_scale(source)
        except NotMetric as exc:
            raise NotMetric(
                f"{exc} A stand-off distance planned against an unscaled surface is a "
                "number of unknown units, so the aircraft would fly the wrong distance "
                "from the structure."
            ) from exc

    vertices, faces = read_surface(source)
    if len(faces) == 0:
        raise NoUsableSurface(
            f"{source.name} carries points but no faces, so no surface normal exists to "
            "stand off along. A path generated from it would invent the orientation of "
            "every surface it claims to photograph."
        )

    centroids, normals, areas = _face_geometry(vertices, faces)
    keep = np.isfinite(normals).all(axis=1) & (areas >= min_face_area_m2)
    vertical = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))
    # vertical is the angle between the normal and the horizontal plane's perpendicular:
    # a wall normal is horizontal, giving 90; a roof normal points up, giving 0.
    keep &= (90.0 - vertical) <= max_vertical_deg

    kept = np.nonzero(keep)[0]
    if kept.size == 0:
        raise NoUsableSurface(
            f"No face of {source.name} is both larger than {min_face_area_m2} m2 and "
            f"within {max_vertical_deg} degrees of vertical. Lower the minimum face "
            "area, or widen the angle to include roofs and soffits."
        )

    # Largest faces first, so thinning keeps the poses that see the most surface.
    order = kept[np.argsort(-areas[kept])]
    poses: list[SurfacePose] = []
    placed: list[np.ndarray] = []
    for index in order:
        position = centroids[index] + normals[index] * standoff_m
        if min_separation_m > 0 and placed:
            distances = np.linalg.norm(np.asarray(placed) - position, axis=1)
            if float(distances.min()) < min_separation_m:
                continue
        placed.append(position)

        # Look back down the normal at the face.
        look = -normals[index]
        yaw = math.degrees(math.atan2(look[0], look[1])) % 360.0
        horizontal = float(math.hypot(look[0], look[1]))
        pitch = math.degrees(math.atan2(look[2], horizontal))
        poses.append(SurfacePose(
            x_m=float(position[0]), y_m=float(position[1]), z_m=float(position[2]),
            yaw_deg=yaw, gimbal_pitch_deg=pitch,
            face_index=int(index), face_area_m2=float(areas[index]),
            standoff_m=float(standoff_m),
        ))

    poses.sort(key=lambda pose: pose.z_m)
    covered = float(areas[[p.face_index for p in poses]].sum())
    return {
        "source": str(source),
        "crs_epsg": None if scale is None else scale.epsg,
        "pose_count": len(poses),
        "poses": [pose.to_dict() for pose in poses],
        "face_count": int(len(faces)),
        "faces_considered": int(kept.size),
        "surface_area_m2": round(float(areas.sum()), 3),
        "covered_area_m2": round(covered, 3),
        "standoff_m": standoff_m,
        "limits": [
            "Capture points are placed along each face's own normal, so the plan follows "
            "the surface as reconstructed -- including anything the reconstruction "
            "bridged or smoothed.",
            f"{int(kept.size)} of {len(faces)} faces met the size and orientation "
            "filters; the rest are not photographed by this plan.",
            "Stand-off is measured from the face centroid, not from the nearest point of "
            "the structure. On a strongly curved surface the nearest obstacle is closer "
            "than the stand-off suggests.",
        ],
    }
