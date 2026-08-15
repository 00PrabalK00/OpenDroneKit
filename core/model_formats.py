"""Interoperable 3D mesh and point-cloud exports.

Mesh formats preserve the projected coordinate frame as explicit OpenDroneKit extras;
LAS/LAZ use the standard CRS VLR. LAZ is real compressed LAS data written by lazrs,
never an uncompressed LAS file with a misleading extension.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelGeometry:
    vertices: np.ndarray
    faces: np.ndarray | None = None
    colors: np.ndarray | None = None
    crs_epsg: int | None = None


def _validated(geometry: ModelGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("3D export requires a non-empty Nx3 vertex array.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("3D export vertices must be finite.")
    faces = np.empty((0, 3), dtype=np.uint32) if geometry.faces is None else np.asarray(
        geometry.faces, dtype=np.int64
    )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be an Mx3 triangle array.")
    if faces.size and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Mesh face index lies outside the vertex array.")
    colors = None
    if geometry.colors is not None:
        source = np.asarray(geometry.colors)
        if source.shape != vertices.shape:
            raise ValueError("Vertex colors must match the Nx3 vertex array.")
        if np.issubdtype(source.dtype, np.floating):
            if not np.all(np.isfinite(source)) or np.any(source < 0) or np.any(source > 1):
                raise ValueError("Floating vertex colors must be finite values from 0 to 1.")
            colors = np.rint(source * 255).astype(np.uint8)
        else:
            if np.any(source < 0) or np.any(source > 255):
                raise ValueError("Integer vertex colors must be values from 0 to 255.")
            colors = source.astype(np.uint8)
    return vertices, faces.astype(np.uint32), colors


def _crs_note(crs_epsg: int | None) -> str:
    return f"EPSG:{int(crs_epsg)}" if crs_epsg is not None else "unreferenced"


def write_obj(path: str | Path, geometry: ModelGeometry) -> str:
    vertices, faces, _ = _validated(geometry)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# OpenDroneKit 3D export\n")
        stream.write(f"# ODK_CRS={_crs_note(geometry.crs_epsg)}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
        for a, b, c in faces:
            stream.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
    return str(target)


def write_ply(path: str | Path, geometry: ModelGeometry) -> str:
    vertices, faces, colors = _validated(geometry)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"comment ODK_CRS {_crs_note(geometry.crs_epsg)}\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        if colors is not None:
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {len(faces)}\n")
        stream.write("property list uchar uint vertex_indices\nend_header\n")
        for index, (x, y, z) in enumerate(vertices):
            color = "" if colors is None else " " + " ".join(str(int(v)) for v in colors[index])
            stream.write(f"{x:.12g} {y:.12g} {z:.12g}{color}\n")
        for a, b, c in faces:
            stream.write(f"3 {int(a)} {int(b)} {int(c)}\n")
    return str(target)


def _gltf_parts(geometry: ModelGeometry, *, embedded: bool) -> tuple[dict[str, Any], bytes]:
    vertices, faces, colors = _validated(geometry)
    origin = np.mean(vertices, axis=0)
    positions = np.asarray(vertices - origin, dtype="<f4")
    chunks: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add(data: bytes, *, target: int, component_type: int, count: int,
            value_type: str, minimum=None, maximum=None, normalized=False) -> int:
        offset = sum(len(chunk) for chunk in chunks)
        padding = (-offset) % 4
        if padding:
            chunks.append(b"\x00" * padding)
            offset += padding
        chunks.append(data)
        view_index = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        accessor: dict[str, Any] = {
            "bufferView": view_index, "componentType": component_type,
            "count": int(count), "type": value_type,
        }
        if minimum is not None:
            accessor["min"] = [float(value) for value in minimum]
        if maximum is not None:
            accessor["max"] = [float(value) for value in maximum]
        if normalized:
            accessor["normalized"] = True
        accessors.append(accessor)
        return len(accessors) - 1

    position_accessor = add(
        positions.tobytes(), target=34962, component_type=5126, count=len(positions),
        value_type="VEC3", minimum=positions.min(axis=0), maximum=positions.max(axis=0),
    )
    attributes = {"POSITION": position_accessor}
    if colors is not None:
        attributes["COLOR_0"] = add(
            np.asarray(colors, dtype=np.uint8).tobytes(), target=34962,
            component_type=5121, count=len(colors), value_type="VEC3", normalized=True,
        )
    primitive: dict[str, Any] = {"attributes": attributes, "mode": 4 if len(faces) else 0}
    if len(faces):
        primitive["indices"] = add(
            np.asarray(faces.reshape(-1), dtype="<u4").tobytes(), target=34963,
            component_type=5125, count=faces.size, value_type="SCALAR",
            minimum=[int(faces.min())], maximum=[int(faces.max())],
        )
    binary = b"".join(chunks)
    crs = {
        "crs_epsg": int(geometry.crs_epsg) if geometry.crs_epsg is not None else None,
        "coordinate_origin": [float(value) for value in origin],
        "axis_units": "metre" if geometry.crs_epsg is not None else "source_units",
    }
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "OpenDroneKit", "extras": {"opendronekit": crs}},
        "scene": 0,
        "scenes": [{"nodes": [0], "extras": {"opendronekit": crs}}],
        "nodes": [{"mesh": 0, "translation": [float(value) for value in origin]}],
        "meshes": [{"primitives": [primitive]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    if embedded:
        document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binary).decode("ascii")
    return document, binary


def write_gltf(path: str | Path, geometry: ModelGeometry) -> str:
    document, _ = _gltf_parts(geometry, embedded=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    return str(target)


def write_glb(path: str | Path, geometry: ModelGeometry) -> str:
    document, binary = _gltf_parts(geometry, embedded=False)
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary_chunk = binary + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total))
        stream.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        stream.write(json_chunk)
        stream.write(struct.pack("<I4s", len(binary_chunk), b"BIN\x00"))
        stream.write(binary_chunk)
    return str(target)


def write_las(path: str | Path, geometry: ModelGeometry, *, compressed: bool = False) -> str:
    vertices, _, colors = _validated(geometry)
    if geometry.crs_epsg is None:
        raise ValueError("LAS/LAZ export requires an explicit CRS.")
    try:
        import laspy
        from pyproj import CRS
    except ImportError as exc:
        raise RuntimeError("LAS/LAZ export requires laspy with the lazrs extra.") from exc

    target = Path(path)
    should_compress = compressed or target.suffix.lower() == ".laz"
    if should_compress and target.suffix.lower() != ".laz":
        raise ValueError("Compressed point clouds must use the .laz extension.")
    if not should_compress and target.suffix.lower() != ".las":
        raise ValueError("Uncompressed point clouds must use the .las extension.")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(vertices.min(axis=0))
    header.add_crs(CRS.from_epsg(int(geometry.crs_epsg)))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    if colors is not None:
        cloud.red = colors[:, 0].astype(np.uint16) * 257
        cloud.green = colors[:, 1].astype(np.uint16) * 257
        cloud.blue = colors[:, 2].astype(np.uint16) * 257
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        cloud.write(target, do_compress=should_compress)
    except Exception as exc:
        if should_compress:
            raise RuntimeError(
                "LAZ compression is unavailable; install laspy[lazrs]. No disguised LAS file was written."
            ) from exc
        raise
    return str(target)


WRITERS = {
    ".obj": write_obj,
    ".ply": write_ply,
    ".gltf": write_gltf,
    ".glb": write_glb,
    ".las": lambda path, geometry: write_las(path, geometry, compressed=False),
    ".laz": lambda path, geometry: write_las(path, geometry, compressed=True),
}


def export_model(path: str | Path, geometry: ModelGeometry) -> str:
    target = Path(path)
    writer = WRITERS.get(target.suffix.lower())
    if writer is None:
        raise ValueError(f"Unsupported 3D format {target.suffix!r}. Supported: {', '.join(WRITERS)}.")
    return writer(target, geometry)
