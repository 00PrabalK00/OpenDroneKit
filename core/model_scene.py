"""The reconstruction as something a viewer can draw, orbit, pick on and measure.

The cockpit's Digital Twin had "Textured Mesh" and "Point Cloud" buttons and nothing
behind them: both fell back to a 2D raster preview, and Save View, Clip and Measure in
3D said the viewer was not built. The products they needed were already on disk --
``reconstruction.ply`` (the cloud COLMAP registered) and ``mesh.ply``/``mesh.obj`` (the
Poisson surface) -- and the saved-view and clip stores were already written against
them. What was missing was a way to get the geometry into the page.

Three rules this module keeps:

* Coordinates are the MODEL FILE's own. Saved views and clips are stored in that frame
  (``export_clipped_model`` cuts ``mesh.obj`` with them), so a viewer working in any
  other frame would save views that open somewhere else and cuts that remove the wrong
  half of the building.
* Nothing is thinned silently. A browser cannot draw every point of a large cloud, so
  the scene carries both the count on disk and the count sent, and says how it was
  reduced.
* A length is metres only when something made it metres. COLMAP writes the cloud in
  structure-from-motion units and records, in ``geo_anchor.json``, the similarity
  transform that turns those units into a projected CRS. That transform is the licence
  to measure; without it a measurement is refused rather than printed in units that look
  like metres and are not.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Which files answer each view, best first. The cloud COLMAP registered is
#: ``reconstruction.ply``; a dense run leaves ``dense/fused.ply``; the custom engine
#: writes ``custom_reconstruction.ply``.
CANDIDATES = {
    "pointcloud": ("reconstruction.ply", "dense/fused.ply", "custom_reconstruction.ply"),
    "mesh": ("mesh.ply", "textured_mesh.obj", "mesh.obj"),
}

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
}


class SceneRefused(ValueError):
    """There is no model to show, or the request cannot be answered honestly."""


# ─────────────────────────────────────────────────────────────── reading models


def find_model(recon_dir: str | Path, kind: str) -> Path | None:
    """The file that answers `kind` ("pointcloud" or "mesh"), or None."""
    if kind not in CANDIDATES:
        raise SceneRefused(f"Unknown 3D view {kind!r}. Use pointcloud or mesh.")
    root = Path(recon_dir)
    for name in CANDIDATES[kind]:
        path = root / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Vertices (N,3), vertex colours (N,3) uint8 or None, and triangles (M,3).

    ASCII and binary, any vertex layout, faces as a vertex_indices list. Polygons larger
    than a triangle are fan-triangulated. Elements other than vertex and face are skipped
    by size, so a file carrying an extra element still reads.
    """
    source = Path(path)
    with source.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise SceneRefused(f"{source.name} ends inside its PLY header.")
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                break
        if not header or header[0] != "ply":
            raise SceneRefused(f"{source.name} is not a PLY file.")

        fmt = "ascii"
        elements: list[dict[str, Any]] = []
        for text in header:
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
            elif parts[0] == "property" and elements:
                if parts[1] == "list":
                    elements[-1]["props"].append(("list", parts[4], parts[2], parts[3]))
                else:
                    elements[-1]["props"].append(("scalar", parts[2], parts[1]))

        vertices = np.zeros((0, 3))
        colors: np.ndarray | None = None
        faces = np.zeros((0, 3), dtype=np.int64)

        if fmt == "ascii":
            for element in elements:
                rows = [handle.readline().decode("ascii", errors="ignore").split()
                        for _ in range(element["count"])]
                if element["name"] == "vertex":
                    vertices, colors = _vertex_columns(
                        _ascii_columns(rows, element["props"]), source.name)
                elif element["name"] == "face":
                    faces = _triangulate([[int(v) for v in row[1:1 + int(row[0])]]
                                          for row in rows if row])
        else:
            endian = "<" if "little" in fmt else ">"
            for element in elements:
                scalar_only = all(p[0] == "scalar" for p in element["props"])
                if scalar_only:
                    dtype = np.dtype([(p[1], endian + _PLY_TYPES.get(p[2], "f4"))
                                      for p in element["props"]])
                    raw = handle.read(dtype.itemsize * element["count"])
                    data = np.frombuffer(raw, dtype=dtype, count=element["count"])
                    if element["name"] == "vertex":
                        vertices, colors = _vertex_columns(
                            {name: data[name] for name in data.dtype.names or ()}, source.name)
                    continue
                if element["name"] == "face" and len(element["props"]) == 1:
                    faces = _read_binary_faces(handle, element, endian)
                    continue
                raise SceneRefused(
                    f"{source.name} has a {element['name']!r} element this reader cannot "
                    "skip; convert it to a plain vertex/face PLY or OBJ.")

    if len(vertices) == 0:
        raise SceneRefused(f"{source.name} contains no vertices.")
    return vertices, colors, faces


def _ascii_columns(rows: list[list[str]], props: list[tuple]) -> dict[str, np.ndarray]:
    names = [p[1] for p in props if p[0] == "scalar"]
    table = np.asarray([[float(v) for v in row[:len(names)]] for row in rows if row],
                       dtype=np.float64)
    if table.size == 0:
        return {}
    return {name: table[:, index] for index, name in enumerate(names) if index < table.shape[1]}


def _vertex_columns(columns: dict[str, np.ndarray],
                    name: str) -> tuple[np.ndarray, np.ndarray | None]:
    if not all(axis in columns for axis in ("x", "y", "z")):
        raise SceneRefused(f"{name} has no x/y/z vertex coordinates.")
    xyz = np.stack([np.asarray(columns[a], dtype=np.float64) for a in ("x", "y", "z")], axis=1)
    rgb = None
    for keys in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(k in columns for k in keys):
            channels = np.stack([np.asarray(columns[k], dtype=np.float64) for k in keys], axis=1)
            if channels.size and channels.max() <= 1.0:
                channels = channels * 255.0
            rgb = np.clip(channels, 0, 255).astype(np.uint8)
            break
    return xyz, rgb


def _read_binary_faces(handle, element: dict[str, Any], endian: str) -> np.ndarray:
    _, _name, count_type, index_type = element["props"][0]
    count_dtype = np.dtype(endian + _PLY_TYPES.get(count_type, "u1"))
    index_dtype = np.dtype(endian + _PLY_TYPES.get(index_type, "i4"))
    total = element["count"]
    if total == 0:
        return np.zeros((0, 3), dtype=np.int64)
    # Fast path: every face a triangle, which is what Open3D and COLMAP write.
    triangle = np.dtype([("n", count_dtype), ("v", index_dtype, (3,))])
    start = handle.tell()
    raw = handle.read(triangle.itemsize * total)
    if len(raw) == triangle.itemsize * total:
        data = np.frombuffer(raw, dtype=triangle, count=total)
        if np.all(data["n"] == 3):
            return data["v"].astype(np.int64)
    # General path: polygons of any size, read one at a time.
    handle.seek(start)
    polygons: list[list[int]] = []
    for _ in range(total):
        n = int(np.frombuffer(handle.read(count_dtype.itemsize), dtype=count_dtype)[0])
        polygons.append(np.frombuffer(handle.read(index_dtype.itemsize * n),
                                      dtype=index_dtype).astype(int).tolist())
    return _triangulate(polygons)


def _triangulate(polygons: list[list[int]]) -> np.ndarray:
    triangles = [[poly[0], poly[i], poly[i + 1]]
                 for poly in polygons if len(poly) >= 3 for i in range(1, len(poly) - 1)]
    return np.asarray(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)


def read_model(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Vertices, colours and triangles from a PLY or an OBJ."""
    source = Path(path)
    if source.suffix.lower() == ".obj":
        from .model_measurement import read_mesh

        vertices, faces = read_mesh(source)
        return vertices, None, faces
    return read_ply(source)


# ───────────────────────────────────────────────────────────────── the anchor


def load_anchor(recon_dir: str | Path):
    """The SfM-to-CRS transform COLMAP recorded, or None when the model is unscaled."""
    path = Path(recon_dir) / "geo_anchor.json"
    if not path.is_file():
        return None
    from .geo import GeoAnchor

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return GeoAnchor(**{k: raw[k] for k in GeoAnchor.__dataclass_fields__ if k in raw})
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


# ────────────────────────────────────────────────────────────────── the scene


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def build_scene(recon_dir: str | Path, kind: str, *, max_points: int = 300_000,
                max_faces: int = 200_000) -> dict[str, Any]:
    """The model for `kind`, clipped by the visible clips and sized for a browser.

    Positions are float32 relative to `center` (file coordinates = position + center), so
    a model already in a projected CRS keeps centimetre precision in a float32 buffer.
    """
    from .model_clipping import ClipStore, combined_mask

    recon = Path(recon_dir)
    source = find_model(recon, kind)
    if source is None:
        wanted = " or ".join(CANDIDATES[kind])
        raise SceneRefused(
            f"No {'mesh' if kind == 'mesh' else 'point cloud'} in this project yet "
            f"({wanted} in the reconstruction folder). Import a dataset and run Process.")

    vertices, colors, faces = read_model(source)
    if kind == "pointcloud":
        faces = np.zeros((0, 3), dtype=np.int64)
    source_vertices, source_faces = int(len(vertices)), int(len(faces))

    clips = [c for c in ClipStore(recon).load() if c.visible]
    if clips:
        if len(faces):
            # The same whole-triangle rule as clip_mesh, which export_clipped_model uses,
            # so what is on screen is what Export writes -- with colours carried along.
            keep = combined_mask(vertices, clips)
            kept_faces = faces[keep[faces].all(axis=1)]
            used = np.unique(kept_faces)
            remap = np.full(len(vertices), -1, dtype=np.int64)
            remap[used] = np.arange(len(used), dtype=np.int64)
            vertices, faces = vertices[used], remap[kept_faces]
            colors = colors[used] if colors is not None else None
        else:
            keep = combined_mask(vertices, clips)
            vertices = vertices[keep]
            colors = colors[keep] if colors is not None else None
    if len(vertices) == 0:
        raise SceneRefused("The visible clips remove the entire model. Hide or remove a clip.")

    reduction = ""
    if len(faces) > max_faces:
        vertices, colors, faces, reduction = _simplify(vertices, colors, faces, max_faces)
    elif not len(faces) and len(vertices) > max_points:
        step = math.ceil(len(vertices) / max_points)
        vertices = vertices[::step]
        colors = colors[::step] if colors is not None else None
        reduction = f"every {step}th point"

    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    local = (vertices - center).astype("<f4")
    radius = float(np.linalg.norm(local, axis=1).max()) if len(local) else 1.0

    anchor = load_anchor(recon)
    if anchor is not None:
        # The direction in the file's frame that the anchor maps to "up" in the CRS.
        up = (anchor.rotation_matrix.T @ np.array([0.0, 0.0, 1.0])).tolist()
        georef = {"epsg": int(anchor.epsg), "scale": float(anchor.scale),
                  "rmse_m": float(anchor.rmse_m), "method": anchor.method}
    else:
        up = [0.0, 0.0, 1.0]
        georef = None

    return {
        "kind": kind,
        "source": source.name,
        "center": [float(v) for v in center],
        "radius": radius,
        "up": [float(v) for v in up],
        "vertex_count": int(len(local)),
        "face_count": int(len(faces)),
        "source_vertex_count": source_vertices,
        "source_face_count": source_faces,
        "reduction": reduction,
        "clips_applied": [c.name for c in clips],
        "positions": _b64(local),
        "colors": _b64(colors.astype(np.uint8)) if colors is not None else None,
        "indices": _b64(faces.astype("<u4")) if len(faces) else None,
        "georeferenced": georef,
        "units": "m" if georef else "sfm",
    }


def _simplify(vertices, colors, faces, max_faces):
    """Fewer triangles for the screen, by quadric decimation when Open3D is present.

    Without Open3D the surface is shown as its first `max_faces` triangles and says so:
    dropping every other triangle would draw a lace curtain and call it the model.
    """
    try:
        import open3d as o3d  # type: ignore

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces.astype(np.int32)))
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        simple = mesh.simplify_quadric_decimation(int(max_faces))
        new_colors = None
        if colors is not None and simple.has_vertex_colors():
            new_colors = np.clip(np.asarray(simple.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        return (np.asarray(simple.vertices), new_colors,
                np.asarray(simple.triangles).astype(np.int64),
                f"quadric decimation to {len(simple.triangles):,} of {len(faces):,} triangles")
    except Exception:  # noqa: BLE001 -- open3d missing or failed; the fallback says so
        kept = faces[:max_faces]
        used = np.unique(kept)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        return (vertices[used], colors[used] if colors is not None else None, remap[kept],
                f"first {max_faces:,} of {len(faces):,} triangles only (Open3D unavailable "
                "to simplify the surface)")


# ─────────────────────────────────────────────────────────────── measurement


def measure_scene(recon_dir: str | Path, kind: str,
                  points_xyz: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Length, height or area between points picked on the model, in metres.

    The points are in the model file's frame. They are carried into the CRS through the
    recorded anchor, and the anchor's own fit error is reported beside the answer: a
    1 m RMSE georeference cannot support a millimetre claim.
    """
    anchor = load_anchor(recon_dir)
    if anchor is None:
        raise SceneRefused(
            "This model has no georeference (geo_anchor.json), so it is in "
            "structure-from-motion units that only look like metres. Reconstruct from "
            "geotagged photographs, or add ground control, to measure on it.")
    pts = np.asarray([[float(v) for v in p[:3]] for p in points_xyz if len(p) >= 3],
                     dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        raise SceneRefused("Measurement points must be finite.")
    metric = anchor.apply(pts, absolute=True)
    limits = [
        f"Georeferenced by {anchor.method} with {anchor.rmse_m:.2f} m RMSE across "
        f"{anchor.inlier_count}/{anchor.sample_count} cameras; absolute positions carry "
        "that error, and picked points carry the reconstruction's own noise.",
    ]
    base = {"crs_epsg": int(anchor.epsg), "georeference_rmse_m": round(float(anchor.rmse_m), 3),
            "points_crs": [[round(float(v), 3) for v in p] for p in metric], "limits": limits}

    if kind in {"length", "height"}:
        if len(metric) < 2:
            raise SceneRefused(f"A {kind} needs two picked points.")
        delta = metric[1] - metric[0]
        horizontal = float(math.hypot(delta[0], delta[1]))
        vertical = float(delta[2])
        return {**base, "kind": kind,
                "slope_distance_m": round(float(np.linalg.norm(delta)), 3),
                "horizontal_distance_m": round(horizontal, 3),
                "vertical_distance_m": round(vertical, 3),
                "inclination_deg": round(math.degrees(math.atan2(vertical, horizontal)), 2)
                if horizontal or vertical else 0.0}
    if kind == "area":
        if len(metric) < 3:
            raise SceneRefused("An area needs at least three picked points.")
        origin = metric[0]
        surface = 0.0
        for i in range(1, len(metric) - 1):
            surface += 0.5 * float(np.linalg.norm(np.cross(metric[i] - origin, metric[i + 1] - origin)))
        x, y = metric[:, 0], metric[:, 1]
        plan = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return {**base, "kind": "area", "surface_area_m2": round(surface, 3),
                "planimetric_area_m2": round(plan, 3),
                "limits": limits + ["The surface area is the fan of triangles through the "
                                    "picked corners, not the mesh surface between them."]}
    raise SceneRefused(f"Unknown measurement {kind!r}. Use length, height or area.")
