"""Cut a reconstruction down to the asset that was surveyed.

A photogrammetric model arrives with everything the camera could see: the neighbouring
building, the car park, the road, the trees on the boundary. The client asked about one
structure. Handing them the raw model makes them find it, and makes every measurement
ambiguous about which roof is being measured.

Two cuts, because they answer different questions:

  polygon   Keep what is inside a footprint drawn on the plan. Removes context.
  plane     Keep what is on one side of an oriented plane. Isolates a facade or exposes
            a section through the structure.

Both are non-destructive. A clip is a named selection stored alongside the model, so the
same reconstruction can carry "clean building", "north facade" and "section at ridge
level" at once, and deleting a clip cannot lose survey data. That matters more than it
sounds: a destructive clip on the only copy of a model turns a presentation decision into
data loss.

No mesh library is used here, for the reason core/model_measurement.py gives for its own
OBJ reader: a deliverable that only works when open3d happens to be installed is not one
the field can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class ClipRefused(ValueError):
    """A clip that cannot be made, or would not mean anything if it were."""


@dataclass(frozen=True)
class ClipPlane:
    """An oriented plane. Points where (p - point) . normal >= 0 are kept.

    Heading, pitch and roll are the operator-facing controls; the normal is derived from
    them so the two can never disagree.
    """

    point: tuple[float, float, float]
    normal: tuple[float, float, float]

    @staticmethod
    def from_orientation(point: Sequence[float], heading_deg: float,
                         pitch_deg: float = 0.0, roll_deg: float = 0.0) -> "ClipPlane":
        """Build a plane from the heading/pitch/roll a viewer exposes as drag handles.

        Heading rotates about Z, pitch about Y, roll about X, applied in that order. Roll
        about the normal itself does not change the cut -- it is accepted because the UI
        offers three handles and refusing one of them would be surprising, not because it
        moves anything.
        """
        if len(point) < 3:
            raise ClipRefused("A clip plane needs a point in the model as [x, y, z].")
        h = math.radians(float(heading_deg))
        p = math.radians(float(pitch_deg))
        # Start pointing along +X, then heading about Z and pitch about Y.
        nx = math.cos(p) * math.cos(h)
        ny = math.cos(p) * math.sin(h)
        nz = -math.sin(p)
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-9:
            raise ClipRefused("That orientation does not define a plane.")
        return ClipPlane(
            point=(float(point[0]), float(point[1]), float(point[2])),
            normal=(nx / length, ny / length, nz / length),
        )


@dataclass
class Clip:
    """A named, non-destructive selection over a model."""

    name: str
    kind: str                       # "polygon" | "plane"
    polygon_xy: list[list[float]] = field(default_factory=list)
    plane: ClipPlane | None = None
    visible: bool = True
    created_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "polygon_xy": [list(p) for p in self.polygon_xy],
            "plane": (
                {"point": list(self.plane.point), "normal": list(self.plane.normal)}
                if self.plane else None
            ),
            "visible": bool(self.visible),
            "created_utc": self.created_utc,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Clip":
        plane_raw = raw.get("plane")
        plane = None
        if plane_raw:
            plane = ClipPlane(
                point=tuple(float(v) for v in plane_raw["point"]),      # type: ignore[arg-type]
                normal=tuple(float(v) for v in plane_raw["normal"]),    # type: ignore[arg-type]
            )
        return Clip(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "polygon")),
            polygon_xy=[list(map(float, p)) for p in (raw.get("polygon_xy") or [])],
            plane=plane,
            visible=bool(raw.get("visible", True)),
            created_utc=str(raw.get("created_utc", "")),
        )


def _closed(ring: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray([[float(p[0]), float(p[1])] for p in ring], dtype=np.float64)
    if len(pts) >= 2 and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    return pts


def points_inside_polygon(points_xyz: np.ndarray, polygon_xy: Sequence[Sequence[float]]) -> np.ndarray:
    """Boolean mask of the points whose XY falls inside the ring.

    Ray casting, vectorised over all points at once: a reconstruction is millions of
    points and a per-point Python loop would make clipping feel broken rather than slow.
    Points exactly on an edge are kept -- a vertex of the drawn footprint should not fall
    out of its own selection.
    """
    ring = _closed(polygon_xy)
    if len(ring) < 4:
        raise ClipRefused("A clip polygon needs at least three distinct corners.")

    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ClipRefused("Points must be an (N, 2) or (N, 3) array.")

    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)

    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        # Does the horizontal ray from the point cross this edge?
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_x = (bx - ax) * (y - ay) / np.where(by != ay, by - ay, np.nan) + ax
        inside ^= straddles & (x < cross_x)
    return inside


def points_in_front_of_plane(points_xyz: np.ndarray, plane: ClipPlane) -> np.ndarray:
    """Boolean mask of points on the keep side of an oriented plane."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ClipRefused("A plane clip needs three-dimensional points.")
    origin = np.asarray(plane.point, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    return ((pts[:, :3] - origin) @ normal) >= 0.0


def apply_clip(points_xyz: np.ndarray, clip: Clip) -> np.ndarray:
    """The mask a single clip selects."""
    if clip.kind == "polygon":
        return points_inside_polygon(points_xyz, clip.polygon_xy)
    if clip.kind == "plane":
        if clip.plane is None:
            raise ClipRefused(f"Clip {clip.name!r} is a plane clip with no plane.")
        return points_in_front_of_plane(points_xyz, clip.plane)
    raise ClipRefused(f"Unknown clip kind: {clip.kind!r}")


def combined_mask(points_xyz: np.ndarray, clips: Iterable[Clip]) -> np.ndarray:
    """What survives every VISIBLE clip.

    Intersection, not union: clips narrow the selection. A polygon around the building
    and a plane at ridge level together mean "this building, below the ridge", which is
    what an operator drawing both of them is asking for. Union would make the second clip
    widen the result, so adding detail would show MORE of the neighbour's car park.

    Hidden clips are ignored rather than deleted, which is what makes a clip a view
    rather than an edit.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    mask = np.ones(len(pts), dtype=bool)
    for clip in clips:
        if not clip.visible:
            continue
        mask &= apply_clip(pts, clip)
    return mask


def clip_mesh(vertices: np.ndarray, faces: np.ndarray,
              clips: Iterable[Clip]) -> tuple[np.ndarray, np.ndarray]:
    """Keep the triangles whose vertices all survive, and reindex.

    A triangle is kept only when every corner is inside. Keeping partly-inside triangles
    leaves geometry hanging outside the boundary the operator drew, and splitting them
    invents vertices that were never measured -- so the cut is on whole triangles and the
    edge is where the data actually stops.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0:
        return verts, tris

    keep_vertex = combined_mask(verts, clips)
    if tris.size == 0:
        return verts[keep_vertex], tris

    keep_face = keep_vertex[tris].all(axis=1)
    kept_tris = tris[keep_face]

    # Reindex: drop vertices no surviving triangle refers to.
    used = np.unique(kept_tris)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return verts[used], remap[kept_tris]


class ClipStore:
    """The clips saved against one model, on disk beside it.

    JSON rather than anything cleverer: a clip is a handful of numbers and a name, and an
    operator should be able to read the file to see what a deliverable was cut to.
    """

    FILENAME = "clips.json"

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / self.FILENAME

    def load(self) -> list[Clip]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [Clip.from_dict(entry) for entry in raw.get("clips", [])]

    def save(self, clips: Sequence[Clip]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"clips": [c.to_dict() for c in clips]}, indent=2),
            encoding="utf-8",
        )

    def add(self, clip: Clip) -> list[Clip]:
        if not clip.name.strip():
            raise ClipRefused("A clip needs a name; that is how it is found again.")
        clips = [c for c in self.load() if c.name != clip.name]
        clip.created_utc = clip.created_utc or datetime.now(timezone.utc).isoformat()
        clips.append(clip)
        self.save(clips)
        return clips

    def remove(self, name: str) -> list[Clip]:
        clips = self.load()
        remaining = [c for c in clips if c.name != name]
        if len(remaining) == len(clips):
            raise ClipRefused(f"No clip named {name!r}.")
        self.save(remaining)
        return remaining

    def set_visible(self, name: str, visible: bool) -> list[Clip]:
        clips = self.load()
        for clip in clips:
            if clip.name == name:
                clip.visible = bool(visible)
                self.save(clips)
                return clips
        raise ClipRefused(f"No clip named {name!r}.")
