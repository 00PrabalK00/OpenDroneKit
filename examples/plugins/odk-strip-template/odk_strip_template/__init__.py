"""Example OpenDroneKit plugin: a strip survey along a centreline.

``strip_survey`` (mission template) turns a centreline -- a road, a canal, a pipeline
-- and a strip width into an area of interest, and asks the built-in planner to fly it
as a high-overlap grid. The plugin only builds the area and chooses options; the
planner, its safety checks and its warnings are OpenDroneKit's own.

The buffer is computed in a local tangent plane centred on the line (metres east and
north of its mean point), which is accurate to well under a metre for strips a few
kilometres long. Longer corridors should be split.
"""

from __future__ import annotations

import math
from typing import Any

from opendronekit.plugins import PluginKind, PluginManifest, PluginSpec

__version__ = "0.1.0"

_EARTH_RADIUS_M = 6_378_137.0


def _to_local(points: list[list[float]], lat0: float, lon0: float) -> list[tuple[float, float]]:
    k = math.cos(math.radians(lat0))
    return [(math.radians(lon - lon0) * _EARTH_RADIUS_M * k,
             math.radians(lat - lat0) * _EARTH_RADIUS_M) for lon, lat in points]


def _to_lonlat(points: list[tuple[float, float]], lat0: float, lon0: float) -> list[list[float]]:
    k = math.cos(math.radians(lat0))
    return [[lon0 + math.degrees(x / (_EARTH_RADIUS_M * k)),
             lat0 + math.degrees(y / _EARTH_RADIUS_M)] for x, y in points]


def buffer_centreline(centreline_lonlat: list[list[float]], width_m: float) -> list[list[float]]:
    """A closed polygon ``width_m`` wide around the polyline (mitred joints)."""
    pts = [[float(p[0]), float(p[1])] for p in centreline_lonlat]
    if len(pts) < 2:
        raise ValueError("A strip needs a centreline of at least two [lon, lat] points.")
    if not width_m > 0:
        raise ValueError("width_m must be positive.")
    lon0 = sum(p[0] for p in pts) / len(pts)
    lat0 = sum(p[1] for p in pts) / len(pts)
    xy = _to_local(pts, lat0, lon0)
    half = width_m / 2.0
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(xy):
        normals = []
        for a, b in ((i - 1, i), (i, i + 1)):
            if 0 <= a and b < len(xy):
                dx, dy = xy[b][0] - xy[a][0], xy[b][1] - xy[a][1]
                length = math.hypot(dx, dy)
                if length == 0:
                    raise ValueError("The centreline repeats a point.")
                normals.append((-dy / length, dx / length))
        nx = sum(n[0] for n in normals)
        ny = sum(n[1] for n in normals)
        norm = math.hypot(nx, ny) or 1.0
        nx, ny = nx / norm, ny / norm
        # Mitre: keep the strip's full width at a bend (capped at 3x for sharp turns).
        cos_half = max(abs(nx * normals[0][0] + ny * normals[0][1]), 1.0 / 3.0)
        d = half / cos_half
        left.append((x + nx * d, y + ny * d))
        right.append((x - nx * d, y - ny * d))
    return _to_lonlat(left + right[::-1], lat0, lon0)


class StripSurveyTemplate:
    """Mission template: a strip of ``width_m`` along ``centreline_lonlat``."""

    def plan_options(self, aoi_lonlat: list[list[float]], **params: Any) -> dict[str, Any]:
        centreline = params.pop("centreline_lonlat", None)
        if not centreline:
            raise ValueError("strip_survey needs centreline_lonlat: the line to follow.")
        width = float(params.pop("width_m", 60.0))
        options: dict[str, Any] = {
            "template": "grid",
            "front_overlap_pct": 80.0,
            "side_overlap_pct": 70.0,
            "altitude_m": 60.0,
        }
        options.update(params)          # the caller's planner options win
        options["aoi_lonlat"] = buffer_centreline(centreline, width)
        return options


manifest = PluginManifest(
    name="odk-strip-template",
    version=__version__,
    api_version="1.1",
    description="Strip survey along a centreline.",
    license="MIT",
    plugins=(
        PluginSpec(PluginKind.MISSION_TYPE, "strip_survey", StripSurveyTemplate,
                   api_version="1.1",
                   description="High-overlap grid over a fixed-width strip along a line."),
    ),
)
