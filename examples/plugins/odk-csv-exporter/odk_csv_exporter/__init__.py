"""Example OpenDroneKit plugin package: one exporter and one report section.

* ``waypoints_csv`` (exporter): the planned waypoints as a plain CSV that a
  spreadsheet, a GIS or a custom ground station can read.
* ``mission_waypoints`` (report section): a table of the project's saved missions,
  taken from the report's own context.

Both only read what OpenDroneKit hands them; neither computes a number of its own.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from opendronekit.plugins import PluginKind, PluginManifest, PluginSpec

__version__ = "0.1.0"


class WaypointCsvExporter:
    """``plan["waypoints"]`` ([lon, lat, alt] each) as one CSV row per waypoint."""

    suffix = ".waypoints.csv"

    def export(self, plan: dict[str, Any], path: Path) -> str:
        waypoints = plan.get("waypoints") or []
        if not waypoints:
            # Refuse rather than write a header-only file that looks like a mission.
            raise ValueError("The plan has no waypoints to export.")
        frame = str(plan.get("altitude_frame") or "relative_to_home")
        gimbal = plan.get("gimbal_tilt_deg")
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "longitude", "latitude", "altitude_m",
                             "altitude_frame", "gimbal_tilt_deg"])
            for index, point in enumerate(waypoints):
                lon, lat = float(point[0]), float(point[1])
                alt = float(point[2]) if len(point) > 2 else float(plan.get("altitude_m") or 0.0)
                writer.writerow([index, f"{lon:.8f}", f"{lat:.8f}", f"{alt:.2f}", frame,
                                 "" if gimbal is None else f"{float(gimbal):.1f}"])
        return str(path)


class MissionWaypointsSection:
    """A report block listing the project's saved missions, as the report knows them."""

    title = "Mission Waypoints (plugin)"

    def build(self, context: dict[str, Any]) -> dict[str, Any]:
        missions = context.get("missions") or []
        if not missions:
            return {"paragraphs": ["No mission has been saved in this project, so there "
                                   "are no waypoints to list."]}
        rows = []
        for m in missions:
            rows.append([str(m.get("name", "")), str(m.get("template", "") or ""),
                         str(m.get("waypoint_count", "")), str(m.get("altitude_m", "")),
                         str(m.get("path_distance_m", ""))])
        return {"paragraphs": [f"{len(missions)} saved mission(s) in this project."],
                "table": {"header": ["Mission", "Template", "Waypoints", "Altitude (m)",
                                     "Path (m)"], "rows": rows}}


manifest = PluginManifest(
    name="odk-csv-exporter",
    version=__version__,
    api_version="1.1",
    description="Waypoint CSV exporter and a mission report section.",
    license="MIT",
    plugins=(
        PluginSpec(PluginKind.EXPORTER, "waypoints_csv", WaypointCsvExporter,
                   api_version="1.1", description="Waypoints as CSV (lon, lat, alt)."),
        PluginSpec(PluginKind.REPORT_SECTION, "mission_waypoints", MissionWaypointsSection,
                   api_version="1.1", description="Table of saved missions."),
    ),
)
