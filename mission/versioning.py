"""Comparing and restoring saved mission versions.

Storing versions is the easy half. The half that gets used is answering "what changed
since last time" -- usually on site, when a mission has been edited once too often and
somebody needs to know whether this is the plan that was signed off.

A raw JSON diff does not answer that. Two recipes differ in dozens of derived fields
whenever one input changes, so the reader has to work out which difference was the
decision and which were its consequences. This produces the operator's version instead:
altitude went from 60 m to 75 m, which cost 48 photographs and two minutes of flying.

Restoring never overwrites. Rolling back to version 3 writes a new version 7 whose
content matches 3, so the record of what was flown in between survives. A rollback that
erased history would make the audit trail unreliable exactly when it matters, which is
after something went wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Fields worth naming in a summary, with how to describe them. Everything else still
# appears in the raw change list, but these are the ones that change what gets flown.
SIGNIFICANT_FIELDS: dict[str, tuple[str, str]] = {
    "altitude_m": ("Altitude", "m"),
    "front_overlap_pct": ("Front overlap", "%"),
    "side_overlap_pct": ("Side overlap", "%"),
    "estimated_gsd_cm": ("Ground sample distance", "cm/px"),
    "estimated_time_min": ("Flight time", "min"),
    "path_distance_m": ("Path distance", "m"),
    "camera": ("Camera", ""),
    "speed_m_s": ("Speed", "m/s"),
    "template": ("Template", ""),
    "flight_direction_deg": ("Flight direction", "deg"),
    "line_spacing_m": ("Line spacing", "m"),
    "capture_spacing_m": ("Capture spacing", "m"),
    "facade_standoff_m": ("Stand-off", "m"),
    "terrain_follow_enabled": ("Terrain following", ""),
}

# Below this a float difference is rounding, not a change an operator made.
FLOAT_EPSILON = 1e-6


@dataclass
class FieldChange:
    """One field that differs between two versions."""

    field: str
    label: str
    before: Any
    after: Any
    unit: str = ""

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        before = _format(self.before)
        after = _format(self.after)
        return f"{self.label}: {before}{unit} -> {after}{unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "label": self.label, "unit": self.unit,
            "before": self.before, "after": self.after,
            "description": self.describe(),
        }


@dataclass
class MissionDiff:
    """What changed between two saved versions of a mission."""

    from_version: int
    to_version: int
    significant: list[FieldChange] = field(default_factory=list)
    other: list[FieldChange] = field(default_factory=list)
    waypoints_before: int = 0
    waypoints_after: int = 0
    geofence_changed: bool = False
    no_fly_changed: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.significant or self.other or self.waypoint_delta
                    or self.geofence_changed or self.no_fly_changed)

    @property
    def waypoint_delta(self) -> int:
        return self.waypoints_after - self.waypoints_before

    def summary(self) -> list[str]:
        """The change described the way an operator would say it."""
        lines = [change.describe() for change in self.significant]

        if self.waypoint_delta:
            direction = "more" if self.waypoint_delta > 0 else "fewer"
            lines.append(
                f"Waypoints: {self.waypoints_before} -> {self.waypoints_after} "
                f"({abs(self.waypoint_delta)} {direction})"
            )
        if self.geofence_changed:
            lines.append("The area of interest was redrawn.")
        if self.no_fly_changed:
            lines.append("No-fly zones changed, which may alter the route.")
        if not lines and self.other:
            lines.append(
                f"{len(self.other)} field(s) changed, none of which alter what is flown."
            )
        if not lines:
            lines.append("No differences.")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "identical": self.is_empty,
            "summary": self.summary(),
            "significant": [c.to_dict() for c in self.significant],
            "other": [c.to_dict() for c in self.other],
            "waypoints_before": self.waypoints_before,
            "waypoints_after": self.waypoints_after,
            "waypoint_delta": self.waypoint_delta,
            "geofence_changed": self.geofence_changed,
            "no_fly_changed": self.no_fly_changed,
        }


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value is None:
        return "unset"
    return str(value)


def _differs(before: Any, after: Any) -> bool:
    if isinstance(before, float) or isinstance(after, float):
        try:
            return abs(float(before) - float(after)) > FLOAT_EPSILON
        except (TypeError, ValueError):
            return before != after
    return before != after


def _summary_of(version: dict[str, Any]) -> dict[str, Any]:
    """The plan summary, whether it arrives nested or already unwrapped."""
    for key in ("plan_summary", "plan_summary_json"):
        value = version.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            import json

            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return version


def _waypoint_count(summary: dict[str, Any]) -> int:
    """How many waypoints a summary records, whichever way it records them.

    save_mission stores the COUNT (plan_summary.waypoints is an int); an unwrapped plan
    carries the list. len() on the count raised TypeError, so once a mission had two
    saved versions, history and compare both failed -- and the cockpit, reading that as
    "nothing", told the operator there were no saved versions.
    """
    value = summary.get("waypoints")
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    count = summary.get("waypoint_count")
    return int(count) if isinstance(count, (int, float)) else 0


def _with_template(version: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """The summary plus the template, which is stored beside it rather than in it."""
    if "template" in summary or not version.get("template"):
        return summary
    return {**summary, "template": version.get("template")}


def diff_versions(older: dict[str, Any], newer: dict[str, Any]) -> MissionDiff:
    """Compare two saved mission versions."""
    before = _with_template(older, _summary_of(older))
    after = _with_template(newer, _summary_of(newer))

    result = MissionDiff(
        from_version=int(older.get("version_num", 0) or 0),
        to_version=int(newer.get("version_num", 0) or 0),
        waypoints_before=_waypoint_count(before),
        waypoints_after=_waypoint_count(after),
    )

    # Geometry is compared as a whole: a redrawn boundary is one decision, not one
    # change per vertex.
    result.geofence_changed = (before.get("polygon") or []) != (after.get("polygon") or [])
    constraints_before = before.get("safety_constraints") or {}
    constraints_after = after.get("safety_constraints") or {}
    result.no_fly_changed = (
        (constraints_before.get("no_fly_polygons") or [])
        != (constraints_after.get("no_fly_polygons") or [])
    )

    skip = {"waypoints", "waypoint_count", "polygon", "geojson", "flight_recipe", "autopilot_commands",
            "safety_constraints", "expected_coverage"}

    for key in sorted(set(before) | set(after)):
        if key in skip:
            continue
        old_value, new_value = before.get(key), after.get(key)
        if not _differs(old_value, new_value):
            continue

        if key in SIGNIFICANT_FIELDS:
            label, unit = SIGNIFICANT_FIELDS[key]
            result.significant.append(
                FieldChange(key, label, old_value, new_value, unit))
        else:
            result.other.append(
                FieldChange(key, key.replace("_", " ").capitalize(), old_value, new_value))

    return result


def restore_version(store: Any, project_id: int, version: dict[str, Any],
                    note: str = "") -> dict[str, Any]:
    """Reinstate an earlier version by saving it again as the newest one.

    History is append-only. Rolling back to version 3 creates version 7 with version 3's
    content, so what was flown between them remains on the record. A rollback that
    deleted the intervening versions would break the audit trail precisely when it is
    needed, which is after an incident.
    """
    source_version = int(version.get("version_num", 0) or 0)
    recipe = version.get("flight_recipe") or version.get("flight_recipe_json") or {}
    summary = _summary_of(version)

    if isinstance(recipe, str):
        import json

        try:
            recipe = json.loads(recipe)
        except json.JSONDecodeError:
            recipe = {}

    explanation = note or f"Restored from version {source_version}."

    return store.save_mission_version(
        project_id=project_id,
        mission_name=str(version.get("mission_name") or "mission"),
        template=str(version.get("template") or "grid"),
        flight_recipe=recipe,
        plan_summary=summary,
        note=explanation,
        parent_version_id=version.get("id"),
    )


def version_history(versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each version alongside what changed from the one before it.

    Version numbers count per mission name, so a project with two missions has two
    version 1s. Each version is compared with the previous version OF THE SAME MISSION;
    comparing "North face v1" with "Roof v1" described a change nobody made.
    """
    ordered = sorted(versions, key=lambda v: (str(v.get("mission_name") or ""),
                                              int(v.get("version_num", 0) or 0)))
    history: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] = {}

    for version in ordered:
        name = str(version.get("mission_name") or "")
        summary = _summary_of(version)
        entry = {
            "mission_name": name,
            "version_num": version.get("version_num"),
            "id": version.get("id"),
            "created_at": version.get("created_at"),
            "note": version.get("note", ""),
            "template": version.get("template"),
            "waypoints": _waypoint_count(summary),
            "altitude_m": summary.get("altitude_m"),
        }
        before = previous.get(name)
        if before is None:
            entry["changes"] = ["First saved version."]
        else:
            entry["changes"] = diff_versions(before, version).summary()
        previous[name] = version
        history.append(entry)

    return history
